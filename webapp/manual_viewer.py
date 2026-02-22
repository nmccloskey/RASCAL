from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple, Union

import streamlit as st


# -----------------------------
# Data structures
# -----------------------------
@dataclass(frozen=True)
class ManualFile:
    rel_path: Path       # e.g., Path("02_policies/02_01_manuals.md")
    abs_path: Path       # full path
    title: str           # best-effort extracted title (H1 or filename)
    text: str            # full markdown text (cached)


TreeNode = Dict[str, Union["TreeNode", ManualFile]]


# -----------------------------
# Helpers: ordering / parsing
# -----------------------------
_NUM_PREFIX_RE = re.compile(r"^(\d+)[_-]")


def numeric_sort_key(name: str) -> Tuple[int, str]:
    """
    Order by leading numeric prefix if present, else after numbered items.
    Example: 01_overview.md < 02_policies < readme.md
    """
    m = _NUM_PREFIX_RE.match(name)
    if m:
        return (int(m.group(1)), name.lower())
    return (10_000, name.lower())


def extract_md_title(md_text: str, fallback: str) -> str:
    """
    Extract first Markdown H1 (# Title). If missing, use fallback.
    """
    for line in md_text.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def read_text_safely(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def is_markdown_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() == ".md"


# -----------------------------
# Indexing the manual directory
# -----------------------------
@st.cache_data(show_spinner=False)
def build_manual_index(manual_dir: str) -> Tuple[TreeNode, Dict[str, ManualFile]]:
    """
    Build:
      - a nested TreeNode for folder/file navigation
      - a flat index {rel_path_str -> ManualFile} for search + selection
    Cached because this can touch many files.
    """
    manual_root = Path(manual_dir).resolve()
    tree: TreeNode = {}
    flat: Dict[str, ManualFile] = {}

    if not manual_root.exists():
        return tree, flat

    # Walk manual directory, include *.md except maybe hidden/temp
    md_paths = sorted(
        [p for p in manual_root.rglob("*.md") if p.is_file()],
        key=lambda p: [numeric_sort_key(part) for part in p.relative_to(manual_root).parts],
    )

    for abs_path in md_paths:
        rel_path = abs_path.relative_to(manual_root)
        rel_str = rel_path.as_posix()

        text = read_text_safely(abs_path)
        title = extract_md_title(text, fallback=abs_path.name)

        mf = ManualFile(rel_path=rel_path, abs_path=abs_path, title=title, text=text)
        flat[rel_str] = mf

        # Insert into tree
        cursor: TreeNode = tree
        parts = list(rel_path.parts)
        for part in parts[:-1]:
            if part not in cursor or not isinstance(cursor[part], dict):
                cursor[part] = {}
            cursor = cursor[part]  # type: ignore[assignment]
        cursor[parts[-1]] = mf

    return tree, flat


# -----------------------------
# Optional: parse tree block from 00_outline.md
# -----------------------------
def extract_tree_block(outline_text: str) -> Optional[str]:
    """
    Extract the fenced code block right after '## Manual Map (Tree)'.

    Returns the literal string inside the ``` block (not parsed),
    or None if not found.
    """
    # Find heading
    idx = outline_text.find("## Manual Map (Tree)")
    if idx == -1:
        return None

    after = outline_text[idx:]
    # Find first fenced block after heading
    m = re.search(r"```([\s\S]*?)```", after)
    if not m:
        return None

    return m.group(1).strip("\n")


# -----------------------------
# UI rendering
# -----------------------------
def render_manual_ui(
    *,
    repo_root: Union[str, Path],
    manual_rel_dir: Union[str, Path] = "manual",
    outline_filename: str = "00_outline.md",
    app_title: str = "📘 Instruction Manual",
) -> None:
    """
    Streamlit UI for modular manuals.

    Expected structure:
      REPO/
        manual/
          00_outline.md   (optional; used for displaying a pretty tree block if present)
          01_*.md
          02_*/
          03_*/
        webapp/
          streamlit_app.py
    """
    repo_root = Path(repo_root).resolve()
    manual_root = (repo_root / manual_rel_dir).resolve()
    outline_path = manual_root / outline_filename

    if "manual_selected" not in st.session_state:
        st.session_state.manual_selected = None  # rel_str
    if "manual_expand_all" not in st.session_state:
        st.session_state.manual_expand_all = False
    if "manual_search" not in st.session_state:
        st.session_state.manual_search = ""

    st.markdown(f"## {app_title}")

    # Load index (cached)
    tree, flat = build_manual_index(str(manual_root))

    if not manual_root.exists():
        st.warning(f"Manual directory not found: {manual_root}")
        return

    if not flat:
        st.warning(f"No markdown files found under: {manual_root}")
        return

    # Optional “pretty” tree block from 00_outline.md
    pretty_tree_block = None
    if outline_path.exists():
        outline_text = read_text_safely(outline_path)
        pretty_tree_block = extract_tree_block(outline_text)

    # Controls row
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        st.session_state.manual_expand_all = st.toggle(
            "Expand all folders",
            value=st.session_state.manual_expand_all,
        )
    with c2:
        if st.button("Open outline"):
            # if outline exists, select it; else select nothing
            if outline_path.exists():
                st.session_state.manual_selected = outline_path.relative_to(manual_root).as_posix()
    with c3:
        st.session_state.manual_search = st.text_input(
            "Search manual",
            value=st.session_state.manual_search,
            placeholder="Type to search titles + content…",
        )

    left, right = st.columns([0.42, 0.58], gap="large")

    # -----------------------------
    # Left pane: map + tree + search
    # -----------------------------
    with left:
        # Manual Map
        with st.expander("🗂 Manual Map (Tree)", expanded=False):
            if pretty_tree_block:
                st.code(pretty_tree_block, language="text")
            else:
                # Fallback: render a simple generated tree preview (filenames only)
                st.code(_render_tree_text(tree), language="text")

        # Search results (click-to-open)
        q = st.session_state.manual_search.strip().lower()
        if q:
            st.markdown("### 🔎 Results")
            results = _search_manual(flat, q, limit=25)

            if not results:
                st.caption("No matches.")
            else:
                for rel_str, score in results:
                    mf = flat[rel_str]
                    # Show a compact row; clicking selects the doc
                    if st.button(f"{mf.rel_path.as_posix()}  —  {mf.title}", key=f"sr_{rel_str}"):
                        st.session_state.manual_selected = rel_str
        else:
            st.markdown("### 📚 Sections")

        # Tree navigation
        _render_tree_nav(
            tree=tree,
            flat=flat,
            expand_all=st.session_state.manual_expand_all,
        )

    # -----------------------------
    # Right pane: breadcrumb + viewer
    # -----------------------------
    with right:
        rel_selected: Optional[str] = st.session_state.manual_selected

        if not rel_selected:
            st.info("Select a manual section from the left panel.")
            return

        if rel_selected not in flat:
            st.warning(f"Selected file not found: {rel_selected}")
            return

        mf = flat[rel_selected]

        # Breadcrumb
        crumbs = ["Manual"] + list(mf.rel_path.parts)
        st.caption(" / ".join(crumbs))

        # Viewer
        st.markdown(f"### {mf.title}")
        st.markdown(mf.text)


# -----------------------------
# Internal UI helpers
# -----------------------------
def _render_tree_text(tree: TreeNode) -> str:
    """
    Generate a clean filename-only tree (like your desired output).
    """
    lines: List[str] = ["Manual Map (Tree)"]

    def walk(node: TreeNode, prefix: str = "") -> None:
        keys = sorted(node.keys(), key=numeric_sort_key)
        for i, name in enumerate(keys):
            last = i == (len(keys) - 1)
            branch = "└── " if last else "├── "
            child = node[name]

            if isinstance(child, dict):
                lines.append(f"{prefix}{branch}{name}/")
                next_prefix = prefix + ("    " if last else "│   ")
                walk(child, next_prefix)
            else:
                lines.append(f"{prefix}{branch}{name}")

    walk(tree)
    return "\n".join(lines)


def _search_manual(flat: Dict[str, ManualFile], q: str, limit: int = 25) -> List[Tuple[str, int]]:
    """
    Very simple scoring:
      +5 if query in title
      +1 per occurrence in text (capped)
    """
    results: List[Tuple[str, int]] = []
    for rel_str, mf in flat.items():
        title_l = mf.title.lower()
        text_l = mf.text.lower()

        score = 0
        if q in title_l:
            score += 5

        # count occurrences, cap so giant files don't dominate
        occ = text_l.count(q)
        score += min(occ, 20)

        if score > 0:
            results.append((rel_str, score))

    # sort by score desc, then numeric-ish path
    results.sort(key=lambda x: (-x[1], [numeric_sort_key(p) for p in Path(x[0]).parts]))
    return results[:limit]


def _render_tree_nav(*, tree: TreeNode, flat: Dict[str, ManualFile], expand_all: bool) -> None:
    """
    Recursive nested-folder expanders.
    Files are buttons that set session_state.manual_selected.
    """
    def walk(node: TreeNode, rel_prefix: Path) -> None:
        keys = sorted(node.keys(), key=numeric_sort_key)
        for name in keys:
            child = node[name]
            if isinstance(child, dict):
                # Folder expander
                with st.expander(f"📁 {name}", expanded=expand_all):
                    walk(child, rel_prefix / name)
            else:
                rel_str = (rel_prefix / name).as_posix()
                mf = flat.get(rel_str)

                # File “open” button
                label = f"📄 {name}"
                if mf and mf.title and mf.title != name:
                    label = f"📄 {name} — {mf.title}"

                if st.button(label, key=f"open_{rel_str}"):
                    st.session_state.manual_selected = rel_str

    walk(tree, Path("."))
