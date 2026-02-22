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
    rel_path: Path
    abs_path: Path
    title: str
    text: str


TreeNode = Dict[str, Union["TreeNode", ManualFile]]
_NUM_PREFIX_RE = re.compile(r"^(\d+)[_-]")


# -----------------------------
# Helpers: ordering / parsing
# -----------------------------
def numeric_sort_key(name: str) -> Tuple[int, str]:
    m = _NUM_PREFIX_RE.match(name)
    if m:
        return (int(m.group(1)), name.lower())
    return (10_000, name.lower())


def read_text_safely(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def extract_md_title(md_text: str, fallback: str) -> str:
    for line in md_text.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


@st.cache_data(show_spinner=False)
def build_manual_index(manual_dir: str) -> Tuple[TreeNode, Dict[str, ManualFile]]:
    manual_root = Path(manual_dir).resolve()
    tree: TreeNode = {}
    flat: Dict[str, ManualFile] = {}

    if not manual_root.exists():
        return tree, flat

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


def extract_tree_block(outline_text: str) -> Optional[str]:
    idx = outline_text.find("## Manual Map (Tree)")
    if idx == -1:
        return None
    after = outline_text[idx:]
    m = re.search(r"```([\s\S]*?)```", after)
    if not m:
        return None
    return m.group(1).strip("\n")


def render_generated_tree_text(tree: TreeNode) -> str:
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


def search_manual(flat: Dict[str, ManualFile], q: str, limit: int = 25) -> List[Tuple[str, int]]:
    q = q.strip().lower()
    if not q:
        return []

    results: List[Tuple[str, int]] = []
    for rel_str, mf in flat.items():
        title_l = mf.title.lower()
        text_l = mf.text.lower()

        score = 0
        if q in title_l:
            score += 5
        score += min(text_l.count(q), 20)

        if score > 0:
            results.append((rel_str, score))

    results.sort(key=lambda x: (-x[1], [numeric_sort_key(p) for p in Path(x[0]).parts]))
    return results[:limit]


# -----------------------------
# Single-pane accordion UI
# -----------------------------
def render_manual_ui_single_pane(
    *,
    repo_root: Union[str, Path],
    manual_rel_dir: Union[str, Path] = "manual",
    outline_filename: str = "00_outline.md",
    expander_label: str = "📘 Show / Hide Instruction Manual",
) -> None:
    repo_root = Path(repo_root).resolve()
    manual_root = (repo_root / manual_rel_dir).resolve()
    outline_path = manual_root / outline_filename

    if "manual_selected" not in st.session_state:
        st.session_state.manual_selected = None
    if "manual_expand_all" not in st.session_state:
        st.session_state.manual_expand_all = False
    if "manual_search" not in st.session_state:
        st.session_state.manual_search = ""

    tree, flat = build_manual_index(str(manual_root))

    if not manual_root.exists():
        st.warning(f"Manual directory not found: {manual_root}")
        return
    if not flat:
        st.warning(f"No markdown files found under: {manual_root}")
        return

    # Pull pretty tree from outline (if present)
    pretty_tree_block = None
    if outline_path.exists():
        pretty_tree_block = extract_tree_block(read_text_safely(outline_path))

    # -------------------------
    # Top-level manual expander
    # -------------------------
    with st.expander(expander_label, expanded=False):
        # Controls row
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            st.session_state.manual_expand_all = st.toggle(
                "Expand all",
                value=st.session_state.manual_expand_all,
            )
        with c2:
            if st.button("Open outline", key="manual_open_outline"):
                if outline_path.exists():
                    st.session_state.manual_selected = outline_path.relative_to(manual_root).as_posix()
        with c3:
            st.session_state.manual_search = st.text_input(
                "Search",
                value=st.session_state.manual_search,
                placeholder="Search titles + content…",
                label_visibility="visible",
            )

        # Tree map preview
        with st.expander("🗂 Manual Map (Tree)", expanded=False):
            if pretty_tree_block:
                st.code(pretty_tree_block, language="text")
            else:
                st.code(render_generated_tree_text(tree), language="text")

        # Search results (optional)
        q = st.session_state.manual_search.strip()
        if q:
            with st.expander("🔎 Search results", expanded=True):
                results = search_manual(flat, q, limit=25)
                if not results:
                    st.caption("No matches.")
                else:
                    for rel_str, _score in results:
                        mf = flat[rel_str]
                        if st.button(f"📄 {mf.rel_path.as_posix()} — {mf.title}", key=f"sr_{rel_str}"):
                            st.session_state.manual_selected = rel_str

        # Highest-level sections accordion:
        # root-level items become expanders (folders) or file buttons (files)
        st.markdown("### 📚 Manual Sections")

        root_keys = sorted(tree.keys(), key=numeric_sort_key)
        for name in root_keys:
            node = tree[name]
            if isinstance(node, dict):
                with st.expander(f"📁 {name}", expanded=st.session_state.manual_expand_all):
                    _render_folder_accordion(
                        node=node,
                        rel_prefix=Path(name),
                        flat=flat,
                        expand_all=st.session_state.manual_expand_all,
                    )
            else:
                rel_str = Path(name).as_posix()
                mf = flat.get(rel_str)
                label = f"📄 {name}"
                if mf and mf.title and mf.title != name:
                    label = f"📄 {name} — {mf.title}"
                if st.button(label, key=f"root_open_{rel_str}"):
                    st.session_state.manual_selected = rel_str

    # -------------------------
    # Content area (below)
    # -------------------------
    rel_selected: Optional[str] = st.session_state.manual_selected
    if not rel_selected:
        st.info("Select a manual section above to view it here.")
        return

    if rel_selected not in flat:
        st.warning(f"Selected file not found: {rel_selected}")
        return

    mf = flat[rel_selected]
    crumbs = ["Manual"] + list(mf.rel_path.parts)
    st.caption(" / ".join(crumbs))
    st.markdown(f"## {mf.title}")
    st.markdown(mf.text)


def _render_folder_accordion(
    *,
    node: TreeNode,
    rel_prefix: Path,
    flat: Dict[str, ManualFile],
    expand_all: bool,
) -> None:
    """
    Recursively render folder contents as nested expanders (subfolders) + buttons (files).
    """
    keys = sorted(node.keys(), key=numeric_sort_key)
    for name in keys:
        child = node[name]
        if isinstance(child, dict):
            with st.expander(f"📁 {name}", expanded=expand_all):
                _render_folder_accordion(
                    node=child,
                    rel_prefix=rel_prefix / name,
                    flat=flat,
                    expand_all=expand_all,
                )
        else:
            rel_str = (rel_prefix / name).as_posix()
            mf = flat.get(rel_str)
            label = f"📄 {name}"
            if mf and mf.title and mf.title != name:
                label = f"📄 {name} — {mf.title}"
            if st.button(label, key=f"open_{rel_str}"):
                st.session_state.manual_selected = rel_str
