"""Command-line entry point for the RASCAL DIAAD wrapper."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass

from rascal import __version__
from rascal.config import ConfigError, init_project, load_project_config
from rascal.diaad_config import DiaadConfigError, write_diaad_config
from rascal.diaad_invocation import DiaadInvocationError
from rascal.planner import PlanError, create_stage_plan, render_plan_json, render_plan_text
from rascal.preflight import (
    PreflightError,
    render_preflight_json,
    render_preflight_text,
    run_preflight,
)
from rascal.profiles import list_profiles
from rascal.runner import (
    RunnerError,
    render_run_result_text,
    run_diaad_passthrough,
    run_stage,
)
from rascal.status import (
    StatusError,
    build_next_report,
    build_status_report,
    render_next_json,
    render_next_text,
    render_status_json,
    render_status_text,
)


IMPLEMENTED_IN_LATER_PASS = (
    "This command is part of the RASCAL DIAAD wrapper MVP, but its behavior "
    "will be implemented in a later pass."
)


@dataclass(frozen=True)
class ParsedCommand:
    """Normalized command details useful for tests and future dispatch."""

    command: str
    branch: str | None = None
    stage: str | None = None
    diaad_args: tuple[str, ...] = ()


def _strip_passthrough_separator(args: list[str]) -> list[str]:
    if args and args[0] == "--":
        return args[1:]
    return args


def build_parser() -> argparse.ArgumentParser:
    """Build the RASCAL CLI parser."""

    parser = argparse.ArgumentParser(
        prog="rascal",
        description="Lab-facing workflow wrapper for DIAAD.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"rascal {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="command")
    subparsers.required = True

    init_parser = subparsers.add_parser(
        "init",
        help="Initialize a RASCAL project.",
    )
    init_parser.add_argument(
        "--profile",
        choices=list_profiles(),
        default="lab_full",
        help="Profile to initialize with (default: lab_full).",
    )
    init_parser.add_argument(
        "--layout",
        choices=("canonical", "legacy"),
        default="canonical",
        help="Project layout to create (default: canonical).",
    )
    init_parser.add_argument(
        "--project",
        default=".",
        help="Project directory to initialize (default: current directory).",
    )
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting generated configuration in a later pass.",
    )

    for name, help_text in (
        ("check", "Run RASCAL preflight checks."),
        ("plan", "Show the planned DIAAD commands for a stage."),
        ("run", "Run a RASCAL stage."),
    ):
        stage_required = name in {"plan", "run"}
        stage_parser = subparsers.add_parser(name, help=help_text)
        stage_parser.add_argument(
            "--branch",
            choices=("common", "monolog", "dialog"),
            required=stage_required,
            help="Workflow branch (default for check: common).",
        )
        stage_parser.add_argument(
            "--stage",
            required=stage_required,
            help="Stage id, such as 4m, 5m_prepare, or 7d (default for check: 1).",
        )
        stage_parser.add_argument(
            "--config",
            default=None,
            help="Path to config/rascal.yaml.",
        )
        if name in {"check", "plan"}:
            stage_parser.add_argument(
                "--format",
                choices=("text", "json"),
                default="text",
                help="Output format (default: text).",
            )
        if name == "plan":
            stage_parser.add_argument(
                "--write-config",
                action="store_true",
                help="Write generated DIAAD config in a later pass.",
            )
        if name == "run":
            stage_parser.add_argument(
                "--dry-run",
                action="store_true",
                help="Plan and preflight without executing DIAAD.",
            )

    status_parser = subparsers.add_parser(
        "status",
        help="Summarize project workflow status.",
    )
    status_parser.add_argument(
        "--config",
        default=None,
        help="Path to config/rascal.yaml.",
    )
    status_parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format (default: text).",
    )

    next_parser = subparsers.add_parser(
        "next",
        help="Recommend the next workflow action.",
    )
    next_parser.add_argument(
        "--config",
        default=None,
        help="Path to config/rascal.yaml.",
    )
    next_parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format (default: text).",
    )

    workflows_parser = subparsers.add_parser(
        "workflows",
        help="Inspect archived workflow manifests.",
    )
    workflows_subparsers = workflows_parser.add_subparsers(
        dest="workflow_command",
        metavar="workflow_command",
    )
    workflows_subparsers.required = True
    workflows_subparsers.add_parser(
        "list",
        help="List archived workflows.",
    )
    workflows_show = workflows_subparsers.add_parser(
        "show",
        help="Show one archived workflow.",
    )
    workflows_show.add_argument("workflow_id", help="Workflow id to show.")
    workflows_show.add_argument(
        "--files",
        action="store_true",
        help="Include referenced docs and script paths.",
    )

    diaad_parser = subparsers.add_parser(
        "diaad",
        help="Pass raw arguments through to DIAAD.",
    )
    diaad_parser.add_argument(
        "diaad_args",
        nargs=argparse.REMAINDER,
        help="Arguments after '--' are passed to DIAAD.",
    )

    asr_parser = subparsers.add_parser(
        "asr",
        help="Run Stage 0 ASR helper utilities.",
    )
    asr_subparsers = asr_parser.add_subparsers(
        dest="asr_command",
        metavar="asr_command",
    )
    asr_subparsers.required = True
    split_audio = asr_subparsers.add_parser(
        "split-audio",
        help="Split .wav files into smaller chunks.",
    )
    split_audio.add_argument("--input", required=True, help="Input directory.")
    split_audio.add_argument("--output", required=True, help="Output directory.")
    split_audio.add_argument(
        "--max-seconds",
        type=int,
        default=60,
        help="Maximum chunk length in seconds (default: 60).",
    )
    combine_chat = asr_subparsers.add_parser(
        "combine-chat-parts",
        help="Combine CHAT part files produced from split audio.",
    )
    combine_chat.add_argument("--input", required=True, help="Input directory.")
    combine_chat.add_argument("--output", required=True, help="Output directory.")

    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments and normalize passthrough details."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "command", None) == "diaad":
        args.diaad_args = _strip_passthrough_separator(args.diaad_args)
    return args


def normalize_command(args: argparse.Namespace) -> ParsedCommand:
    """Convert argparse output into a compact command descriptor."""

    return ParsedCommand(
        command=args.command,
        branch=getattr(args, "branch", None),
        stage=getattr(args, "stage", None),
        diaad_args=tuple(getattr(args, "diaad_args", ()) or ()),
    )


def dispatch(args: argparse.Namespace) -> int:
    """Dispatch parsed arguments.

    Pass 01 intentionally only establishes the CLI shell. Later passes replace
    these placeholders with calls into the wrapper modules.
    """

    command = normalize_command(args)
    if command.command == "init":
        result = init_project(
            args.project,
            profile=args.profile,
            layout=args.layout,
            force=args.force,
        )
        print(f"Initialized RASCAL project: {result.project_root}")
        print(f"Profile: {result.profile_name}")
        print(f"Layout: {result.layout}")
        print(f"Config: {result.config_path}")
        print(f"Directories ensured: {len(result.created_directories)}")
        return 0

    if command.command == "diaad":
        if not command.diaad_args:
            print("No DIAAD arguments supplied. Use: rascal diaad -- <args>")
            return 2
        result = run_diaad_passthrough(command.diaad_args)
        if result.stdout:
            print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
        if result.stderr:
            print(result.stderr, end="" if result.stderr.endswith("\n") else "\n", file=sys.stderr)
        return result.returncode

    if command.command == "check":
        config = load_project_config(args.config)
        report = run_preflight(
            config,
            branch=command.branch or "common",
            stage_id=command.stage or "1",
        )
        print(
            render_preflight_json(report)
            if args.format == "json"
            else render_preflight_text(report)
        )
        return 1 if report.has_errors else 0

    if command.command == "plan":
        config = load_project_config(args.config)
        plan = create_stage_plan(config, command.branch or "", command.stage or "")
        if args.write_config:
            write_diaad_config(config, plan)
        print(render_plan_json(plan) if args.format == "json" else render_plan_text(plan))
        return 0

    if command.command == "run":
        config = load_project_config(args.config)
        result = run_stage(
            config,
            command.branch or "",
            command.stage or "",
            dry_run=args.dry_run,
        )
        print(render_run_result_text(result))
        return result.exit_code

    if command.command == "status":
        config = load_project_config(args.config)
        report = build_status_report(config)
        print(
            render_status_json(report)
            if args.format == "json"
            else render_status_text(report)
        )
        return 0

    if command.command == "next":
        config = load_project_config(args.config)
        status_report = build_status_report(config)
        next_report = build_next_report(status_report)
        print(
            render_next_json(next_report)
            if args.format == "json"
            else render_next_text(next_report)
        )
        return 0

    print(f"RASCAL command parsed: {command.command}")
    if command.branch is not None:
        print(f"Branch: {command.branch}")
    if command.stage is not None:
        print(f"Stage: {command.stage}")
    print(IMPLEMENTED_IN_LATER_PASS)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the RASCAL CLI."""

    try:
        args = parse_args(argv)
        return dispatch(args)
    except ConfigError as exc:
        print(f"RASCAL configuration error: {exc}", file=sys.stderr)
        return 2
    except PlanError as exc:
        print(f"RASCAL planning error: {exc}", file=sys.stderr)
        return 2
    except DiaadInvocationError as exc:
        print(f"RASCAL DIAAD command error: {exc}", file=sys.stderr)
        return 2
    except DiaadConfigError as exc:
        print(f"RASCAL DIAAD config error: {exc}", file=sys.stderr)
        return 2
    except PreflightError as exc:
        print(f"RASCAL preflight error: {exc}", file=sys.stderr)
        return 2
    except RunnerError as exc:
        print(f"RASCAL runner error: {exc}", file=sys.stderr)
        return 2
    except StatusError as exc:
        print(f"RASCAL status error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
