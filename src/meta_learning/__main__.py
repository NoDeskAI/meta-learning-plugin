"""CLI entry point for meta-learning system.

Usage:
    python -m meta_learning run-layer2 [--workspace PATH] [--config PATH]
    python -m meta_learning run-layer3 [--workspace PATH] [--config PATH]
    python -m meta_learning status [--workspace PATH]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from meta_learning.shared.io import (
    list_all_experiences,
    list_pending_signals,
    load_config,
    load_error_taxonomy,
)
from meta_learning.shared.models import MetaLearningConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("meta_learning")


def build_config(args: argparse.Namespace) -> MetaLearningConfig:
    config_path = getattr(args, "config", None)
    if config_path:
        config = load_config(config_path)
    else:
        config = MetaLearningConfig()

    workspace = getattr(args, "workspace", None)
    if workspace:
        config.workspace_root = str(Path(workspace).expanduser())

    return config


def create_llm(config: MetaLearningConfig):
    if config.llm.provider == "openai":
        from meta_learning.shared.llm_openai import OpenAILLM

        return OpenAILLM(config)
    else:
        from meta_learning.shared.llm import StubLLM

        logger.warning("Using StubLLM — set llm.provider='openai' for real LLM calls")
        return StubLLM()


async def cmd_run_layer2(args: argparse.Namespace) -> int:
    config = build_config(args)
    llm = create_llm(config)

    from meta_learning.layer2.orchestrator import Layer2Orchestrator

    orchestrator = Layer2Orchestrator(config, llm)

    if not args.force and not orchestrator.should_trigger():
        logger.info("Layer 2 trigger conditions not met. Use --force to override.")
        return 0

    result = await orchestrator.run_pipeline()
    logger.info("Layer 2 complete: %s", result)
    return 0


async def cmd_run_layer3(args: argparse.Namespace) -> int:
    config = build_config(args)
    llm = create_llm(config)

    from meta_learning.layer3.orchestrator import Layer3Orchestrator

    orchestrator = Layer3Orchestrator(config, llm)
    result = await orchestrator.run_pipeline()
    logger.info(
        "Layer 3 complete: patterns=%d, gaps=%d, recommendations=%d",
        len(result.cross_task_patterns),
        len(result.capability_gaps),
        len(result.memory_recommendations),
    )
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    config = build_config(args)

    pending_signals = list_pending_signals(config)
    all_experiences = list_all_experiences(config)
    taxonomy = load_error_taxonomy(config)
    taxonomy_entries = taxonomy.all_entries()

    print(f"Workspace:          {config.workspace_root}")
    print(f"Signal buffer:      {config.signal_buffer_path}")
    print(f"Pending signals:    {len(pending_signals)}")
    print(f"Total experiences:  {len(all_experiences)}")
    print(f"Taxonomy entries:   {len(taxonomy_entries)}")

    if pending_signals:
        print("\nRecent pending signals:")
        for sig in pending_signals[-5:]:
            print(f"  [{sig.signal_id}] {sig.trigger_reason}: {sig.task_summary[:60]}")

    if taxonomy_entries:
        print("\nTaxonomy entries:")
        for entry in taxonomy_entries[:10]:
            print(f"  [{entry.id}] {entry.name} (confidence={entry.confidence:.2f})")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="meta_learning",
        description="Three-layer self-evolving learning system",
    )
    parser.add_argument(
        "--workspace",
        default="~/.openclaw/workspace",
        help="Workspace root directory (default: ~/.openclaw/workspace)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to meta-learning config YAML",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    l2 = subparsers.add_parser("run-layer2", help="Run Layer 2 near-line consolidation pipeline")
    l2.add_argument("--force", action="store_true", help="Force run even if trigger conditions not met")

    subparsers.add_parser("run-layer3", help="Run Layer 3 offline deep learning pipeline")
    subparsers.add_parser("status", help="Show current meta-learning system status")

    args = parser.parse_args()

    if args.command == "run-layer2":
        return asyncio.run(cmd_run_layer2(args))
    elif args.command == "run-layer3":
        return asyncio.run(cmd_run_layer3(args))
    elif args.command == "status":
        return cmd_status(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
