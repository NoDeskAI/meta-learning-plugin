#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import logging
import sys
import tempfile
from datetime import date, datetime
from pathlib import Path
from textwrap import indent

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

from meta_learning.layer1.quick_think import QuickThinkIndex
from meta_learning.layer1.signal_capture import SignalCapture
from meta_learning.layer2.orchestrator import Layer2Orchestrator
from meta_learning.layer3.orchestrator import Layer3Orchestrator
from meta_learning.shared.io import (
    load_error_taxonomy,
    load_latest_layer3_result,
    list_pending_signals,
)
from meta_learning.shared.llm import StubLLM
from meta_learning.shared.models import (
    ErrorTaxonomy,
    MetaLearningConfig,
    TaskContext,
)
from tests.mock_resources.fixtures import populate_mock_environment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main_test")

SEPARATOR = "=" * 72
THIN_SEP = "-" * 72


def _header(title: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def _sub_header(title: str) -> None:
    print(f"\n{THIN_SEP}")
    print(f"  {title}")
    print(THIN_SEP)


def print_quick_think_result(
    label: str,
    context: TaskContext,
    result,
) -> None:
    hit_marker = "HIT" if result.hit else "---"
    risk = result.risk_level.upper()
    print(f"  [{hit_marker}] (risk={risk:6s}) {context.task_description[:60]}")
    if result.matched_signals:
        print(f"         signals : {', '.join(result.matched_signals)}")
    if result.matched_taxonomy_entries:
        print(f"         taxonomy: {', '.join(result.matched_taxonomy_entries)}")


def print_error_taxonomy(taxonomy: ErrorTaxonomy) -> None:
    if not taxonomy.taxonomy:
        print("  (empty -- no taxonomy entries yet)")
        return
    for domain, subdomains in taxonomy.taxonomy.items():
        for subdomain, entries in subdomains.items():
            for entry in entries:
                print(f"  [{entry.id}] {entry.name}")
                print(f"    confidence : {entry.confidence:.2f}")
                print(f"    trigger    : {entry.trigger[:80]}")
                print(f"    fix_sop    :")
                print(indent(entry.fix_sop, "      "))
                print(f"    prevention : {entry.prevention[:80]}")
                print(f"    keywords   : {', '.join(entry.keywords)}")
                print(f"    source_exps: {', '.join(entry.source_exps)}")
                print()


def print_skill_results(skill_results: list) -> None:
    if not skill_results:
        print("  (no skill suggestions)")
        return
    for sr in skill_results:
        action = sr.action.value.upper()
        target = sr.target_skill or "(none)"
        print(f"  [{action:8s}] target={target}")
        print(f"    description: {sr.changes_description}")
        if sr.new_content:
            preview = sr.new_content[:200].replace("\n", "\n    ")
            print(f"    content preview:")
            print(f"      {preview}")
            if len(sr.new_content) > 200:
                print(f"      ... ({len(sr.new_content)} chars total)")
        if sr.version_bump:
            print(f"    version: {sr.version_bump}")
        print()


def build_task_contexts() -> list[TaskContext]:
    # StubLLM requires: task_type enum value in task_summary, ≥30% keyword overlap in
    # failure_signature for clustering, and ≥3 same-class experiences for taxonomy promotion.
    return [
        TaskContext(
            task_description="coding: Fix TS2345 generic type inference error in UserList component",
            tools_used=["read_file", "edit_file", "tsc"],
            errors_encountered=[
                "TS2345: Argument of type 'string | undefined' is not assignable to parameter of type 'string'",
            ],
            errors_fixed=True,
            step_count=7,
            session_id="ses-mock-ts-type-001",
            extra={
                "resolution": "Added guard clause for undefined params and explicit generic annotation"
            },
        ),
        TaskContext(
            task_description="coding: Fix TS2322 missing prop error in generic component",
            tools_used=["read_file", "edit_file"],
            errors_encountered=[
                "TS2322: Type '{ onClick: () => void; }' is not assignable to type 'Props<User>'",
            ],
            errors_fixed=True,
            step_count=5,
            session_id="ses-mock-ts-type-001",
            extra={"resolution": "Added required data prop to generic component call"},
        ),
        TaskContext(
            task_description="coding: Fix TS2366 generic type constraint error in DataTable",
            tools_used=["read_file", "edit_file", "tsc"],
            errors_encountered=[
                "TS2366: Function lacks ending return statement and return type does not include 'undefined'",
            ],
            errors_fixed=True,
            step_count=6,
            session_id="ses-mock-ts-type-001",
            extra={"resolution": "Added explicit return type with undefined union"},
        ),
        TaskContext(
            task_description="coding: Fix TS2339 property does not exist on generic type",
            tools_used=["read_file", "edit_file", "tsc"],
            errors_encountered=[
                "TS2339: Property 'name' does not exist on type 'T'",
            ],
            errors_fixed=True,
            step_count=4,
            session_id="ses-mock-ts-type-001",
            extra={
                "resolution": "Added generic constraint extends interface with name property"
            },
        ),
        TaskContext(
            task_description="coding: Fix TS2304 cannot find name in module scope",
            tools_used=["read_file", "edit_file", "tsc"],
            errors_encountered=[
                "TS2304: Cannot find name 'ResponseType' in generic handler",
            ],
            errors_fixed=True,
            step_count=3,
            session_id="ses-mock-ts-type-001",
            extra={"resolution": "Added missing import for ResponseType generic"},
        ),
        TaskContext(
            task_description="coding: Fix TS2769 no overload matches generic call",
            tools_used=["read_file", "edit_file", "tsc"],
            errors_encountered=[
                "TS2769: No overload matches this call for generic component",
            ],
            errors_fixed=True,
            step_count=5,
            session_id="ses-mock-ts-type-001",
            extra={"resolution": "Narrowed generic type parameter to match overload"},
        ),
        TaskContext(
            task_description="configuration: Fix DATABASE_URL environment variable not set",
            tools_used=["read_file", "edit_file", "docker"],
            errors_encountered=[
                "configuration error: DATABASE_URL environment variable is not set",
            ],
            errors_fixed=True,
            step_count=6,
            session_id="ses-mock-env-config-006",
            extra={
                "resolution": "Aligned env var naming across docker-compose and app config"
            },
        ),
        TaskContext(
            task_description="configuration: Fix REDIS_URL environment variable missing in staging",
            tools_used=["read_file", "edit_file", "docker"],
            errors_encountered=[
                "configuration error: REDIS_URL environment variable is not set",
            ],
            errors_fixed=True,
            step_count=5,
            session_id="ses-mock-env-config-006",
            extra={"resolution": "Added REDIS_URL to .env.example and docker-compose"},
        ),
        TaskContext(
            task_description="configuration: Fix API_SECRET environment variable not set in production",
            tools_used=["read_file", "edit_file", "docker"],
            errors_encountered=[
                "configuration error: API_SECRET environment variable is not set",
            ],
            errors_fixed=True,
            step_count=4,
            session_id="ses-mock-env-config-006",
            extra={
                "resolution": "Added secret to vault and updated deployment manifest"
            },
        ),
        TaskContext(
            task_description="configuration: Fix SMTP_HOST environment variable not set in CI",
            tools_used=["read_file", "edit_file", "docker"],
            errors_encountered=[
                "configuration error: SMTP_HOST environment variable is not set",
            ],
            errors_fixed=True,
            step_count=3,
            session_id="ses-mock-env-config-006",
            extra={"resolution": "Added SMTP_HOST to CI environment config"},
        ),
        TaskContext(
            task_description="configuration: Fix CACHE_TTL environment variable not set",
            tools_used=["read_file", "edit_file", "docker"],
            errors_encountered=[
                "configuration error: CACHE_TTL environment variable is not set",
            ],
            errors_fixed=True,
            step_count=3,
            session_id="ses-mock-env-config-006",
            extra={"resolution": "Added sensible CACHE_TTL default with env override"},
        ),
        TaskContext(
            task_description="debugging: Fix runtime TypeError reading property of undefined",
            tools_used=["read_file", "edit_file", "browser_dev_tools"],
            errors_encountered=[
                "TypeError: Cannot read properties of undefined (reading 'map')",
            ],
            errors_fixed=True,
            step_count=5,
            session_id="ses-mock-runtime-null-005",
            extra={
                "resolution": "Added optional chaining with nullish coalescing and Zod validation"
            },
        ),
        TaskContext(
            task_description="debugging: Fix runtime TypeError reading property of null object",
            tools_used=["read_file", "edit_file", "browser_dev_tools"],
            errors_encountered=[
                "TypeError: Cannot read properties of null (reading 'forEach')",
            ],
            errors_fixed=True,
            step_count=6,
            session_id="ses-mock-runtime-null-005",
            extra={"resolution": "Added null guard and default empty array fallback"},
        ),
        TaskContext(
            task_description="debugging: Fix runtime TypeError reading property of undefined response",
            tools_used=["read_file", "edit_file", "browser_dev_tools"],
            errors_encountered=[
                "TypeError: Cannot read properties of undefined (reading 'data')",
            ],
            errors_fixed=True,
            step_count=7,
            session_id="ses-mock-runtime-null-005",
            extra={"resolution": "Added response schema validation at API boundary"},
        ),
        TaskContext(
            task_description="debugging: Fix runtime TypeError reading property of undefined in list",
            tools_used=["read_file", "edit_file", "browser_dev_tools"],
            errors_encountered=[
                "TypeError: Cannot read properties of undefined (reading 'length')",
            ],
            errors_fixed=True,
            step_count=4,
            session_id="ses-mock-runtime-null-005",
            extra={"resolution": "Added default empty array before accessing length"},
        ),
        TaskContext(
            task_description="debugging: Fix runtime TypeError reading property of undefined config",
            tools_used=["read_file", "edit_file", "browser_dev_tools"],
            errors_encountered=[
                "TypeError: Cannot read properties of undefined (reading 'baseURL')",
            ],
            errors_fixed=True,
            step_count=5,
            session_id="ses-mock-runtime-null-005",
            extra={
                "resolution": "Added config initialization check before accessing baseURL"
            },
        ),
        TaskContext(
            task_description="Resolve git merge conflicts after rebase",
            tools_used=["git_status", "edit_file", "git_add"],
            user_corrections=["Keep the v2 API endpoint, not v1"],
            step_count=9,
            session_id="ses-mock-git-conflict-002",
        ),
        TaskContext(
            task_description="Deploy application to Kubernetes cluster",
            tools_used=["kubectl", "helm"],
            new_tools=["kubectl", "helm"],
            step_count=4,
            session_id="ses-mock-deploy-k8s",
        ),
        TaskContext(
            task_description="Simple CSS color change that took too many steps",
            tools_used=["read_file", "edit_file"],
            step_count=25,
        ),
    ]


async def main() -> None:
    _header("Meta-Learning Pipeline -- End-to-End Integration Test")
    print(f"  Timestamp: {datetime.now().isoformat()}")

    _header("Step 0: Mock Environment Setup")

    tmp_dir = Path(tempfile.mkdtemp(prefix="meta_learning_test_"))
    logger.info("Workspace: %s", tmp_dir)

    mock_env = populate_mock_environment(tmp_dir)
    config = mock_env.config
    llm = StubLLM()

    print(f"  Temp directory   : {tmp_dir}")
    print(f"  Sessions written : {len(mock_env.session_paths)}")
    for sp in mock_env.session_paths:
        print(f"    - {sp.name}")
    print(f"  Memory files     : {len(mock_env.memory_paths)}")
    for mp in mock_env.memory_paths:
        print(f"    - {mp.name}")

    _header("Step 1: L1 Quick Think -- First Pass (empty taxonomy)")

    taxonomy = ErrorTaxonomy()
    quick_think = QuickThinkIndex(taxonomy, config)

    task_contexts = build_task_contexts()
    print(f"  Evaluating {len(task_contexts)} task contexts:\n")

    first_pass_results = []
    for ctx in task_contexts:
        result = quick_think.evaluate(ctx)
        first_pass_results.append((ctx, result))
        print_quick_think_result("1st", ctx, result)

    hits = sum(1 for _, r in first_pass_results if r.hit)
    print(
        f"\n  Summary: {hits}/{len(task_contexts)} hits (taxonomy is empty, "
        f"only irreversible/new-tool/recent-failure checks active)"
    )

    _header("Step 2: L1 Signal Capture -- Generating Signals")

    signal_capture = SignalCapture(config)
    captured_signals = []

    for ctx in task_contexts:
        signal = signal_capture.evaluate_and_capture(ctx)
        if signal is not None:
            captured_signals.append(signal)
            print(
                f"  [CAPTURED] {signal.signal_id}  "
                f"trigger={signal.trigger_reason.value:20s}  "
                f"keywords={signal.keywords[:5]}"
            )
        else:
            print(f"  [SKIPPED ] {ctx.task_description[:60]}  (no trigger)")

    pending = list_pending_signals(config)
    print(f"\n  Total captured: {len(captured_signals)}")
    print(f"  Pending signals on disk: {len(pending)}")

    if len(pending) == 0:
        print(
            "\n  WARNING: No pending signals -- L2 pipeline will have nothing to process."
        )
        print("  This is unexpected. Check SignalCapture trigger logic.")

    _header("Step 3: L2 Orchestrator Pipeline")
    print("  Running: Materialize -> Consolidate -> Taxonomy -> Skill Evolve\n")

    orchestrator = Layer2Orchestrator(config, llm)

    should_trigger = orchestrator.should_trigger()
    print(f"  should_trigger(): {should_trigger}")
    print(f"  (Running pipeline regardless for demonstration)\n")

    pipeline_result = await orchestrator.run_pipeline()

    _sub_header("L2 Pipeline Results")
    print(f"  Materialized experiences : {pipeline_result.materialized_count}")
    print(f"  Total clusters           : {pipeline_result.total_clusters}")
    print(f"  New taxonomy entries     : {pipeline_result.new_taxonomy_entries}")
    print(f"  Skill updates            : {pipeline_result.skill_updates}")
    print(f"  Completed at             : {pipeline_result.timestamp.isoformat()}")

    _header("Step 4: Refined Error Taxonomy")

    taxonomy = load_error_taxonomy(config)
    entries = taxonomy.all_entries()
    print(f"  Total entries: {len(entries)}\n")
    print_error_taxonomy(taxonomy)

    _header("Step 5: Generated Skill Suggestions")

    skills_dir = Path(config.skills_path)
    skill_dirs: list[Path] = []
    if skills_dir.exists():
        skill_dirs = [d for d in skills_dir.iterdir() if d.is_dir()]
        print(f"  Skill directories created: {len(skill_dirs)}\n")
        for skill_dir in sorted(skill_dirs):
            skill_file = skill_dir / "SKILL.md"
            if skill_file.exists():
                content = skill_file.read_text()
                _sub_header(f"Skill: {skill_dir.name}")
                print(indent(content, "  "))
            else:
                print(f"  [{skill_dir.name}] (no SKILL.md found)")
    else:
        print("  (no skills directory created -- insufficient confidence or evidence)")

    _header("Step 6: L1 Quick Think -- Second Pass (with refined taxonomy)")

    quick_think.update_taxonomy(taxonomy)
    print(f"  Taxonomy now has {len(taxonomy.all_entries())} entries")
    print(f"  Keyword index size: {len(taxonomy.all_keywords())} keywords\n")

    second_pass_results = []
    for ctx in task_contexts:
        qt_result = quick_think.evaluate(ctx)
        second_pass_results.append((ctx, qt_result))
        print_quick_think_result("2nd", ctx, qt_result)

    hits_after = sum(1 for _, r in second_pass_results if r.hit)
    print(
        f"\n  Summary: {hits_after}/{len(task_contexts)} hits "
        f"(was {hits} before taxonomy refinement)"
    )

    _header("Step 7: L3 Offline Deep Learning Pipeline")
    print(
        "  Running: CrossTask Mining -> Capability Gap Detection -> Memory Optimization\n"
    )

    l3_orchestrator = Layer3Orchestrator(config, llm)
    l3_result = await l3_orchestrator.run_pipeline()

    _sub_header("L3 Pipeline Results")
    print(f"  Cross-task patterns      : {len(l3_result.cross_task_patterns)}")
    print(f"  Capability gaps          : {len(l3_result.capability_gaps)}")
    print(f"  Memory recommendations   : {len(l3_result.memory_recommendations)}")
    print(f"  Completed at             : {l3_result.timestamp.isoformat()}")

    if l3_result.cross_task_patterns:
        _sub_header("Cross-Task Patterns")
        for pattern in l3_result.cross_task_patterns:
            types_str = ", ".join(t.value for t in pattern.affected_task_types)
            print(f"  [{pattern.pattern_id}] {pattern.description[:80]}")
            print(f"    task types  : {types_str}")
            print(f"    root cause  : {pattern.shared_root_cause[:80]}")
            print(f"    strategy    : {pattern.meta_strategy[:80]}")
            print(f"    confidence  : {pattern.confidence:.2f}")
            print(f"    evidence    : {len(pattern.source_experience_ids)} experiences")
            print()

    if l3_result.capability_gaps:
        _sub_header("Capability Gaps")
        for gap in l3_result.capability_gaps:
            print(f"  [{gap.gap_id}] ({gap.gap_type}) {gap.description[:80]}")
            print(f"    action   : {gap.suggested_action[:80]}")
            print(f"    priority : {gap.priority:.2f}")
            print(f"    evidence : {len(gap.evidence_ids)} experiences")
            print()

    if l3_result.memory_recommendations:
        _sub_header("Memory Recommendations")
        extract_count = sum(
            1 for r in l3_result.memory_recommendations if r.action.value == "extract"
        )
        prune_count = sum(
            1 for r in l3_result.memory_recommendations if r.action.value == "prune"
        )
        consolidate_count = sum(
            1
            for r in l3_result.memory_recommendations
            if r.action.value == "consolidate"
        )
        print(f"  Extract    : {extract_count}")
        print(f"  Prune      : {prune_count}")
        print(f"  Consolidate: {consolidate_count}")
        print()
        for rec in l3_result.memory_recommendations[:5]:
            print(f"  [{rec.action.value.upper():12s}] {rec.target}")
            print(f"    reason: {rec.reason[:80]}")
            if rec.content:
                print(f"    content: {rec.content[:80]}")
            print()
        if len(l3_result.memory_recommendations) > 5:
            print(f"  ... and {len(l3_result.memory_recommendations) - 5} more")

    l3_saved = load_latest_layer3_result(config)
    print(f"\n  L3 result persisted: {l3_saved is not None}")

    _header("Pipeline Complete -- Summary")

    skill_count = len(skill_dirs)
    print(f"  Mock sessions loaded  : {len(mock_env.session_paths)}")
    print(f"  Mock memories loaded  : {len(mock_env.memory_paths)}")
    print(f"  Task contexts         : {len(task_contexts)}")
    print(f"  Signals captured      : {len(captured_signals)}")
    print(f"  Experiences created   : {pipeline_result.materialized_count}")
    print(f"  Clusters formed       : {pipeline_result.total_clusters}")
    print(f"  Taxonomy entries      : {len(entries)}")
    print(f"  Skills generated      : {skill_count}")
    print(f"  Quick Think hits (before) : {hits}")
    print(f"  Quick Think hits (after)  : {hits_after}")
    print(f"  L3 cross-task patterns    : {len(l3_result.cross_task_patterns)}")
    print(f"  L3 capability gaps        : {len(l3_result.capability_gaps)}")
    print(f"  L3 memory recommendations : {len(l3_result.memory_recommendations)}")
    print(f"  Workspace             : {tmp_dir}")
    print(SEPARATOR)


if __name__ == "__main__":
    asyncio.run(main())
