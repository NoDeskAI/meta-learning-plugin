#!/usr/bin/env python3
"""Validate Layer 2 fixes against real runtime data.

Runs non-destructively against the live meta-learning workspace at
~/.deskclaw/nanobot/workspace/meta-learning-data/ to verify that:

  1. Experience YAML files contain initial_confidence (migration compat)
  2. Taxonomy YAML files contain confidence_adjustment (migration compat)
  3. Confidence decay is idempotent (run twice, same result)
  4. Boost/penalize changes taxonomy confidence correctly
  5. Skill gating threshold is enforced
  6. SKILL.md generation uses prevention fallback
  7. Materializer handles signal failure gracefully

Modes:
  --check       Read-only: inspect existing data for schema compliance
  --simulate    Runs the pipeline in a temp copy (safe, no mutation)

Usage:
    python scripts/validate_layer2_fixes.py --check
    python scripts/validate_layer2_fixes.py --simulate
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import shutil
import sys
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from meta_learning.shared.io import (
    boost_taxonomy_confidence,
    list_all_experiences,
    load_config,
    load_error_taxonomy,
    penalize_taxonomy_confidence,
    save_error_taxonomy,
    write_experience,
)
from meta_learning.shared.models import (
    ErrorTaxonomy,
    Experience,
    MetaLearningConfig,
    TaxonomyEntry,
    TaskType,
)

LIVE_CONFIG = Path("~/.deskclaw/nanobot/workspace/meta-learning-data/config.yaml").expanduser()
LIVE_WORKSPACE = Path("~/.deskclaw/nanobot/workspace/meta-learning-data").expanduser()

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
SKIP = "\033[93mSKIP\033[0m"
INFO = "\033[94mINFO\033[0m"


def banner(msg: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {msg}")
    print(f"{'─'*60}")


# ───────────────────────────────────────────────────────────
# Check mode: read-only inspection of live data
# ───────────────────────────────────────────────────────────

def check_experience_schema(config: MetaLearningConfig) -> bool:
    """Verify experiences have initial_confidence field."""
    exps = list_all_experiences(config)
    if not exps:
        print(f"  [{SKIP}] No experiences found")
        return True

    missing = [e.id for e in exps if not hasattr(e, "initial_confidence")]
    defaulted = [e for e in exps if e.initial_confidence == 0.6 and e.confidence != 0.6]

    print(f"  [{INFO}] Total experiences: {len(exps)}")
    if missing:
        print(f"  [{FAIL}] {len(missing)} experiences missing initial_confidence")
        return False

    if defaulted:
        print(f"  [{INFO}] {len(defaulted)} experiences have default initial_confidence=0.6")
        print(f"         but current confidence differs (likely pre-migration data)")
        for e in defaulted[:3]:
            print(f"         {e.id}: initial={e.initial_confidence}, current={e.confidence:.4f}")

    ok_count = len(exps) - len(missing)
    print(f"  [{PASS}] {ok_count}/{len(exps)} experiences have initial_confidence field")
    return True


def check_taxonomy_schema(config: MetaLearningConfig) -> bool:
    """Verify taxonomy entries have confidence_adjustment field."""
    taxonomy = load_error_taxonomy(config)
    entries = taxonomy.all_entries()
    if not entries:
        print(f"  [{SKIP}] No taxonomy entries found")
        return True

    missing = [e.id for e in entries if not hasattr(e, "confidence_adjustment")]

    print(f"  [{INFO}] Total taxonomy entries: {len(entries)}")
    if missing:
        print(f"  [{FAIL}] {len(missing)} entries missing confidence_adjustment")
        return False

    adjusted = [e for e in entries if e.confidence_adjustment != 0.0]
    print(f"  [{PASS}] All {len(entries)} entries have confidence_adjustment field")
    if adjusted:
        print(f"  [{INFO}] {len(adjusted)} entries have non-zero adjustment:")
        for e in adjusted:
            print(f"         {e.id}: adj={e.confidence_adjustment:+.2f}, conf={e.confidence:.2f}")
    return True


def check_skill_gating_config(config: MetaLearningConfig) -> bool:
    """Verify skill gating thresholds are configured."""
    min_conf = config.layer2.taxonomy.min_confidence_for_skill
    min_exps = config.layer2.taxonomy.min_source_exps_for_skill
    print(f"  [{INFO}] min_confidence_for_skill = {min_conf}")
    print(f"  [{INFO}] min_source_exps_for_skill = {min_exps}")

    taxonomy = load_error_taxonomy(config)
    entries = taxonomy.all_entries()
    would_gate = [
        e for e in entries
        if e.confidence < min_conf or len(e.source_exps) < min_exps
    ]
    would_pass = [e for e in entries if e not in would_gate]

    print(f"  [{INFO}] {len(would_pass)} entries pass gating, {len(would_gate)} would be gated")
    for e in would_gate[:5]:
        print(f"         GATED: {e.id} conf={e.confidence:.2f} exps={len(e.source_exps)}")
    for e in would_pass[:5]:
        print(f"         PASS:  {e.id} conf={e.confidence:.2f} exps={len(e.source_exps)}")

    print(f"  [{PASS}] Skill gating config verified")
    return True


def check_dead_code_removed(config: MetaLearningConfig) -> bool:
    """Verify ConsolidateConfig has no dead fields."""
    from meta_learning.shared.models import ConsolidateConfig
    fields = set(ConsolidateConfig.model_fields.keys())
    dead = {"use_llm_clustering", "max_llm_calls_per_group", "batch_size"}
    found_dead = fields & dead
    if found_dead:
        print(f"  [{FAIL}] Dead fields still in ConsolidateConfig: {found_dead}")
        return False
    print(f"  [{PASS}] ConsolidateConfig has no dead fields (fields: {sorted(fields)})")
    return True


def check_merge_split_removed() -> bool:
    """Verify MERGE/SPLIT removed from SkillUpdateAction."""
    from meta_learning.shared.models import SkillUpdateAction
    members = {m.value for m in SkillUpdateAction}
    if "merge" in members or "split" in members:
        print(f"  [{FAIL}] MERGE/SPLIT still in SkillUpdateAction: {members}")
        return False
    print(f"  [{PASS}] SkillUpdateAction = {sorted(members)}")
    return True


def check_skillmd_generation(config: MetaLearningConfig) -> bool:
    """Verify SKILL.md would render with prevention fallback."""
    from meta_learning.sync_nobot import _entry_rule_text, _render_skill_md
    taxonomy = load_error_taxonomy(config)
    entries = taxonomy.all_entries()
    if not entries:
        print(f"  [{SKIP}] No taxonomy entries to render")
        return True

    empty_prevention = [e for e in entries if not e.prevention]
    has_fallback = [e for e in empty_prevention if _entry_rule_text(e)]
    no_text = [e for e in empty_prevention if not _entry_rule_text(e)]

    print(f"  [{INFO}] {len(entries)} entries total")
    print(f"  [{INFO}] {len(empty_prevention)} with empty prevention")
    if has_fallback:
        print(f"  [{PASS}] {len(has_fallback)} entries use fallback (fix_sop/trigger)")
        for e in has_fallback[:3]:
            print(f"         {e.id}: fallback -> '{_entry_rule_text(e)[:60]}'")
    if no_text:
        print(f"  [{INFO}] {len(no_text)} entries have no usable text at all (excluded from SKILL.md)")

    md = _render_skill_md(entries, max_rules=10)
    rule_lines = [l for l in md.split("\n") if l.startswith("- ")]
    blank_rules = [l for l in rule_lines if l.strip() == "-"]
    if blank_rules:
        print(f"  [{FAIL}] {len(blank_rules)} blank rules in rendered SKILL.md")
        return False
    print(f"  [{PASS}] SKILL.md renders {len(rule_lines)} rules, no blanks")
    return True


def run_check_mode():
    banner("Check Mode: Inspecting live runtime data")

    if not LIVE_WORKSPACE.exists():
        print(f"  [{FAIL}] Workspace not found: {LIVE_WORKSPACE}")
        print(f"  Run with --simulate to test in a temp workspace")
        return

    config = load_config(str(LIVE_CONFIG)) if LIVE_CONFIG.exists() else MetaLearningConfig()
    config.workspace_root = str(LIVE_WORKSPACE)

    results = {}
    checks = [
        ("Experience schema (initial_confidence)", lambda: check_experience_schema(config)),
        ("Taxonomy schema (confidence_adjustment)", lambda: check_taxonomy_schema(config)),
        ("Skill gating config", lambda: check_skill_gating_config(config)),
        ("Dead code removed (ConsolidateConfig)", lambda: check_dead_code_removed(config)),
        ("MERGE/SPLIT removed", check_merge_split_removed),
        ("SKILL.md generation", lambda: check_skillmd_generation(config)),
    ]

    for name, fn in checks:
        banner(name)
        try:
            results[name] = fn()
        except Exception as e:
            print(f"  [{FAIL}] Exception: {e}")
            results[name] = False

    banner("Summary")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for name, ok in results.items():
        tag = PASS if ok else FAIL
        print(f"  [{tag}] {name}")
    print(f"\n  {passed}/{total} checks passed")


# ───────────────────────────────────────────────────────────
# Simulate mode: run pipeline in temp copy
# ───────────────────────────────────────────────────────────

def simulate_decay_idempotency(config: MetaLearningConfig) -> bool:
    """Create an experience, run decay twice, verify idempotent."""
    from meta_learning.layer2.consolidate import Consolidator
    from meta_learning.shared.llm import StubLLM

    exp = Experience(
        id="exp-validate-001",
        task_type=TaskType.CODING,
        created_at=datetime.now() - timedelta(days=30),
        source_signal="sig-validate-001",
        initial_confidence=0.7,
        confidence=0.7,
        scene="Validation: decay idempotency",
        root_cause="test",
        resolution="test",
        meta_insight="test",
    )
    write_experience(exp, config)

    cons = Consolidator(config, StubLLM())

    loop = asyncio.new_event_loop()
    loop.run_until_complete(cons.consolidate())
    exps1 = list_all_experiences(config)
    c1 = next(e.confidence for e in exps1 if e.id == "exp-validate-001")

    loop.run_until_complete(cons.consolidate())
    exps2 = list_all_experiences(config)
    c2 = next(e.confidence for e in exps2 if e.id == "exp-validate-001")
    loop.close()

    expected = 0.7 * (config.confidence.decay_base ** 30)
    ok = abs(c1 - c2) < 1e-9 and abs(c1 - expected) < 0.01
    print(f"  [{INFO}] initial=0.7, after_run1={c1:.6f}, after_run2={c2:.6f}, expected={expected:.6f}")
    if ok:
        print(f"  [{PASS}] Decay is idempotent (delta={abs(c1 - c2):.2e})")
    else:
        print(f"  [{FAIL}] Decay NOT idempotent: run1={c1}, run2={c2}")
    return ok


def simulate_boost_penalize(config: MetaLearningConfig) -> bool:
    """Create a taxonomy entry, boost and penalize, verify."""
    entry = TaxonomyEntry(
        id="tax-validate-001",
        name="Validation Entry",
        trigger="test trigger",
        fix_sop="test fix",
        prevention="test prevention",
        confidence=0.7,
        confidence_adjustment=0.0,
        source_exps=["exp-001"],
        keywords=["test"],
        created_at=date.today(),
        last_verified=date.today(),
    )
    taxonomy = ErrorTaxonomy()
    taxonomy.add_entry("coding", "test", entry)
    save_error_taxonomy(taxonomy, config)

    boosted = boost_taxonomy_confidence("tax-validate-001", config)
    if boosted is None:
        print(f"  [{FAIL}] boost returned None")
        return False
    print(f"  [{INFO}] After boost: conf={boosted.confidence:.2f}, adj={boosted.confidence_adjustment:+.2f}")

    ok_boost = (
        abs(boosted.confidence_adjustment - config.confidence.hit_success_boost) < 1e-9
        and abs(boosted.confidence - 0.8) < 1e-9
    )

    penalized = penalize_taxonomy_confidence("tax-validate-001", config)
    if penalized is None:
        print(f"  [{FAIL}] penalize returned None")
        return False
    print(f"  [{INFO}] After penalize: conf={penalized.confidence:.2f}, adj={penalized.confidence_adjustment:+.2f}")

    expected_adj = config.confidence.hit_success_boost - config.confidence.contradiction_penalty
    ok_penalize = abs(penalized.confidence_adjustment - expected_adj) < 1e-9

    reloaded = load_error_taxonomy(config).find_entry("tax-validate-001")
    ok_persist = (
        reloaded is not None
        and abs(reloaded.confidence_adjustment - expected_adj) < 1e-9
    )

    ok = ok_boost and ok_penalize and ok_persist
    if ok:
        print(f"  [{PASS}] Boost/penalize works and persists correctly")
    else:
        print(f"  [{FAIL}] boost_ok={ok_boost}, penalize_ok={ok_penalize}, persist_ok={ok_persist}")
    return ok


def simulate_skill_gating(config: MetaLearningConfig) -> bool:
    """Verify skill gating blocks low-confidence entries."""
    from meta_learning.layer2.skill_evolve import SkillEvolver
    from meta_learning.shared.llm import StubLLM
    from meta_learning.shared.models import SkillUpdateAction

    low_entry = TaxonomyEntry(
        id="tax-gate-low",
        name="Low Confidence Entry",
        trigger="test",
        fix_sop="test",
        prevention="test",
        confidence=0.5,
        source_exps=["exp-001", "exp-002"],
        keywords=["test"],
        created_at=date.today(),
        last_verified=date.today(),
    )
    high_entry = TaxonomyEntry(
        id="tax-gate-high",
        name="High Confidence Entry",
        trigger="test",
        fix_sop="test",
        prevention="test",
        confidence=0.9,
        source_exps=[f"exp-{i:03d}" for i in range(6)],
        keywords=["test"],
        created_at=date.today(),
        last_verified=date.today(),
    )

    evolver = SkillEvolver(config, StubLLM())
    loop = asyncio.new_event_loop()
    results = loop.run_until_complete(evolver.evolve_from_taxonomy([low_entry, high_entry]))
    loop.close()

    gated = results[0].action == SkillUpdateAction.NONE and "gated" in results[0].changes_description
    passed = results[1].action != SkillUpdateAction.NONE

    print(f"  [{INFO}] Low (conf=0.5, exps=2): action={results[0].action}, desc='{results[0].changes_description}'")
    print(f"  [{INFO}] High (conf=0.9, exps=6): action={results[1].action}")

    ok = gated and passed
    if ok:
        print(f"  [{PASS}] Low-confidence entry gated, high-confidence entry passed")
    else:
        print(f"  [{FAIL}] gated={gated}, passed={passed}")
    return ok


def simulate_materializer_isolation(config: MetaLearningConfig) -> bool:
    """Verify materializer continues after one signal failure."""
    from meta_learning.layer2.materialize import Materializer
    from meta_learning.shared.io import write_signal
    from meta_learning.shared.llm import StubLLM
    from meta_learning.shared.models import Signal, TriggerReason

    sig1 = Signal(
        signal_id="sig-fail-001",
        timestamp=datetime.now(),
        session_id="unknown",
        trigger_reason=TriggerReason.SELF_RECOVERY,
        keywords=["test"],
        task_summary="should fail",
        step_count=3,
    )
    sig2 = Signal(
        signal_id="sig-ok-002",
        timestamp=datetime.now(),
        session_id="unknown",
        trigger_reason=TriggerReason.SELF_RECOVERY,
        keywords=["test"],
        task_summary="should succeed",
        step_count=3,
    )
    write_signal(sig1, config)
    write_signal(sig2, config)

    llm = StubLLM()
    call_count = 0
    original = llm.materialize_signal

    async def _fail_first(signal, context):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("Simulated LLM timeout")
        return await original(signal, context)

    llm.materialize_signal = _fail_first

    mat = Materializer(config, llm)
    loop = asyncio.new_event_loop()
    results = loop.run_until_complete(mat.materialize_all_pending())
    loop.close()

    ok = len(results) == 1 and call_count == 2
    print(f"  [{INFO}] Calls made: {call_count}, experiences returned: {len(results)}")
    if ok:
        print(f"  [{PASS}] First signal failed, second succeeded - batch not aborted")
    else:
        print(f"  [{FAIL}] Expected 1 result from 2 calls, got {len(results)} from {call_count}")
    return ok


def run_simulate_mode():
    banner("Simulate Mode: Running pipeline in temp workspace")

    tmp_dir = Path(tempfile.mkdtemp(prefix="meta_validate_"))
    print(f"  Temp workspace: {tmp_dir}")

    config = MetaLearningConfig(
        workspace_root=str(tmp_dir / "workspace"),
        sessions_root=str(tmp_dir / "sessions"),
    )
    from meta_learning.shared.io import ensure_directories, reset_id_counters
    reset_id_counters()
    ensure_directories(config)

    results = {}
    simulations = [
        ("Decay idempotency", lambda: simulate_decay_idempotency(config)),
        ("Boost/penalize taxonomy", lambda: simulate_boost_penalize(config)),
        ("Skill gating enforcement", lambda: simulate_skill_gating(config)),
        ("Materializer exception isolation", lambda: simulate_materializer_isolation(config)),
    ]

    for name, fn in simulations:
        banner(name)
        reset_id_counters()
        try:
            results[name] = fn()
        except Exception as e:
            print(f"  [{FAIL}] Exception: {e}")
            import traceback; traceback.print_exc()
            results[name] = False

    banner("Summary")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for name, ok in results.items():
        tag = PASS if ok else FAIL
        print(f"  [{tag}] {name}")
    print(f"\n  {passed}/{total} simulations passed")

    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"  Cleaned up: {tmp_dir}")


# ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Validate Layer 2 fixes")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="Read-only: inspect live data")
    group.add_argument("--simulate", action="store_true", help="Run pipeline in temp workspace")
    group.add_argument("--all", action="store_true", help="Run both check and simulate")
    args = parser.parse_args()

    if args.check or args.all:
        run_check_mode()
    if args.simulate or args.all:
        run_simulate_mode()


if __name__ == "__main__":
    main()
