from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pytest

from meta_learning.shared.llm import StubLLM
from meta_learning.shared.models import (
    DetectionChannel,
    ErrorTaxonomy,
    Experience,
    MetaLearningConfig,
    Signal,
    TaskContext,
    TaskType,
    TaxonomyEntry,
)


@pytest.fixture
def tmp_config(tmp_path: Path) -> MetaLearningConfig:
    from meta_learning.shared.io import reset_id_counters

    reset_id_counters()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    return MetaLearningConfig(
        workspace_root=str(workspace),
        sessions_root=str(sessions),
    )


@pytest.fixture
def stub_llm() -> StubLLM:
    return StubLLM()


@pytest.fixture
def sample_signal() -> Signal:
    return Signal(
        signal_id="sig-20260309-001",
        timestamp=datetime(2026, 3, 9, 14, 30),
        session_id="test-session-001",
        memory_date=date(2026, 3, 9),
        detection_channels=[DetectionChannel.SELF_RECOVERY],
        primary_channel=DetectionChannel.SELF_RECOVERY,
        keywords=["TS2345", "generic", "type inference"],
        task_summary="Fix React component TypeScript type error",
        error_snapshot="TS2345: Argument of type X is not assignable",
        resolution_snapshot="Added explicit generic parameter",
        step_count=7,
    )


@pytest.fixture
def sample_experience() -> Experience:
    return Experience(
        id="exp-001",
        task_type=TaskType.CODING,
        created_at=datetime(2026, 3, 9, 15, 0),
        source_signal="sig-20260309-001",
        source_session="test-session-001",
        confidence=0.6,
        scene="User asked to fix TypeScript error",
        failure_signature="TS2345: generic type inference failure",
        root_cause="Insufficient generic constraints",
        resolution="Add explicit generic parameters",
        meta_insight="TypeScript generics need explicit annotation in nested calls",
    )


@pytest.fixture
def sample_taxonomy_entry() -> TaxonomyEntry:
    return TaxonomyEntry(
        id="tax-cod-gen-001",
        name="Generic Type Inference Failure",
        trigger="Nested function calls with generic types",
        fix_sop=(
            "1. Check function signature\n2. Add explicit generics\n3. Verify chain"
        ),
        prevention="Always annotate complex generics explicitly",
        confidence=0.85,
        source_exps=["exp-001", "exp-005", "exp-012"],
        keywords=["TS2345", "generic", "type inference", "assignable"],
        created_at=date(2026, 3, 9),
        last_verified=date(2026, 3, 9),
    )


@pytest.fixture
def sample_taxonomy(sample_taxonomy_entry: TaxonomyEntry) -> ErrorTaxonomy:
    tax = ErrorTaxonomy()
    tax.add_entry("coding", "typescript", sample_taxonomy_entry)
    return tax


@pytest.fixture
def sample_task_context() -> TaskContext:
    return TaskContext(
        task_description="Fix React component TypeScript type error",
        tools_used=["read_file", "edit_file"],
        errors_encountered=["TS2345: Argument of type X is not assignable"],
        errors_fixed=True,
        step_count=7,
        session_id="test-session-001",
    )
