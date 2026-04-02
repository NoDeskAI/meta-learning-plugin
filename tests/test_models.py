from datetime import datetime

from meta_learning.shared.models import (
    ErrorTaxonomy,
    ExperienceCluster,
    ExperienceIndex,
    MetaLearningConfig,
    QuickThinkResult,
    Signal,
    TaskContext,
    TaskType,
    TriggerReason,
)


class TestSignalModel:
    def test_create_minimal(self):
        sig = Signal(
            signal_id="sig-20260309-001",
            timestamp=datetime.now(),
            session_id="abc",
            trigger_reason=TriggerReason.SELF_RECOVERY,
            keywords=["error"],
            task_summary="test",
            step_count=1,
        )
        assert sig.processed is False
        assert sig.error_snapshot is None

    def test_trigger_reason_values(self):
        assert TriggerReason.USER_CORRECTION == "user_correction"
        assert TriggerReason.SELF_RECOVERY == "self_recovery"
        assert TriggerReason.UNRESOLVED_ERROR == "unresolved_error"
        assert TriggerReason.NEW_TOOL == "new_tool"
        assert TriggerReason.EFFICIENCY_ANOMALY == "efficiency_anomaly"


class TestExperienceModel:
    def test_defaults(self, sample_experience):
        assert sample_experience.related_exps == []
        assert sample_experience.promoted_to is None
        assert sample_experience.verification_count == 1

    def test_task_type_values(self):
        assert TaskType.CODING == "coding"
        assert TaskType.UNCLASSIFIED == "_unclassified"


class TestErrorTaxonomy:
    def test_empty_taxonomy(self):
        tax = ErrorTaxonomy()
        assert tax.all_entries() == []
        assert tax.all_keywords() == {}

    def test_add_and_retrieve(self, sample_taxonomy_entry):
        tax = ErrorTaxonomy()
        tax.add_entry("coding", "typescript", sample_taxonomy_entry)
        entries = tax.all_entries()
        assert len(entries) == 1
        assert entries[0].id == "tax-cod-gen-001"

    def test_keyword_index(self, sample_taxonomy):
        kw_map = sample_taxonomy.all_keywords()
        assert "ts2345" in kw_map
        assert "generic" in kw_map
        assert len(kw_map["ts2345"]) == 1


class TestExperienceIndex:
    def test_empty_index(self):
        idx = ExperienceIndex(last_updated=datetime.now())
        assert idx.clusters == []

    def test_with_clusters(self):
        cluster = ExperienceCluster(
            cluster_id="clust-001",
            task_type=TaskType.CODING,
            failure_signature_pattern="TS2345",
            experience_ids=["exp-001", "exp-002"],
        )
        idx = ExperienceIndex(last_updated=datetime.now(), clusters=[cluster])
        assert len(idx.clusters) == 1
        assert idx.clusters[0].cluster_id == "clust-001"


class TestMetaLearningConfig:
    def test_defaults(self):
        cfg = MetaLearningConfig()
        assert cfg.layer1.quick_think.max_latency_ms == 50
        assert cfg.layer2.trigger.min_pending_signals == 2
        assert cfg.layer2.trigger.max_hours_since_last == 8
        assert cfg.confidence.prune_threshold == 0.3
        assert cfg.llm.provider == "stub"

    def test_resolve_paths(self, tmp_config):
        assert "signal_buffer" in tmp_config.signal_buffer_path
        assert "experience_pool" in tmp_config.experience_pool_path


class TestQuickThinkResult:
    def test_no_hit(self):
        result = QuickThinkResult(hit=False)
        assert result.matched_signals == []
        assert result.risk_level == "none"

    def test_hit(self):
        result = QuickThinkResult(
            hit=True,
            matched_signals=["keyword_taxonomy_hit"],
            matched_taxonomy_entries=["tax-001"],
            risk_level="low",
        )
        assert result.hit is True


class TestTaskContext:
    def test_minimal(self):
        ctx = TaskContext(task_description="hello")
        assert ctx.step_count == 0
        assert ctx.errors_fixed is False
        assert ctx.new_tools == []
