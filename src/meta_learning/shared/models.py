from __future__ import annotations

import os
from datetime import date, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


def _env_or(env_key: str, default: str) -> str:
    return os.environ.get(env_key, default)


class TriggerReason(StrEnum):
    ERROR_RECOVERY = "error_recovery"
    USER_CORRECTION = "user_correction"
    NEW_TOOL = "new_tool"
    EFFICIENCY_ANOMALY = "efficiency_anomaly"


class TaskType(StrEnum):
    CODING = "coding"
    DEVOPS = "devops"
    WRITING = "writing"
    DEBUGGING = "debugging"
    CONFIGURATION = "configuration"
    CUSTOMER_SERVICE = "customer_service"
    PROFESSIONAL_DOCUMENT = "professional_document"
    UNCLASSIFIED = "_unclassified"


class ExperimentGroup(StrEnum):
    A = "A"
    B = "B"
    CONTROL = "control"


class ExperimentConfig(BaseModel):
    experiment_id: str = ""
    group: ExperimentGroup = ExperimentGroup.CONTROL
    enabled: bool = False


class Signal(BaseModel):
    signal_id: str
    timestamp: datetime
    session_id: str
    memory_date: date | None = None
    trigger_reason: TriggerReason
    keywords: list[str]
    task_summary: str
    error_snapshot: str | None = None
    resolution_snapshot: str | None = None
    user_feedback: str | None = None
    image_snapshots: list[str] = Field(default_factory=list)
    step_count: int
    processed: bool = False
    experiment_id: str | None = None
    experiment_group: str | None = None


class Experience(BaseModel):
    id: str
    task_type: TaskType
    created_at: datetime
    source_signal: str
    source_session: str | None = None
    source_memory: date | None = None
    confidence: float = 0.6
    verification_count: int = 1
    scene: str
    failure_signature: str | None = None
    root_cause: str
    resolution: str
    meta_insight: str
    related_exps: list[str] = Field(default_factory=list)
    promoted_to: str | None = None


class ExperienceCluster(BaseModel):
    cluster_id: str
    task_type: TaskType
    failure_signature_pattern: str
    experience_ids: list[str]
    promoted_to_taxonomy: str | None = None


class ExperienceIndex(BaseModel):
    last_updated: datetime
    clusters: list[ExperienceCluster] = Field(default_factory=list)


class TaxonomyEntry(BaseModel):
    id: str
    name: str
    trigger: str
    fix_sop: str
    prevention: str
    confidence: float
    source_exps: list[str]
    keywords: list[str] = Field(default_factory=list)
    created_at: date
    last_verified: date


class ErrorTaxonomy(BaseModel):
    taxonomy: dict[str, dict[str, list[TaxonomyEntry]]] = Field(default_factory=dict)

    def all_entries(self) -> list[TaxonomyEntry]:
        entries: list[TaxonomyEntry] = []
        for domain in self.taxonomy.values():
            for sublist in domain.values():
                entries.extend(sublist)
        return entries

    def all_keywords(self) -> dict[str, list[TaxonomyEntry]]:
        kw_map: dict[str, list[TaxonomyEntry]] = {}
        for entry in self.all_entries():
            for kw in entry.keywords:
                kw_lower = kw.lower()
                kw_map.setdefault(kw_lower, []).append(entry)
        return kw_map

    def add_entry(self, domain: str, subdomain: str, entry: TaxonomyEntry) -> None:
        self.taxonomy.setdefault(domain, {}).setdefault(subdomain, []).append(entry)

    def remove_entry(self, entry_id: str) -> bool:
        for domain in self.taxonomy.values():
            for subdomain, entries in domain.items():
                for i, entry in enumerate(entries):
                    if entry.id == entry_id:
                        entries.pop(i)
                        return True
        return False


class QuickThinkConfig(BaseModel):
    irreversible_keywords: list[str] = Field(
        default_factory=lambda: [
            "rm -rf",
            "drop table",
            "force push",
            "git push --force",
            "DELETE FROM",
            "overwrite",
            "truncate",
            "format disk",
            "fdisk",
            "mkfs",
        ]
    )
    max_latency_ms: int = 50


class SignalCaptureConfig(BaseModel):
    efficiency_anomaly_threshold: float = 2.0
    average_step_count: int = 5


class Layer1Config(BaseModel):
    quick_think: QuickThinkConfig = Field(default_factory=QuickThinkConfig)
    signal_capture: SignalCaptureConfig = Field(default_factory=SignalCaptureConfig)


class TriggerConfig(BaseModel):
    min_pending_signals: int = 5
    max_hours_since_last: int = 24


class MaterializeConfig(BaseModel):
    initial_confidence: float = 0.6


class ConsolidateConfig(BaseModel):
    min_cluster_size_for_taxonomy: int = 2
    use_llm_clustering: bool = True
    max_llm_calls_per_group: int = 50
    similarity_threshold: float = 0.3
    batch_size: int = 10


class TaxonomyConfig(BaseModel):
    min_confidence_for_skill: float = 0.8
    min_source_exps_for_skill: int = 5


class Layer2Config(BaseModel):
    trigger: TriggerConfig = Field(default_factory=TriggerConfig)
    materialize: MaterializeConfig = Field(default_factory=MaterializeConfig)
    consolidate: ConsolidateConfig = Field(default_factory=ConsolidateConfig)
    taxonomy: TaxonomyConfig = Field(default_factory=TaxonomyConfig)


class ConfidenceConfig(BaseModel):
    hit_success_boost: float = 0.1
    contradiction_penalty: float = 0.2
    prune_threshold: float = 0.3
    promote_threshold: float = 0.8
    decay_enabled: bool = True
    decay_base: float = 0.95


class DashScopeConfig(BaseModel):
    """Configuration for DashScope multimodal embedding (qwen3-vl-embedding).

    The API key is read from the ``DASHSCOPE_API_KEY`` environment variable by
    default.  A hard-coded fallback is provided **only** for local development;
    rotate / revoke it before deploying to production.
    """

    api_key: str = Field(
        default_factory=lambda: _env_or(
            "DASHSCOPE_API_KEY",
            "sk-dcae1026f5f34f748183bd66fcaaae89",  # dev-only fallback — rotate before prod
        )
    )
    base_url: str = (
        "https://dashscope.aliyuncs.com/api/v1"
        "/services/embeddings/multimodal-embedding/multimodal-embedding"
    )
    model: str = "qwen3-vl-embedding"
    dimension: int = 1024
    enabled: bool = True


class LLMConfig(BaseModel):
    provider: str = "stub"
    model: str = "gpt-4o"
    temperature: float = 0.3
    max_tokens: int = 2000


class QuickThinkResult(BaseModel):
    hit: bool
    matched_signals: list[str] = Field(default_factory=list)
    matched_taxonomy_entries: list[str] = Field(default_factory=list)
    risk_level: str = "none"


class TaskContext(BaseModel):
    """Context for a single task execution, used by Layer 1 signal capture.

    Capability boundary
    -------------------
    All fields MUST originate from **agent-observable data** — full conversation,
    tool call logs (names / args / results), coarse task outcome, and explicit
    user or supervisor corrections.

    Meta-learning is a **strategy and procedure learning system**.  It can learn:
      - tool invocation order and policy compliance decisions (accept / refuse)
      - multi-step procedure completeness (batch operations)
      - verification discipline (check before act)

    It CANNOT learn instance-specific parameters (exact flight numbers, monetary
    amounts, database primary keys) because those are not transferable across
    task instances.

    Fields like ``user_corrections`` may also carry QA supervisor feedback
    (e.g. NL assertion failures that describe *strategy-level* behaviours).
    """

    task_description: str
    tools_used: list[str] = Field(default_factory=list)
    errors_encountered: list[str] = Field(default_factory=list)
    errors_fixed: bool = False
    user_corrections: list[str] = Field(default_factory=list)
    step_count: int = 0
    session_id: str | None = None
    new_tools: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


class MaterializeResult(BaseModel):
    scene: str
    failure_signature: str | None = None
    root_cause: str
    resolution: str
    meta_insight: str
    task_type: TaskType = TaskType.UNCLASSIFIED


class ConsolidateJudgment(BaseModel):
    same_class: bool
    reason: str


class TaxonomyExtraction(BaseModel):
    name: str
    trigger: str
    fix_sop: str
    prevention: str
    keywords: list[str] = Field(default_factory=list)


class SkillUpdateAction(StrEnum):
    APPEND = "append"
    REPLACE = "replace"
    CREATE = "create"
    MERGE = "merge"
    SPLIT = "split"
    NONE = "none"


class SkillEvolveResult(BaseModel):
    action: SkillUpdateAction
    target_skill: str | None = None
    changes_description: str
    new_content: str | None = None
    version_bump: str | None = None


# ---------------------------------------------------------------------------
# Layer 3 models
# ---------------------------------------------------------------------------


class CrossTaskPattern(BaseModel):
    """A pattern discovered across different task types sharing a root cause."""

    pattern_id: str
    description: str
    shared_root_cause: str
    affected_task_types: list[TaskType]
    source_experience_ids: list[str]
    confidence: float
    meta_strategy: str
    created_at: datetime


class CapabilityGap(BaseModel):
    """An identified skill gap the agent should learn."""

    gap_id: str
    gap_type: str  # "failure" | "frequency" | "efficiency"
    description: str
    evidence_ids: list[str]
    suggested_action: str
    priority: float
    created_at: datetime


class MemoryAction(StrEnum):
    EXTRACT = "extract"
    CONSOLIDATE = "consolidate"
    PRUNE = "prune"


class MemoryRecommendation(BaseModel):
    """A recommendation for memory architecture optimization."""

    action: MemoryAction
    target: str
    reason: str
    content: str | None = None


class Layer3Result(BaseModel):
    cross_task_patterns: list[CrossTaskPattern] = Field(default_factory=list)
    capability_gaps: list[CapabilityGap] = Field(default_factory=list)
    memory_recommendations: list[MemoryRecommendation] = Field(default_factory=list)
    timestamp: datetime


class CrossTaskAnalysis(BaseModel):
    """LLM output for cross-task pattern analysis."""

    description: str
    shared_root_cause: str
    meta_strategy: str
    confidence: float


class CapabilityAnalysis(BaseModel):
    """LLM output for capability gap analysis."""

    description: str
    suggested_action: str
    priority: float


class MemoryAnalysis(BaseModel):
    """LLM output for memory architecture analysis."""

    recommendations: list[MemoryRecommendation] = Field(default_factory=list)


class Layer3Config(BaseModel):
    min_experiences_for_cross_task: int = 3
    min_pattern_confidence: float = 0.6
    efficiency_anomaly_factor: float = 2.0
    min_gap_occurrences: int = 3
    prune_confidence_threshold: float = 0.5
    prune_unused_days: int = 30


class MetaLearningConfig(BaseModel):
    workspace_root: str = Field(
        default_factory=lambda: _env_or(
            "OPENCLAW_WORKSPACE_ROOT", "~/.openclaw/workspace"
        )
    )
    sessions_root: str = Field(
        default_factory=lambda: _env_or(
            "OPENCLAW_SESSIONS_DIR", "~/.openclaw/agents/main/sessions"
        )
    )
    signal_buffer_dir: str = "signal_buffer"
    experience_pool_dir: str = "experience_pool"
    error_taxonomy_path: str = "error_taxonomy.yaml"
    skills_dir: str = "skills"
    layer1: Layer1Config = Field(default_factory=Layer1Config)
    layer2: Layer2Config = Field(default_factory=Layer2Config)
    layer3: Layer3Config = Field(default_factory=Layer3Config)
    confidence: ConfidenceConfig = Field(default_factory=ConfidenceConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    dashscope: DashScopeConfig = Field(default_factory=DashScopeConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)

    def resolve_workspace_path(self, relative: str) -> str:
        from pathlib import Path

        root = Path(self.workspace_root).expanduser()
        return str(root / relative)

    @property
    def signal_buffer_path(self) -> str:
        return self.resolve_workspace_path(self.signal_buffer_dir)

    @property
    def experience_pool_path(self) -> str:
        return self.resolve_workspace_path(self.experience_pool_dir)

    @property
    def error_taxonomy_full_path(self) -> str:
        return self.resolve_workspace_path(self.error_taxonomy_path)

    @property
    def skills_path(self) -> str:
        return self.resolve_workspace_path(self.skills_dir)

    @property
    def sessions_full_path(self) -> str:
        from pathlib import Path

        return str(Path(self.sessions_root).expanduser())
