from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from meta_learning.shared.models import MetaLearningConfig
from tests.mock_resources.memory import MEMORY_SCENARIOS, generate_all_memories
from tests.mock_resources.sessions import SESSION_SCENARIOS, generate_all_sessions


@dataclass
class MockEnvironment:
    config: MetaLearningConfig
    sessions_dir: Path
    memory_dir: Path
    session_paths: list[Path]
    memory_paths: list[Path]


def populate_mock_environment(base_dir: Path) -> MockEnvironment:
    workspace = base_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    sessions_dir = base_dir / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    memory_dir = base_dir / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)

    config = MetaLearningConfig(
        workspace_root=str(workspace),
        sessions_root=str(sessions_dir),
    )

    session_paths = generate_all_sessions(sessions_dir)
    memory_paths = generate_all_memories(memory_dir)

    return MockEnvironment(
        config=config,
        sessions_dir=sessions_dir,
        memory_dir=memory_dir,
        session_paths=session_paths,
        memory_paths=memory_paths,
    )


SESSION_ERROR_TYPES: list[str] = [s.error_type for s in SESSION_SCENARIOS]
MEMORY_ERROR_TYPES: list[str] = [m.error_type for m in MEMORY_SCENARIOS]
SESSION_IDS: list[str] = [s.session_id for s in SESSION_SCENARIOS]
