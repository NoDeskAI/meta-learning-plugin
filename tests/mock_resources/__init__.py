"""Mock resources for OpenClaw integration tests.

Generates realistic session (JSONL) and memory (Markdown) directory structures
with diverse error scenarios for testing the meta-learning pipeline.
"""

from __future__ import annotations

from tests.mock_resources.fixtures import (
    populate_mock_environment,
)
from tests.mock_resources.memory import (
    MEMORY_SCENARIOS,
    MockMemory,
    generate_all_memories,
    generate_memory,
)
from tests.mock_resources.sessions import (
    SESSION_SCENARIOS,
    MockSession,
    generate_all_sessions,
    generate_session,
)

__all__ = [
    "SESSION_SCENARIOS",
    "MockSession",
    "generate_all_sessions",
    "generate_session",
    "MEMORY_SCENARIOS",
    "MockMemory",
    "generate_all_memories",
    "generate_memory",
    "populate_mock_environment",
]
