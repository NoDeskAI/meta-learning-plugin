from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass
class MockMemory:
    memory_date: date
    error_type: str
    content: str

    @property
    def filename(self) -> str:
        return f"{self.memory_date.isoformat()}.md"

    def write_to(self, memory_dir: Path) -> Path:
        memory_dir.mkdir(parents=True, exist_ok=True)
        path = memory_dir / self.filename
        path.write_text(self.content, encoding="utf-8")
        return path


def _ts_type_error_memory() -> MockMemory:
    return MockMemory(
        memory_date=date(2026, 3, 9),
        error_type="ts_type_error",
        content="""\
# 2026-03-09 Session Notes

## Task
Fix TypeScript type errors in `UserList.tsx` component.

## Errors Encountered
- **TS2345**: `Argument of type 'string | undefined' is not assignable to parameter of type 'string'`
  - Location: `src/components/UserList.tsx:28`
  - Root cause: `params.id` from Next.js dynamic route can be `undefined`
- **TS2322**: Missing required `data` prop on generic `Props<User>` component

## Resolution
1. Added `if (!userId) return <NotFound />;` guard before calling `getUserById`
2. Passed explicit `data` prop to generic component
3. Verified no remaining type errors via `tsc --noEmit`

## Lessons Learned
- Always check Next.js route params for `undefined` — they aren't guaranteed
- Generic component props need all required fields even in JSX spread patterns
""",
    )


def _git_conflict_memory() -> MockMemory:
    return MockMemory(
        memory_date=date(2026, 3, 10),
        error_type="git_conflict",
        content="""\
# 2026-03-10 Session Notes

## Task
Resolve merge conflicts after rebasing feature branch onto main.

## Conflicts
- `src/api/handlers.py` — API version mismatch (v1 vs v2)
- `src/config/settings.py` — overlapping config additions
- `tests/test_handlers.py` — assertion values outdated

## Resolution
- Kept v2 API endpoints from main, adapted feature branch payload format
- Merged both config additions (no logical overlap)
- Updated test assertions to match v2 response schema

## Lessons Learned
- Rebase frequently to reduce conflict surface
- API version changes must update client payloads AND test fixtures
""",
    )


def _api_param_error_memory() -> MockMemory:
    return MockMemory(
        memory_date=date(2026, 3, 11),
        error_type="api_parameter_error",
        content="""\
# 2026-03-11 Session Notes

## Task
Debug 422 Unprocessable Entity on POST /api/projects endpoint.

## Errors Encountered
- `value_error.datetime` on `deadline` field — client sent Unix timestamp, server expected ISO 8601
- `value_error.number.not_gt` on `team_size` — value was 0, must be > 0

## Resolution
1. Client: convert `deadline` to `.isoformat()` before sending
2. Client: guard `team_size` with `max(1, team_size)`
3. Server: added `@field_validator` to accept both formats

## Lessons Learned
- Always document expected date formats in API schema
- Add Pydantic validators for common format coercions
- Client-side validation should match server-side constraints
""",
    )


def _import_module_error_memory() -> MockMemory:
    return MockMemory(
        memory_date=date(2026, 3, 12),
        error_type="import_module_error",
        content="""\
# 2026-03-12 Session Notes

## Task
Fix `ModuleNotFoundError` on application startup.

## Errors Encountered
- `ModuleNotFoundError: No module named 'src.services.analytics'`
  - `pyproject.toml` maps packages differently than import paths
- `ImportError: circular import` between `services.auth` and `services.users`

## Resolution
1. Fixed import path: `from services.analytics import AnalyticsService`
2. Created missing `services/__init__.py`
3. Broke circular import by extracting shared types to `services/types.py`
4. Used `TYPE_CHECKING` guard for runtime-unnecessary imports

## Lessons Learned
- Verify `pyproject.toml` package mappings match intended import paths
- Circular imports indicate shared types should be extracted
- `TYPE_CHECKING` guards break runtime cycles while keeping type hints
""",
    )


def _runtime_null_error_memory() -> MockMemory:
    return MockMemory(
        memory_date=date(2026, 3, 13),
        error_type="runtime_null_reference",
        content="""\
# 2026-03-13 Session Notes

## Task
Fix production TypeError: Cannot read properties of undefined (reading 'map').

## Errors Encountered
- `data.results` was `undefined` — API v2.3 returns `{}` for empty sets instead of `{"results": []}`
- Component assumed `results` always exists

## Resolution
1. Applied optional chaining: `(data?.results ?? []).map(...)`
2. Added Zod schema validation at API boundary: `DashboardResponseSchema.parse(response.data)`
3. Added Sentry breadcrumb for schema mismatches

## Lessons Learned
- Never trust API response shape — validate at boundary
- Defensive coding with `??` prevents null reference cascades
- Schema validation libraries (Zod) catch contract changes early
""",
    )


def _env_config_error_memory() -> MockMemory:
    return MockMemory(
        memory_date=date(2026, 3, 14),
        error_type="env_config_error",
        content="""\
# 2026-03-14 Session Notes

## Task
Fix deployment failure: 'DATABASE_URL environment variable is not set'.

## Errors Encountered
- Docker Compose uses `DB_URL`, application expects `DATABASE_URL`
- No `.env.example` to document required variables

## Resolution
1. Aligned env var naming: `DATABASE_URL` everywhere
2. Created `.env.example` with all required vars
3. Added startup check printing missing env vars before crash
4. Added `dotenv` support for local dev (gated behind `ENV=development`)

## Lessons Learned
- Standardize env var naming in a single source of truth (`.env.example`)
- Fail fast with descriptive error on missing config
- Gate dev-only tooling behind environment checks
""",
    )


def _permission_denied_memory() -> MockMemory:
    return MockMemory(
        memory_date=date(2026, 3, 15),
        error_type="permission_denied",
        content="""\
# 2026-03-15 Session Notes

## Task
Fix CI pipeline 'Permission denied' when writing to /var/cache/builds.

## Errors Encountered
- Non-root CI runner (uid 1001) lacks write access to `/var/cache/`
- Docker volume mount inherits root ownership

## Resolution
1. Dockerfile: `chown app:app /var/cache/builds` after mkdir
2. CI config: switched to user-writable `$HOME/.cache/builds`
3. Verified permissions with `ls -la`

## Lessons Learned
- Non-root containers need explicit `chown` for mounted volumes
- Prefer user-local paths (`$HOME/.cache`) over system paths in CI
- Always verify file permissions after container image changes
""",
    )


MEMORY_SCENARIOS: list[MockMemory] = [
    _ts_type_error_memory(),
    _git_conflict_memory(),
    _api_param_error_memory(),
    _import_module_error_memory(),
    _runtime_null_error_memory(),
    _env_config_error_memory(),
    _permission_denied_memory(),
]


def generate_memory(scenario_index: int) -> MockMemory:
    return MEMORY_SCENARIOS[scenario_index]


def generate_all_memories(memory_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for memory in MEMORY_SCENARIOS:
        paths.append(memory.write_to(memory_dir))
    return paths
