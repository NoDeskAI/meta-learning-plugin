from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MockSession:
    session_id: str
    error_type: str
    messages: list[dict[str, str]] = field(default_factory=list)

    def to_jsonl(self) -> str:
        return "\n".join(json.dumps(msg, ensure_ascii=False) for msg in self.messages)

    def write_to(self, sessions_dir: Path) -> Path:
        sessions_dir.mkdir(parents=True, exist_ok=True)
        path = sessions_dir / f"{self.session_id}.jsonl"
        path.write_text(self.to_jsonl(), encoding="utf-8")
        return path


def _ts_type_error_session() -> MockSession:
    return MockSession(
        session_id="ses-mock-ts-type-001",
        error_type="ts_type_error",
        messages=[
            {
                "role": "user",
                "content": "Fix the type error in src/components/UserList.tsx",
            },
            {
                "role": "assistant",
                "content": (
                    "I see the error:\n"
                    "```\n"
                    "TS2345: Argument of type 'string | undefined' is not assignable "
                    "to parameter of type 'string'.\n"
                    "  Type 'undefined' is not assignable to type 'string'.\n"
                    "```\n"
                    "The `getUserById` function expects a `string` but `params.id` "
                    "can be `undefined`. Let me add a guard clause."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "I've edited `src/components/UserList.tsx`:\n"
                    "```tsx\n"
                    "const userId = params.id;\n"
                    "if (!userId) return <NotFound />;\n"
                    "const user = getUserById(userId);\n"
                    "```\n"
                    "The type error is resolved."
                ),
            },
            {"role": "user", "content": "Good, also check the generic Props type."},
            {
                "role": "assistant",
                "content": (
                    "Found another issue:\n"
                    "```\n"
                    "TS2322: Type '{ onClick: () => void; }' is not assignable to "
                    "type 'IntrinsicAttributes & Props<User>'.\n"
                    "  Property 'data' is missing in type '{ onClick: () => void; }' "
                    "but required in type 'Props<User>'.\n"
                    "```\n"
                    "Added the missing `data` prop to the component call."
                ),
            },
        ],
    )


def _git_conflict_session() -> MockSession:
    return MockSession(
        session_id="ses-mock-git-conflict-002",
        error_type="git_conflict",
        messages=[
            {
                "role": "user",
                "content": "I have merge conflicts after rebasing onto main. Help me resolve them.",
            },
            {
                "role": "assistant",
                "content": (
                    "Running `git status` shows conflicts in 3 files:\n"
                    "```\n"
                    "both modified:   src/api/handlers.py\n"
                    "both modified:   src/config/settings.py\n"
                    "both modified:   tests/test_handlers.py\n"
                    "```\n"
                    "Let me examine each conflict."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "In `src/api/handlers.py`:\n"
                    "```\n"
                    "<<<<<<< HEAD\n"
                    "    response = await client.post('/api/v2/users', json=payload)\n"
                    "=======\n"
                    "    response = await client.post('/api/v1/users', json=payload)\n"
                    ">>>>>>> feature/user-api\n"
                    "```\n"
                    "Main has migrated to v2 API. I'll keep the v2 endpoint and "
                    "adapt the feature branch's payload format to match."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "Resolved all 3 conflicts:\n"
                    "- `handlers.py`: kept v2 endpoint, updated payload schema\n"
                    "- `settings.py`: merged both config additions (no overlap)\n"
                    "- `test_handlers.py`: updated test assertions for v2 response format\n\n"
                    "Running `git add . && git rebase --continue`."
                ),
            },
            {"role": "user", "content": "Looks right. Continue the rebase."},
            {
                "role": "assistant",
                "content": "Rebase completed successfully. All tests pass on the rebased branch.",
            },
        ],
    )


def _api_param_error_session() -> MockSession:
    return MockSession(
        session_id="ses-mock-api-param-003",
        error_type="api_parameter_error",
        messages=[
            {
                "role": "user",
                "content": "The POST /api/projects endpoint returns 422. Can you debug it?",
            },
            {
                "role": "assistant",
                "content": (
                    "Looking at the error response:\n"
                    "```json\n"
                    "{\n"
                    '  "detail": [\n'
                    "    {\n"
                    '      "loc": ["body", "deadline"],\n'
                    '      "msg": "value is not a valid datetime",\n'
                    '      "type": "value_error.datetime"\n'
                    "    },\n"
                    "    {\n"
                    '      "loc": ["body", "team_size"],\n'
                    '      "msg": "ensure this value is greater than 0",\n'
                    '      "type": "value_error.number.not_gt"\n'
                    "    }\n"
                    "  ]\n"
                    "}\n"
                    "```\n"
                    "Two validation errors: `deadline` expects ISO 8601 datetime "
                    "but received a Unix timestamp, and `team_size` must be > 0."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "Fixed the client code:\n"
                    "```python\n"
                    "payload = {\n"
                    '    "name": project_name,\n'
                    '    "deadline": deadline.isoformat(),  # was: int(deadline.timestamp())\n'
                    '    "team_size": max(1, team_size),    # guard against 0\n'
                    "}\n"
                    "```\n"
                    "The endpoint now returns 201 Created."
                ),
            },
            {
                "role": "user",
                "content": "Also add validation on the server side for the date format.",
            },
            {
                "role": "assistant",
                "content": (
                    "Added a Pydantic validator in `ProjectCreate`:\n"
                    "```python\n"
                    "@field_validator('deadline', mode='before')\n"
                    "@classmethod\n"
                    "def parse_deadline(cls, v):\n"
                    "    if isinstance(v, (int, float)):\n"
                    "        return datetime.fromtimestamp(v, tz=timezone.utc)\n"
                    "    return v\n"
                    "```\n"
                    "Now accepts both ISO strings and Unix timestamps."
                ),
            },
        ],
    )


def _import_module_error_session() -> MockSession:
    return MockSession(
        session_id="ses-mock-import-err-004",
        error_type="import_module_error",
        messages=[
            {
                "role": "user",
                "content": "App crashes on startup with ModuleNotFoundError.",
            },
            {
                "role": "assistant",
                "content": (
                    "The traceback shows:\n"
                    "```\n"
                    "Traceback (most recent call last):\n"
                    '  File "src/main.py", line 3, in <module>\n'
                    "    from src.services.analytics import AnalyticsService\n"
                    "ModuleNotFoundError: No module named 'src.services.analytics'\n"
                    "```\n"
                    "The issue is the import path. The project uses `src` as a "
                    "namespace package but `pyproject.toml` maps it differently. "
                    "Checking the package configuration."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "Found the root cause: `pyproject.toml` has "
                    '`packages = ["src/services"]` but the import uses '
                    "`from src.services.analytics`. After installing with pip, "
                    "the correct import is `from services.analytics import AnalyticsService`.\n\n"
                    "Also, `services/analytics.py` was missing `__init__.py` in the "
                    "`services/` directory. Created it."
                ),
            },
            {
                "role": "user",
                "content": "There's also a circular import between services.auth and services.users.",
            },
            {
                "role": "assistant",
                "content": (
                    "Confirmed circular import:\n"
                    "```\n"
                    "ImportError: cannot import name 'UserService' from partially "
                    "initialized module 'services.users' (most likely due to a "
                    "circular import)\n"
                    "```\n"
                    "Broke the cycle by extracting shared types into "
                    "`services/types.py` and using `TYPE_CHECKING` for the "
                    "runtime-unnecessary imports."
                ),
            },
        ],
    )


def _runtime_null_error_session() -> MockSession:
    return MockSession(
        session_id="ses-mock-runtime-null-005",
        error_type="runtime_null_reference",
        messages=[
            {
                "role": "user",
                "content": "Getting TypeError: Cannot read properties of undefined (reading 'map') in production.",
            },
            {
                "role": "assistant",
                "content": (
                    "The error occurs in `Dashboard.tsx` line 42:\n"
                    "```tsx\n"
                    "const items = data.results.map(r => <Card key={r.id} {...r} />);\n"
                    "```\n"
                    "`data.results` is `undefined` when the API returns an empty "
                    'response body `{}` instead of `{"results": []}`. '
                    "The API contract changed in v2.3."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "Applied defensive coding:\n"
                    "```tsx\n"
                    "const items = (data?.results ?? []).map(r => <Card key={r.id} {...r} />);\n"
                    "```\n"
                    "Also added a response schema validation in the API client:\n"
                    "```ts\n"
                    "const parsed = DashboardResponseSchema.parse(response.data);\n"
                    "```\n"
                    "This ensures `results` is always an array at the boundary."
                ),
            },
            {
                "role": "user",
                "content": "Add a Sentry breadcrumb too so we catch these earlier.",
            },
            {
                "role": "assistant",
                "content": (
                    "Added Sentry integration:\n"
                    "```ts\n"
                    "Sentry.addBreadcrumb({\n"
                    "  category: 'api',\n"
                    "  message: `Dashboard API response missing 'results' field`,\n"
                    "  level: 'warning',\n"
                    "  data: { endpoint: '/api/dashboard', status: response.status },\n"
                    "});\n"
                    "```\n"
                    "Now any schema mismatch is logged before the fallback kicks in."
                ),
            },
        ],
    )


def _env_config_error_session() -> MockSession:
    return MockSession(
        session_id="ses-mock-env-config-006",
        error_type="env_config_error",
        messages=[
            {
                "role": "user",
                "content": "Deployment fails: 'DATABASE_URL environment variable is not set'.",
            },
            {
                "role": "assistant",
                "content": (
                    "The error comes from `src/config.py` line 15:\n"
                    "```python\n"
                    "DATABASE_URL: str = Field(..., env='DATABASE_URL')\n"
                    "```\n"
                    "Using `Field(...)` makes it required with no default. In the "
                    "Docker Compose config, the env var is set as `DB_URL` not "
                    "`DATABASE_URL`. There's a naming mismatch."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "Fixed by aligning the env var names:\n"
                    "- Updated `docker-compose.yml`: `DATABASE_URL=${DATABASE_URL}`\n"
                    "- Added `.env.example` with all required vars documented\n"
                    "- Added a startup check that prints missing env vars before crashing:\n"
                    "```python\n"
                    "missing = [k for k in REQUIRED_ENV if k not in os.environ]\n"
                    "if missing:\n"
                    "    sys.exit(f\"Missing env vars: {', '.join(missing)}\")\n"
                    "```"
                ),
            },
            {"role": "user", "content": "Good. Also add a fallback for local dev."},
            {
                "role": "assistant",
                "content": (
                    "Added `dotenv` support for local development:\n"
                    "```python\n"
                    "if os.getenv('ENV', 'development') == 'development':\n"
                    "    from dotenv import load_dotenv\n"
                    "    load_dotenv()\n"
                    "```\n"
                    "This only activates outside production. The `.env` file is "
                    "gitignored and `.env.example` serves as the template."
                ),
            },
        ],
    )


def _permission_denied_session() -> MockSession:
    return MockSession(
        session_id="ses-mock-perm-denied-007",
        error_type="permission_denied",
        messages=[
            {
                "role": "user",
                "content": "CI pipeline fails with 'Permission denied' when writing to /var/cache/builds.",
            },
            {
                "role": "assistant",
                "content": (
                    "The build step runs:\n"
                    "```\n"
                    "mkdir -p /var/cache/builds && cp -r dist/* /var/cache/builds/\n"
                    "```\n"
                    "The CI runner (uid 1001) doesn't have write permissions to "
                    "`/var/cache/`. The Dockerfile runs as non-root but the volume "
                    "mount inherits root ownership."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "Fixed in two places:\n"
                    "1. Dockerfile: `RUN mkdir -p /var/cache/builds && chown app:app /var/cache/builds`\n"
                    "2. CI config: switched to user-writable path `$HOME/.cache/builds`\n\n"
                    "The pipeline now passes. Verified with `ls -la /var/cache/builds` "
                    "showing correct ownership."
                ),
            },
        ],
    )


SESSION_SCENARIOS: list[MockSession] = [
    _ts_type_error_session(),
    _git_conflict_session(),
    _api_param_error_session(),
    _import_module_error_session(),
    _runtime_null_error_session(),
    _env_config_error_session(),
    _permission_denied_session(),
]


def generate_session(scenario_index: int) -> MockSession:
    return SESSION_SCENARIOS[scenario_index]


def generate_all_sessions(sessions_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for session in SESSION_SCENARIOS:
        paths.append(session.write_to(sessions_dir))
    return paths
