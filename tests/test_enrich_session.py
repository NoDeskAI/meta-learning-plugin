import json
from pathlib import Path

from meta_learning.shared.io import enrich_from_session
from meta_learning.shared.models import MetaLearningConfig


def _make_config(tmp_path: Path) -> MetaLearningConfig:
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


def _write_session(sessions_dir: Path, session_id: str, records: list[dict]) -> Path:
    p = sessions_dir / f"{session_id}.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records))
    return p


class TestEnrichFromSession:
    def test_normal_session(self, tmp_path):
        config = _make_config(tmp_path)
        records = [
            {"role": "user", "content": "修改 .env 文件"},
            {
                "role": "assistant",
                "content": "好的",
                "tool_calls": [
                    {
                        "function": {
                            "name": "read_file",
                            "arguments": json.dumps({"path": "/home/user/.env"}),
                        }
                    }
                ],
            },
            {"role": "tool", "content": 'VAR="hello"'},
            {
                "role": "assistant",
                "content": "现在修改",
                "tool_calls": [
                    {
                        "function": {
                            "name": "edit_file",
                            "arguments": json.dumps(
                                {
                                    "path": "/home/user/.env",
                                    "old_text": 'VAR="hello"',
                                    "new_text": 'VAR="hello"\nVAR2="world"',
                                }
                            ),
                        }
                    }
                ],
            },
            {"role": "tool", "content": "Successfully edited"},
            {"role": "assistant", "content": "已完成"},
        ]
        _write_session(Path(config.sessions_full_path), "sess-001", records)

        result = enrich_from_session("sess-001", config)
        assert result.tools_used == ["read_file", "edit_file"]
        assert result.step_count == 2
        assert result.action_trace is not None
        assert "read_file(/home/user/.env)" in result.action_trace
        assert "edit_file(/home/user/.env)" in result.action_trace
        assert " → " in result.action_trace

    def test_session_not_found(self, tmp_path):
        config = _make_config(tmp_path)
        result = enrich_from_session("nonexistent", config)
        assert result.tools_used == []
        assert result.step_count == 0
        assert result.action_trace is None

    def test_empty_session(self, tmp_path):
        config = _make_config(tmp_path)
        sessions_dir = Path(config.sessions_full_path)
        (sessions_dir / "empty.jsonl").write_text("")

        result = enrich_from_session("empty", config)
        assert result.tools_used == []
        assert result.step_count == 0
        assert result.action_trace is None

    def test_no_tool_calls(self, tmp_path):
        config = _make_config(tmp_path)
        records = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！有什么可以帮你的？"},
        ]
        _write_session(Path(config.sessions_full_path), "chat-only", records)

        result = enrich_from_session("chat-only", config)
        assert result.tools_used == []
        assert result.step_count == 0
        assert result.action_trace is None

    def test_filters_meta_learning_tools(self, tmp_path):
        config = _make_config(tmp_path)
        records = [
            {
                "role": "assistant",
                "content": "操作完成",
                "tool_calls": [
                    {
                        "function": {
                            "name": "edit_file",
                            "arguments": json.dumps({"path": "/tmp/test.py"}),
                        }
                    }
                ],
            },
            {
                "role": "assistant",
                "content": "记录信号",
                "tool_calls": [
                    {
                        "function": {
                            "name": "mcp_meta-learning_capture_signal",
                            "arguments": json.dumps(
                                {"task_description": "test", "user_corrections": ["x"]}
                            ),
                        }
                    }
                ],
            },
        ]
        _write_session(Path(config.sessions_full_path), "with-meta", records)

        result = enrich_from_session("with-meta", config)
        assert result.tools_used == ["edit_file"]
        assert result.step_count == 1
        assert "capture_signal" not in (result.action_trace or "")

    def test_tool_without_path_arg(self, tmp_path):
        config = _make_config(tmp_path)
        records = [
            {
                "role": "assistant",
                "content": "搜索中",
                "tool_calls": [
                    {
                        "function": {
                            "name": "web_search",
                            "arguments": json.dumps({"query": "python datetime"}),
                        }
                    }
                ],
            },
        ]
        _write_session(Path(config.sessions_full_path), "no-path", records)

        result = enrich_from_session("no-path", config)
        assert result.tools_used == ["web_search"]
        assert result.action_trace == "web_search"

    def test_deduplicates_tools(self, tmp_path):
        config = _make_config(tmp_path)
        records = [
            {
                "role": "assistant",
                "content": "读取",
                "tool_calls": [
                    {
                        "function": {
                            "name": "read_file",
                            "arguments": json.dumps({"path": "/a.txt"}),
                        }
                    }
                ],
            },
            {
                "role": "assistant",
                "content": "再读取",
                "tool_calls": [
                    {
                        "function": {
                            "name": "read_file",
                            "arguments": json.dumps({"path": "/b.txt"}),
                        }
                    }
                ],
            },
        ]
        _write_session(Path(config.sessions_full_path), "dedup", records)

        result = enrich_from_session("dedup", config)
        assert result.tools_used == ["read_file"]
        assert result.step_count == 2
        assert result.action_trace == "read_file(/a.txt) → read_file(/b.txt)"

    def test_multiple_tool_calls_in_one_message(self, tmp_path):
        config = _make_config(tmp_path)
        records = [
            {
                "role": "assistant",
                "content": "并行操作",
                "tool_calls": [
                    {
                        "function": {
                            "name": "read_file",
                            "arguments": json.dumps({"path": "/a.txt"}),
                        }
                    },
                    {
                        "function": {
                            "name": "read_file",
                            "arguments": json.dumps({"path": "/b.txt"}),
                        }
                    },
                ],
            },
        ]
        _write_session(Path(config.sessions_full_path), "multi-tc", records)

        result = enrich_from_session("multi-tc", config)
        assert result.tools_used == ["read_file"]
        assert result.step_count == 1
        assert "read_file(/a.txt)" in result.action_trace
        assert "read_file(/b.txt)" in result.action_trace

    def test_skips_metadata_line(self, tmp_path):
        """Session JSONL 第一行通常是 metadata（无 role 字段），应被跳过。"""
        config = _make_config(tmp_path)
        records = [
            {
                "_type": "metadata",
                "key": "agent:main:desk-abc",
                "created_at": "2026-04-01T10:00:00",
            },
            {
                "role": "assistant",
                "content": "操作",
                "tool_calls": [
                    {"function": {"name": "exec", "arguments": "{}"}}
                ],
            },
        ]
        _write_session(Path(config.sessions_full_path), "with-meta-line", records)

        result = enrich_from_session("with-meta-line", config)
        assert result.tools_used == ["exec"]
        assert result.step_count == 1

    def test_action_trace_truncation(self, tmp_path):
        config = _make_config(tmp_path)
        records = [
            {
                "role": "assistant",
                "content": f"step {i}",
                "tool_calls": [
                    {
                        "function": {
                            "name": "read_file",
                            "arguments": json.dumps(
                                {"path": f"/very/long/path/{'x' * 200}/file_{i}.txt"}
                            ),
                        }
                    }
                ],
            }
            for i in range(50)
        ]
        _write_session(Path(config.sessions_full_path), "long-trace", records)

        result = enrich_from_session("long-trace", config)
        assert result.action_trace is not None
        assert len(result.action_trace) <= 2000
        assert result.action_trace.endswith("...")

    def test_nanobot_session_id_resolution(self, tmp_path):
        """session_id='desk-abc' 应能匹配 'agent_main_desk-abc.jsonl'。"""
        config = _make_config(tmp_path)
        records = [
            {
                "role": "assistant",
                "content": "ok",
                "tool_calls": [
                    {"function": {"name": "ls", "arguments": "{}"}}
                ],
            },
        ]
        sessions_dir = Path(config.sessions_full_path)
        _write_session(sessions_dir, "agent_main_desk-abc", records)

        result = enrich_from_session("desk-abc", config)
        assert result.tools_used == ["ls"]
