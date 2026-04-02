import json
from datetime import datetime
from pathlib import Path

from meta_learning.shared.io import (
    ensure_directories,
    list_all_experiences,
    list_pending_signals,
    load_config,
    load_error_taxonomy,
    load_experience_index,
    mark_signal_processed,
    next_experience_id,
    next_signal_id,
    next_taxonomy_id,
    read_experience,
    read_session_context,
    read_signal,
    save_error_taxonomy,
    save_experience_index,
    write_experience,
    write_signal,
)
from meta_learning.shared.models import ExperienceIndex


class TestConfigLoading:
    def test_load_default_when_missing(self, tmp_path):
        cfg = load_config(tmp_path / "nonexistent.yaml")
        assert cfg.llm.provider == "stub"

    def test_load_from_file(self, tmp_path):
        import yaml

        config_data = {"llm": {"provider": "openai", "model": "gpt-4"}}
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)
        cfg = load_config(config_file)
        assert cfg.llm.provider == "openai"
        assert cfg.llm.model == "gpt-4"


class TestSignalIO:
    def test_write_and_read(self, tmp_config, sample_signal):
        path = write_signal(sample_signal, tmp_config)
        assert path.exists()
        loaded = read_signal(path)
        assert loaded.signal_id == sample_signal.signal_id
        assert loaded.detection_channels == sample_signal.detection_channels
        assert loaded.primary_channel == sample_signal.primary_channel

    def test_list_pending_empty(self, tmp_config):
        assert list_pending_signals(tmp_config) == []

    def test_list_pending_filters_processed(self, tmp_config, sample_signal):
        write_signal(sample_signal, tmp_config)
        pending = list_pending_signals(tmp_config)
        assert len(pending) == 1

        mark_signal_processed(sample_signal.signal_id, tmp_config)
        pending = list_pending_signals(tmp_config)
        assert len(pending) == 0

    def test_next_signal_id(self, tmp_config):
        sid = next_signal_id(tmp_config)
        assert sid.startswith("sig-")
        today = datetime.now().strftime("%Y%m%d")
        assert today in sid


class TestExperienceIO:
    def test_write_and_read(self, tmp_config, sample_experience):
        path = write_experience(sample_experience, tmp_config)
        assert path.exists()
        loaded = read_experience(path)
        assert loaded.id == sample_experience.id
        assert loaded.task_type == sample_experience.task_type

    def test_list_all_empty(self, tmp_config):
        assert list_all_experiences(tmp_config) == []

    def test_list_all_with_data(self, tmp_config, sample_experience):
        write_experience(sample_experience, tmp_config)
        exps = list_all_experiences(tmp_config)
        assert len(exps) == 1

    def test_next_experience_id(self, tmp_config):
        eid = next_experience_id(tmp_config)
        assert eid == "exp-001"


class TestExperienceIndexIO:
    def test_load_missing_returns_default(self, tmp_config):
        idx = load_experience_index(tmp_config)
        assert idx.clusters == []

    def test_save_and_load(self, tmp_config):
        idx = ExperienceIndex(last_updated=datetime.now())
        save_experience_index(idx, tmp_config)
        loaded = load_experience_index(tmp_config)
        assert loaded.clusters == []


class TestErrorTaxonomyIO:
    def test_load_missing_returns_empty(self, tmp_config):
        tax = load_error_taxonomy(tmp_config)
        assert tax.all_entries() == []

    def test_save_and_load(self, tmp_config, sample_taxonomy):
        save_error_taxonomy(sample_taxonomy, tmp_config)
        loaded = load_error_taxonomy(tmp_config)
        entries = loaded.all_entries()
        assert len(entries) == 1
        assert entries[0].name == "Generic Type Inference Failure"

    def test_next_taxonomy_id(self, sample_taxonomy):
        tid = next_taxonomy_id(sample_taxonomy, "cod-typ")
        assert tid.startswith("tax-cod-typ-")


class TestSessionContext:
    def test_missing_session(self, tmp_config):
        result = read_session_context("nonexistent", tmp_config)
        assert "not found" in result

    def test_read_jsonl(self, tmp_config):
        sessions_dir = Path(tmp_config.sessions_full_path)
        session_file = sessions_dir / "test-session.jsonl"
        lines = [
            json.dumps({"role": "user", "content": "Fix the bug"}),
            json.dumps({"role": "assistant", "content": "I'll fix the type error"}),
        ]
        session_file.write_text("\n".join(lines))
        result = read_session_context("test-session", tmp_config)
        assert "[user] Fix the bug" in result
        assert "[assistant]" in result


class TestEnsureDirectories:
    def test_creates_dirs(self, tmp_config):
        ensure_directories(tmp_config)
        assert Path(tmp_config.signal_buffer_path).exists()
        assert Path(tmp_config.experience_pool_path).exists()


class TestIdCounterSequential:
    def test_signal_id_sequential_no_recount(self, tmp_config):
        id1 = next_signal_id(tmp_config)
        id2 = next_signal_id(tmp_config)
        num1 = int(id1.rsplit("-", 1)[1])
        num2 = int(id2.rsplit("-", 1)[1])
        assert num2 == num1 + 1

    def test_experience_id_sequential_no_recount(self, tmp_config):
        id1 = next_experience_id(tmp_config)
        id2 = next_experience_id(tmp_config)
        num1 = int(id1.split("-")[1])
        num2 = int(id2.split("-")[1])
        assert num2 == num1 + 1

    def test_id_counter_reset(self, tmp_config):
        from meta_learning.shared.io import reset_id_counters

        _ = next_experience_id(tmp_config)
        reset_id_counters()
        fresh_id = next_experience_id(tmp_config)
        assert fresh_id == "exp-001"
