import json

import pytest
from prime_cli.commands.evals import (
    _has_eval_files,
    _load_eval_directory,
    _validate_eval_path,
)


class TestHasEvalFiles:
    """Tests for _has_eval_files function"""

    def test_has_both_files(self, tmp_path):
        """Directory with both files returns True"""
        (tmp_path / "metadata.json").write_text("{}")
        (tmp_path / "results.jsonl").write_text("")
        assert _has_eval_files(tmp_path) is True

    def test_missing_metadata(self, tmp_path):
        """Directory missing metadata.json returns False"""
        (tmp_path / "results.jsonl").write_text("")
        assert _has_eval_files(tmp_path) is False

    def test_missing_results(self, tmp_path):
        """Directory missing results.jsonl returns False"""
        (tmp_path / "metadata.json").write_text("{}")
        assert _has_eval_files(tmp_path) is False

    def test_empty_directory(self, tmp_path):
        """Empty directory returns False"""
        assert _has_eval_files(tmp_path) is False


class TestValidateEvalPath:
    """Tests for _validate_eval_path function"""

    def test_valid_directory(self, tmp_path):
        """Valid directory with both files returns the path"""
        (tmp_path / "metadata.json").write_text("{}")
        (tmp_path / "results.jsonl").write_text("")
        result = _validate_eval_path(str(tmp_path))
        assert result == tmp_path

    def test_metadata_file_autocorrects_to_parent(self, tmp_path):
        """Passing metadata.json auto-corrects to parent directory"""
        (tmp_path / "metadata.json").write_text("{}")
        (tmp_path / "results.jsonl").write_text("")

        result = _validate_eval_path(str(tmp_path / "metadata.json"))
        assert result == tmp_path

    def test_results_file_autocorrects_to_parent(self, tmp_path):
        """Passing results.jsonl auto-corrects to parent directory"""
        (tmp_path / "metadata.json").write_text("{}")
        (tmp_path / "results.jsonl").write_text("")

        result = _validate_eval_path(str(tmp_path / "results.jsonl"))
        assert result == tmp_path

    def test_metadata_file_without_results_errors(self, tmp_path):
        """Passing metadata.json when results.jsonl missing raises error"""
        (tmp_path / "metadata.json").write_text("{}")

        with pytest.raises(ValueError) as exc_info:
            _validate_eval_path(str(tmp_path / "metadata.json"))

        assert "must contain both metadata.json and results.jsonl" in str(exc_info.value)

    def test_results_file_without_metadata_errors(self, tmp_path):
        """Passing results.jsonl when metadata.json missing raises error"""
        (tmp_path / "results.jsonl").write_text("")

        with pytest.raises(ValueError) as exc_info:
            _validate_eval_path(str(tmp_path / "results.jsonl"))

        assert "must contain both metadata.json and results.jsonl" in str(exc_info.value)

    def test_random_file_errors(self, tmp_path):
        """Passing a random file raises descriptive error"""
        random_file = tmp_path / "random.txt"
        random_file.write_text("hello")

        with pytest.raises(ValueError) as exc_info:
            _validate_eval_path(str(random_file))

        assert "Expected a directory path" in str(exc_info.value)
        assert "random.txt" in str(exc_info.value)

    def test_directory_missing_results(self, tmp_path):
        """Directory with only metadata.json raises specific error"""
        (tmp_path / "metadata.json").write_text("{}")

        with pytest.raises(ValueError) as exc_info:
            _validate_eval_path(str(tmp_path))

        assert "missing results.jsonl" in str(exc_info.value)

    def test_directory_missing_metadata(self, tmp_path):
        """Directory with only results.jsonl raises specific error"""
        (tmp_path / "results.jsonl").write_text("")

        with pytest.raises(ValueError) as exc_info:
            _validate_eval_path(str(tmp_path))

        assert "missing metadata.json" in str(exc_info.value)

    def test_directory_missing_both(self, tmp_path):
        """Empty directory raises error mentioning both files"""
        with pytest.raises(ValueError) as exc_info:
            _validate_eval_path(str(tmp_path))

        assert "missing both metadata.json and results.jsonl" in str(exc_info.value)

    def test_nonexistent_path(self):
        """Nonexistent path raises FileNotFoundError"""
        with pytest.raises(FileNotFoundError) as exc_info:
            _validate_eval_path("/nonexistent/path/xyz123")

        assert "Path not found" in str(exc_info.value)


class TestLoadEvalDirectory:
    """Tests for _load_eval_directory function"""

    def test_loads_valid_eval_data(self, tmp_path):
        """Loads and parses valid eval directory"""
        metadata = {
            "env": "gsm8k",
            "model": "gpt-4",
            "avg_reward": 0.85,
            "avg_accuracy": 0.9,
            "num_examples": 100,
        }
        (tmp_path / "metadata.json").write_text(json.dumps(metadata))

        results = [
            {"id": 0, "reward": 1.0, "answer": "42"},
            {"id": 1, "reward": 0.5, "answer": "18"},
        ]
        (tmp_path / "results.jsonl").write_text("\n".join(json.dumps(r) for r in results))

        data = _load_eval_directory(tmp_path)

        assert data["eval_name"] == "gsm8k-gpt-4"
        assert data["model_name"] == "gpt-4"
        assert data["env"] == "gsm8k"
        assert data["metrics"] == {"reward": 0.85, "accuracy": 0.9}
        assert data["metadata"]["num_examples"] == 100
        assert len(data["results"]) == 2
        assert data["results"][0]["example_id"] == 0
        assert data["results"][1]["example_id"] == 1

    def test_uses_env_id_field(self, tmp_path):
        """Uses env_id field if env is not present"""
        metadata = {"env_id": "math-problems", "model": "claude-3"}
        (tmp_path / "metadata.json").write_text(json.dumps(metadata))
        (tmp_path / "results.jsonl").write_text("")

        data = _load_eval_directory(tmp_path)

        assert data["eval_name"] == "math-problems-claude-3"
        assert data["env"] == "math-problems"

    def test_missing_env_field_raises(self, tmp_path):
        """Raises error if both env and env_id are missing"""
        metadata = {"model": "gpt-4"}
        (tmp_path / "metadata.json").write_text(json.dumps(metadata))
        (tmp_path / "results.jsonl").write_text("")

        with pytest.raises(ValueError) as exc_info:
            _load_eval_directory(tmp_path)

        assert "env_id" in str(exc_info.value)

    def test_missing_model_field_raises(self, tmp_path):
        """Raises error if model field is missing"""
        metadata = {"env": "gsm8k"}
        (tmp_path / "metadata.json").write_text(json.dumps(metadata))
        (tmp_path / "results.jsonl").write_text("")

        with pytest.raises(ValueError) as exc_info:
            _load_eval_directory(tmp_path)

        assert "model" in str(exc_info.value)

    def test_skips_invalid_jsonl_lines_with_warning(self, tmp_path, capsys):
        """Skips invalid JSON lines in results.jsonl and warns"""
        metadata = {"env": "test", "model": "test-model"}
        (tmp_path / "metadata.json").write_text(json.dumps(metadata))

        results_content = '{"id": 0, "reward": 1.0}\ninvalid json line\n{"id": 1, "reward": 0.5}'
        (tmp_path / "results.jsonl").write_text(results_content)

        data = _load_eval_directory(tmp_path)

        # Should only have 2 valid results, skipping the invalid line
        assert len(data["results"]) == 2

        # Should print a warning
        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "Skipped" in captured.out

    def test_converts_id_to_example_id(self, tmp_path):
        """Converts 'id' field to 'example_id' in results"""
        metadata = {"env": "test", "model": "test-model"}
        (tmp_path / "metadata.json").write_text(json.dumps(metadata))

        results = [{"id": 42, "reward": 1.0}]
        (tmp_path / "results.jsonl").write_text(json.dumps(results[0]))

        data = _load_eval_directory(tmp_path)

        assert data["results"][0]["example_id"] == 42
        assert data["results"][0]["id"] == 42  # Original field preserved

    def test_extracts_avg_metrics(self, tmp_path):
        """Extracts avg_* fields into metrics dict"""
        metadata = {
            "env": "test",
            "model": "test-model",
            "avg_reward": 0.75,
            "avg_correctness": 0.8,
            "avg_format_reward": 0.95,
            "other_field": "value",
        }
        (tmp_path / "metadata.json").write_text(json.dumps(metadata))
        (tmp_path / "results.jsonl").write_text("")

        data = _load_eval_directory(tmp_path)

        assert data["metrics"] == {
            "reward": 0.75,
            "correctness": 0.8,
            "format_reward": 0.95,
        }
        assert "avg_reward" not in data["metadata"]
        assert data["metadata"]["other_field"] == "value"

    def test_invalid_metadata_json_raises(self, tmp_path):
        """Raises JSONDecodeError if metadata.json is invalid"""
        (tmp_path / "metadata.json").write_text("not valid json {")
        (tmp_path / "results.jsonl").write_text("")

        with pytest.raises(json.JSONDecodeError):
            _load_eval_directory(tmp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
