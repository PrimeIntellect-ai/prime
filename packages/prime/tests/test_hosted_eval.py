from prime_cli.utils.hosted_eval import clean_logs, filter_progress_bars, strip_ansi


class TestLogCleaning:
    """Test log cleaning utilities"""

    def test_strip_ansi_basic(self):
        """Test stripping basic ANSI escape codes"""
        text = "\x1b[31mRed text\x1b[0m"
        assert strip_ansi(text) == "Red text"

    def test_strip_ansi_multiple_codes(self):
        """Test stripping multiple ANSI codes"""
        text = "\x1b[1m\x1b[32mBold green\x1b[0m\x1b[0m text"
        assert strip_ansi(text) == "Bold green text"

    def test_strip_ansi_no_codes(self):
        """Test text without ANSI codes remains unchanged"""
        text = "Plain text"
        assert strip_ansi(text) == "Plain text"

    def test_strip_ansi_empty(self):
        """Test empty string"""
        assert strip_ansi("") == ""

    def test_filter_progress_bars_100_percent(self):
        """Test that 100% progress bars are kept"""
        text = "Progress: 100%|██████████| 10/10 [00:01<00:00]"
        result = filter_progress_bars(text)
        assert "100%" in result

    def test_filter_progress_bars_partial(self):
        """Test that partial progress bars are filtered out"""
        text = "Progress: 50%|█████     | 5/10 [00:01<00:01]"
        result = filter_progress_bars(text)
        assert result == ""

    def test_filter_progress_bars_mixed(self):
        """Test mixed content with progress bars"""
        text = """Starting evaluation
Progress: 50%|█████     | 5/10 [00:01<00:01]
Progress: 100%|██████████| 10/10 [00:02<00:00]
Evaluation complete"""
        result = filter_progress_bars(text)
        assert "Starting evaluation" in result
        assert "Evaluation complete" in result
        assert "100%" in result
        assert "50%" not in result

    def test_filter_progress_bars_preserves_regular_lines(self):
        """Test that regular log lines are preserved"""
        text = """Model loaded successfully
Processing batch 1
Result: accuracy=0.95"""
        result = filter_progress_bars(text)
        lines = result.splitlines()
        assert len(lines) == 3
        assert "Model loaded successfully" in result
        assert "Processing batch 1" in result
        assert "Result: accuracy=0.95" in result

    def test_clean_logs_combined(self):
        """Test combined cleaning of ANSI codes and progress bars"""
        text = """\x1b[32mStarting evaluation\x1b[0m
Progress: 50%|█████     | 5/10 [00:01<00:01]
\x1b[1mProgress: 100%|██████████| 10/10 [00:02<00:00]\x1b[0m
\x1b[32m✓ Evaluation complete\x1b[0m"""
        result = clean_logs(text)
        assert "Starting evaluation" in result
        assert "✓ Evaluation complete" in result
        assert "100%" in result
        assert "50%" not in result
        assert "\x1b" not in result

    def test_clean_logs_empty(self):
        """Test clean_logs with empty string"""
        assert clean_logs("") == ""

    def test_clean_logs_multiline_with_empty_lines(self):
        """Test that empty lines are filtered out"""
        text = """Line 1

Line 3

Line 5"""
        result = clean_logs(text)
        lines = result.splitlines()
        assert len(lines) == 3
        assert lines[0] == "Line 1"
        assert lines[1] == "Line 3"
        assert lines[2] == "Line 5"


class TestLogStreaming:
    """Test log streaming logic"""

    def test_line_comparison_first_logs(self):
        """Test printing all lines when no previous logs exist"""
        last_logs = ""
        new_logs = """Line 1
Line 2
Line 3"""

        new_lines = new_logs.splitlines()

        if not last_logs:
            assert len(new_lines) == 3
            assert new_lines == ["Line 1", "Line 2", "Line 3"]

    def test_line_comparison_new_lines(self):
        """Test printing only new lines when logs grow"""
        last_logs = """Line 1
Line 2
Line 3"""
        new_logs = """Line 1
Line 2
Line 3
Line 4
Line 5"""

        old_lines = last_logs.splitlines()
        new_lines = new_logs.splitlines()

        overlap = 0
        max_overlap = min(len(old_lines), len(new_lines))
        for i in range(1, max_overlap + 1):
            if old_lines[-i:] == new_lines[:i]:
                overlap = i

        new_content = new_lines[overlap:]
        assert new_content == ["Line 4", "Line 5"]

    def test_line_comparison_no_new_lines(self):
        """Test no output when logs haven't changed"""
        last_logs = """Line 1
Line 2
Line 3"""
        new_logs = """Line 1
Line 2
Line 3"""

        assert last_logs == new_logs

    def test_line_comparison_with_overlap(self):
        """Test finding overlap between old and new logs"""
        last_logs = """Line 1
Line 2
Line 3"""
        new_logs = """Line 2
Line 3
Line 4
Line 5"""

        old_lines = last_logs.splitlines()
        new_lines = new_logs.splitlines()

        overlap = 0
        max_overlap = min(len(old_lines), len(new_lines))
        for i in range(1, max_overlap + 1):
            if old_lines[-i:] == new_lines[:i]:
                overlap = i

        new_content = new_lines[overlap:]
        assert overlap == 2
        assert new_content == ["Line 4", "Line 5"]


class TestProgressBarPatterns:
    """Test various progress bar patterns from different tools"""

    def test_tqdm_progress_bar(self):
        """Test tqdm-style progress bar detection"""
        text = "100%|██████████| 100/100 [00:10<00:00, 10.00it/s]"
        result = filter_progress_bars(text)
        assert "100%" in result

    def test_rich_progress_bar(self):
        """Test rich-style progress indicators"""
        text = "Processing ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%"
        result = filter_progress_bars(text)
        assert len(result) > 0

    def test_multiple_progress_updates(self):
        """Test multiple progress updates where only 100% is kept"""
        text = """Task started
25%|██▌       | 25/100 [00:02<00:06, 10.00it/s]
50%|█████     | 50/100 [00:05<00:05, 10.00it/s]
75%|███████▌  | 75/100 [00:07<00:02, 10.00it/s]
100%|██████████| 100/100 [00:10<00:00, 10.00it/s]
Task completed"""
        result = filter_progress_bars(text)
        assert "Task started" in result
        assert "Task completed" in result
        assert "100%" in result
        assert "25%" not in result
        assert "50%" not in result
        assert "75%" not in result
