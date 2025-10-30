from prime_cli.commands.config import validate_team_id


class TestTeamIdValidation:
    """Test team ID validation logic."""

    def test_valid_team_id(self) -> None:
        """Test that valid team IDs pass validation."""
        valid_id = "cmf0ohr9s0026ilerf3w68s6n"
        assert validate_team_id(valid_id) is True

    def test_valid_team_id_with_uppercase(self) -> None:
        """Test that team IDs with uppercase letters are valid."""
        valid_id = "CMF0OHR9S0026ILERF3W68S6N"
        assert validate_team_id(valid_id) is True

    def test_valid_team_id_mixed_case(self) -> None:
        """Test that team IDs with mixed case are valid."""
        valid_id = "CmF0OhR9s0026IlErF3w68S6n"
        assert validate_team_id(valid_id) is True

    def test_empty_string_is_valid(self) -> None:
        """Test that empty string is valid (personal account)."""
        assert validate_team_id("") is True

    def test_invalid_team_id_too_short(self) -> None:
        """Test that team IDs shorter than 25 characters are invalid."""
        invalid_id = "cmf0ohr9s0026ilerf3w68s6"
        assert validate_team_id(invalid_id) is False

    def test_invalid_team_id_too_long(self) -> None:
        """Test that team IDs longer than 25 characters are invalid."""
        invalid_id = "cmf0ohr9s0026ilerf3w68s6nn"
        assert validate_team_id(invalid_id) is False

    def test_invalid_team_id_with_special_chars(self) -> None:
        """Test that team IDs with special characters are invalid."""
        invalid_id = "cmf0ohr9s0026ilerf3w68s-n"
        assert validate_team_id(invalid_id) is False

    def test_invalid_team_id_with_spaces(self) -> None:
        """Test that team IDs with spaces are invalid."""
        invalid_id = "cmf0ohr9s0026ilerf3w68s n"
        assert validate_team_id(invalid_id) is False

    def test_invalid_team_id_word(self) -> None:
        """Test that regular words are invalid."""
        invalid_id = "intertwine"
        assert validate_team_id(invalid_id) is False

    def test_invalid_team_id_with_underscore(self) -> None:
        """Test that team IDs with underscores are invalid."""
        invalid_id = "cmf0ohr9s0026ilerf3w68s_n"
        assert validate_team_id(invalid_id) is False
