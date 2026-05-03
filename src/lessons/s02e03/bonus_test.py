from src.lessons.s02e03.main import _extract_bonus_token_chars


def test_extract_bonus_token_char_from_string_value() -> None:
    """Single-character token fields should decode as-is."""
    assert _extract_bonus_token_chars({"token": "F"}) == ["F"]


def test_extract_bonus_token_char_from_integer_value() -> None:
    """Integer token fields should decode as printable ASCII."""
    assert _extract_bonus_token_chars({"response": {"token": 70}}) == ["F"]


def test_extract_bonus_token_chars_from_integer_list() -> None:
    """Token lists should decode to multiple printable ASCII characters."""
    assert _extract_bonus_token_chars({"tokens": [70, 76, 65, 71]}) == list("FLAG")


def test_extract_bonus_token_chars_ignores_normal_feedback() -> None:
    """Normal rejected verify feedback without token fields should not decode."""
    payload = {"status": 406, "message": "wrong answer", "hint": "missing components"}

    assert _extract_bonus_token_chars(payload) == []
