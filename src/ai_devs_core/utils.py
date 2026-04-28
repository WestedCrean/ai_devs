import tiktoken


def count_tokens(message: str) -> int:
    """
    Get token count

    Parameters:
        message: str - message to count

    Returns
        int - token count
    """
    try:
        enc = tiktoken.encoding_for_model("gpt-5-2")
        return len(enc.encode(message))
    except Exception:
        return len(message.split())
