from better_profanity import profanity


profanity.load_censor_words()

def is_clean(text: str) -> bool:
    """
    Returns True if text is clean (no profanity detected).
    """
    return not profanity.contains_profanity(text)
