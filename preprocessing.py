"""
preprocessing.py
----------------
Text preprocessing pipeline for ToxiShield AI.
Handles all cleaning steps: HTML, URLs, emojis, punctuation,
stopwords, tokenization, and lemmatization.
"""

import re
import string
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# ── Download required NLTK data ─────────────────────────────────────────────
def download_nltk_resources():
    """Download all required NLTK corpora (idempotent)."""
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for path, pkg in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)

download_nltk_resources()

# ── Initialise shared objects ────────────────────────────────────────────────
_lemmatizer = WordNetLemmatizer()
_stop_words  = set(stopwords.words("english"))

# ── Individual cleaning helpers ──────────────────────────────────────────────
def lowercase(text: str) -> str:
    """Convert all characters to lowercase."""
    return text.lower()


def remove_html_tags(text: str) -> str:
    """Strip HTML / XML tags from text."""
    return re.sub(r"<[^>]+>", " ", text)


def remove_urls(text: str) -> str:
    """Remove HTTP/HTTPS URLs and bare www. links."""
    return re.sub(r"https?://\S+|www\.\S+", " ", text)


def remove_numbers(text: str) -> str:
    """Remove standalone digit sequences."""
    return re.sub(r"\b\d+\b", " ", text)


def remove_punctuation(text: str) -> str:
    """Delete all punctuation characters."""
    return text.translate(str.maketrans("", "", string.punctuation))


def remove_emojis(text: str) -> str:
    """Remove emoji and other non-ASCII unicode symbols."""
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"   # emoticons
        "\U0001F300-\U0001F5FF"   # symbols & pictographs
        "\U0001F680-\U0001F6FF"   # transport & map
        "\U0001F1E0-\U0001F1FF"   # flags
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(" ", text)


def remove_extra_spaces(text: str) -> str:
    """Collapse multiple whitespace characters into a single space."""
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    """Split text into word tokens using NLTK's word tokenizer."""
    return word_tokenize(text)


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Filter out English stopwords from a token list."""
    return [t for t in tokens if t not in _stop_words]


def lemmatize(tokens: list[str]) -> list[str]:
    """Reduce each token to its base lemma form."""
    return [_lemmatizer.lemmatize(t) for t in tokens]


# ── Full pipeline ────────────────────────────────────────────────────────────
def preprocess_text(text: str, return_steps: bool = False):
    """
    Run the complete preprocessing pipeline on *text*.

    Parameters
    ----------
    text : str
        Raw comment string.
    return_steps : bool
        When True, return a dict mapping step-name → intermediate text
        so that each transformation can be shown in the UI.

    Returns
    -------
    str | dict
        Cleaned text string, or dict of steps when return_steps=True.
    """
    if not isinstance(text, str):
        text = str(text)

    steps: dict[str, str] = {}

    text = lowercase(text);            steps["1. Lowercasing"]          = text
    text = remove_html_tags(text);     steps["2. Remove HTML Tags"]     = text
    text = remove_urls(text);          steps["3. Remove URLs"]           = text
    text = remove_numbers(text);       steps["4. Remove Numbers"]        = text
    text = remove_punctuation(text);   steps["5. Remove Punctuation"]    = text
    text = remove_emojis(text);        steps["6. Remove Emojis"]         = text
    text = remove_extra_spaces(text);  steps["7. Remove Extra Spaces"]   = text

    tokens = tokenize(text);           steps["8. Tokenization"]          = str(tokens)
    tokens = remove_stopwords(tokens); steps["9. Stopword Removal"]      = str(tokens)
    tokens = lemmatize(tokens);        steps["10. Lemmatization"]        = str(tokens)

    cleaned = " ".join(tokens);        steps["✅ Final Cleaned Text"]    = cleaned

    return (steps if return_steps else cleaned)


# ── DataFrame helper ─────────────────────────────────────────────────────────
def preprocess_dataframe(df: pd.DataFrame, text_col: str = "comment_text") -> pd.DataFrame:
    """
    Apply the full pipeline to an entire DataFrame column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    text_col : str
        Name of the column that holds raw comment text.

    Returns
    -------
    pd.DataFrame
        Copy of df with an added 'cleaned_text' column.
    """
    df = df.copy()
    df[text_col] = df[text_col].fillna("").astype(str)
    df["cleaned_text"] = df[text_col].apply(preprocess_text)
    return df
