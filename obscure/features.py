"""Per-candidate feature vectors fed to the LinUCB re-ranker.

The bandit learns a linear weight per feature — keep the dimension small and
the features interpretable. ~20 dims is the sweet spot at this scale.
"""

from __future__ import annotations

import numpy as np

from .catalog import Track, obscurity_score

LANGUAGES = (
    "hindi", "tamil", "urdu", "punjabi", "bengali", "malayalam",
    "english", "french", "spanish", "portuguese", "italian",
    "german", "japanese", "turkish", "arabic", "other",
)

GENRE_FAMILIES = (
    "classical", "devotional", "folk", "filmi", "rock",
    "pop", "electronic", "jazz", "world", "metal",
)

# Tag substrings that map a track to a language bucket.
_LANG_HINTS = {
    "hindi": {"hindi", "filmi", "bollywood", "bhangra"},
    "tamil": {"tamil", "carnatic"},
    "urdu": {"urdu", "ghazal", "qawwali"},
    "punjabi": {"punjabi", "sufi"},
    "bengali": {"bengali", "bangla", "rabindra"},
    "malayalam": {"malayalam"},
    "french": {"french", "francophone"},
    "spanish": {"spanish", "latin"},
    "portuguese": {"brazilian", "portuguese", "mpb", "samba", "bossa"},
    "italian": {"italian", "cantautorato"},
    "german": {"german", "krautrock", "kosmische"},
    "japanese": {"japanese", "city pop"},
    "turkish": {"turkish", "anatolian"},
    "arabic": {"arabic", "lebanese", "tarab"},
}

_GENRE_HINTS = {
    "classical": {"classical", "hindustani", "carnatic", "raga", "khayal", "chamber"},
    "devotional": {"devotional", "bhajan", "qawwali", "sufi", "ritual"},
    "folk": {"folk", "manganiyar", "desert blues", "tuareg", "rajasthani", "assamese"},
    "filmi": {"filmi", "bollywood", "soundtrack"},
    "rock": {"rock", "post-rock", "psych rock", "indie rock", "garage"},
    "pop": {"pop", "indipop", "city pop", "indie pop", "dream pop"},
    "electronic": {"electronic", "idm", "techno", "downtempo", "drum and bass",
                   "dubstep", "synth", "ambient", "kosmische"},
    "jazz": {"jazz", "samba jazz", "jazz fusion", "jazz rap"},
    "world": {"world", "balkan", "tuvan", "fusion", "asian underground",
              "afro-cuban", "son cubano"},
    "metal": {"metal", "doom", "progressive metal"},
}


def _hit(tags: list[str], hints: set[str]) -> bool:
    text = " ".join(tags).lower()
    return any(h in text for h in hints)


def language_one_hot(track: Track) -> np.ndarray:
    vec = np.zeros(len(LANGUAGES), dtype="float32")
    for i, lang in enumerate(LANGUAGES):
        if lang == "other":
            continue
        if _hit(track.tags, _LANG_HINTS.get(lang, set())):
            vec[i] = 1.0
    if not vec.any():
        vec[LANGUAGES.index("other")] = 1.0
    return vec


def genre_one_hot(track: Track) -> np.ndarray:
    vec = np.zeros(len(GENRE_FAMILIES), dtype="float32")
    for i, fam in enumerate(GENRE_FAMILIES):
        if _hit(track.tags, _GENRE_HINTS.get(fam, set())):
            vec[i] = 1.0
    if not vec.any():
        # let the bandit see "no genre family matched"
        vec[GENRE_FAMILIES.index("world")] = 1.0
    return vec


def detect_languages(track: Track) -> set[str]:
    """Return the set of language buckets that match this track's tags."""
    return {lang for lang, hints in _LANG_HINTS.items() if _hit(track.tags, hints)}


def feature_vector(track: Track, similarity: float) -> np.ndarray:
    """Build the per-candidate feature vector for LinUCB.

    Layout (dim = 4 + 16 + 10 = 30):
        [bias, similarity, obscurity, year_norm, *language(16), *genre(10)]
    """
    bias = 1.0
    obs = obscurity_score(track)
    year_norm = ((track.year or 2000) - 1960) / 65.0
    year_norm = max(0.0, min(1.0, year_norm))

    head = np.array([bias, similarity, obs, year_norm], dtype="float32")
    return np.concatenate([head, language_one_hot(track), genre_one_hot(track)])


FEATURE_DIM = 4 + len(LANGUAGES) + len(GENRE_FAMILIES)
FEATURE_NAMES = (
    ["bias", "similarity", "obscurity", "year_norm"]
    + [f"lang:{l}" for l in LANGUAGES]
    + [f"genre:{g}" for g in GENRE_FAMILIES]
)
