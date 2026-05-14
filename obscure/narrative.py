"""LLM narrative layer: explain *why* an obscure pick is relevant to the seed.

Runs against a local Ollama instance (default model: gemma3:4b). All calls are
best-effort: if Ollama is unreachable, the app degrades to tag-list display.
"""

from __future__ import annotations

from . import config
from .catalog import Track

_SYSTEM = (
    "You write short, vivid music-recommendation blurbs for a long-tail "
    "discovery app. You never invent facts about the song. You never claim "
    "the song is famous or popular. Use 2-3 sentences. Focus on sonic "
    "neighbors, mood, era, and why a fan of the seed would find this "
    "interesting. No emojis."
)


def _build_prompt(seed: Track, pick: Track, similarity: float, obscurity: float) -> str:
    seed_tags = ", ".join(seed.tags) if seed.tags else "unknown"
    pick_tags = ", ".join(pick.tags) if pick.tags else "unknown"
    return (
        f"Seed track: '{seed.title}' by {seed.artist} (tags: {seed_tags}).\n"
        f"Recommended pick: '{pick.title}' by {pick.artist} (tags: {pick_tags}, "
        f"year: {pick.year or 'unknown'}).\n"
        f"Similarity to seed: {similarity:.2f}. Obscurity: {obscurity:.2f}.\n\n"
        "Write a 2-3 sentence blurb explaining why a fan of the seed would "
        "enjoy this pick. Ground it only in the tags above; do not invent."
    )


def explain(seed: Track, pick: Track, similarity: float, obscurity: float,
            timeout: float = 20.0) -> str:
    """Return the narrative, or a fallback tag-list if Ollama is unavailable."""
    cfg = config.ollama()
    try:
        import ollama
        client = ollama.Client(host=cfg.host)
        resp = client.chat(
            model=cfg.model,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": _build_prompt(seed, pick, similarity, obscurity)},
            ],
            options={"temperature": 0.3, "num_predict": 160},
        )
        text = resp.get("message", {}).get("content", "").strip()
        return text or _fallback(pick)
    except Exception:
        return _fallback(pick)


def _fallback(pick: Track) -> str:
    tags = ", ".join(pick.tags[:5]) if pick.tags else "no tags available"
    return f"({tags})"
