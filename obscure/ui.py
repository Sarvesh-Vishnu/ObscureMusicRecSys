"""Design system + Streamlit component helpers.

Keeps app.py readable by centralizing all CSS, HTML rendering, and the YouTube
video-id resolver. Two layers:

1. `inject_css()` — once-per-session global stylesheet (typography, color
   tokens, card styles, button styles, sticky header, fade-ins).
2. `render_*()` helpers — small, composable functions that emit polished
   markup via st.markdown(unsafe_allow_html=True). Anything interactive (the
   👍/👎 buttons, sliders) stays as native Streamlit widgets, so the bandit
   wiring is unchanged.
"""

from __future__ import annotations

import html
import re
from functools import lru_cache
from urllib.parse import quote_plus

import requests
import streamlit as st


# ============================================================ CSS / design system

CSS = """
<style>
    /* ---------- Color tokens ---------- */
    :root {
        --bg-0:        #0a0e14;
        --bg-1:        #141921;
        --bg-2:        #1c2230;
        --bg-card:     #161b25;
        --border:      #232b3a;
        --text:        #E4E6EB;
        --text-dim:    #8B92A1;
        --text-faint:  #5a6478;
        --accent:      #E8B947;   /* warm vinyl gold */
        --accent-2:    #c79a32;
        --success:     #4ADE80;
        --danger:      #F87171;
    }

    /* ---------- App shell ---------- */
    .stApp {
        background:
          radial-gradient(ellipse at top, #131826 0%, #0a0e14 60%),
          var(--bg-0);
        color: var(--text);
        font-family: ui-sans-serif, system-ui, -apple-system, "Inter",
                     "Helvetica Neue", sans-serif;
    }

    /* tighten the default Streamlit container padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 6rem;
        max-width: 1100px;
    }

    /* ---------- Typography ---------- */
    .hero-title {
        font-family: "Playfair Display", "Source Serif Pro", "Georgia", serif;
        font-weight: 700;
        font-size: 2.6rem;
        letter-spacing: -0.02em;
        line-height: 1.05;
        margin: 0 0 0.25rem 0;
        background: linear-gradient(120deg, #fff 0%, var(--accent) 80%);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-sub {
        color: var(--text-dim);
        font-size: 0.95rem;
        line-height: 1.55;
        max-width: 720px;
        margin: 0 0 1.5rem 0;
    }
    .hero-meta {
        display: flex;
        gap: 1.25rem;
        font-size: 0.8rem;
        color: var(--text-faint);
        margin-bottom: 1.75rem;
    }
    .hero-meta b {
        color: var(--accent);
        font-weight: 600;
    }

    /* ---------- Seed bar (sticky) ---------- */
    .seed-bar {
        position: sticky;
        top: 3.5rem;
        z-index: 99;
        background: rgba(20, 25, 33, 0.92);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 12px 16px;
        margin: 0 0 1.5rem 0;
        font-size: 0.88rem;
        color: var(--text-dim);
    }
    .seed-bar b { color: var(--text); font-weight: 600; }
    .seed-bar .seed-label { color: var(--accent); font-weight: 600;
                            text-transform: uppercase; letter-spacing: 0.08em;
                            font-size: 0.7rem; margin-right: 0.5rem; }

    /* ---------- Recommendation cards ---------- */
    .rec-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1.25rem 1.5rem 1rem 1.5rem;
        margin: 0 0 1.25rem 0;
        transition: border-color 0.2s ease, transform 0.2s ease;
        animation: fadeIn 0.32s ease;
    }
    .rec-card:hover {
        border-color: var(--accent-2);
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(6px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    .rec-rank {
        font-family: "Playfair Display", serif;
        color: var(--accent);
        font-size: 1.5rem;
        font-weight: 700;
        line-height: 1;
        margin-right: 0.6rem;
    }
    .rec-title {
        font-family: "Playfair Display", serif;
        font-size: 1.4rem;
        font-weight: 700;
        color: var(--text);
        margin: 0;
        line-height: 1.2;
    }
    .rec-artist {
        color: var(--text-dim);
        font-size: 1rem;
        margin: 0.1rem 0 0.5rem 0;
    }
    .rec-year {
        color: var(--text-faint);
        font-size: 0.85rem;
        margin-left: 0.6rem;
    }

    /* metric chips */
    .chip-row { display: flex; flex-wrap: wrap; gap: 0.4rem;
                margin: 0.5rem 0 0.75rem 0; }
    .chip {
        display: inline-flex;
        align-items: center;
        padding: 3px 10px;
        background: var(--bg-2);
        color: var(--text-dim);
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 500;
        border: 1px solid var(--border);
    }
    .chip b { color: var(--text); font-weight: 600; margin-left: 4px; }
    .chip-tag {
        background: transparent;
        color: var(--text-dim);
        border-color: var(--bg-2);
        font-size: 0.72rem;
    }
    .chip-accent {
        color: var(--accent);
        border-color: rgba(232, 185, 71, 0.3);
        background: rgba(232, 185, 71, 0.08);
    }

    .rec-blurb {
        font-style: italic;
        color: #cdd2d9;
        line-height: 1.55;
        margin: 0.75rem 0 0.75rem 0;
        padding-left: 0.85rem;
        border-left: 2px solid var(--accent);
    }

    /* ---------- Streamlit native widget restyling ---------- */
    /* Primary buttons */
    .stButton > button {
        background: var(--bg-2);
        color: var(--text);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.4rem 0.95rem;
        font-weight: 500;
        transition: all 0.15s ease;
    }
    .stButton > button:hover {
        border-color: var(--accent);
        color: var(--accent);
    }
    /* the "Recommend" primary button */
    .stButton > button[kind="primary"] {
        background: var(--accent);
        color: #1a1410;
        border-color: var(--accent);
        font-weight: 600;
    }
    .stButton > button[kind="primary"]:hover {
        background: #f4c855;
        color: #1a1410;
    }

    /* Sliders */
    .stSlider [data-baseweb="slider"] > div > div { background: var(--accent); }
    .stSlider [data-baseweb="slider"] > div > div > div { background: var(--accent); }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0d1219;
        border-right: 1px solid var(--border);
    }
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: var(--text) !important;
        font-family: "Playfair Display", serif;
    }
    .sidebar-section-label {
        text-transform: uppercase;
        font-size: 0.7rem;
        letter-spacing: 0.1em;
        color: var(--accent);
        font-weight: 600;
        margin: 1.2rem 0 0.4rem 0;
    }

    /* Links */
    a, a:visited { color: var(--accent); text-decoration: none; }
    a:hover { text-decoration: underline; }

    /* Hide Streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }

    /* Code/inline */
    code {
        background: var(--bg-2);
        color: var(--accent);
        padding: 1px 6px;
        border-radius: 4px;
        font-size: 0.82rem;
    }

    /* Embedded YouTube wrapper */
    .yt-frame {
        position: relative;
        padding-bottom: 56.25%;
        height: 0;
        overflow: hidden;
        border-radius: 12px;
        margin: 0.75rem 0;
        background: #000;
    }
    .yt-frame iframe {
        position: absolute;
        top: 0; left: 0;
        width: 100%; height: 100%;
        border: 0;
    }
    .yt-placeholder {
        background: linear-gradient(135deg, #1c2230 0%, #161b25 100%);
        border: 1px dashed var(--border);
        border-radius: 12px;
        padding: 2.5rem 1rem;
        text-align: center;
        color: var(--text-faint);
        font-size: 0.85rem;
    }
</style>
"""


def inject_css() -> None:
    """Inject the global stylesheet exactly once per session."""
    if st.session_state.get("_css_injected"):
        return
    st.markdown(CSS, unsafe_allow_html=True)
    # Web fonts (Playfair Display for editorial titles, Inter for body)
    st.markdown(
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
        '<link href="https://fonts.googleapis.com/css2?'
        'family=Inter:wght@400;500;600&family=Playfair+Display:wght@600;700&display=swap" '
        'rel="stylesheet">',
        unsafe_allow_html=True,
    )
    st.session_state._css_injected = True


# ============================================================ Hero / header

def render_hero(catalog_size: int, n_events: int, ollama_on: bool,
                spotify_on: bool) -> None:
    badge = lambda label, on: (
        f'<span style="color:{"var(--success)" if on else "var(--text-faint)"};">'
        f'{"●" if on else "○"} {label}</span>'
    )
    st.markdown(
        f"""
        <div>
            <div class="hero-title">Obscure Music RecSys</div>
            <p class="hero-sub">
                A semantic recommender for the long tail. Pick a song, free-text
                what you're in the mood for, or search any track in the world —
                we'll find adjacent obscurities, explain why they fit, and learn
                from your taste as you go.
            </p>
            <div class="hero-meta">
                <span><b>{catalog_size:,}</b> tracks</span>
                <span><b>{n_events}</b> votes this session</span>
                {badge("Ollama", ollama_on)}
                {badge("Spotify", spotify_on)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_seed_bar(seed_label: str, seed_text: str) -> None:
    st.markdown(
        f'<div class="seed-bar"><span class="seed-label">{html.escape(seed_label)}</span>'
        f'<b>{html.escape(seed_text)}</b></div>',
        unsafe_allow_html=True,
    )


# ============================================================ Cards

def render_card_header(rank: int, title: str, artist: str, year: int | None,
                       sim: float, obs: float, score: float,
                       bandit_bonus: float, listeners: int | None,
                       languages: list[str]) -> None:
    year_part = f'<span class="rec-year">· {year}</span>' if year else ""
    chips: list[str] = [
        f'<span class="chip">sim<b>{sim:.2f}</b></span>',
        f'<span class="chip">obscurity<b>{obs:.2f}</b></span>',
        f'<span class="chip chip-accent">score<b>{score:.2f}</b></span>',
    ]
    if bandit_bonus:
        chips.append(f'<span class="chip">bandit<b>{bandit_bonus:.2f}</b></span>')
    if listeners:
        chips.append(f'<span class="chip">~{int(listeners):,} listeners</span>')
    if languages:
        chips.append(f'<span class="chip">{", ".join(languages)}</span>')

    st.markdown(
        f"""
        <div style="margin-bottom: 0.25rem;">
            <span class="rec-rank">{rank}</span>
            <span class="rec-title">{html.escape(title)}</span>
            {year_part}
        </div>
        <div class="rec-artist">{html.escape(artist)}</div>
        <div class="chip-row">{"".join(chips)}</div>
        """,
        unsafe_allow_html=True,
    )


def render_blurb(blurb: str) -> None:
    st.markdown(
        f'<div class="rec-blurb">{html.escape(blurb)}</div>',
        unsafe_allow_html=True,
    )


def render_tags(tags: list[str]) -> None:
    if not tags:
        return
    chips = "".join(f'<span class="chip chip-tag">{html.escape(t)}</span>'
                    for t in tags[:8])
    st.markdown(f'<div class="chip-row">{chips}</div>', unsafe_allow_html=True)


# ============================================================ YouTube embed

_YT_ID_RE = re.compile(r'"videoId":"([a-zA-Z0-9_-]{11})"')


@lru_cache(maxsize=512)
def youtube_video_id(query: str) -> str | None:
    """Scrape the first videoId from YouTube search results.

    No API key, no extra deps — one HTTP GET + regex. Cached per process so
    repeat lookups are free. Returns None on any failure.
    """
    try:
        url = f"https://www.youtube.com/results?search_query={quote_plus(query)}"
        r = requests.get(url, timeout=6,
                         headers={"User-Agent": "Mozilla/5.0"})
        m = _YT_ID_RE.search(r.text)
        return m.group(1) if m else None
    except Exception:
        return None


def render_youtube_embed(title: str, artist: str) -> None:
    """Lazy YouTube embed. We only fetch the video ID when the user expands."""
    vid = youtube_video_id(f"{title} {artist}")
    if not vid:
        st.markdown(
            '<div class="yt-placeholder">Couldn\'t find a YouTube match — '
            'try the search link below.</div>',
            unsafe_allow_html=True,
        )
        return
    st.markdown(
        f'<div class="yt-frame"><iframe '
        f'src="https://www.youtube.com/embed/{vid}?rel=0&modestbranding=1" '
        f'allow="accelerometer; encrypted-media; picture-in-picture" '
        f'allowfullscreen></iframe></div>',
        unsafe_allow_html=True,
    )


# ============================================================ Sidebar helpers

def section_label(text: str) -> None:
    st.markdown(f'<div class="sidebar-section-label">{html.escape(text)}</div>',
                unsafe_allow_html=True)
