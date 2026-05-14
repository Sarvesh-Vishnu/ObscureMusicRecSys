"""Obscure Music RecSys — semantic recommender over a long-tail catalog.

Pipeline (no longer dependent on deprecated Spotify endpoints):
    free-text seed -> sentence-transformer embedding -> FAISS k-NN
    -> blend cosine similarity with an obscurity prior
    -> Ollama gemma3:4b narrative explaining each pick
    -> optional Spotify link/art as display layer
"""

from __future__ import annotations

import streamlit as st

from obscure import catalog as cat
from obscure import config
from obscure import narrative
from obscure import recommender
from obscure import spotify_display


st.set_page_config(page_title="Obscure Music RecSys", page_icon="🎧", layout="wide")


@st.cache_resource(show_spinner=False)
def _load_catalog() -> list[cat.Track]:
    if config.CATALOG_PATH.exists():
        return cat.load_parquet()
    return cat.load_seed()


def _ensure_index_built(tracks: list[cat.Track]) -> bool:
    if config.INDEX_PATH.exists():
        return True
    with st.spinner(f"Building FAISS index over {len(tracks)} tracks (one-time, ~30s)..."):
        from obscure import embeddings as emb
        emb.build_index(tracks)
    return True


def _render_pick(seed_track: cat.Track | None, rec: recommender.Recommendation) -> None:
    t = rec.track
    col_art, col_body = st.columns([1, 4])

    link = spotify_display.lookup(t.title, t.artist)
    with col_art:
        if link and link.image_url:
            st.image(link.image_url, use_container_width=True)
        else:
            st.markdown("`(no art)`")

    with col_body:
        title_md = f"**{t.title}** — *{t.artist}*"
        if t.year:
            title_md += f"  · {t.year}"
        st.markdown(title_md)

        st.caption(
            f"similarity {rec.similarity:.2f}  ·  obscurity {rec.obscurity:.2f}  "
            f"·  blended {rec.score:.2f}"
            + (f"  ·  ~{t.listener_count:,} listeners" if t.listener_count else "")
        )

        if seed_track is not None:
            blurb = narrative.explain(seed_track, t, rec.similarity, rec.obscurity)
            st.write(blurb)

        if t.tags:
            st.markdown(" ".join(f"`{tag}`" for tag in t.tags[:6]))

        if link and link.url:
            st.markdown(f"[Open in Spotify]({link.url})")


def main() -> None:
    st.title("🎧 Obscure Music RecSys")
    st.caption(
        "Semantic recommendations over a curated long-tail catalog. "
        "Self-contained — no Spotify required for recommendations."
    )

    tracks = _load_catalog()
    _ensure_index_built(tracks)

    with st.sidebar:
        st.subheader("Settings")
        k = st.slider("Number of picks", 3, 20, 8)
        obscurity_weight = st.slider(
            "Obscurity vs. similarity", 0.0, 1.0, 0.4, 0.05,
            help="0 = closest semantic match (a normal recommender). "
                 "1 = most obscure track in the catalog. "
                 "0.4 (default) = the sweet spot this app is built for.",
        )
        narrative_on = st.toggle("LLM blurb per pick", value=True,
                                 help=f"Uses Ollama ({config.ollama().model}) locally.")
        st.markdown("---")
        st.markdown(f"**Catalog:** {len(tracks)} tracks")
        st.markdown(f"**Spotify display:** {'on' if config.spotify().enabled else 'off'}")

    mode = st.radio("Seed by", ["Track in catalog", "Free-text description"],
                    horizontal=True, label_visibility="collapsed")

    seed_track: cat.Track | None = None
    seed_text = ""

    if mode == "Track in catalog":
        options = {f"{t.title} — {t.artist}": t for t in tracks}
        pick = st.selectbox("Pick a seed track", list(options.keys()))
        seed_track = options[pick]
        seed_text = seed_track.tag_text()
        st.caption(f"Embedding: *{seed_text}*")
    else:
        seed_text = st.text_input(
            "Describe what you want",
            placeholder="e.g. dreamy Turkish psych with fuzz guitar, 1970s",
        )

    if not seed_text:
        st.info("Pick a seed above to get recommendations.")
        return

    if not st.button("Recommend", type="primary"):
        return

    exclude = {seed_track.title.lower()} if seed_track else set()

    if not narrative_on:
        # Temporarily disable narrative by passing None as seed_track in render loop.
        seed_for_narrative = None
    else:
        seed_for_narrative = seed_track or cat.Track(
            track_id="user_query", title="(your prompt)", artist="—",
            tags=[], description=seed_text,
        )

    with st.spinner("Searching the long tail..."):
        recs = recommender.recommend(
            seed_text=seed_text, tracks=tracks, k=k,
            obscurity_weight=obscurity_weight, exclude_titles=exclude,
        )

    if not recs:
        st.warning("No recommendations — the index may be empty.")
        return

    for i, rec in enumerate(recs, 1):
        st.markdown(f"### {i}.")
        _render_pick(seed_for_narrative, rec)
        st.markdown("---")


if __name__ == "__main__":
    main()
