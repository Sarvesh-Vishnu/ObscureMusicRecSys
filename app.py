"""Obscure Music RecSys — semantic recommender over a long-tail catalog,
re-ranked online by a LinUCB contextual bandit learning from user feedback.

Pipeline:
    free-text or track seed
      -> sentence-transformer embedding
      -> FAISS k-NN
      -> blend(similarity, obscurity)
      -> optional LinUCB re-rank from per-user feedback
      -> optional Ollama gemma3:4b narrative per pick
      -> YouTube link (always) + Spotify link/art (if configured)
"""

from __future__ import annotations

import uuid

import numpy as np
import streamlit as st

from obscure import cache as warmcache
from obscure import catalog as cat
from obscure import config
from obscure import features as feats
from obscure import feedback as fb
from obscure import listen
from obscure import narrative
from obscure import recommender


st.set_page_config(page_title="Obscure Music RecSys", page_icon="🎧", layout="wide")


# ---------------------------------------------------------------- caching

@st.cache_resource(show_spinner=False)
def _catalog() -> cat.Catalog:
    return cat.open_catalog()


def _load_catalog() -> list[cat.Track]:
    """Materialize all tracks. Re-reads each call so warm-cache adds are visible."""
    return _catalog().all_tracks()


@st.cache_resource(show_spinner=False)
def _feedback_store() -> fb.FeedbackStore:
    return fb.FeedbackStore()


def _ensure_index_built(tracks: list[cat.Track]) -> None:
    if config.INDEX_PATH.exists():
        return
    with st.spinner(f"Building FAISS index over {len(tracks)} tracks (one-time, ~30s)..."):
        from obscure import embeddings as emb
        emb.build_index(tracks)


def _session_id() -> str:
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:12]
    return st.session_state.session_id


def _get_bandit(alpha: float) -> fb.LinUCBReranker:
    """Hydrate bandit from this session's feedback events."""
    store = _feedback_store()
    sid = _session_id()
    if (st.session_state.get("_bandit_sid") != sid
            or st.session_state.get("_bandit_alpha") != alpha
            or "bandit" not in st.session_state):
        st.session_state.bandit = fb.hydrate_from_store(store, session_id=sid, alpha=alpha)
        st.session_state._bandit_sid = sid
        st.session_state._bandit_alpha = alpha
    return st.session_state.bandit


def _record_feedback(rec: recommender.Recommendation, reward: float, seed_text: str) -> None:
    store = _feedback_store()
    store.record(_session_id(), seed_text, rec.track.track_id, reward, rec.features)
    # update in-memory bandit immediately so the next click reflects the new state
    bandit = st.session_state.get("bandit")
    if bandit is not None:
        bandit.update(rec.features, reward)
    st.toast(
        f"Recorded {'👍' if reward > 0 else ('👎' if reward < -0.5 else '⏭')} "
        f"for *{rec.track.title}* — bandit updated.",
        icon="🎧",
    )


# ---------------------------------------------------------------- rendering

def _render_pick(idx: int, seed_track: cat.Track | None, rec: recommender.Recommendation,
                 seed_text: str, show_narrative: bool) -> None:
    t = rec.track
    links = listen.for_track(t)

    col_art, col_body = st.columns([1, 4])
    with col_art:
        if links.spotify_art:
            st.image(links.spotify_art, use_container_width=True)
        else:
            st.markdown("`(no art)`")

    with col_body:
        title_md = f"**{t.title}** — *{t.artist}*"
        if t.year:
            title_md += f"  · {t.year}"
        st.markdown(title_md)

        meta_bits = [
            f"similarity {rec.similarity:.2f}",
            f"obscurity {rec.obscurity:.2f}",
            f"score {rec.score:.2f}",
        ]
        if rec.bandit_bonus:
            meta_bits.append(f"bandit {rec.bandit_bonus:.2f}")
        if t.listener_count and not _isnan(t.listener_count):
            meta_bits.append(f"~{int(t.listener_count):,} listeners")
        langs = feats.detect_languages(t)
        if langs:
            meta_bits.append("lang: " + ", ".join(sorted(langs)))
        st.caption("  ·  ".join(meta_bits))

        if show_narrative and seed_track is not None:
            blurb = narrative.explain(seed_track, t, rec.similarity, rec.obscurity)
            st.write(blurb)

        if t.tags:
            st.markdown(" ".join(f"`{tag}`" for tag in t.tags[:6]))

        # listen links
        link_bits = [f"[▶ YouTube]({links.youtube_search})"]
        if links.spotify_url:
            link_bits.append(f"[Spotify]({links.spotify_url})")
        st.markdown(" · ".join(link_bits))

        # feedback buttons
        b1, b2, b3, _ = st.columns([1, 1, 1, 6])
        with b1:
            if st.button("👍", key=f"up_{idx}_{t.track_id}"):
                _record_feedback(rec, fb.REWARDS["up"], seed_text)
        with b2:
            if st.button("👎", key=f"down_{idx}_{t.track_id}"):
                _record_feedback(rec, fb.REWARDS["down"], seed_text)
        with b3:
            if st.button("⏭ Skip", key=f"skip_{idx}_{t.track_id}"):
                _record_feedback(rec, fb.REWARDS["skip"], seed_text)


def _isnan(x) -> bool:
    try:
        return x != x  # NaN is the only value not equal to itself
    except Exception:
        return False


# ---------------------------------------------------------------- sidebar

def _render_sidebar(tracks: list[cat.Track]) -> dict:
    with st.sidebar:
        st.subheader("Settings")
        k = st.slider("Number of picks", 3, 20, 8)
        obscurity_weight = st.slider(
            "Obscurity vs. similarity", 0.0, 1.0, 0.4, 0.05,
            help="0 = pure similarity (a normal recommender). "
                 "1 = pure obscurity. 0.4 = the sweet spot this app is built for.",
        )
        bandit_weight = st.slider(
            "Feedback influence (LinUCB)", 0.0, 1.0, 0.5, 0.05,
            help="How much to weight the contextual bandit learned from your "
                 "👍/👎 history. 0 = ignore feedback.",
        )
        bandit_alpha = st.slider(
            "Exploration (α)", 0.1, 3.0, 1.0, 0.1,
            help="LinUCB exploration coefficient. Higher = surface more uncertain "
                 "(potentially surprising) picks.",
        )
        language_pivot_on = st.toggle(
            "Cross-language pivot",
            value=False,
            help="Filter recommendations to OTHER languages than the seed. "
                 "Find sonic neighbors you literally can't search for in your own language.",
        )
        narrative_on = st.toggle("LLM blurb per pick", value=True,
                                 help=f"Uses Ollama ({config.ollama().model}) locally.")

        st.markdown("---")
        st.markdown(f"**Catalog:** {len(tracks)} tracks")
        st.markdown(f"**Spotify display:** {'on' if config.spotify().enabled else 'off'}")

        # bandit introspection
        store = _feedback_store()
        n_events = store.count(session_id=_session_id())
        st.markdown(f"**Session feedback events:** {n_events}")
        if n_events > 0 and st.checkbox("Show learned weights", value=False):
            bandit = _get_bandit(bandit_alpha)
            theta = bandit.theta()
            order = np.argsort(-np.abs(theta))[:8]
            st.markdown("**Top learned features:**")
            for i in order:
                bar = "▮" * int(abs(theta[i]) * 10) or "·"
                sign = "+" if theta[i] >= 0 else "−"
                st.text(f"{feats.FEATURE_NAMES[i]:18s} {sign} {bar}")

        if n_events > 0 and st.button("Reset session feedback", help="Clear bandit memory for THIS session"):
            with store._conn() as c:  # noqa: SLF001
                c.execute("DELETE FROM feedback WHERE session_id = ?", (_session_id(),))
            st.session_state.pop("bandit", None)
            st.rerun()

    return {
        "k": k,
        "obscurity_weight": obscurity_weight,
        "bandit_weight": bandit_weight,
        "bandit_alpha": bandit_alpha,
        "language_pivot_on": language_pivot_on,
        "narrative_on": narrative_on,
    }


# ---------------------------------------------------------------- main

def main() -> None:
    st.title("🎧 Obscure Music RecSys")
    st.caption(
        "Semantic recommendations over a curated long-tail catalog, re-ranked "
        "by a LinUCB bandit learning from your 👍/👎. Spotify optional. Ollama "
        "optional. YouTube link on every pick so you can actually listen."
    )

    tracks = _load_catalog()
    _ensure_index_built(tracks)

    settings = _render_sidebar(tracks)
    bandit = _get_bandit(settings["bandit_alpha"])

    mode = st.radio(
        "Seed by",
        ["Track in catalog", "Search any track", "Free-text description"],
        horizontal=True, label_visibility="collapsed",
    )

    seed_track: cat.Track | None = None
    seed_text = ""

    if mode == "Track in catalog":
        # Restrict to tracks with descriptive tags (better seed quality).
        options = {f"{t.title} — {t.artist}": t for t in tracks if t.tags}
        pick = st.selectbox(f"Pick a seed track ({len(options)} tagged)",
                            list(options.keys()))
        seed_track = options[pick]
        seed_text = seed_track.tag_text()
        st.caption(f"Embedding: *{seed_text}*")
    elif mode == "Search any track":
        c1, c2 = st.columns(2)
        with c1:
            q_title = st.text_input("Track title", placeholder="Sayonee")
        with c2:
            q_artist = st.text_input("Artist", placeholder="Junoon")

        if q_title or q_artist:
            catalog = _catalog()
            results = catalog.by_text(f"{q_title} {q_artist}", limit=8)
            if results:
                opts = {f"{t.title} — {t.artist}": t for t in results}
                pick = st.selectbox(f"Matches ({len(opts)} in catalog)",
                                    list(opts.keys()))
                seed_track = opts[pick]
                seed_text = seed_track.tag_text()
            elif q_title and q_artist:
                st.caption(
                    f"Not in catalog. Click below to fetch *{q_title}* "
                    f"by *{q_artist}* from MusicBrainz (~2s)."
                )
                if st.button("🔍 Fetch from MusicBrainz"):
                    with st.spinner("Looking up on MusicBrainz..."):
                        result = warmcache.lookup_or_fetch(q_title, q_artist,
                                                           catalog=catalog)
                    if result is None:
                        st.error("MusicBrainz had no match for that title + artist.")
                    else:
                        st.success(
                            f"Fetched: {result.track.title} — {result.track.artist}. "
                            "Rebuilding embedding index..."
                        )
                        # Append to FAISS index incrementally so the new seed is
                        # immediately queryable on the next run.
                        from obscure import embeddings as emb
                        emb.add_to_index([result.track])
                        st.cache_resource.clear()
                        st.rerun()
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

    language_pivot: set[str] | None = None
    if settings["language_pivot_on"] and seed_track is not None:
        language_pivot = feats.detect_languages(seed_track)
        if language_pivot:
            st.info(f"🌐 Excluding seed languages: {', '.join(sorted(language_pivot))}")

    seed_for_narrative = (
        seed_track or cat.Track(track_id="user_query", title="(your prompt)",
                                artist="—", tags=[], description=seed_text)
        if settings["narrative_on"] else None
    )

    with st.spinner("Searching the long tail..."):
        recs = recommender.recommend(
            seed_text=seed_text, tracks=tracks, k=settings["k"],
            obscurity_weight=settings["obscurity_weight"],
            exclude_titles=exclude,
            bandit=bandit, bandit_weight=settings["bandit_weight"],
            language_pivot=language_pivot,
        )

    if not recs:
        st.warning("No recommendations — try loosening filters or a different seed.")
        return

    # Persist the rendered set + seed_text so feedback callbacks can find them.
    st.session_state.last_seed_text = seed_text

    for i, rec in enumerate(recs, 1):
        st.markdown(f"### {i}.")
        _render_pick(i, seed_for_narrative, rec, seed_text, settings["narrative_on"])
        st.markdown("---")


if __name__ == "__main__":
    main()
