"""Obscure Music RecSys — semantic recommender over a long-tail catalog,
re-ranked online by a LinUCB contextual bandit learning from user feedback.

Pipeline:
    free-text / catalog-pick / search-any-track seed
      -> sentence-transformer embedding
      -> FAISS k-NN
      -> blend(similarity, obscurity)
      -> optional LinUCB re-rank from per-user feedback
      -> optional Ollama gemma3:4b narrative per pick
      -> embedded YouTube preview + Spotify link/art (if configured)
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
from obscure import ui


st.set_page_config(page_title="Obscure Music RecSys", page_icon="🎧",
                   layout="wide", initial_sidebar_state="expanded")


# ============================================================ resources

@st.cache_resource(show_spinner=False)
def _catalog() -> cat.Catalog:
    return cat.open_catalog()


def _load_tracks() -> list[cat.Track]:
    """Re-reads each call so warm-cache adds are visible."""
    return _catalog().all_tracks()


@st.cache_resource(show_spinner=False)
def _feedback_store() -> fb.FeedbackStore:
    return fb.FeedbackStore()


def _ensure_index_built(tracks: list[cat.Track]) -> None:
    if config.INDEX_PATH.exists():
        return
    with st.spinner(f"Building FAISS index over {len(tracks)} tracks (~30s, one-time)..."):
        from obscure import embeddings as emb
        emb.build_index(tracks)


def _session_id() -> str:
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:12]
    return st.session_state.session_id


def _get_bandit(alpha: float) -> fb.LinUCBReranker:
    store = _feedback_store()
    sid = _session_id()
    if (st.session_state.get("_bandit_sid") != sid
            or st.session_state.get("_bandit_alpha") != alpha
            or "bandit" not in st.session_state):
        st.session_state.bandit = fb.hydrate_from_store(store, session_id=sid, alpha=alpha)
        st.session_state._bandit_sid = sid
        st.session_state._bandit_alpha = alpha
    return st.session_state.bandit


def _record_feedback(rec: recommender.Recommendation, reward: float,
                     seed_text: str) -> None:
    store = _feedback_store()
    store.record(_session_id(), seed_text, rec.track.track_id, reward, rec.features)
    bandit = st.session_state.get("bandit")
    if bandit is not None:
        bandit.update(rec.features, reward)
    emoji = "👍" if reward > 0 else ("👎" if reward < -0.5 else "⏭")
    st.toast(f"{emoji} {rec.track.title} — bandit updated", icon="🎧")


def _isnan(x) -> bool:
    try:
        return x != x
    except Exception:
        return False


# ============================================================ rendering

def _render_pick(idx: int, seed_track: cat.Track | None,
                 rec: recommender.Recommendation, seed_text: str,
                 show_narrative: bool, show_preview: bool) -> None:
    t = rec.track
    links = listen.for_track(t)
    langs = sorted(feats.detect_languages(t))

    # outer card
    with st.container():
        st.markdown('<div class="rec-card">', unsafe_allow_html=True)

        col_art, col_body = st.columns([1, 3])
        with col_art:
            if links.spotify_art:
                st.image(links.spotify_art, use_container_width=True)
            else:
                st.markdown(
                    '<div style="aspect-ratio:1; background:linear-gradient(135deg,'
                    '#1c2230,#0a0e14); border-radius:12px; display:flex;'
                    'align-items:center; justify-content:center;'
                    'color:var(--text-faint); font-size:2.5rem;">♪</div>',
                    unsafe_allow_html=True,
                )

        with col_body:
            listeners = (None if t.listener_count is None or _isnan(t.listener_count)
                         else int(t.listener_count))
            ui.render_card_header(
                rank=idx, title=t.title, artist=t.artist, year=t.year,
                sim=rec.similarity, obs=rec.obscurity, score=rec.score,
                bandit_bonus=rec.bandit_bonus, listeners=listeners,
                languages=langs,
            )

            if show_narrative and seed_track is not None:
                blurb = narrative.explain(seed_track, t, rec.similarity, rec.obscurity)
                if blurb:
                    ui.render_blurb(blurb)

            ui.render_tags(t.tags)

            # links + buttons row
            link_parts = []
            if links.spotify_url:
                link_parts.append(f'<a href="{links.spotify_url}" target="_blank">Spotify ↗</a>')
            link_parts.append(f'<a href="{links.youtube_search}" target="_blank">YouTube ↗</a>')
            st.markdown(
                f'<div style="margin: 0.6rem 0 0.4rem 0; color: var(--text-faint);">'
                + " · ".join(link_parts) + '</div>',
                unsafe_allow_html=True,
            )

            if show_preview:
                with st.expander("▶ Play preview", expanded=False):
                    ui.render_youtube_embed(t.title, t.artist)

            # feedback row
            b1, b2, b3, _ = st.columns([1, 1, 1, 5])
            with b1:
                if st.button("👍", key=f"up_{idx}_{t.track_id}"):
                    _record_feedback(rec, fb.REWARDS["up"], seed_text)
            with b2:
                if st.button("👎", key=f"down_{idx}_{t.track_id}"):
                    _record_feedback(rec, fb.REWARDS["down"], seed_text)
            with b3:
                if st.button("⏭ Skip", key=f"skip_{idx}_{t.track_id}"):
                    _record_feedback(rec, fb.REWARDS["skip"], seed_text)

        st.markdown('</div>', unsafe_allow_html=True)


# ============================================================ sidebar

def _render_sidebar(catalog: cat.Catalog, n_tracks: int) -> dict:
    with st.sidebar:
        st.markdown("## 🎧 Settings")

        ui.section_label("Ranking")
        k = st.slider("Picks", 3, 20, 8)
        obscurity_weight = st.slider(
            "Obscurity ↔ similarity", 0.0, 1.0, 0.4, 0.05,
            help="0 = closest semantic match. 1 = most obscure. 0.4 = sweet spot.",
        )

        ui.section_label("Feedback loop")
        bandit_weight = st.slider(
            "LinUCB influence", 0.0, 1.0, 0.5, 0.05,
            help="How much to weight what the bandit learned from your 👍/👎.",
        )
        bandit_alpha = st.slider(
            "Exploration α", 0.1, 3.0, 1.0, 0.1,
            help="Higher = surface more uncertain (surprising) picks.",
        )

        ui.section_label("Modes")
        language_pivot_on = st.toggle(
            "Cross-language pivot",
            value=False,
            help="Filter to OTHER languages than the seed. Find sonic neighbors "
                 "you literally can't search for in your own language.",
        )
        narrative_on = st.toggle(
            "LLM blurb per pick", value=True,
            help=f"Uses Ollama ({config.ollama().model}) locally.",
        )
        preview_on = st.toggle(
            "Embedded YouTube preview", value=True,
            help="Each card gets a Play button that embeds the best YouTube match.",
        )

        ui.section_label("Catalog")
        by_src = catalog.count_by_source()
        st.markdown(
            f'<div style="font-size:0.85rem; color:var(--text-dim);">'
            f'<b style="color:var(--text)">{n_tracks:,}</b> total &nbsp;·&nbsp; '
            f'{by_src.get("seed", 0)} curated &nbsp;·&nbsp; '
            f'{by_src.get("crawl", 0)} crawled &nbsp;·&nbsp; '
            f'{by_src.get("cache", 0)} cached'
            '</div>',
            unsafe_allow_html=True,
        )

        # bandit introspection
        store = _feedback_store()
        n_events = store.count(session_id=_session_id())
        ui.section_label("Session")
        st.markdown(
            f'<div style="font-size:0.85rem; color:var(--text-dim);">'
            f'<b style="color:var(--accent)">{n_events}</b> feedback events</div>',
            unsafe_allow_html=True,
        )
        if n_events > 0 and st.checkbox("Show learned weights", value=False):
            bandit = _get_bandit(bandit_alpha)
            theta = bandit.theta()
            order = np.argsort(-np.abs(theta))[:8]
            for i in order:
                bar = "▮" * int(abs(theta[i]) * 10) or "·"
                sign = "+" if theta[i] >= 0 else "−"
                st.text(f"{feats.FEATURE_NAMES[i]:18s} {sign} {bar}")

        if n_events > 0 and st.button("Reset session feedback",
                                      help="Clear bandit memory for this session"):
            with store._conn() as c:  # noqa: SLF001
                c.execute("DELETE FROM feedback WHERE session_id = ?",
                          (_session_id(),))
            st.session_state.pop("bandit", None)
            st.rerun()

        return {
            "k": k, "obscurity_weight": obscurity_weight,
            "bandit_weight": bandit_weight, "bandit_alpha": bandit_alpha,
            "language_pivot_on": language_pivot_on,
            "narrative_on": narrative_on, "preview_on": preview_on,
            "n_events": n_events,
        }


# ============================================================ seed picker

def _seed_picker(catalog: cat.Catalog, tracks: list[cat.Track]
                 ) -> tuple[cat.Track | None, str]:
    """Render the three seed-mode UIs. Returns (seed_track or None, seed_text)."""
    mode = st.radio(
        "How do you want to seed?",
        ["🎵 Pick from catalog", "🔍 Search any track", "✍️ Free-text mood"],
        horizontal=True, label_visibility="collapsed",
    )

    if mode == "🎵 Pick from catalog":
        tagged = [t for t in tracks if t.tags]
        options = {f"{t.title} — {t.artist}": t for t in tagged}
        pick = st.selectbox(
            f"Pick a seed track  ({len(options)} tagged tracks available)",
            list(options.keys()),
        )
        seed_track = options[pick]
        return seed_track, seed_track.tag_text()

    if mode == "🔍 Search any track":
        c1, c2 = st.columns(2)
        with c1:
            q_title = st.text_input("Track title", placeholder="Sayonee")
        with c2:
            q_artist = st.text_input("Artist", placeholder="Junoon")

        if not (q_title or q_artist):
            return None, ""

        results = catalog.by_text(f"{q_title} {q_artist}", limit=8)
        if results:
            opts = {f"{t.title} — {t.artist}": t for t in results}
            pick = st.selectbox(f"{len(opts)} match" + ("es" if len(opts) > 1 else "")
                                + " in catalog", list(opts.keys()))
            seed_track = opts[pick]
            return seed_track, seed_track.tag_text()

        if q_title and q_artist:
            st.markdown(
                f'<div style="padding:0.75rem 1rem; background:var(--bg-1);'
                f'border-radius:10px; border:1px solid var(--border);">'
                f'<span style="color:var(--text-dim);">Not in catalog.</span> '
                f'Fetch <b>{q_title}</b> by <b>{q_artist}</b> from MusicBrainz?'
                f'</div>',
                unsafe_allow_html=True,
            )
            if st.button("🔍 Fetch from MusicBrainz", type="primary"):
                with st.spinner("Looking up on MusicBrainz..."):
                    result = warmcache.lookup_or_fetch(q_title, q_artist,
                                                       catalog=catalog)
                if result is None:
                    st.error("MusicBrainz had no match.")
                else:
                    from obscure import embeddings as emb
                    emb.add_to_index([result.track])
                    st.cache_resource.clear()
                    st.success(
                        f"Added {result.track.title} — {result.track.artist} "
                        "to the catalog. Re-running..."
                    )
                    st.rerun()
        return None, ""

    # free-text mode
    seed_text = st.text_input(
        "Describe what you want",
        placeholder="e.g. dreamy 1970s Turkish psych with fuzz guitar",
    )
    return None, seed_text


# ============================================================ main

def main() -> None:
    ui.inject_css()

    catalog = _catalog()
    tracks = _load_tracks()
    _ensure_index_built(tracks)

    settings = _render_sidebar(catalog, len(tracks))
    bandit = _get_bandit(settings["bandit_alpha"])

    ui.render_hero(
        catalog_size=len(tracks),
        n_events=settings["n_events"],
        ollama_on=True,  # presence-tested at request time, fast enough
        spotify_on=config.spotify().enabled,
    )

    seed_track, seed_text = _seed_picker(catalog, tracks)

    if not seed_text:
        st.markdown(
            '<div style="margin-top:2rem; color:var(--text-faint); font-size:0.9rem;">'
            "Pick a seed above to get recommendations.</div>",
            unsafe_allow_html=True,
        )
        return

    # sticky seed bar — visible while you scroll picks
    seed_label = "seed" if seed_track else "your prompt"
    summary = (f"{seed_track.title} — {seed_track.artist}"
               if seed_track else seed_text)
    ui.render_seed_bar(seed_label, summary)

    if not st.button("Recommend", type="primary", use_container_width=False):
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

    st.session_state.last_seed_text = seed_text

    for i, rec in enumerate(recs, 1):
        _render_pick(i, seed_for_narrative, rec, seed_text,
                     settings["narrative_on"], settings["preview_on"])


if __name__ == "__main__":
    main()
