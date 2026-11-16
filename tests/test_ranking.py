"""Ranking guards for the three v2 failures (see obscure/recommender.py)."""

from obscure import catalog as cat
from obscure import feedback as fb
from obscure import recommender as rec


def T(tid, artist, listeners, tags=("folk",)):
    return cat.Track(track_id=tid, title=f"song {tid}", artist=artist,
                     tags=list(tags), listener_count=listeners)


def ids(recs):
    return [r.track.track_id for r in recs]


def test_pure_similarity_when_obscurity_weight_zero():
    pairs = [(T("a", "A", 10**6), 0.80), (T("b", "B", 100), 0.70),
             (T("c", "C", 1000), 0.75)]
    assert ids(rec.rank_candidates(pairs, k=3, obscurity_weight=0.0)) == ["a", "c", "b"]


def test_obscure_but_irrelevant_track_cannot_enter_outside_pool():
    relevant = [(T(f"r{i}", f"R{i}", 10**6), 0.80 - i * 0.001) for i in range(30)]
    junk = (T("junk", "Cumbia Band", 50), 0.40)  # very obscure, very dissimilar
    out = rec.rank_candidates(relevant + [junk], k=10, obscurity_weight=0.9)
    assert "junk" not in ids(out)


def test_scaling_stops_obscurity_dominating_at_default_weight():
    # Raw v2 blend: 0.6*sim + 0.4*obs would put "deep" first
    # (0.6*0.66 + 0.4*0.70 = 0.676 vs 0.6*0.80 + 0.4*0.17 = 0.548).
    best = (T("best", "A", 10**6), 0.80)       # closest match, mainstream
    mid = (T("mid", "B", 10**4), 0.79)         # nearly as close, fairly obscure
    deep = (T("deep", "C", 10**2 * 4), 0.66)   # weakest match, most obscure
    out = ids(rec.rank_candidates([best, mid, deep], k=3, obscurity_weight=0.4))
    assert out[0] == "mid"     # close + obscure wins
    assert out[-1] == "deep"   # weakest match no longer jumps to the top


def test_cold_bandit_does_not_reorder():
    pairs = [(T(c, c.upper(), 10 ** (i + 2), tags=("folk", "hindi", "sufi")[: i + 1]),
              0.8 - i * 0.02) for i, c in enumerate("abc")]
    plain = ids(rec.rank_candidates(pairs, k=3))
    cold = ids(rec.rank_candidates(pairs, k=3, bandit=fb.LinUCBReranker(),
                                   bandit_weight=1.0))
    assert plain == cold


def test_warm_bandit_learns_a_preference():
    jazz = T("j", "J", 10**4, tags=("jazz",))
    rock = T("r", "R", 10**4, tags=("rock",))
    pairs = [(rock, 0.80), (jazz, 0.79)]
    b = fb.LinUCBReranker(alpha=0.1)
    from obscure import features as feats
    for _ in range(rec.BANDIT_FULL_EVENTS):
        b.update(feats.feature_vector(jazz, 0.79), 1.0)
        b.update(feats.feature_vector(rock, 0.80), -1.0)
    assert ids(rec.rank_candidates(pairs, k=2, bandit=b, bandit_weight=1.0))[0] == "j"


def test_one_track_per_artist():
    pairs = [(T("a1", "Ravi Shankar", 10**5), 0.80),
             (T("a2", "ravi shankar ", 10**5), 0.79),
             (T("b", "Bhimsen Joshi", 10**5), 0.78)]
    assert ids(rec.rank_candidates(pairs, k=3, obscurity_weight=0.0)) == ["a1", "b"]


def test_bandit_ramp():
    assert rec.bandit_ramp(0) == 0.0
    assert rec.bandit_ramp(rec.BANDIT_MIN_EVENTS - 1) == 0.0
    assert 0 < rec.bandit_ramp(rec.BANDIT_MIN_EVENTS) < 1
    assert rec.bandit_ramp(rec.BANDIT_FULL_EVENTS) == 1.0
