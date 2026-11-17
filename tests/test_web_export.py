"""Pure helpers behind the web export: lead artist and iTunes preview matching."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import export_web as E       # noqa: E402
import fetch_previews as F   # noqa: E402


def test_lead_artist_folds_credits():
    assert E.lead_artist("Korn feat. Skrillex") == "korn"
    assert E.lead_artist("Clean Bandit & Jess Glynne") == "clean bandit"
    assert E.lead_artist("Post Malone, Swae Lee") == "post malone"
    assert E.lead_artist("Radiohead") == "radiohead"


def r(artist, title, url="u"):
    return {"artistName": artist, "trackName": title, "previewUrl": url, "artworkUrl100": "a/100x100bb.jpg",
            "trackViewUrl": "l"}


def test_prefers_exact_title_and_upsizes_art():
    m = F.match({"title": "Creep", "artist": "Radiohead"}, [r("Radiohead", "Creep (Acoustic)", "x"), r("Radiohead", "Creep", "y")])
    assert m["url"] == "y" and m["exact"] and m["art"].endswith("300x300bb.jpg")


def test_rejects_other_versions_but_allows_remasters():
    assert F.match({"title": "Creep", "artist": "Radiohead"}, [r("Radiohead", "Creep (Live)")]) is None
    m = F.match({"title": "Mr. Brightside", "artist": "The Killers"}, [r("The Killers", "Mr. Brightside (2004 Remaster)")])
    assert m is not None and not m["exact"]
    m = F.match({"title": "Creep (Acoustic)", "artist": "Radiohead"}, [r("Radiohead", "Creep (Acoustic)")])
    assert m is not None and m["exact"]


def test_requires_the_same_lead_artist():
    assert F.match({"title": "Creep", "artist": "Radiohead"}, [r("Scala & Kolacny Brothers", "Creep")]) is None
    assert F.match({"title": "Sunflower", "artist": "Post Malone & Swae Lee"}, [r("Post Malone & Swae Lee", "Sunflower")])


def test_skips_results_without_a_preview():
    assert F.match({"title": "Creep", "artist": "Radiohead"}, [{**r("Radiohead", "Creep"), "previewUrl": None}]) is None
