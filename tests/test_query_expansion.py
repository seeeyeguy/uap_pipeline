"""Tests for query_expansion.expand_query.

Runs under pytest if present, or standalone:
    .venv/bin/python tests/test_query_expansion.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from query_expansion import MAX_VARIANTS, expand_query  # noqa: E402


def test_original_always_first():
    q = "radar contact over Washington"
    assert expand_query(q)[0] == q


def test_no_match_no_expansion():
    assert expand_query("radar contact over Washington") == [
        "radar contact over Washington"]


def test_empty_and_whitespace():
    assert expand_query("") == [""]
    assert expand_query("   ") == ["   "]


def test_simple_synonym():
    out = expand_query("saucer over lake")
    assert out[0] == "saucer over lake"
    assert "disc over lake" in out
    assert "disk over lake" in out
    # one substitution per variant — never both terms replaced at once
    assert all(v.count("over lake") == 1 for v in out)


def test_exact_token_only_no_substring():
    # "saucers" and "discovery" must NOT trigger saucer/disc expansion
    assert expand_query("saucers discovery") == ["saucers discovery"]


def test_case_insensitive_match():
    out = expand_query("Saucer sighting")
    assert any(v.startswith("disc ") for v in out)


def test_quoted_phrases_untouched():
    out = expand_query('"flying saucer" report')
    assert out == ['"flying saucer" report']


def test_negated_tokens_untouched():
    out = expand_query("lights -saucer")
    assert out == ["lights -saucer"]


def test_phrase_synonym():
    out = expand_query("foo fighter sightings 1944")
    assert out[0] == "foo fighter sightings 1944"
    assert '"kraut fireball" sightings 1944' in out
    assert any("foo-fighter" in v for v in out)


def test_multiword_replacement_is_quoted():
    out = expand_query("airship wave 1897")
    assert '"mystery airship" wave 1897' in out
    assert "dirigible wave 1897" in out


def test_cap_respected():
    q = "saucer craft cigar sphere triangle entity"
    out = expand_query(q)
    assert len(out) <= 1 + MAX_VARIANTS
    assert out[0] == q


def test_custom_cap():
    out = expand_query("saucer craft sphere", max_variants=2)
    assert len(out) == 3


def test_no_duplicate_variants():
    out = expand_query("saucer and saucer")
    assert len(out) == len(set(out))


def test_websearch_or_preserved():
    out = expand_query("saucer OR triangle")
    assert out[0] == "saucer OR triangle"
    assert "disc OR triangle" in out
    assert "saucer OR delta" in out


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failed += 1
                print(f"FAIL {name}: {e}")
    sys.exit(1 if failed else 0)
