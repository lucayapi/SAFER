"""Quote alignment must be literal, traceable and conservative."""

from scripts import annotate_narratives_batch as batch


def _annotation(units, excluded=None):
    return {"document_id": "a1", "units": units, "excluded_spans": excluded or [],
            "issues": [], "needs_review": False}


def _unit(start, end, quote):
    return {"start": start, "end": end, "text": quote, "role": "B",
            "assertion": "AFFIRME", "uncertain": False, "reason": "événement"}


def test_reconcile_exact_quotes_preserves_original_and_provenance():
    raw = "Il travaille. Il tombe."
    original = _annotation([_unit(2, 14, "Il travaille."),
                            _unit(14, 22, "Il tombe.")])
    draft, report = batch.reconcile_annotation(raw, original, "a1")
    assert report["reconciled"]
    assert [(u["start"], u["end"]) for u in draft["units"]] == [(0, 13), (14, 23)]
    assert draft["units"][0]["offset_provenance"] == {
        "model_start": 2, "model_end": 14, "method": "unique_exact_quote"}
    assert original["units"][0]["start"] == 2
    assert draft["review_status"] == "PENDING" and draft["needs_review"] is True
    assert not batch.validate_annotation(raw, draft)


def test_reconcile_blocks_ambiguous_or_missing_quotes():
    raw = "Il tombe. Il tombe."
    for quote, status in (("Il tombe.", "ambiguous_quote"),
                          ("Il glisse.", "quote_not_found")):
        draft, report = batch.reconcile_annotation(raw, _annotation([_unit(99, 110, quote)]), "a1")
        assert draft is None
        assert report["unit_alignments"][0]["status"] == status
    # A correct-looking model offset cannot disambiguate a repeated quote.
    draft, report = batch.reconcile_annotation(raw, _annotation([_unit(0, 9, "Il tombe.")]), "a1")
    assert draft is None
    assert report["unit_alignments"][0]["status"] == "ambiguous_quote"


def test_reconcile_blocks_excluded_overlap_after_relocation():
    raw = "Il travaille. Il tombe."
    original = _annotation([_unit(2, 14, "Il travaille.")],
                           [{"start": 0, "end": 2, "reason": "titre"}])
    draft, report = batch.reconcile_annotation(raw, original, "a1")
    assert draft is None
    assert any("chevauchement" in error for error in report["errors"])
