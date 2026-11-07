import pytest

from app.fusion import reciprocal_rank_fusion


def test_item_in_both_lists_wins():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["c", "d", "a"]], k=60)
    order = [item for item, _ in fused]
    assert order[:2] == ["a", "c"]
    assert set(order) == {"a", "b", "c", "d"}


def test_scores_follow_formula():
    fused = dict(reciprocal_rank_fusion([["a", "b"], ["b"]], k=10))
    assert fused["a"] == pytest.approx(1 / 11)
    assert fused["b"] == pytest.approx(1 / 12 + 1 / 11)


def test_ties_break_on_best_rank_then_first_seen():
    # x and y both score 1/61 + 1/62 and both have a best rank of 1; x was seen first.
    fused = reciprocal_rank_fusion([["x", "y"], ["y", "x"]], k=60)
    assert [i for i, _ in fused] == ["x", "y"]


def test_empty_input():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []
