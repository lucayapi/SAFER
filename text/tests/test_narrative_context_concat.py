from __future__ import annotations

import numpy as np

from scripts.test_narrative_context_concat import make_representation, leave_one_out_context


def test_leave_one_out_context_excludes_current_unit_and_uses_siblings():
    X = np.asarray([[1.0, 3.0], [5.0, 7.0], [9.0, 11.0], [13.0, 15.0]], dtype=np.float32)
    groups = np.asarray(["a", "a", "a", "b"])

    context = leave_one_out_context(X, groups)

    np.testing.assert_allclose(context[0], [7.0, 9.0])
    np.testing.assert_allclose(context[1], [5.0, 7.0])
    np.testing.assert_allclose(context[2], [3.0, 5.0])
    np.testing.assert_array_equal(context[3], [0.0, 0.0])


def test_representations_have_expected_shapes_and_preserve_unit_block():
    X = np.asarray([[1.0, 2.0], [3.0, 4.0], [8.0, 9.0]], dtype=np.float32)
    groups = np.asarray(["a", "a", "b"])

    unit = make_representation(X, groups, "unit_only")
    context = make_representation(X, groups, "context_only")
    combined = make_representation(X, groups, "unit_plus_context")

    assert unit.shape == (3, 2)
    assert context.shape == (3, 2)
    assert combined.shape == (3, 4)
    np.testing.assert_array_equal(combined[:, :2], X)
    np.testing.assert_array_equal(combined[:, 2:], context)
