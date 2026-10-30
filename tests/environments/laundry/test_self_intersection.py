from __future__ import annotations

import numpy as np
import pytest

from laundry.self_intersection import (
    SelfIntersectionAuditCounts,
    build_vertex_neighbourhoods,
)


def test_vertex_neighbourhoods_follow_welded_triangle_edges() -> None:
    triangles = np.asarray(((0, 1, 2), (2, 1, 3), (3, 4, 5)), dtype=np.int32)

    zero_hop = build_vertex_neighbourhoods(
        triangles,
        vertex_count=6,
        graph_hops=0,
    )
    one_hop = build_vertex_neighbourhoods(
        triangles,
        vertex_count=6,
        graph_hops=1,
    )

    assert zero_hop[1] == frozenset((1,))
    assert one_hop[1] == frozenset((0, 1, 2, 3))
    assert 4 not in one_hop[1]
    assert one_hop[3] == frozenset((1, 2, 3, 4, 5))


def test_vertex_neighbourhoods_expand_by_exact_graph_hops() -> None:
    triangles = np.asarray(((0, 1, 2), (2, 3, 4)), dtype=np.int32)

    two_hop = build_vertex_neighbourhoods(
        triangles,
        vertex_count=5,
        graph_hops=2,
    )

    assert two_hop[0] == frozenset(range(5))


def test_vertex_neighbourhood_validation_fails_closed() -> None:
    with pytest.raises(ValueError, match="outside"):
        build_vertex_neighbourhoods(
            np.asarray(((0, 1, 3),), dtype=np.int32),
            vertex_count=3,
            graph_hops=1,
        )
    with pytest.raises(ValueError, match="negative"):
        build_vertex_neighbourhoods(
            np.asarray(((0, 1, 2),), dtype=np.int32),
            vertex_count=3,
            graph_hops=-1,
        )


def test_intersection_count_partition_is_exact() -> None:
    counts = SelfIntersectionAuditCounts(
        nonlocal_count=2,
        topologically_local_count=3,
        raw_count=5,
    )
    assert counts.as_dict() == {
        "nonlocal": 2,
        "topologically_local": 3,
        "raw": 5,
        "newton_candidates": 5,
        "newton_rejected_candidates": 0,
    }
    with pytest.raises(ValueError, match="partition"):
        SelfIntersectionAuditCounts(
            nonlocal_count=1,
            topologically_local_count=1,
            raw_count=3,
        )


def test_intersection_count_exposes_conservative_candidate_rejections() -> None:
    counts = SelfIntersectionAuditCounts(
        nonlocal_count=0,
        topologically_local_count=0,
        raw_count=0,
        candidate_count=4,
    )
    assert counts.as_dict()["newton_candidates"] == 4
    assert counts.as_dict()["newton_rejected_candidates"] == 4
    with pytest.raises(ValueError, match="outnumber"):
        SelfIntersectionAuditCounts(
            nonlocal_count=1,
            topologically_local_count=0,
            raw_count=1,
            candidate_count=0,
        )
