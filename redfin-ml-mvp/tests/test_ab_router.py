"""A/B router must be deterministic and respect traffic split within noise."""
from src.serving.ab_testing import ABRouter


def test_deterministic_routing():
    r = ABRouter(traffic_split=0.5, experiment_id="exp-1")
    key = "request-abc"
    decisions = {r.route(key).variant for _ in range(100)}
    assert len(decisions) == 1


def test_traffic_split_within_tolerance():
    r = ABRouter(traffic_split=0.2, experiment_id="exp-2")
    n = 10_000
    variants = [r.route(f"req-{i}").variant for i in range(n)]
    challenger_share = variants.count("challenger") / n
    assert abs(challenger_share - 0.2) < 0.02, f"split off by too much: {challenger_share}"


def test_zero_split_never_routes_challenger():
    r = ABRouter(traffic_split=0.0, experiment_id="exp-3")
    for i in range(1_000):
        assert r.route(f"req-{i}").variant == "champion"


def test_experiment_id_isolates_buckets():
    r1 = ABRouter(traffic_split=0.5, experiment_id="e1")
    r2 = ABRouter(traffic_split=0.5, experiment_id="e2")
    # Different experiment IDs should give different bucketings
    diffs = sum(r1.route(f"k{i}").variant != r2.route(f"k{i}").variant for i in range(500))
    assert diffs > 100, "Experiments should be independently bucketed"
