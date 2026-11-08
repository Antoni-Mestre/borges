import numpy as np

from borgesbench.metrics import hypervolume_3d, igd_plus, nondominated, path_set_diversity


def test_nondominated_maximization():
    points = np.array([[0.5, 0.5, 0.5], [0.6, 0.6, 0.6], [0.9, 0.2, 0.2]])
    front = nondominated(points)
    assert len(front) == 2
    assert any(np.allclose(row, [0.6, 0.6, 0.6]) for row in front)
    assert any(np.allclose(row, [0.9, 0.2, 0.2]) for row in front)


def test_hypervolume_single_box_and_nested_boxes():
    assert np.isclose(hypervolume_3d([[0.5, 0.5, 0.5]]), 0.125)
    assert np.isclose(hypervolume_3d([[0.5, 0.5, 0.5], [0.4, 0.4, 0.4]]), 0.125)


def test_igd_plus_is_zero_for_identical_front():
    front = [[0.8, 0.4, 0.6], [0.4, 0.8, 0.6]]
    assert np.isclose(igd_plus(front, front), 0.0)


def test_path_set_diversity():
    paths = [["a", "b", "c"], ["a", "b", "d"], ["a", "e", "d"]]
    value = path_set_diversity(paths)
    assert 0.0 < value <= 1.0
