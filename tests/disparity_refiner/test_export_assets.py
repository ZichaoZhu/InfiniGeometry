from pathlib import Path

from training.disparity_refiner.export_assets import public_url


def test_public_url_keeps_the_single_public_data_prefix() -> None:
    root = Path("/tmp/public")
    path = root / "data/exp1/sample/ground_truth.ply"

    assert public_url(path, root) == "/data/exp1/sample/ground_truth.ply"
