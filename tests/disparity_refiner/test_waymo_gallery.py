from training.disparity_refiner.export_waymo_gallery import gallery_html


def test_waymo_gallery_has_selection_and_filters() -> None:
    page = gallery_html()
    assert "复制已选 TFRecord" in page
    assert "location" in page
    assert "addOptions('camera','camera')" in page
    assert "localStorage" in page
    assert "String.fromCharCode(10)" in page
