from training.disparity_refiner.export_eth3d_gallery import gallery_html


def test_eth3d_gallery_has_ten_sample_limit_and_scene_filter() -> None:
    page = gallery_html()
    assert "最多选择 10 张" in page
    assert "复制已选样本 ID" in page
    assert "maxSelection=10" in page
    assert "addOptions('scene','scene')" in page
    assert "localStorage" in page
