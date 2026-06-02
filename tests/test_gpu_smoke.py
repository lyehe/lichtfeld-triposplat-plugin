import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA", allow_module_level=True)
pytest.importorskip("triposplat")


@pytest.mark.gpu
def test_end_to_end_decode_produces_finite_gaussian():
    import sys
    sys.path.insert(0, ".")
    from PIL import Image

    from core import downloads, pipeline_loader
    if not downloads.is_weights_cached():
        pytest.skip("weights not downloaded")
    pipe = pipeline_loader.get_pipeline()
    img = Image.new("RGB", (512, 512), (180, 180, 180))
    g, _prepared = pipe.run(img, seed=0, steps=4, num_gaussians=32768)
    xyz = g.get_xyz
    assert xyz.shape[0] > 0
    assert torch.isfinite(xyz).all()
    assert torch.isfinite(g.get_opacity).all()
