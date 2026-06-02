from core import downloads


def test_required_files_list():
    # Exactly the 5 upstream weight files, with their subdirs.
    assert downloads.REQUIRED_FILES == [
        "diffusion_models/triposplat_fp16.safetensors",
        "vae/triposplat_vae_decoder_fp16.safetensors",
        "clip_vision/dino_v3_vit_h.safetensors",
        "vae/flux2-vae.safetensors",
        "background_removal/birefnet.safetensors",
    ]


def test_is_weights_cached_false_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(downloads, "CKPTS_DIR", tmp_path)
    assert downloads.is_weights_cached() is False


def test_is_weights_cached_true_when_all_present(tmp_path, monkeypatch):
    monkeypatch.setattr(downloads, "CKPTS_DIR", tmp_path)
    for rel in downloads.REQUIRED_FILES:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x" * 2_000_000)  # > 1 MB sanity threshold
    assert downloads.is_weights_cached() is True


def test_ckpt_paths_returns_five_absolute_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(downloads, "CKPTS_DIR", tmp_path)
    paths = downloads.ckpt_paths()
    assert set(paths) == {"ckpt_path", "decoder_path", "dinov3_path",
                          "flux2_vae_encoder_path", "rmbg_path"}
    assert paths["ckpt_path"] == tmp_path / "diffusion_models/triposplat_fp16.safetensors"
