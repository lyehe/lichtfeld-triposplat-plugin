from core.job import JobConfig, latent_signature, num_gaussians_valid


def _cfg(**kw):
    base = dict(image_path="a.png", seed=42, steps=20, guidance_scale=3.0,
               shift=3.0, num_gaussians=262144, erode_radius=1, append=False)
    base.update(kw)
    return JobConfig(**base)


def test_jobconfig_is_frozen():
    cfg = _cfg()
    try:
        cfg.seed = 7  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("JobConfig must be frozen")


def test_signature_changes_when_sampling_inputs_change():
    base = latent_signature(_cfg(), preprocessed_id="m1")
    assert latent_signature(_cfg(seed=43), preprocessed_id="m1") != base
    assert latent_signature(_cfg(steps=30), preprocessed_id="m1") != base
    assert latent_signature(_cfg(guidance_scale=4.0), preprocessed_id="m1") != base
    assert latent_signature(_cfg(shift=2.0), preprocessed_id="m1") != base
    assert latent_signature(_cfg(), preprocessed_id="m2") != base


def test_signature_unchanged_when_only_count_changes():
    base = latent_signature(_cfg(num_gaussians=262144), preprocessed_id="m1")
    assert latent_signature(_cfg(num_gaussians=65536), preprocessed_id="m1") == base


def test_num_gaussians_validation_rounds_to_multiple_of_32():
    assert num_gaussians_valid(262144) == 262144
    assert num_gaussians_valid(65536) == 65536
    assert num_gaussians_valid(100000) == 100000 // 32 * 32 + (32 if (100000 % 32) >= 16 else 0)
    assert num_gaussians_valid(10) == 32768       # clamped up to min
    assert num_gaussians_valid(999999) == 262144  # clamped down to max
