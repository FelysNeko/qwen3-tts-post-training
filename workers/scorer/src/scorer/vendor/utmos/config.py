"""fusion_stage3 config, inference-relevant fields only — values verbatim from
UTMOSv2/utmosv2/config/fusion_stage3.py @ cc2700db (τ=2.5 calibration anchor;
do not touch values)."""

from __future__ import annotations

from types import SimpleNamespace

DATASET_MAP = {
    "bvcc": 0,
    "sarulab": 1,
    "blizzard2008": 2,
    "blizzard2009": 3,
    "blizzard2010-EH1": 4,
    "blizzard2010-EH2": 5,
    "blizzard2010-ES1": 6,
    "blizzard2010-ES3": 7,
    "blizzard2011": 8,
    "somos": 9,
}


def build_cfg() -> SimpleNamespace:
    from torchvision import transforms

    cfg = SimpleNamespace(
        sr=16000,
        phase="prediction",
        dataset=SimpleNamespace(
            specs=[
                SimpleNamespace(
                    mode="melspec",
                    n_fft=4096,
                    hop_length=32,
                    win_length=4096,
                    n_mels=512,
                    shape=(512, 512),
                    norm=80,
                ),
                SimpleNamespace(
                    mode="melspec",
                    n_fft=4096,
                    hop_length=32,
                    win_length=2048,
                    n_mels=512,
                    shape=(512, 512),
                    norm=80,
                ),
                SimpleNamespace(
                    mode="melspec",
                    n_fft=4096,
                    hop_length=32,
                    win_length=1024,
                    n_mels=512,
                    shape=(512, 512),
                    norm=80,
                ),
                SimpleNamespace(
                    mode="melspec",
                    n_fft=4096,
                    hop_length=32,
                    win_length=512,
                    n_mels=512,
                    shape=(512, 512),
                    norm=80,
                ),
            ],
            spec_frames=SimpleNamespace(
                num_frames=2,
                frame_sec=1.4,
                mixup_inner=True,
                mixup_alpha=0.4,
                extend="tile",
            ),
            ssl=SimpleNamespace(duration=3),
            # upstream BUG (cc2700db): predict() sets remove_silent_section=True
            # but restores it to None before lazy __getitem__ runs — the flag has
            # NEVER taken effect. All our historical scores (GVR pilot, seed
            # probes, τ=2.5 calibration) were computed without silence removal,
            # so the vendor must replicate the actual behavior: False.
            remove_silent_section=False,
        ),
        transform={"valid": transforms.Compose([transforms.Resize((512, 512))])},
        model=SimpleNamespace(
            multi_spec=SimpleNamespace(
                backbone="tf_efficientnetv2_s.in21k_ft_in1k",
                pretrained=False,  # full ckpt covers backbone (strict load verified)
                num_classes=1,
                pool_type="catavgmax",
                atten=True,
            ),
            ssl=SimpleNamespace(
                name="facebook/wav2vec2-base", attn=1, freeze=False, num_classes=1
            ),
            ssl_spec=SimpleNamespace(
                ssl_weight="ssl_only_stage2",
                spec_weight="spec_only",
                num_classes=1,
                freeze=False,
            ),
        ),
    )
    return cfg
