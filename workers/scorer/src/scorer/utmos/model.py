"""Model side — verbatim from UTMOSv2/utmosv2/model/{ssl,multi_spec,ssl_multispec}.py
@ cc2700db, inference paths only (SSLExtModel, MultiSpecExtModel, SSLMultiSpecExtModelV2).
Differences from upstream: timm pretrained=False (full ckpt load is strict-verified),
train-only weight cascades skipped (phase != train)."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import timm
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoFeatureExtractor, AutoModel

from scorer.utmos.config import DATASET_MAP

if TYPE_CHECKING:
    from types import SimpleNamespace


class _SSLEncoder(nn.Module):
    def __init__(self, sr: int, model_name: str):
        super().__init__()
        self.sr = sr
        self.processor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, x: tuple[torch.Tensor]) -> tuple[torch.Tensor]:
        x = self.processor(
            [t.cpu().numpy() for t in x],
            sampling_rate=self.sr,
            return_tensors="pt",
        ).to(self.model.device)
        outputs = self.model(**x, output_hidden_states=True)  # type: ignore
        return outputs.hidden_states


class SSLExtModel(nn.Module):
    def __init__(self, cfg: SimpleNamespace):
        super().__init__()
        self.cfg = cfg
        self.encoder = _SSLEncoder(cfg.sr, cfg.model.ssl.name)
        hidden_num, in_features = 13, 768  # facebook/wav2vec2-base
        self.weights = nn.Parameter(F.softmax(torch.randn(hidden_num), dim=0))
        if cfg.model.ssl.attn:
            self.attn = nn.ModuleList(
                [
                    nn.MultiheadAttention(
                        embed_dim=in_features,
                        num_heads=8,
                        dropout=0.2,
                        batch_first=True,
                    )
                    for _ in range(cfg.model.ssl.attn)
                ]
            )
        self.num_dataset = len(DATASET_MAP)
        self.fc: nn.Linear | nn.Identity = nn.Linear(
            in_features * 2 + self.num_dataset, cfg.model.ssl.num_classes
        )

    def forward(self, xt: tuple[torch.Tensor], d: torch.Tensor) -> torch.Tensor:
        xt = self.encoder(xt)
        x: torch.Tensor = sum([t * w for t, w in zip(xt, self.weights)])
        if self.cfg.model.ssl.attn:
            y = x
            for attn in self.attn:
                y, _ = attn(y, y, y)
            x = torch.cat([torch.mean(y, dim=1), torch.max(x, dim=1)[0]], dim=1)
        else:
            x = torch.cat([torch.mean(x, dim=1), torch.max(x, dim=1)[0]], dim=1)
        x = self.fc(torch.cat([x, d], dim=1))
        return x


class MultiSpecExtModel(nn.Module):
    def __init__(self, cfg: SimpleNamespace):
        super().__init__()
        self.cfg = cfg
        self.backbones = nn.ModuleList(
            [
                timm.create_model(
                    cfg.model.multi_spec.backbone,
                    pretrained=cfg.model.multi_spec.pretrained,
                    num_classes=0,
                )
                for _ in range(len(cfg.dataset.specs))
            ]
        )
        for backbone in self.backbones:
            backbone.global_pool = nn.Identity()

        self.weights = nn.Parameter(
            F.softmax(torch.randn(len(cfg.dataset.specs)), dim=0)
        )

        self.pooling = timm.layers.SelectAdaptivePool2d(
            output_size=(None, 1) if self.cfg.model.multi_spec.atten else 1,  # type: ignore
            pool_type=self.cfg.model.multi_spec.pool_type,
            flatten=False,
        )

        if self.cfg.model.multi_spec.atten:
            self.attn = nn.MultiheadAttention(
                embed_dim=cast(int, self.backbones[0].num_features)
                * (2 if self.cfg.model.multi_spec.pool_type == "catavgmax" else 1),
                num_heads=8,
                dropout=0.2,
                batch_first=True,
            )

        fc_in_features = (
            cast(int, self.backbones[0].num_features)
            * (2 if self.cfg.model.multi_spec.pool_type == "catavgmax" else 1)
            * (2 if self.cfg.model.multi_spec.atten else 1)
        )
        self.num_dataset = len(DATASET_MAP)
        self.fc: nn.Linear | nn.Identity = nn.Linear(
            fc_in_features + self.num_dataset, cfg.model.multi_spec.num_classes
        )

    def forward(self, x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        xl = [
            x[:, i, :, :, :].squeeze(1)
            for i in range(
                self.cfg.dataset.spec_frames.num_frames * len(self.cfg.dataset.specs)
            )
        ]
        xl = [
            self.backbones[i % len(self.cfg.dataset.specs)](t) for i, t in enumerate(xl)
        ]
        xl = [
            sum(
                [
                    xl[i * len(self.cfg.dataset.specs) + j] * w
                    for j, w in enumerate(self.weights)
                ]
            )
            for i in range(self.cfg.dataset.spec_frames.num_frames)
        ]
        x = torch.cat(xl, dim=3)
        x = self.pooling(x).squeeze(3)
        if self.cfg.model.multi_spec.atten:
            xt = torch.permute(x, (0, 2, 1))
            y, _ = self.attn(xt, xt, xt)
            x = torch.cat([torch.mean(y, dim=1), torch.max(x, dim=2).values], dim=1)
        x = self.fc(torch.cat([x, d], dim=1))
        return x


class SSLMultiSpecExtModelV2(nn.Module):
    def __init__(self, cfg: SimpleNamespace):
        super().__init__()
        self.cfg = cfg
        self.ssl = SSLExtModel(cfg)
        self.spec_long = MultiSpecExtModel(cfg)
        # upstream loads stage-2 sub-weights only when phase == "train" — skipped here
        if cfg.model.ssl_spec.freeze:
            for param in self.ssl.parameters():
                param.requires_grad = False
            for param in self.spec_long.parameters():
                param.requires_grad = False
        ssl_input = self.ssl.fc.in_features
        spec_long_input = self.spec_long.fc.in_features
        self.ssl.fc = nn.Identity()
        self.spec_long.fc = nn.Identity()

        self.num_dataset = len(DATASET_MAP)

        self.fc = nn.Linear(
            cast(int, ssl_input) + cast(int, spec_long_input) + self.num_dataset,
            cfg.model.ssl_spec.num_classes,
        )

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, d: torch.Tensor
    ) -> torch.Tensor:
        x1 = self.ssl(x1, torch.zeros(x1.shape[0], self.num_dataset).to(x1.device))
        x2 = self.spec_long(
            x2, torch.zeros(x1.shape[0], self.num_dataset).to(x1.device)
        )
        x = torch.cat([x1, x2, d], dim=1)
        x = self.fc(x)
        return x
