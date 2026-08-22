"""WER scoring: Qwen3-ASR-1.7B-hf, greedy, batch-8 — verbatim inference path
from playground/qwen_asr_probe.py (RTFx 52-66, rerun delta 0.000)."""

from __future__ import annotations

import torch

from qwen3_tts_post_training.reward.text import cer, normalize


class ASRScorer:
    def __init__(self, model_id: str, device: str, batch_size: int = 8):
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self.device = device
        self.batch = batch_size
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = (
            AutoModelForMultimodalLM.from_pretrained(
                model_id, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
            )
            .to(device)
            .eval()
        )

    @torch.inference_mode()
    def transcribe(self, wavs: list[str]) -> dict[str, str]:
        texts: dict[str, str] = {}
        for i in range(0, len(wavs), self.batch):
            chunk = wavs[i : i + self.batch]
            inputs = self.processor.apply_transcription_request(
                audio=chunk, language=["Chinese"] * len(chunk)
            ).to(self.model.device, self.model.dtype)
            out = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
            gen = out[:, inputs["input_ids"].shape[1] :]
            for w, t in zip(
                chunk, self.processor.decode(gen, return_format="transcription_only")
            ):
                texts[w] = t
        return texts

    def score(self, wavs: list[str], texts_ref: list[str | None]) -> list[dict]:
        transcripts = self.transcribe(wavs)
        results = []
        for wav, ref in zip(wavs, texts_ref):
            hyp = transcripts[wav]
            results.append(
                {
                    "transcript": hyp,
                    "cer": cer(normalize(ref), normalize(hyp)) if ref else None,
                }
            )
        return results
