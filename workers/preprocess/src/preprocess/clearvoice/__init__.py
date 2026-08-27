"""Vendored subset of ClearVoice (modelscope/ClearerVoice-Studio) — only what
the MossFormer2_SE_48K speech-enhancement path needs, replacing the `clearvoice`
pip package (and its CWD-relative checkpoint quirk, GPU auto-pick side effect,
and heavy dep tree).

Layout:
- `mossformer2_se/` — model defs, BYTE-FAITHFUL vs upstream @ main (2026-08-27);
  state-dict keys must keep matching the published ckpt. Do not "fix" style.
- `decode.py` — stft/istft/compute_fbank ported from clearvoice/utils/misc.py +
  the MossFormer2_SE_48K decode loop from clearvoice/utils/decode_batch.py
  (short-input path verbatim; the >20s sliding-window path reimplements
  upstream's evident intent — the upstream version crashes: 3-index access on
  a 2-D output tensor, and the stride increment sits inside the batch loop).
- `load.py` — checkpoint loading ported from networks.py::SpeechModel.

Config mirrors config/inference/MossFormer2_SE_48K.yaml (see decode.py).
External deps of the vendored tree: einops, rotary-embedding-torch.
(one deviation from byte-faithful: upstream's dead `from torchinfo import
summary` in mossformer2_block.py is commented out and the dep dropped.)
"""
