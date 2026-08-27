"""SFT backend (planned): entry + loss live here.

Consumes the shared kernel (`ModelWrapper.collate` +
`.teacher_forcing`) and adds the official SFT objective:
CE(`talker_logits`, `codec_0_labels[:, 1:]`) + 0.3 · CE(`sub_talker_logits`,
`talker_codec_ids[:, 1:]`), plus the dataset wrapper (jsonl → items, shared
reference-audio mel) and ckpt export.
"""
