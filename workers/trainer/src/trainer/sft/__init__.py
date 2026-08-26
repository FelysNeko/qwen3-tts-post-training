"""SFT backend (planned): entry + loss live here.

Consumes the shared kernel (`trainer.batch.collate` +
`TrainerModel.teacher_forcing`) and adds the official SFT objective:
CE(`sem_logits`, `codec_0_labels[:, 1:]`) + 0.3 · CE(`sub_logits`,
`codes_flat[:, 1:]`), plus the dataset wrapper (jsonl → items, shared
reference-audio mel) and ckpt export.
"""
