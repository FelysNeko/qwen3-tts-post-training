# Qwen3-TTS-Post-Training

SFT and GRPO pipeline for Qwen3-TTS. Work in progress.

I only understand some high-level stuff, so I rely heavily on coding agents to help me implement the details and verify correctness. This is a vibe-coded project, but still dominated by my will.

## Sep 6, 2026

The preprocessing pipeline, hardware-accelerated rollout sampler, async reward model, and teacher-forcing computation are mostly audited and close to their final structure. However, the training loop and metrics collection are temporary. The goal is not simply to implement a machine learning algorithm, but an entire infrastructure enabling faster experiments. Why am I doing this? It's for Cyrene, obviously.

I spent some time making it support multiple speakers; at least SFT works well. I'm still doing ablations on the GRPO reward design to fix the following failure modes:

- Speaker identity drifting and poor character error rate on long audio generation
- Unstable generation for out-of-distribution content

我来设计、我来许愿、我来审计。我任账单刷爆余额，因你而在，聆听往昔的涟漪。一切献给——德谬歌！

## References

The following papers helped me a lot. Also, ask LLMs.

- [FlowTTS-GRPO](https://arxiv.org/abs/2606.23190)
- [Qwen3-TTS Technical Report](https://arxiv.org/abs/2601.15621)
- [Fish Audio S2 Technical Report](https://arxiv.org/abs/2603.08823)

## License

Distributed under the terms of the [LICENSE](LICENSE).

## Copyright

© All rights reserved by FelysNeko
