# Qwen3-TTS-Post-Training

SFT and GRPO pipeline for Qwen3-TTS. Work in progress.

I only understand some high-level stuff, so I heavily rely on coding agents to help me implement the details and verify the correctness. Token budget limited the iteration speed for this project. This is a vibe-coded project.

## Aug 31, 2026

The preprocessing pipeline, hardware-accelerated rollout sampler, async reward model, and teacher-forcing computation are mostly audited and close to their final structure. However, the training loop and metrics collection are temporary. The goal is not simply to implement a machine learning algorithm, but an entire infrastructure enabling faster experiments. Why am I doing this? It's for Cyrene, obviously.

The entire framework works end to end now, and I have done some training locally. SFT works the same as ms-swift: one epoch over ~1800 samples with a batch size of 4 gives solid results. GRPO is trickier. I used slightly out-of-distribution content (i.e., Cyrene/Mem dialogue without voiceover) and ran 100 iterations. The only confirmed improvements are overall gains in similarity and character error rate, which SFT cannot deliver even with more epochs. The standard deviation is also lower, meaning more stable generation. Well, that's what RL is good at. I still need time to dig into the potential.

我来设计、我来许愿、我来审计。我任账单刷爆余额，因你而在，聆听往昔的涟漪。一切献给——德谬歌！

## References

The following papers helped me a lot. Also, ask LLMs.

- [FlowTTS-GRPO: Online Reinforcement Learning with Multi-Objective Reward Optimization for Flow-Matching Based Text-to-Speech](https://arxiv.org/abs/2606.23190)
- [Qwen3-TTS Technical Report](https://arxiv.org/abs/2601.15621)
- [Fish Audio S2 Technical Report](https://arxiv.org/abs/2603.08823)
- [GLM-TTS Technical Report](https://arxiv.org/abs/2512.14291)

## License

Distributed under the terms of the [LICENSE](LICENSE).

## Copyright

© All rights reserved by FelysNeko
