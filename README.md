# Qwen3-TTS-Post-Training

SFT and RL pipeline for Qwen3-TTS.

## Aug 27, 2026

The hardware-accelerated rollout sampler, async reward model, and teacher-forcing computation are mostly audited and close to their final structure. However, the training loop, preprocessing pipeline, and metrics collection are either temporary or not yet migrated to this repository. The goal is not simply implement a machine learning algorithm, but an entire infrastructure enabling faster experiments. Why I'm doing this? It's for Cyrene, obviously.

I only understand some high-level stuff, so I heavily rely on coding agents to help me implement the details and verify the correctness. Token budget limited the iteration speed for this project.

Everything should wrap up in early September.

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
