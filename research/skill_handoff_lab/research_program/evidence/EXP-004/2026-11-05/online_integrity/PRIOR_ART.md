# Primary-source checks used this cycle

- [AWAC paper and project](https://awacrl.github.io/), [actual trainer](https://github.com/rail-berkeley/rlkit/blob/master/rlkit/torch/sac/awac_trainer.py): examined advantage-weighted policy update and separate BC terms. Persistent success BC here is not a full AWAC reproduction and not original.
- [RLPD paper](https://arxiv.org/abs/2302.02948), [actual online training loop](https://github.com/ikostrikov/rlpd/blob/main/train_finetuning.py): examined own-policy interaction, separate offline/online buffers, update-to-data accounting and independent evaluation seeds. This cycle uses inherited ARES discrete Option-SAC, not a claim of full RLPD replication. Old primitive replay is not silently mixed with event-Option data.
- Prior Option-Critic/RecoveryChaining/BCP/source review remains in the preceding continuation_option evidence. Option commitment, target-frame corrections and success BC are mature foundations, not a new paper contribution.

No third-party implementation copied into new code. URLs were read2026-11-05; the original authors' benchmark gains are not transferred to ARES.
