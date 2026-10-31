# Frozen critic ranking: actual first suffix states

**CPU-only diagnostic; zero new physics steps and zero optimizer updates.**
Models and source hashes, all 24 per-state Q pairs, action gradients and physical trace hashes are in [CRITIC_RANK.json](CRITIC_RANK.json). [Reproduction script](critic_rank_diagnosis.py) uses pinned local SB3 2.7.0 and each checkpoint's saved ContractEffect source.

The first 21-dimensional suffix observation is repeated four times exactly as `reset` does; six zeros or six derived features produce the actual 90-input policy observation. Both policies were evaluated on the **same** SAC first-suffix observations for 729000–729003 × bias[-1.2,0,1.2]mm. Their own first observations and the zero-policy / SAC-zero-Z physical comparison observations match exactly (maximum absolute difference **0**). Reconstructed deterministic actions reproduce every logged first action to within 1e-5. All physical trace SHA256 values were checked.

| Frozen model | Actual full-task success | min-Q ranks actor above zero | min-Q ranks actor above actor-zero-Z | Mean Q(actor)−Q(zero) | Mean Q(actor)−Q(actor-zero-Z) |
|---|---:|---:|---:|---:|---:|
| Ordinary SAC | 1/12 | 12/12 | 12/12 | 0.30882 | 0.03153 |
| Effect-feature SAC | 3/12 | 12/12 | 12/12 | 0.27063 | 0.07388 |

Mean first normalized actions (Y, Z, advance): ordinary **(-0.7373, -0.7623, 0.9374)**, effect **(-0.8281, -0.8437, 0.9405)**. These are first-suffix actions, not whole-trajectory averages. Both learned policies initially command large negative lateral offsets, including Z although the injected bias is only Y.

Actual full-policy geometry-zero control succeeds **4/12**; actual same frozen ordinary-SAC policy with Z projected to zero succeeds **7/12**. Those paths begin from identical recorded suffix states. At zero bias, geometry succeeds4/4, ordinarySAC1/4, effectSAC1/4, and ordinarySAC-zero-Z3/4. These provide physical evidence that the learned correction can degrade nominal success, and that removing an unnecessary action component rescues several complete tasks. **The projected actor is a distinct deployment controller and its7/12 is not the original SAC score.** No effect-SAC-zero-Z physical result is claimed here.

Interpretation: the critics' local preference is consistent with the actor producing an unhelpful Z correction, and fails to align with the observed improvement from projecting Z out of the full ordinary-SAC rollout. This is actionable value-learning/representation evidence, not proof that critic rank alone caused every failure or that LayerNorm will fix it. The Q comparisons change one first action while implicitly assuming later learned behavior; the physical comparisons change the policy for the entire suffix, so they are **not exact one-action counterfactual Q targets**.

Soft Q contains discounted future entropy and reward, not a probability of task success. Q magnitudes cannot be compared across separately trained models as calibration. Twelve conditions contain only four independent initial scenes. These bounded1024-decision pilots are not mature SAC baselines, and3/12 versus1/12 does not establish independent method novelty or generalization.

Reproduce from repository root (CPU):

```bash
CUDA_VISIBLE_DEVICES='' /home/andyee/robot-research/ARES-Skilllab/envs/method-evolution-20261028/bin/python \
 research/skill_handoff_lab/research_program/evidence/EXP-002/2026-10-31/critic_rank_diagnosis.py \
 --runs /home/andyee/robot-research/ARES-Skilllab/runs/research_program/EXP-002 \
 --sb3 /home/andyee/robot-research/ARES-Skilllab/external/sb3-2.7.0 \
 --out /tmp/CRITIC_RANK_reproduced.json
```
