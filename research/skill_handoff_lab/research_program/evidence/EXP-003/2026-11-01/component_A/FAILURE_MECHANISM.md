# Full4096: measured failure mechanism, not a generic training diagnosis

CPU analysis of all36 final-evaluation trajectories, SHA-verified. Five failures are paired with the nearest successful bias condition of the same physical initial seed and family; every pair has the same initial-state SHA. **The observed bias/approach differs, so these are not same-state single-action causal interventions.** Full metrics and reproducible script are in `FAILURE_MECHANISM.json` and `../diagnose_full_failures_cpu.py`. No GPU/physics/update was added.

Tail means the last40 **servo steps**, excluding hold/probes; progress uses the last10 skill decisions. Successful tail windows occur earlier than failed ones, so comparisons are not matched in time.

| Failure seed / bias(Y,Z)mm | Median lateral tracking error(mm) | Forward allowed | Median contact force(N) | Last10-decision axial progress(mm) | Nearest successful bias |
|---|---:|---:|---:|---:|---|
|732001 / (0,0)|2.945|0%|0|-0.108|(-1.2,0)|
|732002 / (+1.2,-1.2)|1.541|10%|13.66|-0.056|(-1.2,-1.2)|
|732003 / (+1.2,-1.2)|2.004|5%|14.72|-0.012|(-1.2,-1.2)|
|732004 / (+1.2,-1.2)|1.390|0%|13.82|+0.079|(-1.2,-1.2)|
|732005 / (+1.2,0)|1.344|5%|18.02|-0.002|(0,0)|

All five never satisfy native success and never enter hold. Four contact cases remain near16mm axial entrance; the zero-bias732001 failure is still20mm away and has zero median contact force. It should not be merged with contact-jamming failures.

**Which limits matter?** The lateral0.4mm alignment gate blocks90–100% of these late servo steps. Orientation medians are0.005–0.013rad below0.03rad; only732002 has5% orientation blocking, others0%. Per-joint increment clipping occurs0–2.5%; requested and clipped policy actions are identical. Thus pervasive joint clipping/pose-orientation failure is unsupported. However bounded advance reaches its4mm cap in50–100% of late steps, and some policy outputs are near normalized bounds; “no action saturation” would be too strong.

For the four positive-Y-bias failures, the policy's mean positive Y correction puts its requested target **+1.74 to+2.03mm from the true hole axis**, worsening the injected+1.2mm offset. This is an analyst calculation from logged bias, never an actor input. Combined with13.7–18N measured contact and near-zero net progress, this supports a harmful action-direction/contact mismatch. It does not prove the correction would succeed if simply reversed.

**Forward-allowed is not actual physical authority.** The neighboring successes also have only0–10.5% allowed forward steps in their tails, and lateral medians1.39–2.27mm. For732001/(-1.2,0), three decisions with forward_allowed exactly0 still move axially>0.3mm, with a maximum5.62mm. The shared controller retains integral bias; tracking dynamics and terminal/probe motion also contribute. Therefore a low forward flag alone neither explains failure nor certifies no axial physical effect. Gate relaxation is not justified by this dataset.

Policy target variation is also not a sufficient diagnosis: failed lateral target changes have medians0.43–1.93mm per decision, while successful neighbors show1.05–1.98mm. “Jitter causes all failures” is unsupported without an intervention.

## Next discriminating experiment

Keep the frozen checkpoint and common controller. Separately test the zero-contact tracking failure and the four contact mismatches, using a declared short persistent-action window or actual measured-effect-based correction alongside the strongest search baseline. Measure requested target, controller integral contribution, realized TCP/tool motion and complete-task rescue/regression. A frozen-action duration ablation can reveal whether repeated changing targets conceal useful authority; an equal-budget strong search contrast can reveal whether directional contact information is already sufficient without learning.

The available evidence warrants **action-direction and controller-effect diagnosis**, not an assertion that more training, a looser lateral gate or additional stopping adjustments will solve these failures. Every intervention remains development analysis and must preserve solved cases. This component remains ordinary RL infrastructure, not a new multi-skill method.

The nominal controller's `bias_p` integral state is not explicitly included in the92-dimensional policy observation; four-frame history may only partially recover it. This is known **controller memory**, not evidence of hidden physical parameters or a novel POMDP mechanism. A future Markov-sufficiency test should first expose that known integrator state identically to every baseline/candidate and distinguish it from unobserved friction or attachment. The frozen controller is not changed in this diagnosis.
