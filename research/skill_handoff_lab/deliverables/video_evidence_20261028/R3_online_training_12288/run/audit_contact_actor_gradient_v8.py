"""First real on-policy rollout diagnostic; performs no optimization or simulation.

At identical pre-update weights, compare the score gradients of full latent
Gaussian and execution-active marginal log probabilities. This diagnoses actual
Actor updates, not algorithm performance. Inactive-action unit fixtures remain
separate from this real-rollout record.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch


def audit_first_rollout(model, output, sample_count=512):
    output = Path(output)
    report_path = output / 'FIRST_REAL_ROLLOUT_GRADIENT.json'
    if report_path.exists():
        return
    buffer = model.rollout_buffer
    assert buffer.full, 'Only a completed on-policy rollout is eligible'
    actions_np = np.asarray(buffer.actions).reshape(-1, model.action_space.shape[0])[:sample_count]
    observations_np = np.asarray(buffer.observations).reshape(-1, model.observation_space.shape[0])[:sample_count]
    advantages_np = np.asarray(buffer.advantages).reshape(-1)[:sample_count]
    old_log_prob_np = np.asarray(buffer.log_probs).reshape(-1)[:sample_count]
    device = model.device
    observations = torch.as_tensor(observations_np, device=device)
    actions = torch.as_tensor(actions_np, device=device)
    mask = observations[:, -actions.shape[-1]:].detach()
    assert bool(torch.all((mask == 0) | (mask == 1)))
    advantages = torch.as_tensor(advantages_np, device=device)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    policy = model.policy
    was_training = policy.training
    rng_before = torch.get_rng_state().clone()
    policy.set_training_mode(False)
    try:
        configured = policy.get_distribution(observations)
        component_log_prob = configured.distribution.log_prob(actions)
        full_log_prob = component_log_prob.sum(-1)
        active_log_prob = (component_log_prob * mask).sum(-1)
        # Detached old values put both counterfactual estimators at ratio=1.
        # No relabeling of stored PPO old probabilities for actual training.
        full_loss = -(advantages * (full_log_prob - full_log_prob.detach()).exp()).mean()
        active_loss = -(advantages * (active_log_prob - active_log_prob.detach()).exp()).mean()
        named = [(name, parameter) for name, parameter in policy.named_parameters()
                 if parameter.requires_grad and not name.startswith(('value_net.', 'mlp_extractor.value_net.'))]
        parameters = [p for _, p in named]
        full_grad = torch.autograd.grad(full_loss, parameters, retain_graph=True, allow_unused=True)
        active_grad = torch.autograd.grad(active_loss, parameters, allow_unused=True)
        def flatten(values):
            return torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten()
                              for p, g in zip(parameters, values)])
        gf, ga = flatten(full_grad), flatten(active_grad)
        expected = configured.log_prob(actions).detach()
        old_error = (expected - torch.as_tensor(old_log_prob_np, device=device)).abs().max().item()
        assert old_error < 1e-4, 'Diagnostic is not at the rollout sampling policy'
        norms = {'full_latent_gradient_l2': gf.norm().item(), 'active_gradient_l2': ga.norm().item(),
                 'gradient_difference_l2': (gf - ga).norm().item(),
                 'gradient_cosine': float(torch.dot(gf, ga) / (gf.norm() * ga.norm()))
                    if float(gf.norm() * ga.norm()) > 0 else None}
        np.savez_compressed(output / 'first_real_rollout_gradient_samples.npz',
            observations=observations_np, actions=actions_np, advantages=advantages_np,
            old_log_probs=old_log_prob_np, active_mask=mask.cpu().numpy(),
            full_latent_gradient=gf.cpu().numpy(), active_gradient=ga.cpu().numpy())
        model.save(output / 'policy_before_first_update.zip')
        source = Path(__file__)
        (output / 'first_rollout_gradient_source.py').write_bytes(source.read_bytes())
        report = {'evidence': 'ACTUAL_ON_POLICY_ROLLOUT_SAME_WEIGHT_GRADIENT_DIAGNOSTIC',
            'sample_selection': 'first samples in time-major env-minor rollout order; no success selection',
            'samples': len(actions_np), 'online_interactions_at_audit': model.num_timesteps,
            'configured_policy_class': type(policy).__name__,
            'active_counts_per_action_dimension': mask.sum(0).cpu().tolist(),
            'all_inactive_samples': int((mask.sum(-1) == 0).sum()),
            'stored_old_log_probability_max_error': old_error, **norms,
            'gradient_parameter_names': [name for name, _ in named],
            'torch_cpu_rng_unchanged': bool(torch.equal(rng_before, torch.get_rng_state())),
            'additional_environment_interactions': 0, 'optimizer_steps_performed_by_audit': 0,
            'limitations': ['Single pre-update score-gradient diagnostic, not a performance comparison',
                'PPO clipping away from ratio=1 may alter estimator behavior',
                'Active dimensions retain the usual latent Gaussian treatment of action clipping',
                'The sample advantage is normalized only for this diagnostic; training remains unchanged']}
        report['sha256'] = {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
            for name in ['first_real_rollout_gradient_samples.npz', 'policy_before_first_update.zip',
                         'first_rollout_gradient_source.py']}
        report_path.write_text(json.dumps(report, indent=2) + '\n')
    finally:
        policy.set_training_mode(was_training)
