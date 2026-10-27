"""Execution-active Gaussian likelihood for a stateful contact controller.

The last action_dim observation entries are the PRE-ACTION execution mask.
Inactive samples are integrated out, not evaluated at the stored old parameter.
This is an execution-semantics ablation, not a new PPO unbiasedness theorem.
"""
import torch
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import DiagGaussianDistribution


class ExecutionGaussian(DiagGaussianDistribution):
    def __init__(self, action_dim, mask):
        super().__init__(action_dim); self.mask=mask
    def log_prob(self, actions):
        return (self.distribution.log_prob(actions)*self.mask).sum(dim=1)
    def entropy(self):
        return (self.distribution.entropy()*self.mask).sum(dim=1)


class ExecutionActivePolicy(ActorCriticPolicy):
    def _distribution(self, latent_pi, obs):
        mean=self.action_net(latent_pi)
        mask=obs[:,-self.action_space.shape[0]:].clamp(0,1)
        return ExecutionGaussian(self.action_space.shape[0],mask).proba_distribution(mean,self.log_std)
    def forward(self, obs, deterministic=False):
        features=self.extract_features(obs)
        if self.share_features_extractor:latent_pi,latent_vf=self.mlp_extractor(features)
        else:latent_pi=self.mlp_extractor.forward_actor(features[0]);latent_vf=self.mlp_extractor.forward_critic(features[1])
        distribution=self._distribution(latent_pi,obs);actions=distribution.get_actions(deterministic=deterministic)
        return actions.reshape((-1,*self.action_space.shape)),self.value_net(latent_vf),distribution.log_prob(actions)
    def evaluate_actions(self, obs, actions):
        features=self.extract_features(obs)
        if self.share_features_extractor:latent_pi,latent_vf=self.mlp_extractor(features)
        else:latent_pi=self.mlp_extractor.forward_actor(features[0]);latent_vf=self.mlp_extractor.forward_critic(features[1])
        distribution=self._distribution(latent_pi,obs)
        return self.value_net(latent_vf),distribution.log_prob(actions),distribution.entropy()
    def get_distribution(self, obs):
        features=super(ActorCriticPolicy,self).extract_features(obs,self.pi_features_extractor)
        return self._distribution(self.mlp_extractor.forward_actor(features),obs)
