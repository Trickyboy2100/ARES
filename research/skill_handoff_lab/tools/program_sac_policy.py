"""Known critic-only LayerNorm SAC baseline, independently implemented.

Not a DAWN reproduction and not a new algorithm. Install before SAC constructs
its critic optimizer so online/target parameters and optimizer agree.
"""
from torch import nn
from stable_baselines3.sac.policies import SACPolicy

class CriticLayerNormPolicy(SACPolicy):
    def make_critic(self, features_extractor=None):
        critic=super().make_critic(features_extractor)
        networks=[]
        for i,network in enumerate(critic.q_networks):
            modules=[];last_linear=max(j for j,m in enumerate(network) if isinstance(m,nn.Linear))
            for j,module in enumerate(network):
                modules.append(module)
                if isinstance(module,nn.Linear) and j!=last_linear:modules.append(nn.LayerNorm(module.out_features))
            replacement=nn.Sequential(*modules);critic.add_module(f'qf{i}',replacement);networks.append(replacement)
        critic.q_networks=networks
        return critic.to(self.device)
