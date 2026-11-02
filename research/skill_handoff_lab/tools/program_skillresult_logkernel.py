"""Same CEM updates as pinned-port kernel, ranking log probabilities directly.
Standard numerical repair, not a new planning algorithm. Critics return log Q.
Visited scores/values are logs; final values/probability are probabilities.
"""
from program_stap_kernel import *
def stable_port_plan(observation,skeleton,policies,dynamics,num_iterations=8,num_samples=128,num_elites=16,standard_deviation=1.,keep_elites_fraction=0.,population_decay=1.,momentum=0.):
    """Independent compact port: identical RNG order, Q product, elite updates."""
    count_elite=max(2,min(num_elites,num_samples//2));keep=int(keep_elites_fraction*count_elite+.5)
    low=torch.stack([torch.from_numpy(policies[p.idx_policy].action_space.low) for p in skeleton]);high=torch.stack([torch.from_numpy(policies[p.idx_policy].action_space.high) for p in skeleton])
    n=num_samples*sum(policies[p.idx_policy].action_space.shape[0] for p in skeleton)
    obs=torch.from_numpy(observation);_,mean=dynamics.rollout(obs,skeleton,policies);std=standard_deviation*.5*(high-low)
    best=None;visited=[];iterations=[];calls=1
    with torch.no_grad():
        for it in range(num_iterations):
            samples=torch.distributions.Normal(mean,std).sample((n,));samples=torch.clip(samples,low,high)
            if it>0:samples[:keep]=elites[:keep]
            samples[keep]=mean
            constants=[ConstantAgent(samples[:,t],policies[p.idx_policy]) for t,p in enumerate(skeleton)]
            states,_=dynamics.rollout(obs,skeleton,constants,batch_size=n,time_index=True);calls+=n
            values=torch.stack([policies[p.idx_policy].critic.predict(dynamics.decode(states[:,t],primitive=p),samples[:,t]) for t,p in enumerate(skeleton)],-1)
            score=values.sum(-1);idx=score.topk(count_elite).indices;elites=samples[idx];winner=idx[0]
            if best is None or score[winner]>best['log_success']:
                best=dict(actions=samples[winner].numpy().copy(),states=states[winner].numpy().copy(),log_success=float(score[winner]),p_success=float(torch.exp(score[winner])),values=torch.exp(values[winner]).numpy().copy())
            mean=momentum*mean+(1-momentum)*elites.mean(0);std=(momentum*std+(1-momentum)*elites.std(0)).clamp(min=1e-4)
            iterations.append(dict(iteration=it,population=n,elite_indices=idx.tolist(),mean=mean.tolist(),std=std.tolist(),best_probability=best['p_success']))
            visited.append(dict(actions=samples.numpy(),states=states.numpy(),scores=score.numpy(),values=values.numpy()))
            n=max(int(population_decay*n+.5),2*count_elite)
    return dict(**best,iterations=iterations,visited_actions=np.concatenate([v['actions'] for v in visited]),visited_states=np.concatenate([v['states'] for v in visited]),p_visited_success=np.concatenate([v['scores'] for v in visited]),visited_values=np.concatenate([v['values'] for v in visited]),trajectory_queries=calls,skill_model_queries=calls*len(skeleton))
