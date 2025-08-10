from typing import Dict, Tuple, Optional, Union
import torch

from rich.progress import track
from collections import deque
from omnisafe.adapter import OnPolicyAdapter
from omnisafe.common.logger import Logger
from omnisafe.models.actor_critic.constraint_actor_critic import ConstraintActorCritic
from omnisafe.utils.config import Config
from omnisafe.algorithms.on_policy.mice.mice_buffer import MICEVectorBuffer


class MICEAdapter(OnPolicyAdapter):
    def __init__(self, env_id: str, num_envs: int, seed: int, cfgs: Config):
        super().__init__(env_id, num_envs, seed, cfgs)
        self._ep_discount_ci: torch.Tensor
        self._ep_discount_ci_list = deque(maxlen=cfgs.logger_cfgs.window_lens)
        self._kernel_cluster_distance=0.008
        self._kernel_epsilon=0.0001


    def _reset_log(self, idx: Optional[int] = None) -> None:
        """Reset log."""
        if idx is None:
            self._ep_ret = torch.zeros(self._env.num_envs)
            self._ep_cost = torch.zeros(self._env.num_envs)
            self._ep_len = torch.zeros(self._env.num_envs)
            self._ep_discount_ci = torch.zeros(self._env.num_envs)
        else:
            self._ep_ret[idx] = 0.0
            self._ep_cost[idx] = 0.0
            self._ep_len[idx] = 0.0
            self._ep_discount_ci[idx] = 0.0


    def rollout(  # pylint: disable=too-many-locals
        self,
        steps_per_epoch: int,
        agent: ConstraintActorCritic,
        buffer: MICEVectorBuffer,
        flashbulb_memory,
        logger: Logger,
        rpnet,
        epoch,
    ) -> None:
        self._reset_log()

        obs, _ = self.reset()
        self._ep_discount_ci = self._ep_discount_ci.to(obs.device)

        for step in track(
            range(steps_per_epoch),
            description=f'Processing rollout for epoch: {logger.current_epoch}...',
        ):
            act, value_r, value_c, logp = agent.step(obs)
            next_obs, reward, cost, terminated, truncated, info = self.step(act)

            self._log_value(reward=reward, cost=cost, info=info)
            ep_cost = self._ep_cost.clone()

            if self._cfgs.algo_cfgs.use_cost:
                logger.store(**{'Value/cost': value_c})
            logger.store(**{'Value/reward': value_r})

            with torch.no_grad():
                emb_obs = rpnet(obs)
            for idx in range(self._env.num_envs):
                if info.get('original_cost', cost)[idx] > 0:
                    flashbulb_memory.unsafe_states[idx].append(emb_obs[idx])
                    
            intrinsic_cost = torch.zeros_like(cost)
            
            for idx in range(self._env.num_envs):
                if len(flashbulb_memory.unsafe_states[idx]) > self._cfgs.algo_cfgs.k_knn:
                    dist = torch.tensor(
                        [torch.dist(c_state, emb_obs[idx]) for c_state in flashbulb_memory.unsafe_states[idx]], device=obs.device
                    )
                    topk_dist, _ = torch.topk(
                        dist, self._cfgs.algo_cfgs.k_knn, largest=False, sorted=True
                    )
                    dist_tensor = topk_dist / torch.mean(topk_dist)
                    dist_tensor = torch.max(
                        dist_tensor - self._kernel_cluster_distance, torch.tensor(0.0, device=obs.device)
                    )

                    kernel = self._kernel_epsilon / (dist_tensor + self._kernel_epsilon)
                    ci = torch.sqrt(torch.sum(kernel))
                    ci = torch.where(torch.isnan(ci), torch.tensor(0.0, device=obs.device), ci)

                    intrinsic_cost[idx] = self._cfgs.algo_cfgs.intrinsic_factor * (self._cfgs.algo_cfgs.cost_gamma**epoch) * ci
            self._ep_discount_ci += intrinsic_cost

            buffer.store(
                obs=obs,
                act=act,
                reward=reward,
                cost=cost,
                value_r=value_r,
                value_c=value_c,
                logp=logp,
                intrinsic_costs=intrinsic_cost,
                time_step=self._ep_len - 1,
            )

            obs = next_obs
            epoch_end = step >= steps_per_epoch - 1

            if epoch_end:
                num_dones = int(terminated.contiguous().sum())
                if self._env.num_envs - num_dones:
                    logger.log(
                        f'\nWarning: trajectory cut off when rollout by epoch\
                            in {self._env.num_envs - num_dones} of {self._env.num_envs} environments.',
                    )

            for idx, (done, time_out) in enumerate(zip(terminated, truncated)):
                if epoch_end or done or time_out:
                    last_value_r = torch.zeros(1)
                    last_value_c = torch.zeros(1)
                    if not done:
                        if epoch_end:
                            _, last_value_r, last_value_c, _ = agent.step(obs[idx])
                        if time_out:
                            _, last_value_r, last_value_c, _ = agent.step(
                                info['final_observation'][idx]
                            )
                        last_value_r = last_value_r.unsqueeze(0)
                        last_value_c = last_value_c.unsqueeze(0)

                    if done or time_out:
                        self._log_metrics(
                            logger, idx
                        )

                        self._ep_discount_ci_list.append(self._ep_discount_ci[idx].item())
                        self._reset_log(idx)

                        self._ep_ret[idx] = 0.0
                        self._ep_cost[idx] = 0.0
                        self._ep_len[idx] = 0.0
                        self._ep_discount_ci[idx] = 0.0

                    buffer.finish_path(last_value_r, last_value_c, idx, self._cfgs.model_cfgs.critic.lr)
        return flashbulb_memory, self._ep_discount_ci_list