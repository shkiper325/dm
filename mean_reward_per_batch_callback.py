import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class MeanRewardPerBatchCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_rollout_end(self) -> None:
        # Считаем среднюю награду по всем шагам последнего батча (rollout)
        # rollout_buffer.rewards — это список наград shape=(n_steps * n_envs,)
        rewards = np.array(self.model.rollout_buffer.rewards)
        mean_reward = rewards.mean()
        # Записываем в TensorBoard под меткой "train/mean_reward_batch"
        self.logger.record("train/mean_reward_batch", mean_reward)
    
    def _on_step(self):
        return super()._on_step()