import env

import torch
from torch import nn
import numpy as np
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from model_save_callback import SaveEveryNStepsCallback

class CustomCNNFeatureExtractor(BaseFeaturesExtractor):
    """
    CNN-экстрактор признаков для входного тензора формы (C=128, H=24, W=80).
    Архитектура: несколько сверточных слоёв + GAP → вектор размерности features_dim.
    """
    def __init__(self, observation_space, features_dim: int = 128):
        """
        :param observation_space: gym.Space с shape = (128, 24, 80)
        :param features_dim: размер выходного вектора признаков
        """
        super().__init__(observation_space, features_dim)
        # получаем каналы, высоту и ширину из пространства наблюдений
        n_channels, height, width = observation_space.shape

        # сверточная часть
        self.cnn = nn.Sequential(
            # 1-й сверточный блок, сохраняет (H, W)
            nn.Conv2d(n_channels, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # 2-й сверточный блок, сохраняет (H, W)
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # 3-й сверточный блок, переход к нужному числу каналов
            nn.Conv2d(128, features_dim, kernel_size=1, stride=1),
            nn.ReLU(),
            # глобальный средний пул по пространственным осям → (features_dim, 1, 1)
            nn.AdaptiveAvgPool2d((1, 1)),
            # расплющивание в (batch_size, features_dim)
            nn.Flatten()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        :param observations: тензор формы (batch_size, 128, 24, 80)
        :return: тензор формы (batch_size, features_dim)
        """
        return self.cnn(observations)



policy_kwargs = dict(
    net_arch=dict(pi=[128],
                  vf=[128]),
    features_extractor_class=CustomCNNFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=128),
)

# Создаем папку для чекпоинтов
os.makedirs("./checkpoints", exist_ok=True)

model = PPO(
    policy="MlpPolicy",      # можно оставить стандартную MlpPolicy, фичи вытянет наш экстрактор
    env=env.RogueEnv(),
    policy_kwargs=policy_kwargs,
    verbose=1,
    learning_rate=3e-4,      # Уменьшил learning_rate для стабильности
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    #gae_lambda=0.95,
    #clip_range=0.2,
    #ent_coef=0.01,         # Добавил энтропию для исследования
    device='cpu',           # Принудительно используем CPU как рекомендовано
    n_epochs=100,            # Увеличил количество эпох для лучшей сходимости
    tensorboard_log = './tb/'
)

# Очищаем файл rewards.txt перед началом обучения
with open("rewards.txt", 'w') as f:
    f.write("")

print("Начинаем обучение...")
model.learn(total_timesteps=100, callback=SaveEveryNStepsCallback(save_freq=50, save_path="./checkpoints", name_prefix="ppo_rogue"))

model.save("ppo_custom_env")

print("Тестируем обученную модель...")
# Создаем новый экземпляр среды для тестирования
test_env = env.RogueEnv()
obs, info = test_env.reset()
for i in range(20):  # Уменьшил количество шагов для тестирования
    action, _ = model.predict(obs, deterministic=True)
    action = int(action)  # Преобразуем numpy array в int
    obs, reward, terminated, truncated, info = test_env.step(action)
    print(f"Step {i+1}: Action={action}, Reward={reward}, Terminated={terminated}, Truncated={truncated}")
    test_env.render()
    if terminated or truncated:
        print("Episode ended, resetting...")
        obs, info = test_env.reset()
    print("-" * 40)