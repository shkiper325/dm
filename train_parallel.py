#!/usr/bin/env python3
"""
Пример параллельного обучения PPO с несколькими средами RogueEnv.
"""

import env
import os
import torch
from torch import nn
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from model_save_callback import SaveEveryNStepsCallback


class CustomCNNFeatureExtractor(BaseFeaturesExtractor):
    """CNN-экстрактор признаков для входного тензора формы (C=128, H=24, W=80)."""
    
    def __init__(self, observation_space, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        n_channels, height, width = observation_space.shape

        self.cnn = nn.Sequential(
            # 1-й сверточный блок
            nn.Conv2d(n_channels, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # 2-й сверточный блок  
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # 3-й сверточный блок
            nn.Conv2d(128, features_dim, kernel_size=1, stride=1),
            nn.ReLU(),
            # Глобальный средний пул
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.cnn(observations)


def make_rogue_env(rank: int = 0, max_steps: int = 50):
    """Фабрика для создания среды RogueEnv."""
    def _init():
        # Каждая среда будет иметь уникальный экземпляр RogueInterface
        rogue_env = env.RogueEnv(max_steps=max_steps)
        return rogue_env
    return _init


def train_parallel_ppo():
    """Обучение PPO с параллельными средами."""
    print("Настройка параллельного обучения PPO...")
    
    # Параметры
    num_envs = 12  # Количество параллельных сред
    max_steps_per_env = 50  # Максимальное количество шагов в эпизоде
    total_timesteps = 2000  # Общее количество шагов обучения
    
    # Создаем папки
    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    
    print(f"Создание {num_envs} параллельных сред...")
    
    # Создаем векторизованную среду
    # Используем DummyVecEnv (в одном процессе) для простоты отладки
    # Для лучшей производительности можно использовать SubprocVecEnv
    vec_env = DummyVecEnv([
        make_rogue_env(rank=i, max_steps=max_steps_per_env) 
        for i in range(num_envs)
    ])
    
    print("Настройка модели PPO...")
    
    # Конфигурация политики
    policy_kwargs = dict(
        net_arch=dict(pi=[128, 64, 32], vf=[128, 64, 32]),
        features_extractor_class=CustomCNNFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=128),
    )
    
    # Создание модели PPO
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=3e-4,  # Стандартный learning rate для PPO
        n_steps=128,  # Количество шагов на среду для сбора данных
        batch_size=256,  # Размер батча (должен быть <= n_steps * num_envs)
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        device='cpu',
        tensorboard_log="./logs/"
    )
    
    # Настройка коллбеков
    checkpoint_callback = CheckpointCallback(
        save_freq=100,  # Сохранять каждые 100 шагов
        save_path="./checkpoints/",
        name_prefix="ppo_rogue_parallel"
    )
    
    # Очищаем файл rewards.txt
    with open("rewards.txt", 'w') as f:
        f.write("")
    
    print("Начинаем обучение...")
    print(f"Параметры:")
    print(f"  - Количество сред: {num_envs}")
    print(f"  - Шагов на среду: {max_steps_per_env}")
    print(f"  - Общих timesteps: {total_timesteps}")
    print(f"  - N_steps: {model.n_steps}")
    print(f"  - Batch size: {model.batch_size}")
    
    # Обучение
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    # Сохранение финальной модели
    model.save("ppo_rogue_parallel_final")
    print("Модель сохранена")
    
    # Тестирование обученной модели
    print("\nТестирование обученной модели...")
    test_trained_model(model, vec_env, num_episodes=2)
    
    # Закрытие сред
    vec_env.close()
    print("Векторизованная среда закрыта")


def test_trained_model(model, vec_env, num_episodes: int = 1):
    """Тестирование обученной модели."""
    for episode in range(num_episodes):
        print(f"\n--- Тестовый эпизод {episode + 1} ---")
        
        obs = vec_env.reset()
        episode_rewards = [0] * vec_env.num_envs
        episode_lengths = [0] * vec_env.num_envs
        
        for step in range(50):  # Максимум 50 шагов на тест
            actions, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = vec_env.step(actions)
            
            # Обновляем статистику
            for i in range(vec_env.num_envs):
                episode_rewards[i] += rewards[i]
                episode_lengths[i] += 1
            
            print(f"Шаг {step + 1}: actions={actions}, rewards={rewards}")
            
            # Проверяем завершение эпизодов
            if any(dones):
                for i, done in enumerate(dones):
                    if done:
                        print(f"Среда {i}: эпизод завершен, reward={episode_rewards[i]:.2f}, длина={episode_lengths[i]}")
                        episode_rewards[i] = 0
                        episode_lengths[i] = 0
        
        print(f"Эпизод {episode + 1} завершен")


def main():
    """Основная функция."""
    print("Параллельное обучение Rogue с PPO")
    print("=" * 40)
    
    try:
        train_parallel_ppo()
        print("\n✅ Обучение успешно завершено!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Обучение прервано пользователем")
        
    except Exception as e:
        print(f"\n❌ Ошибка при обучении: {e}")
        raise


if __name__ == "__main__":
    main()
