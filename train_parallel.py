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
from mean_reward_per_batch_callback import MeanRewardPerBatchCallback

SNAPSHOT_NAME = "checkpoints/ppo_rogue_parallel_98304_steps.zip"

class CustomCNNFeatureExtractor(BaseFeaturesExtractor):
    """
    Улучшенный CNN-экстрактор для входного тензора формы (C=128, H=24, W=80).
    Использует embedding для сжатия one-hot кодирования и современные техники.
    """
    
    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_channels, height, width = observation_space.shape

        # Embedding слой для сжатия one-hot кодирования
        self.embedding = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=1),  # 128→64: сжимаем one-hot
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        # Улучшенная CNN архитектура с batch normalization
        self.cnn = nn.Sequential(
            # 1-й сверточный блок
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # 2-й сверточный блок с большим kernel для лучшего receptive field
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),  # 24x80 → 12x40
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # 3-й сверточный блок
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 12x40 → 6x20
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            # 4-й сверточный блок для финального сжатия
            nn.Conv2d(512, features_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features_dim),
            nn.ReLU(),
            
            # Глобальный средний пул
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            
            # Dropout для регуляризации
            nn.Dropout(0.2)
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Сначала сжимаем one-hot кодирование
        x = self.embedding(observations)  # (B, 128, 24, 80) → (B, 64, 24, 80)
        # Затем применяем CNN
        return self.cnn(x)  # (B, 64, 24, 80) → (B, features_dim)


def make_rogue_env(rank: int = 0, max_steps: int = 1024):
    """Фабрика для создания среды RogueEnv с правильными параметрами."""
    def _init():
        # Каждая среда будет иметь уникальный экземпляр RogueInterface
        rogue_env = env.RogueEnv(max_steps=max_steps)
        return rogue_env
    return _init


def train_parallel_ppo():
    """Обучение PPO с параллельными средами и исправленными гиперпараметрами."""
    print("Настройка параллельного обучения PPO...")
    
    # Исправленные параметры
    num_envs = 8  # Уменьшено для стабильности
    max_steps_per_env = 1024  # Увеличено для более длинных эпизодов
    total_timesteps = 500000# if SNAPSHOT_NAME is None else (500000 - int(SNAPSHOT_NAME.split('_')[-2]))  # Увеличено время обучения
    
    # Создаем папки
    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    
    print(f"Создание {num_envs} параллельных сред...")
    
    # Создаем векторизованную среду
    # Используем DummyVecEnv для отладки, можно переключить на SubprocVecEnv для производительности
    vec_env = DummyVecEnv([
        make_rogue_env(rank=i, max_steps=max_steps_per_env) 
        for i in range(num_envs)
    ])
    
    print("Настройка модели PPO с исправленными гиперпараметрами...")
    
    # Улучшенная конфигурация политики
    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 256, 128],  # Более глубокая policy сеть
            vf=[256, 256, 128]   # Более глубокая value сеть
        ),
        features_extractor_class=CustomCNNFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),  # Увеличен размер признаков
        activation_fn=nn.ReLU,
        normalize_images=False,  # Наши данные уже нормализованы
    )
    
    # Проверяем доступность GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Используем устройство: {device}")
    
    # Создание модели PPO с правильными гиперпараметрами
    if SNAPSHOT_NAME is not None:
        model = PPO.load(SNAPSHOT_NAME, env=vec_env)
        print('Model loaded from snapshot:', SNAPSHOT_NAME)
    else:
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            
            # Исправленные гиперпараметры
            learning_rate=1e-4,  # Более консервативный learning rate
            n_steps=512,  # Увеличено для лучших траекторий
            batch_size=256,  # Правильный размер: n_steps * num_envs / 16
            n_epochs=10,  # Больше эпох обучения на батч
            
            # Важные PPO параметры
            gamma=0.995,  # Высокий discount для долгосрочного планирования
            gae_lambda=0.95,  # GAE параметр
            clip_range=0.2,  # PPO clipping
            ent_coef=0.005,  # Entropy coefficient для исследования
            vf_coef=0.5,  # Value function coefficient
            max_grad_norm=0.5,  # Gradient clipping
            
            device=device,
            tensorboard_log="./tb/",
            seed=42  # Для воспроизводимости
        )
        print('Cold start model created')
    
    # Валидация гиперпараметров
    print("\nВалидация гиперпараметров:")
    print(f"  n_steps: {model.n_steps}")
    print(f"  batch_size: {model.batch_size}")
    print(f"  num_envs: {num_envs}")
    print(f"  rollout_buffer_size: {model.n_steps * num_envs}")
    
    # Проверки
    rollout_size = model.n_steps * num_envs
    assert model.batch_size <= rollout_size, f"batch_size ({model.batch_size}) > rollout_size ({rollout_size})"
    assert rollout_size % model.batch_size == 0, f"rollout_size не кратен batch_size"
    print("  ✅ Все проверки пройдены!")
    
    # Настройка коллбеков
    checkpoint_callback = CheckpointCallback(
        save_freq=max(2000, num_envs * model.n_steps),  # Сохранять каждые ~1000 шагов
        save_path="./checkpoints/",
        name_prefix="ppo_rogue_parallel"
    )

    reward_per_batch_callback = MeanRewardPerBatchCallback(verbose=1)
    
    # Очищаем файл rewards.txt
    with open("rewards.txt", 'w') as f:
        f.write("")
    
    print("\nНачинаем обучение...")
    print(f"Параметры:")
    print(f"  - Количество сред: {num_envs}")
    print(f"  - Шагов на среду за эпизод: {max_steps_per_env}")
    print(f"  - Общих timesteps: {total_timesteps}")
    print(f"  - N_steps (rollout): {model.n_steps}")
    print(f"  - Batch size: {model.batch_size}")
    print(f"  - Learning rate: {model.learning_rate}")
    print(f"  - Gamma: {model.gamma}")
    
    # Обучение
    try:
        start_timesteps = 0
        if SNAPSHOT_NAME is not None:
            # Извлекаем количество шагов из имени checkpoint'а
            import re
            match = re.search(r'(\d+)_steps', SNAPSHOT_NAME)
            if match:
                start_timesteps = int(match.group(1))
        
        remaining_timesteps = total_timesteps - start_timesteps
        if remaining_timesteps > 0:
            model.learn(
                total_timesteps=remaining_timesteps,
                callback=[checkpoint_callback, reward_per_batch_callback],
                progress_bar=True
            )
        else:
            print("Модель уже обучена на указанное количество шагов")
            
    except KeyboardInterrupt:
        print("\n⚠️ Обучение прервано пользователем")
    
    # Сохранение финальной модели
    model.save("ppo_rogue_parallel_final")
    print("Модель сохранена как 'ppo_rogue_parallel_final'")
    
    # Тестирование обученной модели
    print("\nТестирование обученной модели...")
    test_trained_model(model, vec_env, num_episodes=3)
    
    # Закрытие сред
    vec_env.close()
    print("Векторизованная среда закрыта")


def test_trained_model(model, vec_env, num_episodes: int = 1):
    """Тестирование обученной модели с улучшенной статистикой."""
    print(f"\n{'='*50}")
    print(f"ТЕСТИРОВАНИЕ ОБУЧЕННОЙ МОДЕЛИ")
    print(f"{'='*50}")
    
    for episode in range(num_episodes):
        print(f"\n--- Тестовый эпизод {episode + 1}/{num_episodes} ---")
        
        obs = vec_env.reset()
        episode_rewards = [0.0] * vec_env.num_envs
        episode_lengths = [0] * vec_env.num_envs
        active_envs = [True] * vec_env.num_envs
        
        max_test_steps = 100  # Максимум шагов для теста
        
        for step in range(max_test_steps):
            # Предсказание действий
            actions, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = vec_env.step(actions)
            
            # Обновляем статистику только для активных сред
            for i in range(vec_env.num_envs):
                if active_envs[i]:
                    episode_rewards[i] += rewards[i]
                    episode_lengths[i] += 1
                    
                    # Если эпизод завершен
                    if dones[i]:
                        print(f"  Среда {i}: завершен на шаге {step+1}")
                        print(f"    Награда: {episode_rewards[i]:.3f}")
                        print(f"    Длина: {episode_lengths[i]}")
                        if 'player_position' in infos[i]:
                            print(f"    Позиция игрока: {infos[i]['player_position']}")
                        if 'visited_positions' in infos[i]:
                            print(f"    Посещено позиций: {infos[i]['visited_positions']}")
                        active_envs[i] = False
            
            # Краткий отчет каждые 20 шагов
            if (step + 1) % 20 == 0:
                active_count = sum(active_envs)
                avg_reward = sum(episode_rewards) / len(episode_rewards)
                print(f"  Шаг {step+1}: активных сред={active_count}, средняя награда={avg_reward:.3f}")
            
            # Если все среды завершены
            if not any(active_envs):
                print(f"  Все среды завершены на шаге {step+1}")
                break
        
        # Финальная статистика эпизода
        print(f"\nСтатистика эпизода {episode + 1}:")
        print(f"  Средняя награда: {sum(episode_rewards) / len(episode_rewards):.3f}")
        print(f"  Макс награда: {max(episode_rewards):.3f}")
        print(f"  Мин награда: {min(episode_rewards):.3f}")
        print(f"  Средняя длина: {sum(episode_lengths) / len(episode_lengths):.1f}")
    
    print(f"\n{'='*50}")
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print(f"{'='*50}")


def main():
    """Основная функция с улучшенной обработкой ошибок."""
    print("🎮 Параллельное обучение Rogue с PPO")
    print("=" * 40)
    
    # Проверяем системные требования
    print("Проверка системных требований...")
    
    # Проверка PyTorch и CUDA
    print(f"PyTorch версия: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA доступна: {torch.cuda.get_device_name(0)}")
        print(f"CUDA память: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("CUDA недоступна, используем CPU")
    
    # Проверка доступной RAM
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / 1e9
        print(f"Доступная RAM: {ram_gb:.1f} GB")
        if ram_gb < 8:
            print("⚠️ Предупреждение: менее 8GB RAM может быть недостаточно")
    except ImportError:
        print("Не удалось проверить RAM (установите psutil)")
    
    print("-" * 40)
    
    try:
        train_parallel_ppo()
        print("\n✅ Обучение успешно завершено!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Обучение прервано пользователем (Ctrl+C)")
        print("Checkpoint'ы сохранены в ./checkpoints/")
        
    except Exception as e:
        print(f"\n❌ Ошибка при обучении: {e}")
        import traceback
        traceback.print_exc()
        print("\nПроверьте:")
        print("1. Установлена ли игра rogue: /usr/games/rogue")
        print("2. Доступен ли tmux")
        print("3. Достаточно ли памяти")
        raise


if __name__ == "__main__":
    main()
