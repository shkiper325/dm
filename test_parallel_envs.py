#!/usr/bin/env python3
"""
Тест параллельного обучения с несколькими средами RogueEnv.
"""

import env
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np

def test_single_env():
    """Тест одной среды."""
    print("=== Тест одной среды ===")
    
    rogue_env = env.RogueEnv(max_steps=10)
    
    try:
        obs, info = rogue_env.reset()
        print(f"Начальное наблюдение: shape={obs.shape}, info={info}")
        
        for step in range(5):
            action = rogue_env.action_space.sample()
            obs, reward, terminated, truncated, info = rogue_env.step(action)
            print(f"Шаг {step+1}: action={action}, reward={reward}, terminated={terminated}, truncated={truncated}")
            
            if terminated or truncated:
                obs, info = rogue_env.reset()
                print("Эпизод перезапущен")
    
    finally:
        rogue_env.close()
        print("Среда закрыта\n")

def worker_env(env_id: int, num_steps: int):
    """Рабочая функция для тестирования среды в потоке."""
    print(f"Среда {env_id}: создание")
    rogue_env = env.RogueEnv(max_steps=20)
    
    try:
        print(f"Среда {env_id}: запуск")
        obs, info = rogue_env.reset()
        
        total_reward = 0
        for step in range(num_steps):
            action = rogue_env.action_space.sample()
            obs, reward, terminated, truncated, info = rogue_env.step(action)
            total_reward += reward
            
            if step % 5 == 0:
                print(f"Среда {env_id}: шаг {step}, reward={reward}")
            
            if terminated or truncated:
                print(f"Среда {env_id}: эпизод завершен на шаге {step}, перезапуск")
                obs, info = rogue_env.reset()
        
        print(f"Среда {env_id}: завершена, общий reward={total_reward}")
        return total_reward
        
    except Exception as e:
        print(f"Среда {env_id}: ошибка - {e}")
        return 0
    finally:
        rogue_env.close()
        print(f"Среда {env_id}: очищена")

def test_parallel_envs():
    """Тест параллельных сред."""
    print("=== Тест параллельных сред ===")
    
    num_envs = 3
    steps_per_env = 15
    
    # Используем ThreadPoolExecutor для параллельного выполнения
    with ThreadPoolExecutor(max_workers=num_envs) as executor:
        # Запускаем все среды параллельно
        futures = []
        for i in range(num_envs):
            future = executor.submit(worker_env, i+1, steps_per_env)
            futures.append(future)
        
        # Собираем результаты
        total_rewards = []
        for i, future in enumerate(futures):
            reward = future.result()
            total_rewards.append(reward)
            print(f"Среда {i+1} завершилась с reward={reward}")
    
    print(f"Средний reward: {np.mean(total_rewards):.2f}")
    print("Все параллельные среды завершены\n")

def test_env_info():
    """Тест получения информации о средах."""
    print("=== Тест информации о средах ===")
    
    envs = []
    for i in range(3):
        rogue_env = env.RogueEnv(max_steps=10)
        envs.append(rogue_env)
        info = rogue_env.iface.get_session_info()
        print(f"Среда {i+1}: {info}")
    
    # Очистка
    for i, rogue_env in enumerate(envs):
        rogue_env.close()
        print(f"Среда {i+1} очищена")
    print()

def main():
    """Основная функция тестирования."""
    print("Тестирование параллельных сред RogueEnv\n")
    
    try:
        # Тест одной среды
        test_single_env()
        
        # Тест информации о средах
        test_env_info()
        
        # Тест параллельных сред
        test_parallel_envs()
        
        print("✅ Все тесты параллельных сред прошли успешно!")
        
    except Exception as e:
        print(f"❌ Ошибка в тестах: {e}")
        raise

if __name__ == "__main__":
    main()
