#!/usr/bin/env python3
"""
Тест среды RogueEnv перед запуском обучения.
"""

import env
import numpy as np

def test_rogue_env():
    """Простой тест среды."""
    print("Создаем среду...")
    rogue_env = env.RogueEnv(max_steps=5)
    
    try:
        print("Тестируем reset()...")
        obs, info = rogue_env.reset()
        print(f"Observation shape: {obs.shape}")
        print(f"Info: {info}")
        
        print("\nТестируем несколько шагов...")
        for i in range(3):
            action = rogue_env.action_space.sample()  # Случайное действие
            print(f"\nШаг {i+1}: действие = {action}")
            
            obs, reward, terminated, truncated, info = rogue_env.step(action)
            print(f"Reward: {reward}")
            print(f"Terminated: {terminated}")
            print(f"Truncated: {truncated}")
            print(f"Info: {info}")
            print(f"Obs shape: {obs.shape}")
            
            if terminated or truncated:
                print("Эпизод завершен, перезапускаем...")
                obs, info = rogue_env.reset()
                
        print("\nТест прошел успешно!")
        
    except Exception as e:
        print(f"Ошибка в тесте: {e}")
        raise
    finally:
        print("Закрываем среду...")
        rogue_env.close()

if __name__ == "__main__":
    test_rogue_env()
