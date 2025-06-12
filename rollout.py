#!/usr/bin/env python3
"""
Скрипт для демонстрации поведения обученного агента в среде Rogue.
Загружает модель из checkpoint и запускает rollout с визуализацией.
"""

import argparse
import os
import sys
import time
from typing import Optional

import numpy as np
from stable_baselines3 import PPO

import env


def parse_args():
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description="Демонстрация поведения обученного агента в Rogue",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "checkpoint_path",
        type=str,
        help="Путь к .zip файлу с checkpoint'ом модели"
    )
    
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Количество эпизодов для демонстрации"
    )
    
    parser.add_argument(
        "--max_steps",
        type=int,
        default=50,
        help="Максимальное количество шагов в эпизоде"
    )
    
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Задержка между шагами в секундах"
    )
    
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Использовать детерминистическую политику (без случайности)"
    )
    
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Не отображать игровое поле (только статистика)"
    )
    
    return parser.parse_args()


def clear_screen():
    """Очистка экрана терминала."""
    os.system('clear' if os.name == 'posix' else 'cls')


def print_action_info(action: int, step: int):
    """Вывод информации о действии."""
    action_names = {
        0: "ВЛЕВО (h)",
        1: "ВНИЗ (j)", 
        2: "ВВЕРХ (k)",
        3: "ВПРАВО (l)"
    }
    
    action_name = action_names.get(action, f"НЕИЗВЕСТНО ({action})")
    print(f"Шаг {step}: Действие = {action_name}")


def run_rollout(model: PPO, 
                env_instance: env.RogueEnv, 
                num_episodes: int,
                max_steps: int,
                delay: float,
                deterministic: bool,
                render: bool) -> None:
    """
    Запуск rollout для демонстрации поведения агента.
    
    Args:
        model: Загруженная модель PPO
        env_instance: Экземпляр среды RogueEnv
        num_episodes: Количество эпизодов
        max_steps: Максимальное количество шагов в эпизоде
        delay: Задержка между шагами
        deterministic: Использовать детерминистическую политику
        render: Отображать игровое поле
    """
    
    total_rewards = []
    total_steps = []
    
    for episode in range(num_episodes):
        print(f"\n{'='*60}")
        print(f"ЭПИЗОД {episode + 1}/{num_episodes}")
        print(f"{'='*60}")
        
        # Сброс среды
        obs, info = env_instance.reset()
        episode_reward = 0.0
        episode_steps = 0
        
        print(f"Начальное состояние среды (эпизод {episode + 1}):")
        if render:
            env_instance.render()
            print(f"Информация: {info}")
        
        # Пауза перед началом
        if delay > 0:
            time.sleep(delay * 2)
        
        # Основной цикл эпизода
        for step in range(max_steps):
            # Предсказание действия
            action, _states = model.predict(obs, deterministic=deterministic)
            # Преобразуем numpy массив в скаляр, если необходимо
            if isinstance(action, np.ndarray):
                action = action.item()
            
            if render:
                clear_screen()
                print(f"ЭПИЗОД {episode + 1}/{num_episodes}")
                print_action_info(action, step + 1)
                print("-" * 40)
            
            # Выполнение действия
            obs, reward, terminated, truncated, info = env_instance.step(action)
            
            episode_reward += reward
            episode_steps += 1
            
            # Отображение состояния
            if render:
                env_instance.render()
                print("-" * 40)
                print(f"Награда за шаг: {reward:.4f}")
                print(f"Общая награда: {episode_reward:.4f}")
                print(f"Информация: {info}")
                
                if terminated:
                    print("🏁 ЭПИЗОД ЗАВЕРШЕН СРЕДОЙ")
                elif truncated:
                    print("⏰ ЭПИЗОД ПРЕРВАН ПО ВРЕМЕНИ")
            else:
                # Краткий вывод без отображения поля
                print(f"Шаг {step + 1}: действие={action}, награда={reward:.4f}, "
                      f"общая_награда={episode_reward:.4f}")
            
            # Проверка завершения эпизода
            if terminated or truncated:
                break
                
            # Задержка между шагами
            if delay > 0:
                time.sleep(delay)
        
        # Статистика эпизода
        total_rewards.append(episode_reward)
        total_steps.append(episode_steps)
        
        print(f"\n📊 РЕЗУЛЬТАТЫ ЭПИЗОДА {episode + 1}:")
        print(f"   Общая награда: {episode_reward:.4f}")
        print(f"   Количество шагов: {episode_steps}")
        print(f"   Средняя награда за шаг: {episode_reward/max(episode_steps, 1):.4f}")
        
        # Пауза между эпизодами
        if episode < num_episodes - 1 and delay > 0:
            print(f"\nПауза перед следующим эпизодом...")
            time.sleep(delay * 3)
    
    # Общая статистика
    print(f"\n{'='*60}")
    print("ОБЩАЯ СТАТИСТИКА")
    print(f"{'='*60}")
    print(f"Количество эпизодов: {num_episodes}")
    print(f"Средняя награда: {np.mean(total_rewards):.4f} ± {np.std(total_rewards):.4f}")
    print(f"Минимальная награда: {np.min(total_rewards):.4f}")
    print(f"Максимальная награда: {np.max(total_rewards):.4f}")
    print(f"Среднее количество шагов: {np.mean(total_steps):.1f} ± {np.std(total_steps):.1f}")
    print(f"Общее количество шагов: {np.sum(total_steps)}")


def main():
    """Главная функция."""
    args = parse_args()
    
    # Проверка существования файла checkpoint'а
    if not os.path.exists(args.checkpoint_path):
        print(f"❌ Ошибка: Файл checkpoint'а не найден: {args.checkpoint_path}")
        sys.exit(1)
    
    if not args.checkpoint_path.endswith('.zip'):
        print(f"❌ Ошибка: Файл checkpoint'а должен иметь расширение .zip")
        sys.exit(1)
    
    print("🚀 Запуск демонстрации обученного агента")
    print(f"📁 Checkpoint: {args.checkpoint_path}")
    print(f"🎮 Эпизодов: {args.episodes}")
    print(f"👣 Макс. шагов: {args.max_steps}")
    print(f"⏱️  Задержка: {args.delay}s")
    print(f"🎯 Детерминистический: {args.deterministic}")
    print(f"👁️  Отображение: {not args.no_render}")
    
    try:
        # Создание среды
        print("\n🏗️  Создание среды...")
        rogue_env = env.RogueEnv(max_steps=args.max_steps)
        
        # Загрузка модели
        print(f"📤 Загрузка модели из {args.checkpoint_path}...")
        model = PPO.load(args.checkpoint_path)
        print("✅ Модель успешно загружена")
        
        # Информация о модели
        print(f"📋 Информация о модели:")
        print(f"   Политика: {model.policy}")
        print(f"   Размерность наблюдений: {model.observation_space}")
        print(f"   Размерность действий: {model.action_space}")
        
        # Запуск демонстрации
        print("\n🎬 Начинаем демонстрацию...")
        run_rollout(
            model=model,
            env_instance=rogue_env,
            num_episodes=args.episodes,
            max_steps=args.max_steps,
            delay=args.delay,
            deterministic=args.deterministic,
            render=not args.no_render
        )
        
        print("\n✅ Демонстрация завершена успешно!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Демонстрация прервана пользователем")
        
    except Exception as e:
        print(f"\n❌ Ошибка во время демонстрации: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    finally:
        # Закрытие среды
        try:
            rogue_env.close()
            print("🔒 Среда закрыта")
        except:
            pass


if __name__ == "__main__":
    main()
