#!/usr/bin/env python3
"""
Тест одновременной работы нескольких экземпляров RogueInterface.
"""

import time
import threading
from rogue_iface_tmux import RogueInterface

def test_single_instance():
    """Тест одного экземпляра."""
    print("=== Тест одного экземпляра ===")
    
    game = RogueInterface(cmd="/usr/games/rogue")
    print(f"Создан экземпляр: {game.get_session_info()}")
    
    try:
        game.restart()
        print("Игра запущена")
        
        # Несколько действий
        for i, action in enumerate(["h", "j", "k", "l"]):
            game.key(action)
            print(f"Действие {i+1}: {action}")
            time.sleep(0.1)
        
        state = game.state()
        print(f"Получено состояние: {len(state)} строк")
        
    finally:
        game.cleanup()
        print("Экземпляр очищен\n")

def test_multiple_instances():
    """Тест нескольких экземпляров одновременно."""
    print("=== Тест нескольких экземпляров ===")
    
    # Создаем несколько экземпляров
    games = []
    for i in range(3):
        game = RogueInterface(cmd="/usr/games/rogue")
        games.append(game)
        print(f"Экземпляр {i+1}: {game.get_session_info()}")
    
    try:
        # Запускаем все игры
        for i, game in enumerate(games):
            game.restart()
            print(f"Игра {i+1} запущена")
        
        # Делаем действия в каждой игре
        for round_num in range(3):
            print(f"\n--- Раунд {round_num + 1} ---")
            for i, game in enumerate(games):
                actions = ["h", "j", "k", "l"]
                action = actions[round_num % len(actions)]
                game.key(action)
                print(f"Игра {i+1}: действие '{action}'")
                time.sleep(0.05)
            
            # Получаем состояния
            for i, game in enumerate(games):
                state = game.state()
                non_empty_lines = sum(1 for line in state if line.strip())
                print(f"Игра {i+1}: {non_empty_lines} непустых строк")
    
    finally:
        # Очищаем все экземпляры
        for i, game in enumerate(games):
            game.cleanup()
            print(f"Игра {i+1} очищена")
        print()

def worker_thread(thread_id: int, num_actions: int):
    """Рабочая функция для тестирования в потоках."""
    print(f"Поток {thread_id}: создание экземпляра")
    game = RogueInterface(cmd="/usr/games/rogue")
    
    try:
        print(f"Поток {thread_id}: запуск игры ({game.session})")
        game.restart()
        
        # Выполняем действия
        actions = ["h", "j", "k", "l"]
        for i in range(num_actions):
            action = actions[i % len(actions)]
            game.key(action)
            print(f"Поток {thread_id}: действие {i+1}/{num_actions} - '{action}'")
            time.sleep(0.1)
        
        # Получаем финальное состояние
        state = game.state()
        non_empty = sum(1 for line in state if line.strip())
        print(f"Поток {thread_id}: завершен, {non_empty} непустых строк")
        
    except Exception as e:
        print(f"Поток {thread_id}: ошибка - {e}")
    finally:
        game.cleanup()
        print(f"Поток {thread_id}: очищен")

def test_threaded_instances():
    """Тест экземпляров в разных потоках."""
    print("=== Тест в потоках ===")
    
    threads = []
    num_threads = 3
    actions_per_thread = 4
    
    # Создаем и запускаем потоки
    for i in range(num_threads):
        thread = threading.Thread(
            target=worker_thread, 
            args=(i+1, actions_per_thread),
            name=f"RogueWorker-{i+1}"
        )
        threads.append(thread)
        thread.start()
    
    # Ждем завершения всех потоков
    for thread in threads:
        thread.join()
    
    print("Все потоки завершены\n")

def main():
    """Основная функция тестирования."""
    print("Тестирование множественных экземпляров RogueInterface\n")
    
    try:
        # Тест одного экземпляра
        test_single_instance()
        
        # Тест нескольких экземпляров
        test_multiple_instances()
        
        # Тест в потоках
        test_threaded_instances()
        
        print("✅ Все тесты прошли успешно!")
        
    except Exception as e:
        print(f"❌ Ошибка в тестах: {e}")
        raise

if __name__ == "__main__":
    main()
