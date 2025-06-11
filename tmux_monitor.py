#!/usr/bin/env python3
"""
Утилита для мониторинга и управления tmux сессиями RogueInterface.
"""

import subprocess
import sys
import time
from typing import List, Dict


def get_all_tmux_sessions() -> List[Dict[str, str]]:
    """Получает список всех tmux сессий."""
    try:
        result = subprocess.run(
            ["tmux", "list-sessions", "-F", "#{session_name}:#{session_created}:#{session_attached}"],
            capture_output=True,
            text=True,
            check=True
        )
        
        sessions = []
        for line in result.stdout.strip().split('\n'):
            if line:
                name, created, attached = line.split(':')
                sessions.append({
                    'name': name,
                    'created': created,
                    'attached': attached == '1'
                })
        return sessions
    
    except subprocess.CalledProcessError:
        return []


def get_rogue_sessions() -> List[Dict[str, str]]:
    """Получает только Rogue сессии."""
    all_sessions = get_all_tmux_sessions()
    return [s for s in all_sessions if s['name'].startswith('rogue')]


def kill_session(session_name: str) -> bool:
    """Убивает указанную сессию."""
    try:
        subprocess.run(["tmux", "kill-session", "-t", session_name], check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def kill_all_rogue_sessions():
    """Убивает все Rogue сессии."""
    rogue_sessions = get_rogue_sessions()
    killed_count = 0
    
    for session in rogue_sessions:
        if kill_session(session['name']):
            print(f"✓ Убита сессия: {session['name']}")
            killed_count += 1
        else:
            print(f"✗ Ошибка при убивании сессии: {session['name']}")
    
    print(f"Убито сессий: {killed_count}/{len(rogue_sessions)}")


def monitor_sessions():
    """Мониторинг сессий в реальном времени."""
    print("Мониторинг tmux сессий Rogue (Ctrl+C для выхода)")
    print("=" * 60)
    
    try:
        while True:
            rogue_sessions = get_rogue_sessions()
            
            # Очищаем экран
            print("\033[2J\033[H", end="")
            
            print(f"Активных Rogue сессий: {len(rogue_sessions)}")
            print("=" * 60)
            
            if rogue_sessions:
                print(f"{'Имя сессии':<35} {'Создана':<15} {'Прикреплена':<12}")
                print("-" * 60)
                
                for session in rogue_sessions:
                    attached = "Да" if session['attached'] else "Нет"
                    created_time = time.strftime('%H:%M:%S', time.localtime(int(session['created'])))
                    print(f"{session['name']:<35} {created_time:<15} {attached:<12}")
            else:
                print("Нет активных Rogue сессий")
            
            print(f"\nОбновлено: {time.strftime('%H:%M:%S')}")
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\nМониторинг остановлен")


def show_help():
    """Показывает справку по командам."""
    print("Утилита управления tmux сессиями RogueInterface")
    print("=" * 50)
    print("Команды:")
    print("  list     - Показать все Rogue сессии")
    print("  monitor  - Мониторинг сессий в реальном времени")
    print("  kill-all - Убить все Rogue сессии")
    print("  kill <name> - Убить конкретную сессию")
    print("  help     - Показать эту справку")
    print()


def main():
    """Основная функция."""
    if len(sys.argv) < 2:
        show_help()
        return
    
    command = sys.argv[1].lower()
    
    if command == "list":
        rogue_sessions = get_rogue_sessions()
        print(f"Найдено Rogue сессий: {len(rogue_sessions)}")
        
        if rogue_sessions:
            print(f"{'Имя сессии':<35} {'Создана':<15} {'Прикреплена':<12}")
            print("-" * 60)
            
            for session in rogue_sessions:
                attached = "Да" if session['attached'] else "Нет"
                created_time = time.strftime('%H:%M:%S %d.%m', time.localtime(int(session['created'])))
                print(f"{session['name']:<35} {created_time:<15} {attached:<12}")
    
    elif command == "monitor":
        monitor_sessions()
    
    elif command == "kill-all":
        print("Убиваем все Rogue сессии...")
        kill_all_rogue_sessions()
    
    elif command == "kill":
        if len(sys.argv) < 3:
            print("Ошибка: укажите имя сессии")
            print("Использование: python tmux_monitor.py kill <session_name>")
            return
        
        session_name = sys.argv[2]
        if kill_session(session_name):
            print(f"✓ Сессия {session_name} убита")
        else:
            print(f"✗ Ошибка при убивании сессии {session_name}")
    
    elif command == "help":
        show_help()
    
    else:
        print(f"Неизвестная команда: {command}")
        show_help()


if __name__ == "__main__":
    main()
