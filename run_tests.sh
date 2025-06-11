#!/bin/bash

# Скрипт для запуска тестов RogueInterface

echo "Установка зависимостей для тестирования..."
pip install -r requirements-test.txt

echo "Запуск тестов..."
python -m pytest test_rogue_interface.py -v

echo "Запуск тестов с покрытием кода (если доступен coverage)..."
if command -v coverage &> /dev/null; then
    coverage run -m pytest test_rogue_interface.py
    coverage report -m --include="rogue_iface_tmux.py"
else
    echo "Coverage не установлен. Установите его командой: pip install coverage"
fi
