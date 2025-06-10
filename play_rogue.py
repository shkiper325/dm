#!/usr/bin/env python3
"""
Interactive harness for testing RogueInterface.

* Отображает актуальный 24×80 экран игры.
* Считывает нажатия клавиш без ожидания <Enter>.
* Пересылает их в работающий процесс rogue.

Выход — клавиша q (или Q).
"""

import os
import sys
import tty
import termios
from PIL import Image

from rogue_iface_tmux import RogueInterface        # класс из предыдущего ответа


# ---------- low-level input ------------------------------------------------- #

def get_key() -> str:
    """
    Прочитать одну «клавишу» из stdin в raw-режиме и вернуть её как str.

    • Для обычных символов (a, $, пробел …) это одна буква.  
    • Для Escape-последовательностей (стрелки и пр.) вернётся
      вся последовательность целиком, например «\x1b[A».
    """
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)               # мгновенный ввод без буферизации
        ch = sys.stdin.read(1)
        if ch == "\x1b":             # возможно, начинается ESC-последовательность
            # Стандартные стрелки — ещё два байта, читаем их если есть
            ch += sys.stdin.read(2)
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ---------- main loop ------------------------------------------------------- #

def main() -> None:
    game = RogueInterface()
    game.restart()

    try:
        i = 0
        while True:
            # os.system("clear")

            for row in game.state():
                print("".join(row))
            print("\nPress in-game keys; Q to quit.", flush=True)

            key = get_key()
            if key.lower() == "q":
                break
            game.key(key)

            # img = Image.fromarray(game.screenshot())   # Pillow ожидает H×W×C, dtype=uint8
            # img.save(os.path.join('images', f"output_{i}.png"), format="PNG")
            # i += 1
    finally:
        print("\nExiting…")


if __name__ == "__main__":
    main()
