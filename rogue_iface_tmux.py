#!/usr/bin/env python3
"""rogue_tmux_interface.py

Упрощённая обёртка над классической консольной игрой *rogue*,
работающая через tmux‑сессию и не требующая ни `pexpect`, ни `pyte`.

Подходит для скриптов, автоматизации и RL‑агентов: позволяет
перезапускать игру, посылать клавиши и считывать текущее состояние
экрана как список строк фиксированного размера 24×80.

Пример использования
--------------------
>>> from rogue_tmux_interface import RogueTmuxInterface
>>> game = RogueTmuxInterface()
>>> game.restart()
>>> game.key("h")           # шаг влево
>>> screen = game.state()    # список из 24 строк
"""
from __future__ import annotations

import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import List, Optional
import random
import threading

__all__ = ["RogueInterface"]


class RogueInterface:
    """Мини‑API поверх *rogue* через tmux.

    Параметры
    ---------
    session : str, optional
        Имя tmux‑сессии, в которой будет запущена игра. Если не указано,
        генерируется уникальное имя с использованием времени и случайного числа.
    cmd : str, optional
        Команда запуска *rogue* (если не в $PATH, укажите полный путь).
    rows, cols : int, optional
        Размер экрана, *rogue* по умолчанию использует 24×80.
    """

    DEFAULT_ROWS = 24
    DEFAULT_COLS = 80
    
    # Блокировка для thread-safe генерации уникальных имен сессий
    _session_counter_lock = threading.Lock()
    _session_counter = 0

    def __init__(
        self,
        session: Optional[str] = None,
        cmd: str = "rogue",
        rows: int = DEFAULT_ROWS,
        cols: int = DEFAULT_COLS,
    ) -> None:
        # Генерируем уникальное имя сессии если не указано
        if session is None:
            with self._session_counter_lock:
                self._session_counter += 1
                session = f"rogue_{int(time.time())}_{self._session_counter}_{random.randint(1000, 9999)}"
        
        self.session = session
        self.cmd = cmd
        self.rows = rows
        self.cols = cols

        # Проверяем наличие tmux при инициализации, чтобы заранее кинуть ошибку 
        if not shutil.which("tmux"):
            raise EnvironmentError("tmux not found in PATH; install it first.")

    # --------------------------------------------------------------------- util

    def _run(self, *args: str, check: bool = True, capture: bool = False, suppress_stderr: bool = False) -> subprocess.CompletedProcess:
        """Запускает `tmux` с переданными аргументами.

        Возвращает CompletedProcess. Если capture=True, stdout возвращается
        строкой; иначе вывод идёт в /dev/null.
        """
        cmd = ["tmux", *args]
        kwargs = {"check": check}
        
        if capture:
            kwargs.update({"text": True, "stdout": subprocess.PIPE})
        
        if suppress_stderr:
            kwargs["stderr"] = subprocess.DEVNULL
            
        return subprocess.run(cmd, **kwargs)

    def _session_exists(self) -> bool:
        """True если tmux‑сессия уже существует."""
        result = subprocess.run([
            "tmux",
            "has-session",
            "-t",
            self.session,
        ], stderr=subprocess.DEVNULL)  # Подавляем вывод ошибок
        return result.returncode == 0

    # ----------------------------------------------------------------- public

    def restart(self) -> None:
        """Убивает (если нужно) и запускает *rogue* в новой tmux‑сессии."""
        # Сносим старую сессию целиком, чтобы всегда начинать с чистого экрана
        if self._session_exists():
            self._run("kill-session", "-t", self.session, check=False, suppress_stderr=True)

        # Формируем команду: выставляем переменные окружения для curses
        env_prefix = (
            f"env COLUMNS={self.cols} LINES={self.rows} TERM=xterm TERMINFO={os.environ.get('TERMINFO', '/usr/share/terminfo')} "
        )
        start_cmd = env_prefix + shlex.quote(self.cmd)

        # Создаём детачнутую сессию нужного размера
        self._run(
            "new-session",
            "-d",                # detached
            "-s",
            self.session,
            "-x",
            str(self.cols),
            "-y",
            str(self.rows),
            start_cmd,
        )

        # Немного ждём, чтобы игра успела инициализироваться
        time.sleep(0.3)
    # --------------------------------------------------------------------- I/O

    def key(self, keys: str, enter: bool = False) -> None:
        """Посылает набор символов в *rogue*.

        Параметры
        ---------
        keys : str
            Строка клавиш. Каждая буква/символ отправляется как отдельный
            *keypress* (tmux `send-keys`).
        enter : bool, optional
            Если True, добавляет ENTER после ключей (по умолчанию False).
        """
        if not self._session_exists():
            # Если сессия умерла, перезапускаем
            print(f"Session {self.session} not found, restarting...")
            self.restart()

        # tmux send-keys принимает последовательность аргументов: символы, затем специальные
        send_args: List[str] = ["send-keys", "-t", f"{self.session}:0.0"]
        send_args.extend(list(keys))
        if enter:
            send_args.append("Enter")
        
        try:
            self._run(*send_args)
        except subprocess.CalledProcessError:
            # Если команда не сработала, возможно сессия умерла
            print(f"Failed to send keys to session {self.session}, restarting...")
            self.restart()
            self._run(*send_args)

        # Небольшая пауза, чтобы игра обработала ввод
        time.sleep(0.05)

    def state(self) -> List[str]:
        """Считывает видимые 24×80 символов экрана.

        Возвращает список ровно из `self.rows` строк длиной `self.cols`.
        """
        if not self._session_exists():
            print(f"Session {self.session} not found, restarting...")
            self.restart()

        try:
            # -J склеивает перенёсы строк curses, -S -24 берёт последние 24 строки
            cp = self._run(
                "capture-pane",
                "-pJ",
                "-S",
                f"-{self.rows}",
                "-t",
                f"{self.session}:0.0",
                capture=True,
            )
            raw_text = cp.stdout
        except subprocess.CalledProcessError:
            print(f"Failed to capture pane from session {self.session}, restarting...")
            self.restart()
            cp = self._run(
                "capture-pane",
                "-pJ",
                "-S",
                f"-{self.rows}",
                "-t",
                f"{self.session}:0.0",
                capture=True,
            )
            raw_text = cp.stdout

        lines = raw_text.splitlines()
        # # Гарантируем нужное количество строк/столбцов
        # lines = lines[-self.rows :]  # точно не больше rows
        # lines += [""] * (self.rows - len(lines))  # если вдруг меньше
        # padded = [line.ljust(self.cols)[: self.cols] for line in lines]
        ret = [line + " " * (self.cols - len(line)) for line in lines]
        return ret

    def get_session_info(self) -> dict:
        """Возвращает информацию о текущей сессии."""
        return {
            "session_name": self.session,
            "cmd": self.cmd,
            "rows": self.rows,
            "cols": self.cols,
            "exists": self._session_exists()
        }

    def cleanup(self) -> None:
        """Принудительно закрывает сессию и освобождает ресурсы."""
        if self._session_exists():
            self._run("kill-session", "-t", self.session, check=False, suppress_stderr=True)
            print(f"Session {self.session} closed")

    # ----------------------------------------------------------------- dunder

    def __enter__(self):
        self.restart()
        return self

    def __exit__(self, exc_type, exc, tb):
        # Закрываем сессию при выходе из контекста
        self.cleanup()

    def __del__(self):
        """Деструктор для автоматической очистки ресурсов."""
        try:
            self.cleanup()
        except:
            # Игнорируем ошибки при удалении объекта
            pass


# ---------------------------------------------------------------------------
# Вспомогательное: тихий импорт shutil тут, чтобы не тащить в начало if py ≤3.9
import shutil  # noqa: E402, isort:skip

