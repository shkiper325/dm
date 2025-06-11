#!/usr/bin/env python3
"""Тесты для RogueInterface класса.

Тестирует основную функциональность обёртки над rogue через tmux.
"""

import pytest
import subprocess
import time
import shutil
from unittest.mock import Mock, patch, MagicMock
from rogue_iface_tmux import RogueInterface


class TestRogueInterface:
    """Тестовый класс для RogueInterface."""

    def test_init_default_params(self):
        """Тест инициализации с параметрами по умолчанию."""
        with patch('shutil.which', return_value='/usr/bin/tmux'):
            interface = RogueInterface()
            assert interface.cmd == "rogue"
            assert interface.rows == 24
            assert interface.cols == 80
            assert interface.session.startswith("rogue")

    def test_init_custom_params(self):
        """Тест инициализации с пользовательскими параметрами."""
        with patch('shutil.which', return_value='/usr/bin/tmux'):
            interface = RogueInterface(
                session="test_session",
                cmd="/usr/games/rogue",
                rows=30,
                cols=120
            )
            assert interface.session == "test_session"
            assert interface.cmd == "/usr/games/rogue"
            assert interface.rows == 30
            assert interface.cols == 120

    def test_init_no_tmux(self):
        """Тест что инициализация падает если tmux не найден."""
        with patch('shutil.which', return_value=None):
            with pytest.raises(EnvironmentError, match="tmux not found in PATH"):
                RogueInterface()

    @patch('subprocess.run')
    def test_run_basic(self, mock_run):
        """Тест базового вызова _run."""
        with patch('shutil.which', return_value='/usr/bin/tmux'):
            interface = RogueInterface()
            mock_run.return_value = Mock()
            
            interface._run("test", "args")
            
            mock_run.assert_called_once_with(
                ["tmux", "test", "args"], 
                check=True
            )

    @patch('subprocess.run')
    def test_run_with_capture(self, mock_run):
        """Тест _run с захватом вывода."""
        with patch('shutil.which', return_value='/usr/bin/tmux'):
            interface = RogueInterface()
            mock_run.return_value = Mock(stdout="test output")
            
            result = interface._run("test", capture=True)
            
            mock_run.assert_called_once_with(
                ["tmux", "test"], 
                check=True, 
                text=True, 
                stdout=subprocess.PIPE
            )
            assert result.stdout == "test output"

    @patch('subprocess.run')
    def test_session_exists_true(self, mock_run):
        """Тест _session_exists когда сессия существует."""
        with patch('shutil.which', return_value='/usr/bin/tmux'):
            interface = RogueInterface(session="test_session")
            mock_run.return_value = Mock(returncode=0)
            
            result = interface._session_exists()
            
            assert result is True
            mock_run.assert_called_once_with([
                "tmux", "has-session", "-t", "test_session"
            ])

    @patch('subprocess.run')
    def test_session_exists_false(self, mock_run):
        """Тест _session_exists когда сессия не существует."""
        with patch('shutil.which', return_value='/usr/bin/tmux'):
            interface = RogueInterface(session="test_session")
            mock_run.return_value = Mock(returncode=1)
            
            result = interface._session_exists()
            
            assert result is False

    @patch('time.sleep')
    @patch.object(RogueInterface, '_run')
    @patch.object(RogueInterface, '_session_exists')
    def test_restart_new_session(self, mock_exists, mock_run, mock_sleep):
        """Тест restart для новой сессии."""
        with patch('shutil.which', return_value='/usr/bin/tmux'):
            interface = RogueInterface(session="test_session", cmd="rogue")
            mock_exists.return_value = False
            
            interface.restart()
            
            # Проверяем что вызывается создание новой сессии
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0]
            assert call_args[0] == "new-session"
            assert "-d" in call_args
            assert "-s" in call_args
            assert "test_session" in call_args
            assert "-x" in call_args
            assert "80" in call_args
            assert "-y" in call_args
            assert "24" in call_args
            
            mock_sleep.assert_called_once_with(0.3)

    @patch('time.sleep')
    @patch.object(RogueInterface, '_run')
    @patch.object(RogueInterface, '_session_exists')
    def test_restart_kill_existing_session(self, mock_exists, mock_run, mock_sleep):
        """Тест restart с удалением существующей сессии."""
        with patch('shutil.which', return_value='/usr/bin/tmux'):
            interface = RogueInterface(session="test_session")
            mock_exists.return_value = True
            
            interface.restart()
            
            # Проверяем что вызывается и kill-session и new-session
            assert mock_run.call_count == 2
            first_call = mock_run.call_args_list[0][0]
            assert first_call[0] == "kill-session"
            assert "-t" in first_call
            assert "test_session" in first_call

    @patch('time.sleep')
    @patch.object(RogueInterface, '_run')
    @patch.object(RogueInterface, '_session_exists')
    def test_key_basic(self, mock_exists, mock_run, mock_sleep):
        """Тест отправки клавиш."""
        with patch('shutil.which', return_value='/usr/bin/tmux'):
            interface = RogueInterface(session="test_session")
            mock_exists.return_value = True
            
            interface.key("hjkl")
            
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0]
            assert call_args[0] == "send-keys"
            assert "-t" in call_args
            assert "test_session:0.0" in call_args
            assert "h" in call_args
            assert "j" in call_args
            assert "k" in call_args
            assert "l" in call_args
            
            mock_sleep.assert_called_once_with(0.05)

    @patch('time.sleep')
    @patch.object(RogueInterface, '_run')
    @patch.object(RogueInterface, '_session_exists')
    def test_key_with_enter(self, mock_exists, mock_run, mock_sleep):
        """Тест отправки клавиш с Enter."""
        with patch('shutil.which', return_value='/usr/bin/tmux'):
            interface = RogueInterface(session="test_session")
            mock_exists.return_value = True
            
            interface.key("q", enter=True)
            
            call_args = mock_run.call_args[0]
            assert "q" in call_args
            assert "Enter" in call_args

    @patch.object(RogueInterface, '_session_exists')
    def test_key_no_session(self, mock_exists):
        """Тест key без активной сессии."""
        with patch('shutil.which', return_value='/usr/bin/tmux'):
            interface = RogueInterface()
            mock_exists.return_value = False
            
            with pytest.raises(RuntimeError, match="Session not running"):
                interface.key("h")

    @patch.object(RogueInterface, '_run')
    @patch.object(RogueInterface, '_session_exists')
    def test_state_basic(self, mock_exists, mock_run):
        """Тест получения состояния экрана."""
        with patch('shutil.which', return_value='/usr/bin/tmux'):
            interface = RogueInterface(session="test_session")
            mock_exists.return_value = True
            
            # Имитируем вывод tmux capture-pane
            mock_output = "Line 1\nLine 2\nLine 3"
            mock_run.return_value = Mock(stdout=mock_output)
            
            result = interface.state()
            
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0]
            assert call_args[0] == "capture-pane"
            assert "-pJ" in call_args
            assert "-S" in call_args
            assert "-24" in call_args
            assert "-t" in call_args
            assert "test_session:0.0" in call_args
            
            # Проверяем что строки дополняются пробелами до нужной длины
            assert len(result) == 3
            assert len(result[0]) == 80
            assert result[0].startswith("Line 1")

    @patch.object(RogueInterface, '_session_exists')
    def test_state_no_session(self, mock_exists):
        """Тест state без активной сессии."""
        with patch('shutil.which', return_value='/usr/bin/tmux'):
            interface = RogueInterface()
            mock_exists.return_value = False
            
            with pytest.raises(RuntimeError, match="Session not running"):
                interface.state()

    @patch.object(RogueInterface, '_run')
    @patch.object(RogueInterface, '_session_exists')
    @patch.object(RogueInterface, 'restart')
    def test_context_manager_enter(self, mock_restart, mock_exists, mock_run):
        """Тест входа в контекстный менеджер."""
        with patch('shutil.which', return_value='/usr/bin/tmux'):
            interface = RogueInterface()
            
            result = interface.__enter__()
            
            mock_restart.assert_called_once()
            assert result is interface

    @patch.object(RogueInterface, '_run')
    @patch.object(RogueInterface, '_session_exists')
    def test_context_manager_exit(self, mock_exists, mock_run):
        """Тест выхода из контекстного менеджера."""
        with patch('shutil.which', return_value='/usr/bin/tmux'):
            interface = RogueInterface(session="test_session")
            mock_exists.return_value = True
            
            interface.__exit__(None, None, None)
            
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0]
            assert call_args[0] == "kill-session"
            assert "-t" in call_args
            assert "test_session" in call_args

    @patch.object(RogueInterface, '_run')
    @patch.object(RogueInterface, '_session_exists')
    def test_context_manager_exit_no_session(self, mock_exists, mock_run):
        """Тест выхода из контекстного менеджера без сессии."""
        with patch('shutil.which', return_value='/usr/bin/tmux'):
            interface = RogueInterface()
            mock_exists.return_value = False
            
            interface.__exit__(None, None, None)
            
            # Не должно вызываться kill-session если сессии нет
            mock_run.assert_not_called()

    @patch('time.sleep')
    @patch.object(RogueInterface, '_run')
    @patch.object(RogueInterface, '_session_exists')
    def test_context_manager_full_workflow(self, mock_exists, mock_run, mock_sleep):
        """Тест полного рабочего процесса с контекстным менеджером."""
        with patch('shutil.which', return_value='/usr/bin/tmux'):
            mock_exists.side_effect = [False, True, True, True]  # для restart, key, state, exit
            mock_run.return_value = Mock(stdout="test line")
            
            with RogueInterface(session="test_session") as game:
                game.key("h")
                state = game.state()
            
            # Проверяем что все операции выполнились
            assert mock_run.call_count >= 3  # new-session, send-keys, capture-pane, kill-session
            assert len(state) == 1
            assert state[0].startswith("test line")


class TestRogueInterfaceIntegration:
    """Интеграционные тесты (требуют установленный tmux)."""
    
    @pytest.mark.skipif(not shutil.which("tmux"), reason="tmux not available")
    def test_real_tmux_session_lifecycle(self):
        """Тест реального жизненного цикла tmux сессии."""
        interface = RogueInterface(session="test_integration_session")
        
        try:
            # Проверяем что сессии нет
            assert not interface._session_exists()
            
            # Создаём сессию (без rogue, просто shell)
            interface._run(
                "new-session", "-d", "-s", "test_integration_session", 
                "-x", "80", "-y", "24", "bash"
            )
            time.sleep(0.1)
            
            # Проверяем что сессия создалась
            assert interface._session_exists()
            
            # Отправляем команду и проверяем состояние
            interface._run("send-keys", "-t", "test_integration_session:0.0", "echo", "hello", "Enter")
            time.sleep(0.1)
            
            result = interface._run("capture-pane", "-pJ", "-t", "test_integration_session:0.0", capture=True)
            assert "hello" in result.stdout
            
        finally:
            # Убираем сессию
            interface._run("kill-session", "-t", "test_integration_session", check=False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
