import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Any
import gym
from gym import spaces

from env import RogueEnv


class TestRogueEnv(unittest.TestCase):
    """Тесты для класса RogueEnv."""

    def setUp(self) -> None:
        """Настройка тестового окружения."""
        self.max_steps = 100
        
        # Создаем mock для RogueInterface
        with patch('env.RogueInterface') as mock_iface:
            self.mock_iface_instance = Mock()
            mock_iface.return_value = self.mock_iface_instance
            self.env = RogueEnv(max_steps=self.max_steps)

    def test_init(self) -> None:
        """Тест инициализации среды."""
        # Проверяем, что пространства действий и наблюдений инициализированы правильно
        self.assertIsInstance(self.env.action_space, spaces.Discrete)
        self.assertEqual(self.env.action_space.n, 4)
        
        self.assertIsInstance(self.env.observation_space, spaces.Box)
        self.assertEqual(self.env.observation_space.shape, (24, 80))
        
        # Проверяем, что max_steps установлен правильно
        self.assertEqual(self.env.max_steps, self.max_steps)
        
        # Проверяем, что интерфейс создан
        self.assertIsNotNone(self.env.iface)

    def test_non_empty_method(self) -> None:
        """Тест метода _non_empty."""
        # Создаем тестовое состояние
        test_state = [
            "  hello  ",
            "   world ",
            "         ",
            "test     "
        ]
        
        result = self.env._non_empty(test_state)
        # Ожидаем: h,e,l,l,o,w,o,r,l,d = 10 символов (исключая пробелы и последнюю строку)
        expected = 10
        self.assertEqual(result, expected)

    def test_non_empty_empty_state(self) -> None:
        """Тест метода _non_empty с пустым состоянием."""
        test_state = [
            "         ",
            "         ",
            "         "
        ]
        
        result = self.env._non_empty(test_state)
        self.assertEqual(result, 0)

    def test_reset(self) -> None:
        """Тест метода reset."""
        # Настраиваем mock для возврата тестового состояния
        mock_state = [['t', 'e', 's', 't'] * 20 for _ in range(24)]
        self.mock_iface_instance.state.return_value = mock_state
        
        # Вызываем reset
        observation = self.env.reset()
        
        # Проверяем, что restart был вызван
        self.mock_iface_instance.restart.assert_called_once()
        
        # Проверяем, что curr_step сброшен
        self.assertEqual(self.env.curr_step, 0)
        
        # Проверяем, что last_score установлен
        self.assertIsNotNone(self.env.last_score)
        
        # Проверяем, что возвращается правильное наблюдение
        self.assertEqual(observation, mock_state)

    def test_step_valid_actions(self) -> None:
        """Тест метода step с валидными действиями."""
        # Настраиваем mock
        mock_state = [['.' for _ in range(80)] for _ in range(24)]
        self.mock_iface_instance.state.return_value = mock_state
        
        # Инициализируем среду
        self.env.reset()
        
        # Тестируем каждое действие
        for action in [0, 1, 2, 3]:
            with self.subTest(action=action):
                obs, reward, done, info = self.env.step(action)
                
                # Проверяем, что возвращается правильная структура
                self.assertEqual(obs, mock_state)
                self.assertIsInstance(reward, (int, float))
                self.assertIsInstance(done, bool)
                self.assertIsInstance(info, dict)
                
                # Проверяем содержимое info
                self.assertIn("score", info)
                self.assertIn("steps", info)
                self.assertIsInstance(info["score"], (int, float))
                self.assertIsInstance(info["steps"], int)

    def test_step_invalid_action(self) -> None:
        """Тест метода step с невалидным действием."""
        # Настраиваем mock
        mock_state = [['.' for _ in range(80)] for _ in range(24)]
        self.mock_iface_instance.state.return_value = mock_state
        
        # Инициализируем среду
        self.env.reset()
        
        # Тестируем невалидное действие
        with self.assertRaises(ValueError) as cm:
            self.env.step(5)
        
        self.assertIn("Invalid action", str(cm.exception))

    def test_step_action_mapping(self) -> None:
        """Тест правильного маппинга действий."""
        # Настраиваем mock
        mock_state = [['.' for _ in range(80)] for _ in range(24)]
        self.mock_iface_instance.state.return_value = mock_state
        
        # Инициализируем среду
        self.env.reset()
        
        # Тестируем маппинг действий
        action_map = {
            0: "h",  # Влево
            1: "j",  # Вниз
            2: "k",  # Вверх
            3: "l"   # Вправо
        }
        
        for action, expected_key in action_map.items():
            with self.subTest(action=action):
                self.env.step(action)
                # Проверяем, что правильная клавиша была отправлена
                self.mock_iface_instance.key.assert_called_with(expected_key)

    def test_step_done_condition(self) -> None:
        """Тест условия завершения эпизода."""
        # Настраиваем mock
        mock_state = [['.' for _ in range(80)] for _ in range(24)]
        self.mock_iface_instance.state.return_value = mock_state
        
        # Создаем среду с маленьким max_steps
        with patch('env.RogueInterface') as mock_iface:
            mock_iface.return_value = self.mock_iface_instance
            env = RogueEnv(max_steps=2)
        
        env.reset()
        
        # Первый шаг - не должен быть done
        obs, reward, done, info = env.step(0)
        self.assertFalse(done)
        self.assertEqual(info["steps"], 1)
        
        # Второй шаг - должен быть done
        obs, reward, done, info = env.step(1)
        self.assertTrue(done)
        self.assertEqual(info["steps"], 2)

    def test_render(self) -> None:
        """Тест метода render."""
        # Настраиваем mock для возврата тестового состояния
        mock_state = [
            ['H', 'e', 'l', 'l', 'o'],
            ['W', 'o', 'r', 'l', 'd']
        ]
        self.mock_iface_instance.state.return_value = mock_state
        
        # Перехватываем print для проверки вывода
        with patch('builtins.print') as mock_print:
            self.env.render()
            
            # Проверяем, что print был вызван для каждой строки
            expected_calls = [
                unittest.mock.call('Hello'),
                unittest.mock.call('World')
            ]
            mock_print.assert_has_calls(expected_calls)

    def test_render_with_mode(self) -> None:
        """Тест метода render с параметром mode."""
        mock_state = [['t', 'e', 's', 't']]
        self.mock_iface_instance.state.return_value = mock_state
        
        with patch('builtins.print') as mock_print:
            self.env.render(mode="human")
            mock_print.assert_called_once_with('test')

    def test_metadata(self) -> None:
        """Тест metadata класса."""
        self.assertIn("render.modes", RogueEnv.metadata)
        self.assertIn("human", RogueEnv.metadata["render.modes"])

    def test_gym_env_inheritance(self) -> None:
        """Тест того, что RogueEnv наследуется от gym.Env."""
        self.assertIsInstance(self.env, gym.Env)

    def test_step_count_increment(self) -> None:
        """Тест инкремента счетчика шагов."""
        # Настраиваем mock
        mock_state = [['.' for _ in range(80)] for _ in range(24)]
        self.mock_iface_instance.state.return_value = mock_state
        
        self.env.reset()
        
        # Делаем несколько шагов и проверяем счетчик
        for i in range(1, 4):
            obs, reward, done, info = self.env.step(0)
            self.assertEqual(self.env.curr_step, i)
            self.assertEqual(info["steps"], i)


if __name__ == '__main__':
    unittest.main()
