from typing import Any, Tuple, Dict, List
import gym
from gym import spaces
import numpy as np
from rogue_iface_tmux import RogueInterface


def list2onehot(state: List[str], rows: int = 24, cols: int = 80) -> np.ndarray:
    """
    Преобразует список строк в трехмерный numpy массив с one-hot кодированием.
    
    Args:
        state (List[str]): Список строк, представляющий состояние игры
        rows (int): Количество строк (по умолчанию 24)
        cols (int): Количество столбцов (по умолчанию 80)
    
    Returns:
        np.ndarray: Трехмерный массив размером 128 x rows x cols с one-hot кодированием
                   каждого ASCII символа
    """
    # Создаем трехмерный массив заполненный нулями в формате (128, rows, cols)
    result = np.zeros((128, rows, cols), dtype=np.float32)
    
    # Проходим по каждой позиции в state
    for row in range(min(rows, len(state))):
        line = state[row]
        for col in range(min(cols, len(line))):
            char = line[col]
            # Получаем ASCII код символа (7-бит, 0-127)
            ascii_code = ord(char) & 0x7F  # Маскируем до 7 бит для безопасности
            # Устанавливаем соответствующий бит в one-hot вектор
            result[ascii_code, row, col] = 1.0
    
    return result

class RogueEnv(gym.Env):
    """
    RL-среда для игры Rogue, совместимая с Ray RLlib.
    Пока методы не реализованы, используется заглушка.
    """
    metadata = {"render.modes": ["human"]}

    def _non_empty(self, state):
        ret = 0
        for line in state[:-1]:  # Исправлено: state[:-1] вместо state[state[:-1]]
            for ch in line:
                if ch != " ":
                    ret += 1
        return ret

    def __init__(self, max_steps: int) -> None:
        print(f'max_steps: {max_steps}')
        super().__init__()
        # Определение пространства действий и наблюдений.
        # Пока приведены примерные значения, которые можно изменить под конкретные требования.
        self.action_space: spaces.Space = spaces.Discrete(4)  # Пример: 4 возможных действия.
        # Изменяем observation_space для one-hot кодирования: (128, 24, 80)
        self.observation_space: spaces.Space = spaces.Box(low=0, high=1, shape=(128, 24, 80), dtype=np.float32)

        self.iface = RogueInterface(cmd="/usr/games/rogue")  # Явно указываем путь к rogue

        self.max_steps = max_steps
        self.curr_step = 0
        self.last_score = 0

    def reset(self, seed=None, options=None) -> Tuple[Any, Dict[str, Any]]:
        """
        Сбрасывает состояние среды в начальное состояние.

        Returns:
            Tuple containing the initial observation and info dict.
        """
        super().reset(seed=seed)
        
        self.iface.restart()
        self.curr_step = 0
        obs = self.iface.state()
        self.last_score = self._non_empty(obs)

        info = {"score": self.last_score, "steps": self.curr_step}
        return list2onehot(obs), info

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """
        Выполняет действие в среде.

        Args:
            action (Any): Действие, выбранное агентом.

        Returns:
            A tuple containing:
                - next observation,
                - reward,
                - terminated (done due to environment conditions),
                - truncated (done due to time limit),
                - info (additional info).
        """

        action_map = {
            0: "h",  # Влево
            1: "j",  # Вниз
            2: "k",  # Вверх
            3: "l"   # Вправо
        }

        if action not in action_map:
            raise ValueError(f"Invalid action: {action}. Must be one of {list(action_map.keys())}.")
        
        self.iface.key(action_map[action])
        obs = self.iface.state()

        new_score = self._non_empty(obs)
        reward = new_score - self.last_score
        self.last_score = new_score

        self.curr_step += 1
        
        # Разделяем terminated (завершение из-за условий среды) и truncated (завершение по времени)
        terminated = False  # Пока нет условий естественного завершения
        truncated = self.curr_step >= self.max_steps

        info = {"score": new_score, "steps": self.curr_step}

        return list2onehot(obs), reward, terminated, truncated, info

    def render(self, mode: str = "human") -> None:
        
        """
        Отображает текущее состояние среды.
        
        Args:
            mode (str): Режим отображения.
        """
        for row in self.iface.state():
            print("".join(row))
     
    def close(self) -> None:
        """
        Закрывает среду и освобождает ресурсы.
        """
        if hasattr(self, 'iface') and self.iface._session_exists():
            self.iface._run("kill-session", "-t", self.iface.session, check=False)

    def __del__(self):
        """Деструктор для автоматической очистки ресурсов."""
        try:
            self.close()
        except:
            pass  # Игнорируем ошибки при удалении объекта