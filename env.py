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
        np.ndarray: Трехмерный массив размером rows x cols x 128 с one-hot кодированием
                   каждого ASCII символа
    """
    # Создаем трехмерный массив заполненный нулями
    result = np.zeros((rows, cols, 128), dtype=np.float32)
    
    # Проходим по каждой позиции в state
    for row in range(min(rows, len(state))):
        line = state[row]
        for col in range(min(cols, len(line))):
            char = line[col]
            # Получаем ASCII код символа (7-бит, 0-127)
            ascii_code = ord(char) & 0x7F  # Маскируем до 7 бит для безопасности
            # Устанавливаем соответствующий бит в one-hot вектор
            result[row, col, ascii_code] = 1.0
    
    return result


def onehot2list(onehot_array: np.ndarray) -> List[str]:
    """
    Преобразует трехмерный numpy массив с one-hot кодированием обратно в список строк.
    
    Args:
        onehot_array (np.ndarray): Трехмерный массив размером rows x cols x 128 
                                  с one-hot кодированием ASCII символов
    
    Returns:
        List[str]: Список строк, восстановленный из one-hot представления
    """
    rows, cols, _ = onehot_array.shape
    result = []
    
    for row in range(rows):
        line = ""
        for col in range(cols):
            # Находим индекс максимального элемента (should be 1 in one-hot)
            ascii_code = np.argmax(onehot_array[row, col])
            # Преобразуем ASCII код обратно в символ
            char = chr(ascii_code)
            line += char
        result.append(line)
    
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

    def __init__(self, max_steps) -> None:
        super().__init__()
        # Определение пространства действий и наблюдений.
        # Пока приведены примерные значения, которые можно изменить под конкретные требования.
        self.action_space: spaces.Space = spaces.Discrete(4)  # Пример: 4 возможных действия.
        self.observation_space: spaces.Space = spaces.Box(low=0, high=255, shape=(24, 80), dtype=int)

        self.iface = RogueInterface()

        self.max_steps = max_steps

    def reset(self) -> Any:
        """
        Сбрасывает состояние среды в начальное состояние.

        Returns:
            The initial observation of the environment.
        """

        self.iface.restart()
        self.curr_step = 0
        self.last_score = self._non_empty(self.iface.state())

        return list2onehot(self.iface.state())

    def step(self, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """
        Выполняет действие в среде.

        Args:
            action (Any): Действие, выбранное агентом.

        Returns:
            A tuple containing:
                - next observation,
                - reward,
                - done (flag indicating termination),
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
        if self.curr_step >= self.max_steps:
            done = True
        else:
            done = False

        info = {"score": new_score, "steps": self.curr_step}

        return list2onehot(obs), reward, done, info

    def render(self, mode: str = "human") -> None:
        
        """
        Отображает текущее состояние среды.
        
        Args:
            mode (str): Режим отображения.
        """
        for row in self.iface.state():
            print("".join(row))