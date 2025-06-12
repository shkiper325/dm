from typing import Any, Tuple, Dict, List, Set, Optional, Optional, Set
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
        # self.main_score = 0

        self.plus_tracker = TerminalScreen()

    def _find_player(self, state: List[str]) -> Tuple[int, int]:
        """
        Находит позицию игрока '@' в состоянии игры.
        
        Args:
            state (List[str]): Состояние игры, представленное списком строк.
        
        Returns:
            Tuple[int, int]: Координаты игрока (row, col).
        """
        for i, row in enumerate(state):
            j = row.find('@')
            if j != -1:
                return i, j
        raise ValueError("Игрок '@' не найден в состоянии игры.")

    def reset(self, seed=None, options=None) -> Tuple[Any, Dict[str, Any]]:
        """
        Сбрасывает состояние среды в начальное состояние.

        Returns:
            Tuple containing the initial observation and info dict.
        """
        super().reset(seed=seed)

        self.iface.restart()
        obs = self.iface.state()
        # self.visited = set(self._find_player(obs))  # Инициализируем посещённые позиции
        self.plus_tracker.reset()
        
        self.door_score = self.plus_tracker.dist(obs)
        # self.main_score = self._non_empty(obs)

        self.curr_step = 0

        info = {"steps": self.curr_step}

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

        # new_main_score = self._non_empty(obs)
        # main_diff = np.tanh(new_main_score - self.main_score)
        # self.main_score = new_main_score

        new_door_score = self.plus_tracker.dist(obs)
        door_diff = new_door_score - self.door_score
        self.door_score = new_door_score

        reward = -door_diff  # Используем разницу в расстоянии до '+'

        self.curr_step += 1
        
        # Разделяем terminated (завершение из-за условий среды) и truncated (завершение по времени)
        terminated = False  # Пока нет условий естественного завершения
        truncated = self.curr_step >= self.max_steps

        info = {"steps": self.curr_step}

        # return list2onehot(obs), main_diff + door_diff, terminated, truncated, info
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

class TerminalScreen:
    """
    Класс для отслеживания позиций игрока '@' на терминальном экране и вычисления
    минимального L1-расстояния до непосещённых символов '+'.
    """

    def __init__(self):
        # Множество координат '+' (row, col), которые уже были посещены игроком
        self.visited: Set[Tuple[int, int]] = set()
        # Прошлое состояние экрана (для детекции посещения '+')
        self._last_grid: Optional[List[str]] = None
        # Прошлая позиция игрока '@'
        self._last_pos: Optional[Tuple[int, int]] = None

    def reset(self) -> None:
        """
        Сбросить историю посещённых '+'.
        После вызова reset() все '+' на экране будут считаться непосещёнными.
        """
        self.visited.clear()
        self._last_grid = None
        self._last_pos = None

    def dist(self, grid: List[str]) -> int:
        """
        Рассчитать минимальное L1-расстояние от текущей позиции игрока '@'
        до любого символа '+' в grid, который ещё не был посещён.
        При этом, если на предыдущем шаге игрок пришёл в клетку, где в прошлой версии grid
        стоял '+', то эта координата будет помечена как посещённая.

        :param grid: список строк одинаковой длины, представляющих экран.
                     На нём ровно один '@' и ноль или более '+'.
        :return: минимальное расстояние до непосещённого '+', или 0, если таких нет.
        """
        # Поиск текущей позиции '@'
        current_pos = None
        for i, row in enumerate(grid):
            j = row.find('@')
            if j != -1:
                current_pos = (i, j)
                break
        if current_pos is None:
            raise ValueError("Входной grid не содержит символа '@'")

        # Если у нас есть прошлый grid и игрок переместился
        if self._last_grid is not None and self._last_pos is not None:
            # Если на прошлой grid в точке current_pos был '+', то отметим её как посещённую
            i_cur, j_cur = current_pos
            if (i_cur, j_cur) != self._last_pos:
                if self._last_grid[i_cur][j_cur] == '+':
                    self.visited.add((i_cur, j_cur))

        # Обновляем сохранённое прошлое состояние
        self._last_grid = list(grid)  # строки неизменяемы, копирование списка достаточно
        self._last_pos = current_pos

        # Сбор всех '+' на экране, исключая уже посещённые
        targets = []
        for i, row in enumerate(grid):
            for j, ch in enumerate(row):
                if ch == '+' and (i, j) not in self.visited:
                    targets.append((i, j))

        if not targets:
            return 0

        # Вычисляем минимальное L1-расстояние
        ci, cj = current_pos
        min_dist = min(abs(ci - ti) + abs(cj - tj) for ti, tj in targets)
        return min_dist