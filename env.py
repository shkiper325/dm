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
        
        # Трекинг для системы наград
        self.visited_positions = set()  # Посещенные позиции
        self.found_items = set()  # Найденные предметы/сокровища
        self.last_item_found_step = 0  # Шаг последнего найденного предмета
        self.stairs_positions = set()  # Позиции лестниц
        self.doors_positions = set()  # Позиции дверей
        self.last_exploration_reward = 0  # Последняя награда за исследование
        
        self.plus_tracker = TerminalScreen()

    def _count_hash_symbols(self, state: List[str]) -> int:
        """
        Подсчитывает количество символов '#' на карте.
        
        Args:
            state (List[str]): Состояние игры, представленное списком строк.
        
        Returns:
            int: Количество символов '#' на карте.
        """
        hash_count = 0
        for line in state:
            hash_count += line.count('#')
        return hash_count

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
    
        self.curr_step = 0
        
        # Сброс трекинга для системы наград
        self.visited_positions.clear()
        self.found_items.clear()
        self.last_item_found_step = 0
        self.stairs_positions.clear()
        self.doors_positions.clear()
        self.last_exploration_reward = 0
        
        # Добавляем начальную позицию игрока в посещенные
        try:
            player_pos = self._find_player(obs)
            self.visited_positions.add(player_pos)
        except ValueError:
            pass  # Игрок может быть не найден в начальном состоянии

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
        
        # Сохраняем старое состояние для расчета награды
        old_state = self.iface.state()
        old_pos = self._find_player(old_state)
        
        # Выполняем действие
        self.iface.key(action_map[action])
        
        # Получаем новое состояние
        new_state = self.iface.state()
        new_pos = self._find_player(new_state)
        
        # Рассчитываем награду с помощью новой системы
        reward = self._calculate_reward(old_state, new_state, old_pos, new_pos)

        self.curr_step += 1
        
        # Разделяем terminated (завершение из-за условий среды) и truncated (завершение по времени)
        terminated = False  # Пока нет условий естественного завершения
        truncated = self.curr_step >= self.max_steps

        info = {
            "steps": self.curr_step,
            "visited_positions": len(self.visited_positions),
            "found_items": len(self.found_items),
            "player_position": new_pos
        }

        return list2onehot(new_state), reward, terminated, truncated, info

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

    def _get_items_and_features(self, state: List[str]) -> Dict[str, List[Tuple[int, int]]]:
        """
        Находит все важные символы на карте (предметы, лестницы, двери и т.д.).
        
        Args:
            state (List[str]): Состояние игры
            
        Returns:
            Dict[str, List[Tuple[int, int]]]: Словарь с позициями различных символов
        """
        features = {
            'treasures': [],  # Сокровища: $, *
            'items': [],      # Предметы: !, ?, ), ], =
            'stairs': [],     # Лестницы: %
            'doors': [],      # Двери: +
            'enemies': [],    # Враги: буквы A-Z, a-z (кроме @)
        }
        
        treasure_symbols = {'$', '*'}
        item_symbols = {'!', '?', ')', ']', '='}
        
        for i, row in enumerate(state):
            for j, char in enumerate(row):
                if char in treasure_symbols:
                    features['treasures'].append((i, j))
                elif char in item_symbols:
                    features['items'].append((i, j))
                elif char == '%':
                    features['stairs'].append((i, j))
                elif char == '+':
                    features['doors'].append((i, j))
                elif char.isalpha() and char != '@':
                    features['enemies'].append((i, j))
                    
        return features
    
    def _calculate_reward(self, old_state: List[str], new_state: List[str], 
                         old_pos: Tuple[int, int], new_pos: Tuple[int, int]) -> float:
        """
        Рассчитывает награду на основе изменения состояния игры.
        
        Args:
            old_state: Предыдущее состояние
            new_state: Новое состояние  
            old_pos: Предыдущая позиция игрока
            new_pos: Новая позиция игрока
            
        Returns:
            float: Общая награда за шаг
        """
        reward = 0.0
        
        # 1. Базовая награда за выживание
        reward += 0.01
        
        # 2. Штраф за столкновение со стеной (нет движения)
        if old_pos == new_pos:
            reward -= 0.1
            return reward  # Ранний возврат при столкновении
        
        # 3. Награда за исследование новых территорий
        if new_pos not in self.visited_positions:
            reward += 0.15  # Значительная награда за новые клетки
            self.visited_positions.add(new_pos)
        else:
            # Штраф за повторное посещение (убывает с количеством посещений)
            visit_count = sum(1 for pos in self.visited_positions if pos == new_pos)
            reward -= 0.02 * min(visit_count, 5)  # Максимальный штраф 0.1
        
        # 4. Награды за нахождение предметов и сокровищ
        old_features = self._get_items_and_features(old_state)
        new_features = self._get_items_and_features(new_state)
        
        # Проверяем, собрал ли игрок что-то (предмет исчез с карты)
        old_treasures = set(old_features['treasures'])
        new_treasures = set(new_features['treasures'])
        collected_treasures = old_treasures - new_treasures
        
        old_items = set(old_features['items'])
        new_items = set(new_features['items'])
        collected_items = old_items - new_items
        
        # Большая награда за сокровища
        if collected_treasures:
            reward += 1.0 * len(collected_treasures)
            self.last_item_found_step = self.curr_step
            
        # Средняя награда за предметы
        if collected_items:
            reward += 0.5 * len(collected_items)
            self.last_item_found_step = self.curr_step
        
        # 5. Прогресс к важным целям (лестницы, двери)
        if new_features['stairs']:
            # Награда за приближение к лестницам
            min_stairs_dist = min(abs(new_pos[0] - pos[0]) + abs(new_pos[1] - pos[1]) 
                                for pos in new_features['stairs'])
            old_min_stairs_dist = float('inf')
            if old_features['stairs']:
                old_min_stairs_dist = min(abs(old_pos[0] - pos[0]) + abs(old_pos[1] - pos[1]) 
                                        for pos in old_features['stairs'])
            
            if old_min_stairs_dist != float('inf') and min_stairs_dist < old_min_stairs_dist:
                reward += 0.05  # Награда за приближение к лестнице
        
        if new_features['doors']:
            # Награда за приближение к дверям
            min_door_dist = min(abs(new_pos[0] - pos[0]) + abs(new_pos[1] - pos[1]) 
                              for pos in new_features['doors'])
            old_min_door_dist = float('inf')
            if old_features['doors']:
                old_min_door_dist = min(abs(old_pos[0] - pos[0]) + abs(old_pos[1] - pos[1]) 
                                      for pos in old_features['doors'])
            
            if old_min_door_dist != float('inf') and min_door_dist < old_min_door_dist:
                reward += 0.03  # Меньшая награда за приближение к двери
        
        # 6. Штраф за бездействие (долго не находил предметы)
        steps_since_item = self.curr_step - self.last_item_found_step
        if steps_since_item > 50:
            reward -= 0.01 * min(steps_since_item - 50, 100) / 100  # Растущий штраф
        
        # 7. Небольшой штраф за близость к врагам (опционально)
        if new_features['enemies']:
            min_enemy_dist = min(abs(new_pos[0] - pos[0]) + abs(new_pos[1] - pos[1]) 
                               for pos in new_features['enemies'])
            if min_enemy_dist <= 2:  # Если враг очень близко
                reward -= 0.02 * (3 - min_enemy_dist)  # Штраф увеличивается при приближении
        
        return reward

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