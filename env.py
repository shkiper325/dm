from typing import Any, Tuple, Dict
import gym
from gym import spaces
from rogue_iface_tmux import RogueInterface


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

        return self.iface.state()

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

        return obs, reward, done, info

    def render(self, mode: str = "human") -> None:
        
        """
        Отображает текущее состояние среды.
        
        Args:
            mode (str): Режим отображения.
        """
        for row in self.iface.state():
            print("".join(row))