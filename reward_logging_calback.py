# class RewardLoggingCallback(BaseCallback):
#     """
#     Логирует среднюю награду за последние `log_freq` шагов обучения.

#     :param log_freq: сколько шагов усреднять перед записью
#     :param log_file: путь к файлу, куда вести лог
#     :param verbose: 0 — без вывода в консоль, 1 — печать каждого события логирования
#     """
#     def __init__(self, log_freq: int = 100, log_file: str = "rewards.txt", verbose: int = 0):
#         super().__init__(verbose)
#         self.log_freq = log_freq
#         self.log_file = log_file
#         self._reward_buffer = []  # накапливаем награды с момента последней записи

#     def _on_step(self) -> bool:
#         """
#         Вызывается библиотекой после каждого env-шага.
#         """
#         # В self.locals["rewards"] приходит список наград из всех параллельных окружений
#         self._reward_buffer.extend(self.locals.get("rewards", []))

#         # Достаточно данных для вычисления среднего?
#         if len(self._reward_buffer) >= self.log_freq:
#             avg_reward = float(np.mean(self._reward_buffer[:self.log_freq]))

#             # Записываем "<текущий_шаг>,<средняя_награда>\n"
#             with open(self.log_file, "a", encoding="utf-8") as f:
#                 f.write(f"{self.num_timesteps},{avg_reward}\n")

#             if self.verbose > 0:
#                 print(
#                     f"[RewardLoggingCallback] step={self.num_timesteps:>7} "
#                     f"avg_reward(last {self.log_freq}) = {avg_reward:.3f}"
#                 )

#             # очищаем буфер только на размер log_freq
#             del self._reward_buffer[:self.log_freq]

#         # True → продолжить обучение
#         return True