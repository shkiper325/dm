import os

from stable_baselines3.common.callbacks import BaseCallback

class SaveEveryNStepsCallback(BaseCallback):
    """Callback for *Stable Baselines 3* that saves the model every ``save_freq`` calls to :py:meth:`_on_step`.

    Since :py:meth:`_on_step` is executed once **per environment step**, this is effectively a checkpoint
    every *N* steps of experience collected.

    Parameters
    ----------
    save_freq : int
        Interval (in environment steps) between two checkpoints.
    save_path : str
        Directory where the checkpoint files will be written. It is created if it does not exist.
    name_prefix : str, optional
        Base filename for checkpoints (``<name_prefix>_<step>.zip``). Defaults to ``"model"``.
    verbose : int, optional
        0 = silent, 1 = info messages. Defaults to 0.
    """

    def __init__(self, *, save_freq: int = 100, save_path: str = "./checkpoints", name_prefix: str = "model", verbose: int = 0):
        super().__init__(verbose)
        assert save_freq > 0, "save_freq must be positive"
        self.save_freq = save_freq
        self.save_path = os.path.abspath(save_path)
        self.name_prefix = name_prefix

        # Create directory once here to avoid race conditions in multi‑process setups
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:  # noqa: D401,E501  (Stable Baselines 3 expects this exact signature)
        """This method is called by the model wrapper after every environment step."""
        # self.n_calls = number of times the callback has been called so far (starts at 1)
        if self.n_calls % self.save_freq == 0:
            checkpoint_file = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}.zip")
            self.model.save(checkpoint_file)  # type: ignore[attr-defined]

            if self.verbose:
                print(f"[SaveEveryNStepsCallback] Saved model to {checkpoint_file}")

        # Returning True lets training continue. Return False to interrupt.
        return True