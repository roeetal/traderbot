from logging import getLogger, INFO

import numpy as np

from trade.defaults import Stateful

logger = getLogger(__name__)
logger.setLevel(INFO)


class KalmanOLS(Stateful):

    def __init__(self, path: str):
        super(KalmanOLS, self).__init__(path)
        try:
            state = self.load_state()
            self.delta = state["delta"]
            self.nu = state["nu"]
            self.w = self._deserialize(state["w"])
            self.theta = self._deserialize(state["theta"])
            self.R = self._deserialize(state["R"])
            self.C = self._deserialize(state["C"])
        except FileNotFoundError:
            logger.debug("Initialising state with defaults...")
            self.delta = 1e-4
            self.nu = 1e-3
            self.w = self.delta / (1 - self.delta) * np.eye(2)
            self.theta = np.zeros((2, 1))
            self.R = np.random.rand(2, 2) / 100
            self.C = None

    def get_state(self):
        return {
            "delta": self.delta,
            "nu": self.nu,
            "w": self._serialize(self.w),
            "theta": self._serialize(self.theta),
            "R": self._serialize(self.R),
            "C": self._serialize(self.C),
        }

    def __call__(self, y1: float, y2: float):
        F = np.array([[y1, 1]])
        self.R = self.R if self.C is None else self.C + self.w
        yp = (F.dot(self.theta)).item()
        et = y2 - yp
        Qt = (F.dot(self.R).dot(F.T)).item() + self.nu
        At = self.R.dot(F.T) / Qt
        self.theta += At * et
        self.C = self.R - At.dot(F).dot(self.R)

        return et, Qt, self.theta[0, 0]

    @staticmethod
    def _serialize(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        return x

    @staticmethod
    def _deserialize(x):
        if isinstance(x, list):
            return np.asarray(x)
        return x
