import json
from abc import ABC
from pathlib import Path
from typing import Dict, Any

from logging import getLogger, INFO

logger = getLogger(__name__)
logger.setLevel(INFO)


class Stateful(ABC):

    def __init__(self, path: str):
        self.path = Path(path)

    def save_state(self) -> None:
        logger.info(f"Saving state to {str(self.path)}")
        with open(str(self.path.absolute()), 'w+') as f:
            json.dump(self.get_state(), f)

    def load_state(self) -> Dict[str, Any]:
        logger.info(f"Loading state from {str(self.path)}")
        with open(str(self.path.absolute()), 'r+') as f:
            state = json.load(f)
        return state

    def get_state(self) -> Dict[str, Any]:
        raise NotImplementedError("Get internal state!")

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Model must predict!")
