from typing import Any
import numpy as np

ACTION_SPACE = 90

class ContinuousAct:
    def __init__(self):
        self.ACTION_LEN = 8 # public variable

    def parse_actions(self, action=np.ndarray) -> int:
        return action
