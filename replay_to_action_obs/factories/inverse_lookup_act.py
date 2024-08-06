from typing import Any
import numpy as np

ACTION_SPACE = 90

class InverseLookupAct:
    def __init__(self, bins=None):
        if bins is None:
            self.bins = [(-1, 0, 1)] * 5
        elif isinstance(bins[0], (float, int)):
            self.bins = [bins] * 5
        else:
            assert len(bins) == 5, "Need bins for throttle, steer, pitch, yaw and roll"
            self.bins = bins
        self._lookup_table = self.make_lookup_table(self.bins)
        # print(self._lookup_table)
        # quit()
        self._inverse_lookup_table = {
            tuple(action): i for i, action in enumerate(self._lookup_table)
        }

    @staticmethod
    def make_lookup_table(bins):
        actions = []
        # Ground
        for throttle in bins[0]:
            for steer in bins[1]:
                for boost in (0, 1):
                    for handbrake in (0, 1):
                        if boost == 1 and throttle != 1:
                            continue
                        actions.append(
                            [throttle or boost, steer, 0, steer, 0, 0, boost, handbrake]
                        )
        # [ 1 -1  0  0  0  0  1  0]

        # Aerial
        for pitch in bins[2]:
            for yaw in bins[3]:
                for roll in bins[4]:
                    for jump in (0, 1):
                        for boost in (0, 1):
                            if jump == 1 and yaw != 0:  # Only need roll for sideflip
                                continue
                            if pitch == roll == jump == 0:  # Duplicate with ground
                                continue
                            # Enable handbrake for potential wavedashes
                            handbrake = jump == 1 and (
                                pitch != 0 or yaw != 0 or roll != 0
                            )
                            actions.append(
                                [boost, yaw, pitch, yaw, roll, jump, boost, handbrake]
                            )
        actions = np.array(actions)
        return actions

        # [throttle or boost, steer, 0, steer, 0, 0, boost, handbrake]
        # [boost, yaw, pitch, yaw, roll, jump, boost, handbrake]
        # actual:
        # [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]

        # [ 1 -1  0  0  0  0  1  0]
        # (0.0, -1.0, -1.0, -1.0, -1.0, 0.0, 1.0, 0.0)

    def round_actions(self, action: np.ndarray) -> np.ndarray:
        # throttle, steer, pitch, yaw, roll
        bins = np.array([-0.25, 0.25])
        action[:5] = np.digitize(action[:5], bins) - 1
        # jump, boost, handbrake
        action[5:] = action[5:] > 0.5

        throttle, steer, pitch, yaw, roll, jump, boost, handbrake = range(8)
        # if we're ground
        if action[pitch] == action[roll] == action[jump] == 0:
            # set yaw to the steer value
            action[yaw] = action[steer]
            # set throttle to 1 if we're boosting
            if action[boost] == 1:
                action[throttle] = 1
        else:  # if we're in the air
            # set throttle to the boost value
            action[throttle] = action[boost]
            # set steer to the yaw value
            action[steer] = action[yaw]
            # enable handbrake for wavedashes
            action[handbrake] = action[jump] == 1 and (
                action[pitch] != 0 or action[yaw] != 0 or action[roll] != 0
            )
            # change yaw jump to roll jump
            if action[jump] == 1 and action[yaw] != 0:
                action[roll] = action[yaw]
                action[yaw] = 0
                action[steer] = 0

        return action

    def parse_actions(self, action=np.ndarray) -> int:
        try:
            action = self.round_actions(action)
        except Exception as e:
            print(action)
            quit()
        return self._inverse_lookup_table[tuple(action)]
    
    def __repr__(self) -> str:
        return_str = ""
        for i, action in enumerate(self._lookup_table):
            return_str += f"{i}: {action}\n"
        return return_str
