# Baesd on https://github.com/AechPro/rocket-league-gym-sim/blob/main/rlgym_sim/utils/obs_builders/advanced_obs.py
# But modeified for building dataset from replays, not to be used in the environment
# since it doesn't inherit ObsBuilder

import math
import numpy as np
from typing import Any, List
from rlgym.rocket_league import common_values
from rlgym.rocket_league.api import Car, GameState, PhysicsObject
from rlgym.api import AgentID

OBS_SPACE = 99

class SingleFrameObs:
    POS_STD = 2300 # If you read this and wonder why, ping Rangler in the dead of night.
    ANG_STD = math.pi

    def __init__(self):
        pass

    def reset(self, initial_state: GameState):
        pass

    def build_obs(self, agent: AgentID, state: GameState) -> np.ndarray:
        car = state.cars[agent]
        if car.team_num == common_values.ORANGE_TEAM:
            inverted = True
            ball = state.inverted_ball
            pad_timers = state.inverted_boost_pad_timers
        else:
            inverted = False
            ball = state.ball
            pad_timers = state.boost_pad_timers
        pads = pad_timers == 0

        obs = [ball.position / self.POS_STD,
               ball.linear_velocity / self.POS_STD,
               ball.angular_velocity / self.ANG_STD,
               pads]

        physics = self._add_player_to_obs(obs, car, ball, inverted)

        allies = []
        enemies = []

        for other, other_car in state.cars.items():
            if other == agent:
                continue

            if other_car.team_num == car.team_num:
                team_obs = allies
            else:
                team_obs = enemies

            other_physics = self._add_player_to_obs(team_obs, other_car, ball, inverted)

            # Extra info
            team_obs.extend([
                (other_physics.position - physics.position) / self.POS_STD,
                (other_physics.linear_velocity - physics.linear_velocity) / self.POS_STD
            ])

        obs.extend(allies)
        obs.extend(enemies)
        return np.concatenate(obs)

    def _add_player_to_obs(self, obs: List, car: Car, ball: PhysicsObject, inverted: bool):
        if inverted:
            physics = car.inverted_physics
        else:
            physics = car.physics

        rel_pos = ball.position - physics.position
        rel_vel = ball.linear_velocity - physics.linear_velocity

        obs.extend([
            rel_pos / self.POS_STD,
            rel_vel / self.POS_STD,
            physics.position / self.POS_STD,
            physics.forward,
            physics.up,
            physics.linear_velocity / self.POS_STD,
            physics.angular_velocity / self.ANG_STD,
            [car.boost_amount,
             int(car.on_ground),
             int(car.can_flip),
             int(car.is_demoed)]])

        return physics