# Baesd on https://github.com/AechPro/rocket-league-gym-sim/blob/main/rlgym_sim/utils/obs_builders/advanced_obs.py
# But modeified for building dataset from replays, not to be used in the environment
# since it doesn't inherit ObsBuilder

import math
import numpy as np
from typing import Any, List
from rlgym.rocket_league import common_values
from rlgym.rocket_league.api import Car, GameState, PhysicsObject
from rlgym.api import AgentID

OBS_SPACE = 111  # This is wrong, I'm not sure what it's supposed to be


class SingleFramePyrObs:
    POS_STD = (
        2300  # If you read this and wonder why, ping Rangler in the dead of night.
    )
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

        obs = [
            ball.position / self.POS_STD,
            ball.linear_velocity / self.POS_STD,
            ball.angular_velocity / self.ANG_STD,
            pads,
        ]

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
            team_obs.extend(
                [
                    (other_physics.position - physics.position) / self.POS_STD,
                    (other_physics.linear_velocity - physics.linear_velocity)
                    / self.POS_STD,
                ]
            )

        obs.extend(allies)
        obs.extend(enemies)
        return np.concatenate(obs)

    def _add_player_to_obs(
        self, obs: List, car: Car, ball: PhysicsObject, inverted: bool
    ):
        if inverted:
            physics = car.inverted_physics
        else:
            physics = car.physics

        rel_pos = ball.position - physics.position
        rel_vel = ball.linear_velocity - physics.linear_velocity

        obs.extend(
            [
                rel_pos / self.POS_STD,
                rel_vel / self.POS_STD,
                physics.position / self.POS_STD,
                physics.forward,
                physics.up,
                physics.left,
                physics.linear_velocity / self.POS_STD,
                physics.angular_velocity / self.ANG_STD,
                [
                    physics.pitch,
                    physics.yaw,
                    physics.roll,
                    car.boost_amount,
                    int(car.on_ground),
                    int(car.can_flip),
                    int(car.is_demoed),
                ],
            ]
        )

        return physics


"""
0,1,2: ball_position.x,y,z
3,4,5: ball_linear_velocity.x,y,z
6,7,8: ball_angular_velocity.x,y,z
9 - 42: pads,
43,44,45: player_rel_pos.x,y,z
46,47,48: player_rel_vel.x,y,z
49,50,51: player_pos.x,y,z
52,53,54: player_forward.x,y,z
55,56,57: player_up.x,y,z
58,59,60: player_left.x,y,z
61,62,63: player_linear_velocity.x,y,z
64,65,66: player_angular_velocity.x,y,z
67,68,69: player.pyr
70: player_boost
71: player_on_ground
72: player_can_flip
73: player_is_demoed
74,75,76: opp_rel_pos.x,y,z
77,78,79: opp_rel_vel.x,y,z
80,81,82: opp_pos.x,y,z
83,84,85: opp_forward.x,y,z
86,87,88: opp_up.x,y,z
89,90,91: opp_left.x,y,z
92,93,94: opp_linear_velocity.x,y,z
95,96,97: opp_angular_velocity.x,y,z
98,99,100: opp.pyr
101: opp_boost
102: opp_on_ground
103: opp_can_flip
104: opp_is_demoed
105,106,107: player_opp_rel_pos.x,y,z
108,109,110: player_opp_rel_vel.x,y,z
"""
