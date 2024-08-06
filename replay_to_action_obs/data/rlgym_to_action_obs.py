from typing import List, Tuple
import numpy as np
from tqdm.rich import tqdm

from rlgym_tools.replays.convert import ReplayFrame

from replay_to_action_obs.factories import SingleFrameObs, InverseLookupAct


def rlgym_frames_to_action_obs(
    frames: List[ReplayFrame],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Return: [action], [obs] for each player in the replay
    action: torch.Tensor of shape (n, 1), where n is the number of frames
    obs: torch.Tensor of shape (n, len_obs), where n is the number of frames
    """
    
    obs_builder = SingleFrameObs()
    action_parser = InverseLookupAct()

    agent_ids = list(frames[0].state.cars.keys())
    n_frames = len(frames)
    len_obs = len(obs_builder.build_obs(agent_ids[0], frames[0].state))

    actions = [np.zeros((n_frames, 1)) for _ in agent_ids]
    obs = [np.zeros((n_frames, len_obs)) for _ in agent_ids]

    for frame_idx, frame in enumerate(frames):
        for i, agent_id in enumerate(agent_ids):
            action = frame.actions[agent_id]
            if len(action.shape) == 2:
                action = action[0]  # take the first action bc it's repeated tick_skip times
            obs[i][frame_idx] = obs_builder.build_obs(agent_id, frame.state)
            actions[i][frame_idx] = action_parser.parse_actions(action)
            
    return actions, obs


if __name__ == "__main__":
    import os
    from replay_to_action_obs.data import download_replays, replay_to_rlgym_frames

    # download a replay file
    replay_dir = "dataset/replays"
    ids = download_replays(replay_dir=replay_dir, count=1, verbose=True)

    # convert replay file to rlgym frames
    replay_path = os.path.join(replay_dir, f"{ids[0]}.replay")
    frames = replay_to_rlgym_frames(replay_path)

    # for i in range(20):
    #     actions_dict = frames[i].actions
    #     for agent_id, action in actions_dict.items():
    #         print(f"Frame {i}, Agent {agent_id}: {action[0]}", end=" ")
    #     print()
    # quit()

    # convert frames to action and obs
    action, obs = rlgym_frames_to_action_obs(frames)

    # save actions
    np.save("dataset/actions.npy", action)
