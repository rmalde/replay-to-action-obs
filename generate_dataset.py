from typing import List, Tuple
import numpy as np
from tqdm.rich import tqdm
import os

from rlgym_tools.replays.convert import ReplayFrame

from replay_to_action_obs.factories import (
    SingleFrameObs,
    InverseLookupAct,
    ACTION_SPACE,
    OBS_SPACE,
)
from replay_to_action_obs.data import download_replays, replay_to_rlgym_frames
from replay_to_action_obs.data.util import zip_dataset
import argparse


def make_dirs(dataset_dir: str) -> None:
    def _make_if_not_exists(dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        return dir_path

    dataset_dir = _make_if_not_exists(dataset_dir)
    replay_dir = _make_if_not_exists(os.path.join(dataset_dir, "replays"))
    actions_dir = _make_if_not_exists(os.path.join(dataset_dir, "actions"))
    obs_dir = _make_if_not_exists(os.path.join(dataset_dir, "obs"))
    return dataset_dir, replay_dir, actions_dir, obs_dir


def gen_dataset(dataset_dir, count: int = 2, verbose: bool = False) -> None:
    dataset_dir, replay_dir, actions_dir, obs_dir = make_dirs(dataset_dir)

    ids = download_replays(replay_dir=replay_dir, count=count, verbose=verbose)

    obs_builder = SingleFrameObs()
    action_parser = InverseLookupAct()

    idx_to_replay_id = []
    idx = 0

    print("Processing replays...")

    for replay_id in tqdm(ids):

        replay_path = os.path.join(replay_dir, f"{replay_id}.replay")
        try:
            frames = replay_to_rlgym_frames(replay_path)
        except Exception as e:
            print(f"Error processing {replay_id}: {e}")
            print("Skipping...")
            continue

        agent_ids = list(frames[0].state.cars.keys())
        n_agents = len(agent_ids)
        n_frames = len(frames)
        len_obs = len(obs_builder.build_obs(agent_ids[0], frames[0].state))

        actions = [np.zeros((n_frames, 1)) for _ in agent_ids]
        obs = [np.zeros((n_frames, len_obs)) for _ in agent_ids]

        for frame_idx, frame in enumerate(frames):
            for i, agent_id in enumerate(agent_ids):
                action = frame.actions[agent_id]
                if len(action.shape) == 2:
                    # take the first action bc it's repeated tick_skip times
                    action = action[0]
                obs[i][frame_idx] = obs_builder.build_obs(agent_id, frame.state)
                actions[i][frame_idx] = action_parser.parse_actions(action)

        for i in range(len(agent_ids)):
            np.save(
                os.path.join(actions_dir, f"{idx:05}.npy"),
                actions[i],
            )
            np.save(
                os.path.join(obs_dir, f"{idx:05}.npy"),
                obs[i],
            )
            idx_to_replay_id.append(replay_id)
            idx += 1

    # save idx to replay id mapping as csv
    with open(os.path.join(dataset_dir, "idx_to_replay_id.csv"), "w") as f:
        f.write("idx,replay_id\n")
        for idx, replay_id in enumerate(idx_to_replay_id):
            f.write(f"{idx},{replay_id}\n")

    print("Zipping dataset...")
    zip_dataset(dataset_dir)

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset")
    parser.add_argument("--count", type=int, default=4, help="Number of replays to process")
    args = parser.parse_args()

    DATASET_NAME = "ssl-1v1-100"
    gen_dataset(os.path.join("dataset", DATASET_NAME), count=args.count, verbose=True)
