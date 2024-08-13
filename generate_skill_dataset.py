from typing import List, Tuple
import numpy as np
from tqdm.rich import tqdm
import os
from concurrent.futures import ProcessPoolExecutor
import argparse
import json

from rlgym_tools.replays.convert import ReplayFrame
from ballchasing.constants import Rank

from replay_to_action_obs.factories import SingleFrameObs, SingleFramePyrObs, InverseLookupAct
from replay_to_action_obs.data import (
    download_replays,
    replay_to_rlgym_frames,
    rlgym_frames_to_action_obs,
)
from replay_to_action_obs.data.util import zip_dataset


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

rank_to_skill = {
    Rank.BRONZE_1: 0.0,
    Rank.SILVER_1: 0.111,
    Rank.GOLD_1: 0.222,
    Rank.PLATINUM_1: 0.333,
    Rank.DIAMOND_1: 0.444,
    Rank.CHAMPION_1: 0.555,
    Rank.GRAND_CHAMPION_1: 0.666,
    Rank.GRAND_CHAMPION_2: 0.777,
    Rank.GRAND_CHAMPION_3: 0.888,
    Rank.SUPERSONIC_LEGEND: 1.0,
}

def get_ranks():
    # min rank, max rank
    return [
        (Rank.BRONZE_1, Rank.BRONZE_3),
        (Rank.SILVER_1, Rank.SILVER_3),
        (Rank.GOLD_1, Rank.GOLD_3),
        (Rank.PLATINUM_1, Rank.PLATINUM_3),
        (Rank.DIAMOND_1, Rank.DIAMOND_3),
        (Rank.CHAMPION_1, Rank.CHAMPION_3),
        (Rank.GRAND_CHAMPION_1, Rank.GRAND_CHAMPION_1),
        (Rank.GRAND_CHAMPION_2, Rank.GRAND_CHAMPION_2),
        (Rank.GRAND_CHAMPION_3, Rank.GRAND_CHAMPION_3),
        (Rank.SUPERSONIC_LEGEND, Rank.SUPERSONIC_LEGEND),
    ]


def process_replay(replay_path, idx, actions_dir, obs_dir, rank):
    try:
        frames = replay_to_rlgym_frames(replay_path)
    except Exception as e:
        print(f"Error processing {replay_path}: {e}")
        print("Skipping...")
        return None

    actions, obs = rlgym_frames_to_action_obs(frames)
    agent_ids = list(frames[0].state.cars.keys())

    filenames = []
    for j in range(len(agent_ids)):
        filename = f"{idx:07}_{j}"
        action_path = os.path.join(actions_dir, f"{filename}.npz")
        obs_path = os.path.join(obs_dir, f"{filename}.npz")
        np.savez_compressed(action_path, array=actions[j])
        np.savez_compressed(obs_path, array=obs[j])
        filenames.append(filename)
    return filenames, rank

def gen_dataset(
    dataset_dir,
    count: int = 2,
    use_downloaded_replays: bool = False,
    verbose: bool = False,
) -> None:
    dataset_dir, replay_dir, actions_dir, obs_dir = make_dirs(dataset_dir)

    replay_paths = []
    ranks = []
    if use_downloaded_replays:
        # in replays dir there are subdirs for each rank
        ranks = [d for d in os.listdir(replay_dir) if os.path.isdir(os.path.join(replay_dir, d))]
        for rank in ranks:
            for f in os.listdir(os.path.join(replay_dir, rank)):
                if f.endswith(".replay"):
                    replay_paths.append(os.path.join(replay_dir, rank, f))
                    ranks.append(rank)
    else:
        for min_rank, max_rank in get_ranks():
            rank_dir = os.path.join(replay_dir, f"{min_rank}")
            ids = download_replays(replay_dir=rank_dir, count=count, min_rank=min_rank, max_rank=max_rank, verbose=verbose)
            replay_paths.extend([os.path.join(rank_dir, f"{replay_id}.replay") for replay_id in ids])
            ranks.extend([rank] * len(ids))
            
    print("Processing replays...")
    filename_to_rank = {}
    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = []
        for i, (replay_path, rank) in enumerate(zip(replay_paths, rank)):
            futures.append(executor.submit(process_replay, replay_path, i, actions_dir, obs_dir, rank))
        with tqdm(total=len(futures), desc="Processing replays") as pbar:
            for future in futures:
                res = future.result()
                if res is not None:
                    filenames, rank = res
                    for filename in filenames:
                        filename_to_rank[filename] = rank
                pbar.update(1)
    
    # json dump filename_to_rank
    with open(os.path.join(dataset_dir, "filename_to_rank.json"), "w") as f:
        json.dump(filename_to_rank, f)

    print("Zipping dataset...")
    zip_dataset(dataset_dir)

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset")
    # choose one or the other
    parser.add_argument(
        "--count", type=int, default=4, help="Number of replays to process"
    )
    parser.add_argument(
        "--use-downloaded-replays",
        action="store_true",
        default=False,
        help="Use already downloaded replays",
    )
    args = parser.parse_args()

    DATASET_NAME = "1v1-skill"
    gen_dataset(
        os.path.join("dataset", DATASET_NAME),
        count=args.count,
        use_downloaded_replays=args.use_downloaded_replays,
        verbose=True,
    )
