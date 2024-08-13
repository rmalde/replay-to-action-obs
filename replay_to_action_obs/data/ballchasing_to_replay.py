# for the ballchasing api: https://github.com/Rolv-Arild/python-ballchasing/blob/master/ballchasing/api.py
from dotenv import load_dotenv
import os
from typing import List, Iterator
import pprint
import warnings

from tqdm.rich import tqdm
from tqdm import TqdmExperimentalWarning

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

import ballchasing
from ballchasing.constants import Playlist, Rank, Season, MatchResult


load_dotenv()
BALLCHASING_API_KEY = os.getenv("BALLCHASING_API_KEY")
api = None
if BALLCHASING_API_KEY is not None:
    api = ballchasing.Api(BALLCHASING_API_KEY)


def get_replay_dicts(count: int = 2, min_rank=None, max_rank=None) -> Iterator[dict]:
    if min_rank is None:
        min_rank = Rank.SUPERSONIC_LEGEND
    if max_rank is None:
        max_rank = Rank.SUPERSONIC_LEGEND
    return api.get_replays(
        # player_name="retals",
        playlist=Playlist.RANKED_DUELS,
        season=Season.SEASON_13_FTP,
        min_rank=min_rank,
        max_rank=max_rank,
        count=count,
    )


def download_replays(
    replay_dir: str, count: int = 2, min_rank=None, max_rank=None, verbose: bool = False
) -> List[str]:
    print(f"Downloading replays between {min_rank} and {max_rank} to {replay_dir}")

    if not os.path.exists(replay_dir):
        os.makedirs(replay_dir)

    ids = []
    with tqdm(total=count) as pbar:
        for replay in get_replay_dicts(
            count=count, min_rank=min_rank, max_rank=max_rank
        ):
            if verbose:
                orange_players = ", ".join(
                    [player["name"] for player in replay["orange"]["players"]]
                )
                blue_players = ", ".join(
                    [player["name"] for player in replay["blue"]["players"]]
                )
                print(f"{orange_players} vs. {blue_players}")

            api.download_replay(replay["id"], replay_dir)
            ids.append(replay["id"])

            pbar.update(1)
    return ids


if __name__ == "__main__":
    ids = download_replays(replay_dir="dataset/replays", count=10, verbose=True)
    print("Downloaded replay ids: ", ids)
