from typing import List
from rlgym_tools.replays.parsed_replay import ParsedReplay
from rlgym_tools.replays.convert import replay_to_rlgym, ReplayFrame


def replay_to_rlgym_frames(replay_file_path: str) -> List[ReplayFrame]:
    parsed_replay = ParsedReplay.load(replay_file_path)
    return list(replay_to_rlgym(parsed_replay))


if __name__ == "__main__":
    # download replay file
    import os
    from replay_to_action_obs.data import download_replays


    replay_dir = "dataset/replays"
    ids = download_replays(replay_dir=replay_dir, count=1, verbose=True)

    # convert replay file to rlgym
    replay_path = os.path.join(replay_dir, f"{ids[0]}.replay")
    frames = replay_to_rlgym_frames(replay_path)
    print(frames[0])
