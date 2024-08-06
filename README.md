# Replay to Action Obs

Downloads rocket league replays from ballchasing.com, and converts it into actions and observation tensors for training.
NOTE: This will only work on Windows (not even WSL), because RLGym requires Windows. 

# Installation
```
pip install -e .
conda install pytorch -c pytorch
```

Put `BALLCHASING_API_KEY` in a `.env` file at the top level

# Running
```
python generate_dataset.py count=100
```