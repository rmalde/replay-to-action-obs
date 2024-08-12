# Replay to Action Obs

Downloads rocket league replays from ballchasing.com, and converts it into actions and observation tensors for training.
NOTE: This will only work on Windows (not even WSL), because RLGym requires Windows. 

# Installation
```
pip install -e .
conda install pytorch -c pytorch
```

Put `BALLCHASING_API_KEY` in a `.env` file at the top level

Make one change to the rlgym library
In `rlgym/rocket_league/api/physics_object.py` 
Change the `euler_angles` function to:
```python
    @property
    def euler_angles(self) -> np.ndarray:
        if self._euler_angles is None:
            if self._quaternion is not None:
                self._euler_angles = quat_to_euler(self._quaternion)
            elif self._rotation_mtx is not None:
                quat = rotation_to_quaternion(self._rotation_mtx)
                self._euler_angles = quat_to_euler(quat)
            else:
                raise ValueError
        return self._euler_angles
```

# Running
```
python generate_dataset.py count=100
```