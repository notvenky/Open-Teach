import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation, Slerp
from scipy.signal import butter, lfilter

# Define the filters as per provided classes
class OldFilter:
    def __init__(self, state, comp_ratio=0.3):
        self.pos_state = state[:3]
        self.ori_state = state[3:7]
        self.comp_ratio = comp_ratio

    def __call__(self, next_state):
        self.pos_state = np.array(self.pos_state) * self.comp_ratio + np.array(next_state[:3]) * (1 - self.comp_ratio)
        ori_interp = Slerp([0, 1], Rotation.from_quat(
            np.array([self.ori_state, next_state[3:7]])))
        self.ori_state = ori_interp([1 - self.comp_ratio])[0].as_quat()
        return np.concatenate([self.pos_state, self.ori_state])
    
class OldVecFilter:
    def __init__(self, state, comp_ratio=0.3):
        self.pos_state = state[:3]
        self.ori_state = state[3:7]
        self.comp_ratio = comp_ratio

    def __call__(self, next_state):
        print(self.pos_state, next_state[:3])
        self.pos_state = np.array(self.pos_state[:3]) * self.comp_ratio + np.array(next_state[:3]) * (1 - self.comp_ratio)
        ori_interp = Slerp([0, 1], Rotation.from_rotvec(
            np.stack([self.ori_state, next_state[3:7]], axis=0)),)
        self.ori_state = ori_interp([1 - self.comp_ratio])[0].as_rotvec()
        return np.concatenate([self.pos_state, self.ori_state])
    
class Filter:
    def __init__(self, state, comp_ratio=0.6, cutoff=0.1, fs=0.6, order=2):
        self.pos_state = state[:3]
        self.ori_state = state[3:7]
        self.comp_ratio = comp_ratio
        self.cutoff = cutoff
        self.fs = fs
        self.order = order
        self.b, self.a = butter(self.order, self.cutoff / (0.5 * fs), btype='low')
        self.pos_history = [self.pos_state]

    def __call__(self, next_state):
        self.pos_history.append(next_state[:3])
        filtered_pos = lfilter(self.b, self.a, np.array(self.pos_history), axis=0)
        self.pos_state = filtered_pos[-1]

        ori_interp = Slerp([0, 1], Rotation.from_rotvec(
            np.stack([self.ori_state, next_state[3:7]], axis=0)))
        self.ori_state = ori_interp([1 - self.comp_ratio])[0].as_rotvec()

        return np.concatenate([self.pos_state, self.ori_state])

np.random.seed(0)
n_samples = 100
positions = np.cumsum(np.random.randn(n_samples, 3), axis=0)
orientations = Rotation.random(n_samples).as_rotvec()  # Generating rotation vectors
states = np.hstack([positions, orientations])

# Initial state
initial_state = np.concatenate([positions[0], orientations[0]])

# Instantiate filters
old_vec_filter = OldVecFilter(initial_state)
new_filter = Filter(initial_state)

# Apply filters
old_filtered_states = [old_vec_filter(states[i]) for i in range(n_samples)]
new_filtered_states = [new_filter(states[i]) for i in range(n_samples)]

# Plotting
fig, axs = plt.subplots(2, 1, figsize=(10, 10))

# Position plot
axs[0].plot(positions[:, 0], label='Original X')
axs[0].plot([state[0] for state in old_filtered_states], label='Old Vec Filter X')
axs[0].plot([state[0] for state in new_filtered_states], label='Butterworth Filter X')
axs[0].set_title('Position Filters')
axs[0].legend()

# Orientation plot (showing one component of the rotation vector for simplicity)
axs[1].plot(orientations[:, 0], label='Original RotVec X')
axs[1].plot([state[3] for state in old_filtered_states], label='Old Vec Filter RotVec X')
axs[1].plot([state[3] for state in new_filtered_states], label='Butterworth Filter RotVec X')
axs[1].set_title('Orientation Filters')
axs[1].legend()

plt.tight_layout()
plt.show()