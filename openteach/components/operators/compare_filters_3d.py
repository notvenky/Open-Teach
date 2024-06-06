import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation, Slerp
from scipy.signal import butter, lfilter
from mpl_toolkits.mplot3d import Axes3D

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
    
class Filter:
    def __init__(self, state, comp_ratio=0.3, cutoff=0.1, fs=0.6, order=2):
        self.pos_state = state[:3]
        self.ori_state = state[3:7]
        self.comp_ratio = comp_ratio
        self.cutoff = cutoff
        self.fs = fs
        self.order = order
        self.b, self.a = butter(self.order, self.cutoff, fs=self.fs, btype='low')
        self.pos_history = [self.pos_state]

    def __call__(self, next_state):
        self.pos_history.append(next_state[:3])
        filtered_pos = lfilter(self.b, self.a, np.array(self.pos_history), axis=0)
        self.pos_state = filtered_pos[-1]
        ori_interp = Slerp([0, 1], Rotation.from_quat(
            np.array([self.ori_state, next_state[3:7]])))
        self.ori_state = ori_interp([1 - self.comp_ratio])[0].as_quat()
        return np.concatenate([self.pos_state, self.ori_state])

# Generate simulated position and orientation data
np.random.seed(0)
n_samples = 100
positions = np.cumsum(np.random.randn(n_samples, 3), axis=0)
orientations = Rotation.random(n_samples).as_quat()
states = np.hstack([positions, orientations])

# Initial state
initial_state = np.concatenate([positions[0], orientations[0]])

# Instantiate filters
old_filter = OldFilter(initial_state)
new_filter = Filter(initial_state)

# Apply filters
old_filtered_states = [old_filter(states[i]) for i in range(n_samples)]
new_filtered_states = [new_filter(states[i]) for i in range(n_samples)]

# Preparing 3D plots for positions and orientations
fig = plt.figure(figsize=(14, 12))

# 3D plot for positions
ax1 = fig.add_subplot(211, projection='3d')
ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], label='Original Position')
ax1.plot([state[0] for state in old_filtered_states], [state[1] for state in old_filtered_states], [state[2] for state in old_filtered_states], label='Old Filter Position')
ax1.plot([state[0] for state in new_filtered_states], [state[1] for state in new_filtered_states], [state[2] for state in new_filtered_states], label='New Filter Position')
ax1.set_title('3D Position Comparison')
ax1.legend()
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

# Convert quaternions to Euler angles for a better visualization
euler_original = Rotation.from_quat(orientations).as_euler('xyz', degrees=True)
euler_old_filter = Rotation.from_quat([state[3:7] for state in old_filtered_states]).as_euler('xyz', degrees=True)
euler_new_filter = Rotation.from_quat([state[3:7] for state in new_filtered_states]).as_euler('xyz', degrees=True)

fig = plt.figure(figsize=(14, 12))

# 3D plot for Euler angles (orientation)
ax = fig.add_subplot(111, projection='3d')
ax.plot(euler_original[:, 0], euler_original[:, 1], euler_original[:, 2], label='Original Euler Angles')
ax.plot(euler_old_filter[:, 0], euler_old_filter[:, 1], euler_old_filter[:, 2], label='Old Filter Euler Angles')
ax.plot(euler_new_filter[:, 0], euler_new_filter[:, 1], euler_new_filter[:, 2], label='New Filter Euler Angles')
ax.set_title('3D Euler Angle Comparison')
ax.legend()
ax.set_xlabel('Roll (Degrees)')
ax.set_ylabel('Pitch (Degrees)')
ax.set_zlabel('Yaw (Degrees)')

plt.tight_layout()
plt.show()