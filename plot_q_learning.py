import pickle
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


MARK_EVERY = (0, 1)
MARKER_SIZE = 25.0
LINE_WIDTH = 6.0
FONT_SIZE = 36.0
FONT_SIZE_TICK = 32.0
FIG_SIZE = (16,9)
DPI = 160
GRID = False
FONT_WEIGHT = "bold"


# Load the results
results = None
path = "./results/q_learning/results.pkl"
with open(path, "rb") as f:
    results = pickle.load(f)


# Plot the reward list
Path("./results/q_learning/figures").mkdir(parents=True, exist_ok=True)
reward_list = results["reward_list"]
print("len(reward_list):", len(reward_list))
reward_array = np.asarray(reward_list, dtype=float)
mean_reward_array = np.mean(reward_array.reshape(-1, 500), axis=1)
print(mean_reward_array)
fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
plt.setp(ax.get_xticklabels(), fontsize=FONT_SIZE_TICK, fontweight=FONT_WEIGHT)
plt.setp(ax.get_yticklabels(), fontsize=FONT_SIZE_TICK, fontweight=FONT_WEIGHT)
ax.yaxis.get_offset_text().set_fontsize(20)
ax.yaxis.get_offset_text().set_fontweight(FONT_WEIGHT)
ax.xaxis.get_offset_text().set_fontsize(20)
ax.xaxis.get_offset_text().set_fontweight(FONT_WEIGHT)
# ax.set_yticks([ -0.2, 0.0, 0.2 ])
# plt.locator_params(axis='x', nbins=3)
ax.plot(range(1, len(mean_reward_array) + 1), mean_reward_array, 'o', markevery=MARK_EVERY, linestyle='-', color='#009E73', label="Q-learning", markersize=MARKER_SIZE, linewidth=LINE_WIDTH)
ax.legend(shadow=True, loc=(0.35, 0.12), ncol=2, fontsize=24.0)
#     legend = ax.legend(shadow=True, loc='best', fontsize=24.0)
plt.grid(GRID)
# plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
ax.set_xlabel(r'Episode', fontsize=FONT_SIZE, fontweight=FONT_WEIGHT)
ax.set_ylabel(r'Reward', fontsize=FONT_SIZE, fontweight=FONT_WEIGHT)
# Save the figure
fig.savefig('./results/q_learning/figures/q_learning.pdf', format='pdf', bbox_inches='tight')
fig.savefig('./results/q_learning/figures/q_learning.png', format='png', bbox_inches='tight')
