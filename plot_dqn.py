import argparse
import pickle
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


COLOR = "#1e8bc3"
MARK_EVERY = (0, 1)
MARKER_SIZE = 12.0
LINE_WIDTH = 4.0
FONT_SIZE = 36.0
FONT_SIZE_TICK = 32.0
FIG_SIZE = (16,9)
DPI = 160
GRID = False
FONT_WEIGHT = "normal"
SMOOTH_INTERVAL = 100


# Load the results
results = None
path = "./results/dqn/results.pkl"
with open(path, "rb") as f:
    results = pickle.load(f)


episode_reward_array = results["episode_reward_array"]
print("episode_reward_array.shape:", episode_reward_array.shape)


# Plot the reward list
Path("./results/dqn/figures").mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
plt.setp(ax.get_xticklabels(), fontsize=FONT_SIZE_TICK, fontweight=FONT_WEIGHT)
plt.setp(ax.get_yticklabels(), fontsize=FONT_SIZE_TICK, fontweight=FONT_WEIGHT)
ax.yaxis.get_offset_text().set_fontsize(20)
ax.yaxis.get_offset_text().set_fontweight(FONT_WEIGHT)
ax.xaxis.get_offset_text().set_fontsize(20)
ax.xaxis.get_offset_text().set_fontweight(FONT_WEIGHT)
# ax.set_yticks([ -0.2, 0.0, 0.2 ])
# plt.locator_params(axis='x', nbins=3)
ax.plot(range(1, len(mean_episode_reward_array) + 1), mean_episode_reward_array, 'o', linestyle='-', color=COLOR, alpha=0.6, label="Episode reward", markersize=MARKER_SIZE, linewidth=LINE_WIDTH)
# ax.legend(shadow=True, loc=(0.35, 0.12), ncol=1, fontsize=24.0)
ax.legend(shadow=True, loc='lower right', fontsize=24.0)
plt.grid(GRID)
# plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
ax.set_xlabel(r'Episode', fontsize=FONT_SIZE, fontweight=FONT_WEIGHT)
ax.set_ylabel(r'Reward', fontsize=FONT_SIZE, fontweight=FONT_WEIGHT)
# Save the figure
fig.savefig('./results/dqn/figures/dqn.pdf', format='pdf', bbox_inches='tight')
fig.savefig('./results/dqn/figures/dqn.png', format='png', bbox_inches='tight')
