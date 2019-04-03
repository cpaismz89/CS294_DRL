# Importations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pickle
import argparse
parser = argparse.ArgumentParser()

# Arguments
parser.add_argument('--file_name1', '-fl1', type = str, default=None)
parser.add_argument('--file_name2', '-fl2', type = str, default=None)
parser.add_argument('--plot_name', '-pl', type = str, default="Test")
parser.add_argument('--Ram', action='store_true')
parser.add_argument('--DQ', action='store_true')
args = parser.parse_args()

# Load files (Vanilla and Double Q-Learning)
vanilla_data = pickle.load(open(args.file_name1+'.pkl','rb'))
double_data = pickle.load(open(args.file_name2+'.pkl','rb'))

pixel_t = vanilla_data["log_t"]
vanilla_mean_rewards = vanilla_data["log_mean_reward"]
vanilla_best_rewards = vanilla_data["log_best_mean"]
double_mean_rewards = double_data["log_mean_reward"]
double_best_rewards = double_data["log_best_mean"]

# Figure Size
plot = plt.figure(figsize = (15, 9))

# Font sizes
plt.rcParams['font.size'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 12
 
# axes
ax = plt.subplot(111)                    
ax.spines["top"].set_visible(False) 
ax.spines["right"].set_visible(False)
ax.get_xaxis().tick_bottom() 
ax.get_yaxis().tick_left() 

# Plot
vanilla_mean_rew, = plt.plot(pixel_t, vanilla_mean_rewards, label='Mean 100-Episode Reward (Vanilla Q)')
vanilla_best_rew, = plt.plot(pixel_t, vanilla_best_rewards, label='Best Mean Reward (Vanilla Q)')
double_mean_rew, = plt.plot(pixel_t, double_mean_rewards, label='Mean 100-Episode Reward (Double Q)')
double_best_rew, = plt.plot(pixel_t, double_best_rewards, label='Best Mean Reward (Double Q)')

# Title
plt.suptitle('Vanilla vs Double Q-Learning Performance comparison: ' + args.plot_name + ' from Pixels', fontsize=20) 

# Labels
plt.xlabel('Timesteps')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.ylabel('Reward')

# Legend
plt.legend()

# Save to PDF
pp = PdfPages(args.plot_name + '.pdf')
pp.savefig(plot)
pp.close()
