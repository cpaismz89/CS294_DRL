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
parser.add_argument('--file_name3', '-fl3', type = str, default=None)
parser.add_argument('--file_name4', '-fl4', type = str, default=None)
parser.add_argument('--file_name5', '-fl5', type = str, default=None)
parser.add_argument('--plot_name', '-pl', type = str, default="Test")
parser.add_argument('--Ram', action='store_true')
parser.add_argument('--DQ', action='store_true')
parser.add_argument('--best_reward', action='store_true')
args = parser.parse_args()

# Load files (Vanilla and Double Q-Learning)
f_data1 = pickle.load(open(args.file_name1+'.pkl','rb'))
f_data2 = pickle.load(open(args.file_name2+'.pkl','rb'))
f_data3 = pickle.load(open(args.file_name3+'.pkl','rb'))
f_data4 = pickle.load(open(args.file_name4+'.pkl','rb'))
f_data5 = pickle.load(open(args.file_name5+'.pkl','rb'))

pixel_t = np.array(f_data1["log_t"])[:395]
f1_mean_rewards = f_data1["log_mean_reward"]
f2_mean_rewards = f_data2["log_mean_reward"]
f3_mean_rewards = f_data3["log_mean_reward"]
f4_mean_rewards = f_data4["log_mean_reward"]
f5_mean_rewards = f_data5["log_mean_reward"]

f1_best_rewards = f_data1["log_best_mean"]
f2_best_rewards = f_data2["log_best_mean"]
f3_best_rewards = f_data3["log_best_mean"]
f4_best_rewards = f_data4["log_best_mean"]
f5_best_rewards = f_data5["log_best_mean"]

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
f1_mean_rew, = plt.plot(pixel_t, np.array(f1_mean_rewards)[:395], label='Mean 100-Episode Reward (Vanilla Q)')
f2_mean_rew, = plt.plot(pixel_t, f2_mean_rewards, label='Mean 100-Episode Reward (LR = 0.001)')
f3_mean_rew, = plt.plot(pixel_t, f3_mean_rewards, label='Mean 100-Episode Reward (LR = 0.05)')
f4_mean_rew, = plt.plot(pixel_t, f4_mean_rewards, label='Mean 100-Episode Reward (LR = 0.1)')
f5_mean_rew, = plt.plot(pixel_t, f5_mean_rewards, label='Mean 100-Episode Reward (LR = 100)')

if args.best_reward:
    f1_best_rew, = plt.plot(pixel_t, np.array(f1_best_rewards)[:395], label='Best Mean Reward (Vanilla Q)')
    f2_best_rew, = plt.plot(pixel_t, f2_best_rewards, label='Best Mean Reward (LR = 0.001')
    f3_best_rew, = plt.plot(pixel_t, f3_best_rewards, label='Best Mean Reward (LR = 0.05)')
    f4_best_rew, = plt.plot(pixel_t, f4_best_rewards, label='Best Mean Reward (LR = 0.1)')
    f5_best_rew, = plt.plot(pixel_t, f5_best_rewards, label='Best Mean Reward (LR = 100)')

# Title
plt.suptitle('Learning Rate Performance comparison: ' + args.plot_name + ' from Pixels', fontsize=20) 

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
