# Importations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pickle
import argparse
parser = argparse.ArgumentParser()

# Arguments
parser.add_argument('--file_name', type = str, default=None)
parser.add_argument('--plot_name', type = str, default="Test")
parser.add_argument('--Ram', action='store_true')
parser.add_argument('--DQ', action='store_true')
args = parser.parse_args()

# Load files
data = pickle.load(open(args.file_name+'.pkl','rb'))
tsteps = data["log_t"]
meanR = data["log_mean_reward"]
bestR = data["log_best_mean"]

# Figure size
plot= plt.figure(figsize = (15, 9))

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
mean_rew, = plt.plot(tsteps, meanR, label='Mean 100-Episode Reward')
best_rew, = plt.plot(tsteps, bestR, label='Best Mean Reward')

# Titles
if args.Ram:
    if args.DQ == False:
        plt.suptitle('Q-Learning Performance on ' + args.file_name +' from Ram', fontsize=20) 
    else:
        plt.suptitle('Double Q-Learning Performance on ' + args.file_name +' from Ram', fontsize=20) 
else:
    if args.DQ == False:
        plt.suptitle('Q-Learning Performance on ' + args.file_name +' from Pixels', fontsize=20) 
    else:
        plt.suptitle('Double Q-Learning Performance on ' + args.file_name +' from Pixels', fontsize=20) 

# Labels		
plt.xlabel('Timesteps')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.ylabel('Reward')
plt.legend()

# Save PDF
pp = PdfPages(args.plot_name + '.pdf')
pp.savefig(plot)
pp.close()
