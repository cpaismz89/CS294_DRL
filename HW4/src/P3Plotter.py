# CS 294-112 HW4
# Author: Cristobal Pais
# Date:    October 2018

# Importations
import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
%matplotlib inline

# Plotter function
def Plotter(x, y, Path, namePlot, Title="MBRL with on-policy data collection: ReturnAVG vs Iteration"):
    # Figure size
    plt.figure(figsize = (15, 9)) 

    # Font sizes
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['figure.titlesize'] = 14

    # axes
    ax = plt.subplot(111)                    
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left() 
    
    # Title
    plt.title(Title)
    
    # Plot
    plt.plot(x, y, '-o', alpha = 0.5, label = "ReturnAVG")
    plt.xlabel(r'Iteration (n)')
    plt.ylabel(r'Return AVG')
    plt.legend()
    
    # Save
    plt.savefig(Path + namePlot + ".pdf", bbox_inches='tight')

# Main Program 
# Path for data folder (Hard-coded)
Path = "C:/Users/Lenovo/Desktop/UCBerkeley/5to Semestre/"
Path += "CS294/HW4/data/HalfCheetah_q3_21-10-2018_00-36-59/"

# Read Data as csv (Pandas)
P3DF = pd.read_csv(Path + "log.csv", sep=",")
#P3DF

# Plot
namePlot = "P3a"
x = P3DF["Itr"].values
y = P3DF["ReturnAvg"].values

# Generate the graph
Plotter(x[1:], y[1:], Path, namePlot, Title="MBRL with on-policy data collection: ReturnAVG vs Iteration")