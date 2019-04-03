# coding: utf-8
# Plotter main script for CS294-112 HW1
# Author: Cristobal Pais
# Date: 9/5/2018

# Importations
import pandas as pd 
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages


# Plot: AVGReturns vs NRolls
def EvolutionPlot(AVGReturns, STDs, NRolls, Instance=None, filePlot=None, EXPAVGReturns=[], EXPSTDs=[] ):
    # Plot with error bars
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

    # Title and labels
    plt.title(Instance+': AVG Return vs N°Rollouts')
    plt.xlabel(r'N°Rollouts')
    plt.ylabel(r'Average Returns')

    # Ticks 
    #plt.xticks(NRolls)
    xScale = [i for i in np.arange(1,90,10)]
    plt.xticks(xScale, NRolls, size='small')
    #plt.yticks()
    
    # Best AVG Return x-axis
    idx = [np.argwhere(AVGReturns == np.max(AVGReturns))[0]][0]
    BestAVGReturnX = xScale[int(idx)]
    BestAVGReturnY = np.max(AVGReturns)
    
    # Annotation factor
    annFactor = 50.5
    if Instance == "Reacher-v2":
        annFactor = 2
    if Instance == "HalfCheetah-v2":
        annFactor = 15.5
        
    # Plot
    ax.scatter(xScale, AVGReturns, alpha = 0.5, label = 'BC AVG Returns', color = "red")
    plt.plot(xScale, AVGReturns, alpha = 0.5, color = "red")
    ax.scatter(BestAVGReturnX, BestAVGReturnY, alpha=1.0, label = 'Best BC AVG Return', 
               color = "red", marker='*', s=150)
    ax.annotate("[" + str(NRolls[idx][0]) + "," + str(np.round(BestAVGReturnY, 3)) + "]", 
                xy = (BestAVGReturnX + 0.25, BestAVGReturnY + annFactor))
    
    # Extra: Error Bar 
    x = xScale
    y = AVGReturns
    yerr = STDs
    ax.errorbar(x, y, yerr = yerr, ecolor = 'red', lw = 2, 
                capsize = 5, fmt='none', capthick = 2, alpha = 0.5)

    # If Expert average results are provided
    if len(EXPAVGReturns) > 0:
        # Best AVG Return x-axis
        idx = [np.argwhere(EXPAVGReturns == np.max(EXPAVGReturns))[0]][0]
        BestEXPAVGReturnX = xScale[int(idx)]
        BestEXPAVGReturnY = np.max(EXPAVGReturns)
        
        
        ax.scatter(xScale, EXPAVGReturns, alpha = 0.5, color="blue", label = 'EXP AVG Returns')
        plt.plot(xScale, EXPAVGReturns, alpha = 0.5, color="blue")
        ax.scatter(BestEXPAVGReturnX, BestEXPAVGReturnY, alpha=0.5, label = 'Best EXP AVG Return', 
                   color = "blue", marker = '*', s=150)
        ax.annotate("[" + str(NRolls[idx][0]) + "," + str(np.round(BestEXPAVGReturnY,3)) + "]", 
                    xy = (BestEXPAVGReturnX + 0.25, BestEXPAVGReturnY + annFactor))
    
        # Extra: Error Bar 
        x = xScale
        y = EXPAVGReturns
        yerr = EXPSTDs
        ax.errorbar(x, y, yerr = yerr, ecolor = 'blue', lw = 2, 
                    capsize = 5, fmt='none', capthick = 2, alpha = 0.5)


    
    
    # Add legend
    plt.legend()

    # Save plot as PDF
    Path = os.getcwd() + "/Plots/"
    if filePlot == None:
        plt.savefig(Path + "Plot1.pdf", bbox_inches='tight')
    else:
        plt.savefig(Path + filePlot, bbox_inches='tight')

    # Return plot as an object
    return plt

# Plot: AVGReturns vs HyperParameter
def EvolutionPlotHyper(AVGReturns, STDs, HyperParameter, Instance="", HyperName="", BatchSize=64, NRolls=20):
    # Plot with error bars
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

    # Ticks 
    xScale = [i for i in np.arange(1,len(HyperParameter)*100,100)]
    plt.xticks(xScale, HyperParameter)
    
    # Title and labels
    plt.title(Instance+': AVG Return vs '+HyperName + " (NRolls="+str(NRolls)+", BatchSize="+str(BatchSize)+")")
    plt.xlabel(HyperName)
    plt.ylabel(r'Average Returns')

    # Best AVG Return x-axis
    # Best AVG Return x-axis
    idx = [np.argwhere(AVGReturns == np.max(AVGReturns))[0]][0]
    BestAVGReturnX = xScale[int(idx)]
    BestAVGReturnY = np.max(AVGReturns)
       
    # Plot
    ax.scatter(xScale, AVGReturns, alpha = 0.5, label = 'AVG Returns', color="red")
    ax.scatter(BestAVGReturnX, BestAVGReturnY, alpha=0.5, label = 'Best AVG Return', color = "red", marker="*", s=150)
    plt.plot(xScale, AVGReturns, alpha = 0.5, color="red")
    ax.annotate("[" + str(HyperParameter[idx][0]) + "," + str(np.round(BestAVGReturnY,1)) + "]", 
                xy = (BestAVGReturnX - 1, BestAVGReturnY + 50.08))

    # Extra: Error Bar 
    x = xScale
    y = AVGReturns
    yerr = STDs
    ax.errorbar(x, y, yerr = yerr, ecolor = 'red', lw = 2, 
                capsize = 5, fmt='none', capthick = 2, alpha = 0.2)

    # Add legend
    plt.legend()

    # Save plot as PDF
    Path = os.getcwd() + "/Plots/"
    plt.savefig(Path + Instance+"_H"+HyperName+"_NRoll"+str(NRolls)+"_BS"+str(BatchSize)+".pdf", bbox_inches='tight')

    # Return plot as an object
    return plt

# Plot: DAgger Evolution
def DAggerEvolution(DAVGReturns, EXPAVGReturns, BCAVGReturns, DASTDs, EXPSTD, BCSTD, 
                    DAggerIters, Instance="", Epochs=1000, NRolls=20):
    # Plot with error bars
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

    # Title and labels
    plt.title(Instance+': AVG Return vs DAgger Iterations (Epochs='+str(Epochs)+', NRolls='+str(NRolls)+')')
    plt.xlabel('DAgger Iteration')
    plt.ylabel(r'Average Returns')

    # Ticks 
    plt.xticks(DAggerIters)
    
    # Plots
    # DAgger Evolution
    ax.scatter(DAggerIters, DAVGReturns, alpha = 0.5, color="green", label='DAgger AVG Returns')
    plt.plot(DAggerIters, DAVGReturns, alpha = 0.5, color="green")
    
    # Error Bar 
    x = [i for i in DAggerIters]
    y = DAVGReturns
    yerr = DASTDs
    ax.errorbar(x, y, yerr = yerr, ecolor = 'green', lw = 2, capsize = 5, fmt='none', capthick = 2, alpha = 0.5)

    
    # Expert Policy
    ax.scatter(DAggerIters, np.full(fill_value=EXPAVGReturns, shape = len(DAggerIters)), 
               alpha = 0.5, color="blue", label = 'Expert AVG Return') 
    
    # Error Bar 
    y = np.full(fill_value=EXPAVGReturns, shape = len(DAggerIters))
    yerr = np.full(fill_value=EXPSTD, shape=len(DAggerIters))
    ax.errorbar(x, y, yerr = yerr, ecolor = 'blue', lw = 2, capsize = 5, fmt='none', capthick = 2, alpha = 0.5)

    plt.plot(DAggerIters, [EXPAVGReturns for i in range(len(DAggerIters))], 
               alpha = 0.5, color="blue") 
    
    # BC Policy
    ax.scatter(DAggerIters, [BCAVGReturns for i in range(len(DAggerIters))], 
               alpha = 0.5, color="red", label = 'BC AVG Return') 
    plt.plot(DAggerIters, [BCAVGReturns for i in range(len(DAggerIters))], 
               alpha = 0.5, color="red") 
     # Error Bar 
    y = np.full(fill_value=BCAVGReturns, shape = len(DAggerIters))
    yerr = np.full(fill_value=BCSTD, shape=len(DAggerIters))
    ax.errorbar(x, y, yerr = yerr, ecolor = 'red', lw = 2, capsize = 5, fmt='none', capthick = 2, alpha = 0.5)

    
    # Add legend
    plt.legend()

    # Save plot as PDF
    Path = os.getcwd() + "/Plots/"
    plt.savefig(Path + "PlotDAgger_"+Instance+"_Epoch"+str(Epochs)+"_NRoll"+str(NRolls)+".pdf", bbox_inches='tight')

    # Return plot as an object
    return plt

# Main code for generating plots
def GeneratePlot(Instance="Hopper-v2", Path=None):
    fileName = Path + "EXPData_"+Instance+".csv"
    EXPDF = pd.read_csv(fileName)
    EXPDF = EXPDF.sort_values(["NRoll", "TSteps"], ascending=["True", "True"])
    print(EXPDF)

    fileName = Path + "BCData_"+Instance+".csv"
    BCDF = pd.read_csv(fileName)
    BCDF = BCDF[(BCDF["Epochs"] == 1000) & (BCDF["Batch_size"] == 64)]
    BCDF = BCDF.sort_values(["NRoll", "TSteps"], ascending=["True", "True"])
    print(BCDF)

    # Test Evolution PLot
    EXPAVGReturns = EXPDF["AVGReturn"].values
    EXPSTDs = EXPDF["STD"].values
    NRolls = EXPDF["NRoll"].values

    BCAVGReturns = BCDF["AVGReturn"].values
    BCSTDs = BCDF["STD"].values
    NRollsBC = BCDF["NRoll"].values

    filename = "BC_"+Instance+".pdf"
    
    EvolutionPlot(BCAVGReturns, BCSTDs, NRolls, Instance, filename, EXPAVGReturns, EXPSTDs)

# Main code for generating plots
def GenerateDFs(Path=None, Batch_size=64, Epochs=1000):
    Instances = ["Hopper-v2", "Ant-v2", "Reacher-v2", "Walker2d-v2", "HalfCheetah-v2", "Humanoid-v2"]
    EXPDFs = {}
    BCDFs = {}
    
    for i in Instances:
        fileName = "EXPData_"+i+".csv"
        if Path != None:
            fileName = Path + fileName
        EXPDF = pd.read_csv(fileName)
        EXPDF = EXPDF.sort_values(["NRoll", "TSteps"], ascending=["True", "True"])
        EXPDFs[i] = EXPDF

        fileName = "BCData_"+i+".csv"
        if Path != None:
            fileName = Path + fileName
        BCDF = pd.read_csv(fileName)
        BCDF = BCDF[(BCDF["Epochs"] == Epochs) & (BCDF["Batch_size"] == Batch_size)]
        BCDF = BCDF.sort_values(["NRoll", "TSteps"], ascending=["True", "True"])
        BCDFs[i] = BCDF

    return EXPDFs, BCDFs

# DAgger comparison function (with plot)
def DAggerComparison(Instance, DataPath=None, NRolls=20, Epochs=1000, BatchSize=64, Relu=False, MLAYER=False):
    # Expert
    fileName = DataPath + "EXPData_"+Instance+".csv"
    DFEXP = pd.read_csv(fileName)
    print(DFEXP[DFEXP["NRoll"] == NRolls])
    DFEXP = DFEXP[DFEXP["NRoll"] == NRolls]
    DFEXP.reset_index(inplace=True, drop="index")
    EXPReturn, EXPSTD = DFEXP.loc[0, ["AVGReturn", "STD"]].values
    print("EXP values:", EXPReturn, EXPSTD)

    # BC and DAgger
    if Relu == True:
        fileName = DataPath + "DAggerData_RELU"+Instance+".csv"
    elif MLAYER == True:
        fileName = DataPath + "DAggerData_MLAYER"+Instance+".csv"
    else:
        fileName = DataPath + "DAggerData_"+Instance+".csv"
    DFDA_BC = pd.read_csv(fileName)
    DFDA_BC = DFDA_BC[(DFDA_BC["NRoll"] == NRolls) & (DFDA_BC["Epochs"]==Epochs) & (DFDA_BC["Batch_size"] == BatchSize)]
    DFDA_BC = DFDA_BC.sort_values(["DAggerIters", "Epochs"], ascending=["True", "True"])
    DFDA_BC.reset_index(inplace=True, drop="index")
    print(DFDA_BC)
    BCReturn, BCSTD = DFDA_BC.loc[0,["AVGReturn", "STD"]].values
    
    print("BC (Initial iter DAgger):", BCReturn, BCSTD)
    
    # Instance RELU
    if Relu == True:
        Instance = "RELU"+Instance
    elif MLAYER == True:
        Instance = "MLAYER"+Instance
    
    # Plot
    DAggerEvolution(DFDA_BC["AVGReturn"].values, 
                EXPReturn, BCReturn, 
                DFDA_BC["STD"].values, EXPSTD, BCSTD, 
                DFDA_BC["DAggerIters"].values, Instance=Instance,
                Epochs=Epochs, NRolls=NRolls)

# Data folder 
DataPath = os.getcwd() + "/Outputs/"
				
# Plots folder
PlotPath = os.getcwd() + "/Plots/"
if not os.path.exists(PlotPath):
    os.makedirs(PlotPath)

# Print-out Expert vs BC plots
for instance in ["Ant-v2", "Hopper-v2", "Reacher-v2", "Walker2d-v2", "HalfCheetah-v2", "Humanoid-v2"]:
    GeneratePlot(instance,DataPath)

# Table Appendix: Create directory if needed
TablePath = os.getcwd() + "/Tables/"
if not os.path.exists(TablePath):
    os.makedirs(TablePath)
    
# Read Results
EXPDFs, BCDFs = GenerateDFs(DataPath)

# Generate LaTeX table (Appendix)
df = pd.DataFrame()
Rolls = ['1', '5', '10', '15', '20', '25', '50', '75', '100']

idx = 0
for i in Rolls:
    for metric in ["AVGReturn","STD"]:
        for agent in ["Expert","BC"]:
            if  agent == "Expert":
                Aux = EXPDFs
            else:
                Aux = BCDFs
            
            df = df.append({"NRolls":i,"Metric" : metric,
                            "Agent":agent, 
                            "Ant":np.round(Aux["Ant-v2"][metric].values[idx],1),
                            "HalfCheetah":np.round(Aux["HalfCheetah-v2"][metric].values[idx],1),
                            "Hopper": np.round(Aux["Hopper-v2"][metric].values[idx],1),
                            "Humanoid": np.round(Aux["Humanoid-v2"][metric].values[idx],1),
                            "Reacher":np.round(Aux["Reacher-v2"][metric].values[idx],1),
                            "Walker2d":np.round(Aux["Walker2d-v2"][metric].values[idx],1)}, ignore_index=True)
    idx = idx + 1
# Set indexes and generate txt with latex table
df.set_index(['NRolls', 'Metric', 'Agent', 'Ant', 'HalfCheetah', 
              'Hopper', 'Humanoid', 'Reacher', 'Walker2d'], inplace=True,)
df.to_latex(TablePath + "TableAppendix.txt", multirow = True, escape=False, float_format='${:,.2f}'.format )


# In[405]:


# Table Q22
# Read Results
EXPDFs, BCDFs = GenerateDFs(DataPath)

# Generate LaTeX table (Q2.2)
df = pd.DataFrame()
Rolls = ['1', '5', '10', '15', '20', '25', '50', '75', '100']

idx = 0
for i in Rolls:
    for metric in ["AVGReturn","STD"]:
        for agent in ["Expert","BC"]:
            if  agent == "Expert":
                Aux = EXPDFs
            else:
                Aux = BCDFs
            
            df = df.append({"NRolls":i,"Metric" : metric,
                            "Agent":agent, 
                            "HalfCheetah":np.round(Aux["HalfCheetah-v2"][metric].values[idx],1),
                            "Humanoid": np.round(Aux["Humanoid-v2"][metric].values[idx],1)}, ignore_index=True)
    idx = idx + 1
# Set indexes and generate txt with latex table
df.set_index(['NRolls', 'Metric', 'Agent', 'HalfCheetah', 'Humanoid'], inplace=True,)
df.to_latex(TablePath + "TableQ22.txt", multirow = True, escape=False, float_format='${:,.2f}'.format )


# In[417]:


# HyperParameter Plot (Epochs, BS=64, NR=20)
Instance = "Humanoid-v2"
fileName = "BCData_"+Instance+".csv"
DFHumanoid = pd.read_csv(DataPath + fileName)
DFHumanoid = DFHumanoid.sort_values(["NRoll", "TSteps", "Epochs", "Batch_size"], ascending=["True", "True", "True", "True"])
DFHumanoid = DFHumanoid[(DFHumanoid["Batch_size"] == 64) & (DFHumanoid["NRoll"] == 20)]
#display(DFHumanoid)
EvolutionPlotHyper(DFHumanoid["AVGReturn"].values, DFHumanoid["STD"].values,
                   DFHumanoid["Epochs"].values, Instance=Instance, HyperName="Epochs",
                   BatchSize=64, NRolls=20)


# In[420]:


# HyperParameter Plot (Epochs, BS=32, NR=20)
Instance = "Humanoid-v2"
fileName = "BCData_"+Instance+".csv"
DFHumanoid = pd.read_csv(DataPath + fileName)
DFHumanoid = DFHumanoid.sort_values(["NRoll", "TSteps", "Epochs", "Batch_size"], ascending=["True", "True", "True", "True"])
DFHumanoid = DFHumanoid[(DFHumanoid["Batch_size"] == 32) & (DFHumanoid["NRoll"] == 20)]
#display(DFHumanoid)
EvolutionPlotHyper(DFHumanoid["AVGReturn"].values, DFHumanoid["STD"].values,
                   DFHumanoid["Epochs"].values, Instance=Instance, HyperName="Epochs",
                   BatchSize=32, NRolls=20)


# Instance
for instance in ["Ant-v2", "Hopper-v2", "Reacher-v2", "Walker2d-v2", "HalfCheetah-v2", "Humanoid-v2"]:
    DAggerComparison(instance, DataPath=DataPath, NRolls=20, BatchSize=64, Epochs=1000)

# RELU Instance
Instance = "Humanoid-v2"
DAggerComparison(Instance, DataPath=DataPath, Epochs=1000, BatchSize=64, NRolls=20, Relu=True)

# MLAYER Instance
Instance = "Humanoid-v2"
DAggerComparison(Instance, DataPath=DataPath, Epochs=1000, BatchSize=64, NRolls=20, MLAYER=True)