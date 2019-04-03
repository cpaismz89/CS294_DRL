###############################################################################################################################################################
	HW3 CS294-112 
	Author: Cristobal Pais 
	Date: October 2018 

	Note: All plots for Part2(pdf files) are generated in a folder "Plots" created inside the current working directory, see the attached SamplePlots and Data for examples.
	Note2: Our MultiPlot.py wrapper requires two arguments (1) -re = regular expression for filtering folders, (2) -o name of the output graph (.pdf)
###############################################################################################################################################################

Instructions
I) Problem 1: Atari games
1) Q1 Q-Learning: run the following commands
# Lunar Lander
# Model training and figures are generated using the following commands (double_q = True/False inside dqn.py, line 35):
python run_dqn_lander.py

# Generate the plots      
python Plotter.py --fileName=Q-LunarLander_HuberLoss  
                  --plotName=Q-LunarLander_HuberLoss
python Plotter.py --fileName=Q-LunarLander_MSE  
                  --plotName=Q-LunarLander_MSE
python Plotter.py --fileName=DoubleQ-LunarLanding-HuberLoss  
                  --plotName=DoubleQ-LunarLanding-HuberLoss
python Plotter.py --fileName=DoubleQ-LunarLanding-MSE  
                  --plotName=DoubleQ-LunarLanding-MSE

# Pong Ram
# Model training and figures are generated using the following commands (double_q = True/False inside dqn.py, line 35):
python run_dqn_ram.py 

# Generate the plots     
python Plotter.py --fileName=Q-PongRAM_HuberLoss  
                  --plotName=Q-PongRAM_HuberLoss
python Plotter.py --fileName=Q-PongRAM_MSE  
                  --plotName=Q-PongRAM_MSE
python Plotter.py --fileName=DoubleQ-PongRAM-HuberLoss  
                  --plotName=DoubleQ-PongRAM-HuberLoss
python Plotter.py --fileName=DoubleQ-PongRAM-MSE  
                  --plotName=DoubleQ-PongRAM-MSE
				  
# Pong Pixels
# Model training and figures are generated using the following commands 
# (changing the loss function in lines 205,206 inside the dqn.py file):
python run_dqn_atari.py 

# Generate the plots 
python Plotter.py --fileName=Q-PongPixel_HuberLoss  
                  --plotName=Q-PongPixel_HuberLoss
python Plotter.py --fileName=Q-PongPixel_MSE  
                  --plotName=Q-PongPixel_MSE

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

2) Q2 Double Q-Learning Pong from Pixels: 
# Model training and figures are generated using the following commands (modifying double_q = True/False inside dqn.py, lines 35 or 188): 
python run_dqn_atari.py    
 
# Generate the plots 
python Plotter.py --fileName=DoubleQ-PongPixel_HuberLoss  
                  --plotName=DoubleQ-PongPixel_HuberLoss
				  
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

3) Q3 Hyper-Parameters setting: 
# All experiments were performed by modifying the learning rate scheduling proposed inside the run_dqn_atary.py 
# file (lines 38-43) and simply running the script.
python run_dqn_atari.py 

# Generate the plots 
# Plots of this section are generated using the following commands (not adding the best_reward flag if needed)
python PlotterVS_Hyper.py --file_name1=Q-Pong-HuberLoss
                          --file_name2=Q-Pong-LR000001
                          --file_name3=Q-Pong-LR0001 
                          --file_name4=Q-Pong-LR0005
                          --file_name5=Q-Pong-LR005
                          --file_name6=Q-Pong-LR01 
                          --file_name7=Q-Pong-LR100 
                          --plot_name=Pong_HyperVS_Best 
                          --best_reward
						  
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

II) Problem 2: Actor-Critic
1) Q1: CartPole
# We run the experiments:
python  train_ac_f18.py CartPole-v0 -n 100 -b 1000 -e 3 --exp_name CartPole1_1 -ntu 1 -ngsptu 1
python train_ac_f18.py CartPole-v0 -n 100 -b 1000 -e 3 --exp_name CartPole100_1 -ntu 100 -ngsptu 1
python train_ac_f18.py CartPole-v0 -n 100 -b 1000 -e 3 --exp_name CartPole1_100 -ntu 1 -ngsptu 100
python train_ac_f18.py CartPole-v0 -n 100 -b 1000 -e 3 --exp_name CartPole10_10 -ntu 10 -ngsptu 10

# Generate the plots 
python MultiPlot.py -re="CartPole" -o=Cartpole_All

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

2) Q2: InvertedPendulum and HalfCheetah
# We run the experiments (modifying the initialization of logstd to ones or zeros in line 144 of train_ac_f18.py file)
python train_ac_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.95 
                       -n 100 -e 3 -l 2 -s 64 -b 5000 -lr 0.01 
                       --exp_name ip_10_10 -ntu 10 -ngsptu 10

python train_ac_f18.py HalfCheetah-v2 -ep 150 --discount 0.90 
                       -n 100 -e 3 -l 2 -s 32 -b 30000 -lr 0.02 
                       --exp_name hc_10_10 -ntu 10 -ngsptu 10

python MultiPlot.py -re="ip_10_10_ones" -o=IP_10_10
python MultiPlot.py -re="ip_10_10__zeros" -o=IPzeros_10_10
python MultiPlot.py -re="hc_10_10_ones" -o=HC_10_10
python MultiPlot.py -re="hc_10_10_zeros" -o=HCzeros_10_10
					   
					   
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------					   
					   
III) Bonus: modifying the structure of the Critic network
# Running the experiments 
python train_ac_f18_Bonus.py InvertedPendulum-v2 -ep 1000 
                             --discount 0.95 
                             -n 100 -e 3 -l 2 -s 64 -b 5000 -lr 0.01 
                             --exp_name ip_20_20_NN -ntu 20 -ngsptu 20

python train_ac_f18_Bonus.py HalfCheetah-v2 -ep 150 --discount 0.90 
                             -n 100 -e 3 -l 2 -s 32 -b 30000 -lr 0.02 
                             --exp_name hc_20_20_NN -ntu 20 -ngsptu 20

# Generate plot 
python MultiPlot.py -re="hc_20_20_NN" -o=HC_20_20_bonus
python MultiPlot.py -re="ip_20_20_NN" -o=IP_20_20_bonus							 
