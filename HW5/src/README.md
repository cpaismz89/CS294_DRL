###############################################################################################################################################################
	HW5c CS294-112 
	Author: Cristobal Pais 
	Date: November 2018 

	Note: All plots (pdf files) are generated in a folder "Plots" created inside the current working directory, see the attached SamplePlots and Data for examples.
###############################################################################################################################################################

Instructions
1) Problem 1: run the following commands
# Contextual tasks
python train_policy.py 'pm-obs' --exp_name P1 --history 1 -lr 5e-5 
                       -n 200 --num_tasks 4
					   
# Generate graph
python plot.py data/P1 --plotName P1_NN

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

2) Problem 2: NN  vs RNN
## NN
python train_policy.py 'pm' --exp_name P2_NN_H1 --history 1 
                            --discount 0.90 -lr 5e-4 -n 60
python train_policy.py 'pm' --exp_name P2_NN_H5 --history 5 
                            --discount 0.90 -lr 5e-4 -n 60
python train_policy.py 'pm' --exp_name P2_NN_H10 --history 10 
                            --discount 0.90 -lr 5e-4 -n 60
python train_policy.py 'pm' --exp_name P2_NN_H15 --history 15 
                            --discount 0.90 -lr 5e-4 -n 60
python train_policy.py 'pm' --exp_name P2_NN_H30 --history 30 
                            --discount 0.90 -lr 5e-4 -n 60
python train_policy.py 'pm' --exp_name P2_NN_H45 --history 45 
                            --discount 0.90 -lr 5e-4 -n 45
python train_policy.py 'pm' --exp_name P2_NN_H60 --history 60 
                            --discount 0.90 -lr 5e-4 -n 60
python train_policy.py 'pm' --exp_name P2_NN_H100 --history 100 
                            --discount 0.90 -lr 5e-4 -n 60
python train_policy.py 'pm' --exp_name P2_NN_H200 --history 200 
                            --discount 0.90 -lr 5e-4 -n 60

## RNN
python train_policy.py 'pm' --exp_name P2_RNN_H1 --history 1 
                            --discount 0.90 -lr 5e-4 -n 60
                            --recurrent
python train_policy.py 'pm' --exp_name P2_RNN_H5 --history 5 
                            --discount 0.90 -lr 5e-4 -n 60
                            --recurrent
python train_policy.py 'pm' --exp_name P2_RNN_H10 --history 10 
                            --discount 0.90 -lr 5e-4 -n 60
                            --recurrent
python train_policy.py 'pm' --exp_name P2_RNN_H15 --history 15 
                            --discount 0.90 -lr 5e-4 -n 60
                            --recurrent
python train_policy.py 'pm' --exp_name P2_RNN_H30 --history 30 
                            --discount 0.90 -lr 5e-4 -n 60
                            --recurrent
python train_policy.py 'pm' --exp_name P2_RNN_H45 --history 45 
                            --discount 0.90 -lr 5e-4 -n 60
                            --recurrent
python train_policy.py 'pm' --exp_name P2_RNN_H60 --history 60 
                            --discount 0.90 -lr 5e-4 -n 60
                            --recurrent
python train_policy.py 'pm' --exp_name P2_RNN_H100 --history 100 
                            --discount 0.90 -lr 5e-4 -n 60
                            --recurrent
python train_policy.py 'pm' --exp_name P2_RNN_H200 --history 200 
                            --discount 0.90 -lr 5e-4 -n 60
                            --recurrent
							
## Plot (example)
python plot.py data/Folder_1_NN ... data/Folder_n_NN --plotName P2_NN
python plot.py data/Folder_1_RNN ... data/Folder_n_RNN --plotName P2_RNN

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

3) Problem 3: Generalization 
# Note: Special train_policy_P3.py file is included in order to separate the questions, including the extra argument/flag
# sqs for the granularity (square size) of the checkerboard
python train_policy_P3.py 'pm' --exp_name P3_H1_SQ1 --history 1 
                               --discount 0.90 -lr 5e-4 -n 60 --sqs 1
                               --recurrent
python train_policy_P3.py 'pm' --exp_name P3_H1_SQ2 --history 1 
                               --discount 0.90 -lr 5e-4 -n 60 --sqs 2
                               --recurrent
python train_policy_P3.py 'pm' --exp_name P3_H1_SQ5 --history 1 
                               --discount 0.90 -lr 5e-4 -n 60 --sqs 5
                               --recurrent          
python train_policy_P3.py 'pm' --exp_name P3_H1_SQ10 --history 1 
                               --discount 0.90 -lr 5e-4 -n 60 --sqs 10
                               --recurrent          							   
python train_policy_P3.py 'pm' --exp_name P3_H60_SQ1 --history 60 
                            --discount 0.90 -lr 5e-4 -n 60 --sqs 1
                            --recurrent
python train_policy_P3.py 'pm' --exp_name P3_H60_SQ2 --history 60 
                            --discount 0.90 -lr 5e-4 -n 60 --sqs 2
                            --recurrent
python train_policy_P3.py 'pm' --exp_name P3_H60_SQ5 --history 60 
                            --discount 0.90 -lr 5e-4 -n 60 --sqs 5
                            --recurrent          
python train_policy_P3.py 'pm' --exp_name P3_H60_SQ10 --history 60 
                            --discount 0.90 -lr 5e-4 -n 60 --sqs 10
                            --recurrent          

## Plot (Example)							
python plot_P3.py data/P3_H<h>_SQ<sq> --plotName P3_RNN<h>_SQ<sq> 
                  --label RNN_H<h>_SQ<sq>


