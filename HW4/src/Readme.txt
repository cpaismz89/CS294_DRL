###############################################################################################################################################################
	HW4 CS294-112 
	Author: Cristobal Pais 
	Date: October 2018 

	Note: Question 3a plot is generated via the auxiliary script P3Plotter.py attached 	
###############################################################################################################################################################

Instructions
I) Problem 1
# Running the experiments and generate the plots
python main.py q1
						  
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

II) Problem 2
# Running the experiments and generate the log
python main.py q2

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

III) Problem 3a
# Running the experiments and generate the plot
python main.py q3
python P3Plotter.py
					   
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------					   
					   
IV) Problem 3b
# Running the experiments and generate the plots

python main.py q3 --exp_name horizon10 --mpc_horizon 10
python main.py q3 --exp_name horizon15 --mpc_horizon 15
python main.py q3 --exp_name horizon20 --mpc_horizon 20
python plot.py --exps HalfCheetah_q3_horizon10 HalfCheetah_q3_horizon15 HalfCheetah_q3_horizon20 --save HalfCheetah_q3_mpc_horizon

python main.py q3 --exp_name action128 --num_random_action_selection 128
python main.py q3 --exp_name action4096 --num_random_action_selection 4096
python main.py q3 --exp_name action16384 --num_random_action_selection 16384
python plot.py --exps HalfCheetah_q3_action128 HalfCheetah_q3_action4096 HalfCheetah_q3_action16384 --save HalfCheetah_q3_actions

python main.py q3 --exp_name layers1 --nn_layers 1
python main.py q3 --exp_name layers2 --nn_layers 2
python main.py q3 --exp_name layers3 --nn_layers 3
python plot.py --exps HalfCheetah_q3_layers1 HalfCheetah_q3_layers2 HalfCheetah_q3_layers3 --save HalfCheetah_q3_nn_layers

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------