###############################################################################################################################################################
	HW2 CS294-112 
	Author: Cristobal Pais 
	Date: September 2018 

	Note: All plots (pdf files) are generated in a folder "Plots" created inside the current working directory, see the attached SamplePlots and Data for examples.
	Note2: Our MultiPlot.py wrapper requires two arguments (1) -re = regular expression for filtering folders, (2) -o name of the output graph (.pdf)
###############################################################################################################################################################

Instructions
1) Problem 4: run the following commands (or execute CartPole.sh)
# SB experiments
python train_pg_f18.py CartPole-v0 -n 100 -b 1000 -e 3 -dna --exp_name sb_no_rtg_dna
python train_pg_f18.py CartPole-v0 -n 100 -b 1000 -e 3 -rtg -dna --exp_name sb_rtg_dna
python train_pg_f18.py CartPole-v0 -n 100 -b 1000 -e 3 -rtg --exp_name sb_rtg_na
python MultiPlot.py -re "^sb.*\_CartPole" -o CartPoleSB 

# LB experiments
python train_pg_f18.py CartPole-v0 -n 100 -b 5000 -e 3 -dna --exp_name lb_no_rtg_dna
python train_pg_f18.py CartPole-v0 -n 100 -b 5000 -e 3 -rtg -dna --exp_name lb_rtg_dna
python train_pg_f18.py CartPole-v0 -n 100 -b 5000 -e 3 -rtg --exp_name lb_rtg_na
python MultiPlot.py -re "^lb.*\_CartPole" -o CartPoleLB 

# Generate full graph
python MultiPlot.py -re "CartPole" -o CartPoleAll 

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

2) Problem 5: run the InvertedPendulum.sh bash in order to generate all the data folders we used for searching the optimal b* and r* values.
#!/bin/bash
LR=0.001
BS=8

for i in {1..128};
	do
		for j in {1..100};
			do
				echo "---------------------- BS: "${BS}"  Learning-Rate: "${LR}" -------------------------"
				echo ""
				python train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 3 -l 2 -s 64 -b ${BS} -lr ${LR} -rtg --exp_name ip_b${BS}_r${LR} >> Log_InvertedPendulum_script.txt
				echo ""
				LR="$(echo ${LR} + 0.001 | bc -l)"
			done
		BS="$(echo ${BS} + 8| bc -l)"	
		LR=0.001
	done

Then, following the visual analysis, append ipOPT to those instances that satisfy the maximum score condition and are not dominated (see report). Finally, use the following command for 
generating the plots included in the report:

# All instances (relevant subset)
python MultiPlot.py -re "ipOPT" -o ipAll 

# Example 
python MultiPlot.py -re "ip_b8_r.012" -o BestInvertedPendulum 

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

3) Problem 7: run the following commands for performing training our agent and generate the corresponding plot (LunarLanding.sh):
# Train the agent
python train_pg_f18.py LunarLanderContinuous-v2 -ep 1000 --discount 0.99 -n 100 -e 3 -l 2 -s 64 -b 40000 -lr 0.005 -rtg --nn_baseline --exp_name ll_b40000_r0.005

# Generate the plot 
python MultiPlot.py -re "ll" -o LunarLanding 

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

4) Problem 8: First, we run all the combinations of b,r suggested (Half-Cheetah.sh):
python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b 10000 -lr 0.005 -rtg --nn_baseline --exp_name hc_b50000_r0005_tuning
python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b 30000 -lr 0.005 -rtg --nn_baseline --exp_name hc_b30000_r0005_tuning
python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b 50000 -lr 0.005 -rtg --nn_baseline --exp_name hc_b50000_r0005_tuning
python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b 10000 -lr 0.01 -rtg --nn_baseline --exp_name hc_b10000_r001_tuning
python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b 30000 -lr 0.01 -rtg --nn_baseline --exp_name hc_b30000_r001_tuning
python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b 50000 -lr 0.01 -rtg --nn_baseline --exp_name hc_b50000_r001_tuning
python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b 10000 -lr 0.02 -rtg --nn_baseline --exp_name hc_b10000_r002_tuning
python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b 30000 -lr 0.02 -rtg --nn_baseline --exp_name hc_b30000_r002_tuning
python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b 50000 -lr 0.02 -rtg --nn_baseline --exp_name hc_b50000_r002_tuning

Then, we run all te remaining experiments with the optimal (b*,r*) = (50000, 0.02) combination (Half-Cheetah-Best.sh):
python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b 50000 -lr 0.02 --exp_name hc_b50000_r002_vanilla_best
python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b 50000 -lr 0.02 -rtg --exp_name hc_b50000_r002_rtg_best
python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b 50000 -lr 0.02 --nn_baseline --exp_name hc_b50000_r002_bl_best
python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b 50000 -lr 0.02 -rtg --nn_baseline --exp_name hc_b50000_r002_rtg_bl_best

Finally, we generate the two relevant plots:
python MultiPlot.py -re "hc.*tuning" -o HCTuning 
python MultiPlot.py -re "hc.*best" -o HCBest 

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

4) Bonus: GAE-lambda implementation results presented in the report are obtained with the following commands (alternatively, we can execute Walker2d.sh file included):
python train_pg_f18.py Walker2d-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b 50000 -lr 0.02 -rtg --nn_baseline -gl 0 --exp_name w2d_b50000_r002_rtg_bl_gae0
python train_pg_f18.py Walker2d-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b 50000 -lr 0.02 -rtg --nn_baseline -gl 0.7 --exp_name w2d_b50000_r002_rtg_bl_ngae07
python train_pg_f18.py Walker2d-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b 50000 -lr 0.02 -rtg --nn_baseline -gl 0.9 --exp_name w2d_b50000_r002_rtg_bl_ngae09
python train_pg_f18.py Walker2d-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b 50000 -lr 0.02 -rtg --nn_baseline -gl 0.95 --exp_name w2d_b50000_r002_rtg_bl_ngae095
python train_pg_f18.py Walker2d-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b 50000 -lr 0.02 -rtg --nn_baseline -gl 0.96 --exp_name w2d_b50000_r002_rtg_bl_ngae096
python train_pg_f18.py Walker2d-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b 50000 -lr 0.02 -rtg --nn_baseline -gl 0.97 --exp_name w2d_b50000_r002_rtg_bl_ngae097
python train_pg_f18.py Walker2d-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b 50000 -lr 0.02 -rtg --nn_baseline -gl 0.98 --exp_name w2d_b50000_r002_rtg_bl_ngae098
python train_pg_f18.py Walker2d-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b 50000 -lr 0.02 -rtg --nn_baseline -gl 0.99 --exp_name w2d_b50000_r002_rtg_bl_ngae099
python train_pg_f18.py Walker2d-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b 50000 -lr 0.02 -rtg --nn_baseline -gl 1.0 --exp_name w2d_b50000_r002_rtg_bl_ngae1
python train_pg_f18.py Walker2d-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b 50000 -lr 0.02 -rtg --nn_baseline --exp_name w2d_b50000_r002_rtg_bl_ngae
python train_pg_f18.py Walker2d-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b 50000 -lr 0.02 -rtg --nn_baseline -gl 0.5 --exp_name w2d_b50000_r002_rtg_bl_gae05



