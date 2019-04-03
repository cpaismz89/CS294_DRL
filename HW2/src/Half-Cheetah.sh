#!/bin/bash
LR=0.01
BS=10000

for i in {1..3};
	do
		for j in {1..2};
			do
				echo "---------------------- BS: "${BS}"  Learning-Rate: "${LR}" -------------------------"
				echo ""
				python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b ${BS} -lr ${LR} -rtg --nn_baseline --exp_name hc_b${BS}_r${LR}_rtg_bl_tuning
				echo ""
				LR="$(echo ${LR} + 0.01 | bc -l)"
			done
		BS="$(echo ${BS} + 20000| bc -l)"	
		LR=0.01
	done
	

LR=0.005
BS=10000

for i in {1..3};
	do
		echo "---------------------- BS: "${BS}"  Learning-Rate: "${LR}" -------------------------"
		echo ""
		python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b ${BS} -lr ${LR} -rtg --nn_baseline --exp_name hc_b${BS}_r${LR}_rtg_bl_tuning
		echo ""
		BS="$(echo ${BS} + 20000| bc -l)"	
	done

