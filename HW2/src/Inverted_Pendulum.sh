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
