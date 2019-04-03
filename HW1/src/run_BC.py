#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
Modified by: Cristobal Pais (cpaismz@berkeley.edu)
Date: September 2018
Course: CS294
"""
# General importations
import os
import pickle
import tensorflow as tf
import numpy as np
import pandas as pd
import tf_util
import gym
import load_policy
from tqdm import tqdm

# Model class importations
from DRL_BCModel import *

# Main function 
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--max_timesteps', type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    with tf.Session():
        tf_util.initialize()

        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        for i in tqdm(range(args.num_rollouts)):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = policy_fn(obs[None,:])
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}

        print("Expert Data dimensions:")
        print("Obs:", expert_data["observations"].shape)
        print("Act:", expert_data["actions"].shape)
        
        
        #####################################################################################
        #
        #                                     BC Model
        #
        #####################################################################################
        # Generate the DRL_BCModel class object
        DRLNModel = DRL_BCModel(expert_data["observations"], expert_data["actions"],
                                args.envname, "BC", args.batch_size)
        
        # Train the model
        DRLNModel.train(epochs=args.epochs, TrainData = None, TestData = None, nIter=None)
        
        # Get Returns from BC
        BCreturns = []
        BCactions = []
        BCobservations = []
        
        # RollOut loop
        for i in tqdm(range(args.num_rollouts)):
            obs = env.reset()
            done = False
            totalr = 0
            steps = 0
            while not done:
                action = DRLNModel.Sampling(obs, nNet=args.batch_size)
                BCobservations.append(obs)
                BCactions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            BCreturns.append(totalr)

        print('BC returns', BCreturns)
        print('BC mean return', np.mean(BCreturns))
        print('BC std of return', np.std(BCreturns))
        
        # Save data (name, mean returns, std returns, number of rollouts, number of steps)
        toSave = pd.DataFrame({"Name": args.envname, "AVGReturn": np.mean(BCreturns), "STD": np.std(BCreturns), 
                               "NRoll": args.num_rollouts, "TSteps": max_steps, 
                               "Epochs": args.epochs, "Batch_size":args.batch_size}, index=[0])
        print("Generating .csv file with relevant info for plotting")
        print(toSave)
        
        # Outputs folder
        OutPath = os.getcwd() + "/Outputs/"
        if not os.path.exists(OutPath):
            os.makedirs(OutPath)
        fileName = OutPath + "BCData_"+args.envname+".csv"
        
        if os.path.isfile(fileName):
            toSave.to_csv(fileName, header=False, index=False, mode='a')
        else:
            toSave.to_csv(fileName, header=True, index=False, mode='a')
        
        # Save expert data
        ExpPath = os.getcwd() + "/experts_data/"
        if not os.path.exists(ExpPath):
            os.makedirs(ExpPath)
            
        with open(os.path.join('experts_data', args.envname + '.pkl'), 'wb') as f:
            pickle.dump(expert_data, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()