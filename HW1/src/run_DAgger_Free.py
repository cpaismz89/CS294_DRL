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
from DRL_BCModel_Free import *

# Main function 
def main():
    # Parse main arguments/inputs
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
    parser.add_argument('--dagger_iters', type=int, default=5)
    parser.add_argument('--RELU', action='store_true')
    parser.add_argument('--MLAYER', action='store_true')
    args = parser.parse_args()

    # Load expert policy
    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    # Expert Policy Data recollection
    with tf.Session():
        tf_util.initialize()
        
        # Create environment
        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        # Empty vectors for recording data
        returns = []
        observations = []
        actions = []
        
        # Expert Rollouts
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

        # Print-out information
        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        # Keep Expert Data in dictionary
        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}
        
        # Expert observations and actions
        TrainObs = expert_data["observations"]
        TrainActs = expert_data["actions"]

        # Show expert data dimension
        print("Expert Data dimensions:")
        print("Obs:", expert_data["observations"].shape)
        print("Act:", expert_data["actions"].shape)
        
        #####################################################################################
        #
        #                                     DAgger Model
        #
        #####################################################################################
        # Generate the DRL_BCModel class object (Initial training)
        if args.MLAYER == True:
            print("MultiLayer model (5 layers)")
            NLayers = 5
            NNet = [128,64,64,64,128]   # Initial + Hidden Layers
            Activations = [tf.nn.tanh for i in range(len(NNet))]
        elif args.RELU == True:
            print("RELU activations model")
            NLayers = 2
            NNet = [64,32]
            Activations = [tf.nn.relu for i in range(len(NNet))] 
        
        DRLNModel = DRL_BCModel_Free(TrainObs, TrainActs, args.envname, "DAgger", NNet, NLayers, args.batch_size, Activations)
        
        # Train the model
        print("Training the model (Initial train)", "\nEpochs:", args.epochs, "\nBatch-Size:", args.batch_size, "\nNumber of Hidden layers:", NLayers-1, "\nActivation Rules:", Activations)
        DRLNModel.train(epochs=args.epochs, TrainData = None, TestData = None, nIter=None)
        
        # Get Returns from DA
        DAreturns = []
        DAstds = []
        DAactions = []
        DAobservations = []
        
        # Print-out info
        print("\n------ DAgger loop ------")
        
        # DAgger loop
        for i in tqdm(range(args.dagger_iters)):
            newObs = []
            newAct = []
            totalr = 0
            DARollreturns = []
            
            # DAgger Rollouts
            for j in tqdm(range(args.num_rollouts)):
                print('iter', j)
                obs = env.reset()
                done = False
                totalr = 0.
                steps = 0
                while not done:
                    action = DRLNModel.Sampling(obs, nNet=args.batch_size)
                    DAobservations.append(obs)
                    DAactions.append(action)
                    obs, r, done, _ = env.step(action)
                    totalr += r
                    
                    # Get corrected actions from expert policy
                    correctedAct = policy_fn(obs[None, :])
                    newObs.append(obs)
                    newAct.append(correctedAct)
                    
                    steps += 1
                    if args.render:
                        env.render()
                    if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                    if steps >= max_steps:
                        break
                DARollreturns.append(totalr)
                
                print('DA returns (RollOut)', DARollreturns)
                print('DA mean return', np.mean(DARollreturns))
                print('DA std of return', np.std(DARollreturns))
                    
            DAreturns.append(np.mean(DARollreturns))
            DAstds.append(np.std(DARollreturns))
            
            # Update observations and train model: expert labels for correction
            print("Training DAgger iteration", i)
            TrainObs = np.concatenate((TrainObs, newObs), axis=0)
            TrainActs = np.concatenate((TrainActs, newAct), axis=0)
            DRLNModel.train(epochs=args.epochs, TrainData=TrainObs, TestData=TrainActs, nIter=i)
                
                
        
        
        # Save data (name, mean returns, std returns, number of rollouts, number of steps)
        toSave = pd.DataFrame({"Name": args.envname, "AVGReturn": DAreturns, "STD": DAstds, 
                               "NRoll": np.repeat(args.num_rollouts, args.dagger_iters), "TSteps": np.repeat(max_steps, args.dagger_iters), 
                               "Epochs": np.repeat(args.epochs, args.dagger_iters), "Batch_size":np.repeat(args.batch_size, args.dagger_iters),
                               "DAggerIters": np.arange(1,args.dagger_iters+1)})
        print("Generating .csv file with relevant info for plotting")
        print(toSave)
        
        # Outputs folder
        OutPath = os.getcwd() + "/Outputs/"
        if not os.path.exists(OutPath):
            os.makedirs(OutPath)
        
        if args.RELU == True:
            fileName = OutPath + "DAggerData_RELU"+args.envname+".csv"
        elif args.MLAYER == True:
            fileName = OutPath + "DAggerData_MLAYER"+args.envname+".csv"
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
