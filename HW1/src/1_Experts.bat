FOR %%D IN ("Ant-v2", "HalfCheetah-v2", "Hopper-v2", "Humanoid-v2", "Reacher-v2", "Walker2d-v2") DO (
	FOR %%A IN (1,5,10,15,20,25,50,75,100) DO (
		python run_expert.py experts/%%D.pkl %%D --num_rollouts=%%A --max_timesteps=1000 
	)
)