FOR %%A IN (10,50,100,500,2000,3000,5000,10000) DO (
	python run_BC.py experts/Humanoid-v2.pkl Humanoid-v2 --num_rollouts=20 --max_timesteps=1000 --batch_size=64 --epochs=%%A
)

FOR %%A IN (10,50,100,500,1000,2000,3000,5000,10000) DO (
	python run_BC.py experts/Humanoid-v2.pkl Humanoid-v2 --num_rollouts=20 --max_timesteps=1000 --batch_size=32 --epochs=%%A
)