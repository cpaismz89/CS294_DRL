FOR %%D IN ("Ant-v2", "HalfCheetah-v2", "Hopper-v2", "Humanoid-v2", "Reacher-v2", "Walker2d-v2") DO (
	python run_DAgger.py experts/%%D.pkl %%D --num_rollouts=20 --max_timesteps=1000 --epochs=1000 --batch_size=64 --dagger_iters=10
)