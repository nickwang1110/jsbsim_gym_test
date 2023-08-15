import gymnasium as gym
import jsbsim_gym.jsbsim_gym # This line makes sure the environment is registered
from os import path
from jsbsim_gym.features import JSBSimFeatureExtractor
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env

policy_kwargs = dict(
    features_extractor_class=JSBSimFeatureExtractor
)

env = gym.make("JSBSim-v0",render_mode='human')

#check_env(env)

log_path = path.join(path.abspath(path.dirname(__file__)), 'logs')

try:
    model = SAC('MlpPolicy', env, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log=log_path, gradient_steps=-1, device='cuda')
    model.learn(1000)
finally:
    model.save("models/jsbsim_sac")
    model.save_replay_buffer("models/jsbsim_sac_buffer")