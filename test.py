import gymnasium as gym
import jsbsim_gym.jsbsim_gym # This line makes sure the environment is registered
import imageio as iio
import numpy as np
from os import path
from jsbsim_gym.features import JSBSimFeatureExtractor
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env

policy_kwargs = dict(
    features_extractor_class=JSBSimFeatureExtractor
)

env = gym.make("JSBSim-v0",render_modes='rgb_array')
#check_env(env)


model = SAC.load("models/jsbsim_sac", env)

mp4_writer = iio.get_writer("video.mp4", format="ffmpeg", fps=30)
gif_writer = iio.get_writer("video.gif", format="gif", fps=5)
obs,info = env.reset()
done = False
step = 0
while not done:
    render_data = env.render()
    mp4_writer.append_data(render_data)
    if step % 6 == 0:
        gif_writer.append_data(render_data[::2,::2,:])

    action, _ = model.predict(obs, deterministic=True)
    obs, _, done, _, _ = env.step(action)
    step += 1
mp4_writer.close()
gif_writer.close()
env.close()