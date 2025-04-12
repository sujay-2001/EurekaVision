"""Evaluating an RL Agent using stable_baselines3 
"""

from absl import app
from absl import flags
import os
import json
import numpy as np
import gymnasium as gym
import pygame
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.utils import set_random_seed
import matplotlib.pyplot as plt
import importlib
from PIL import Image
import logging
from collections import defaultdict

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "model_name", 
    "PPO", 
    "Model Type of the RL agent. Default is DQN"
)

flags.DEFINE_string(
    "env_module_name", 
    "acrobot_mistral", 
    "Environment module name. Default is cartpole_gpt"
)

flags.DEFINE_string(
    "env_module_path", 
    "./../envs/acrobot_mistral.py", 
    "Path to the environment module. Default is ../envs/cartpole_gpt.py"
)

flags.DEFINE_string(
    "env_class_name", 
    "AcrobotEnv", 
    "Environment class name. Default is CartPoleEnv"
)

flags.DEFINE_string(
    "model_path", 
    None, 
    "Path of folder containing trained model weights.", 
    required=False
)

flags.DEFINE_string(
    "trajectory_dir", 
    None, 
    "Directory to save trajectories",
    required=False
)

flags.DEFINE_integer(
    "testing_episodes", 
    1, 
    "Number of testing episodes."
)

flags.DEFINE_bool(
    "render", 
    False, 
    "Set to render the testing episodes."
)

flags.DEFINE_integer(
    "seed", 
    42, 
    "Global RNG seed."
)

# Set up basic logging configuration.
logging.basicConfig(
    level=logging.INFO,  # Adjust the level as needed (DEBUG, INFO, WARNING, etc.)
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def wrap_env(
    env: gym.Env, 
    render_wrap: bool = True, 
    order_enforce: bool = False, 
    env_check: bool = False, 
    time_limit: bool = True, 
    max_episode_steps: int = 500
):
    """Wrap the given environment in requested gym wrappers"""
    if time_limit:
        env = gym.wrappers.TimeLimit(
            env,
            max_episode_steps=max_episode_steps
        )
    if env_check:
        env = gym.wrappers.PassiveEnvChecker(env)
    if order_enforce:
        env = gym.wrappers.OrderEnforcing(env, render_wrap)
    if render_wrap:
        env = gym.wrappers.HumanRendering(env)
    return env

def plot_transitions_and_save_prompt(obs_path, episode_rewards, reward_components, states, episode_dir):
    """
    Creates and saves a plot of state transitions and generates a prompt .txt file.

    Args:
        obs_path: Path to the JSON file containing the observation space.
        episode_rewards: A list of rewards obtained in the episode.
        states: A list of observed states during the episode.
        episode_dir: Directory to save the plot and prompt file.

    Returns:
        None
    """
    plt.figure()
    plt.plot(episode_rewards)
    plt.xlabel("Timestep")
    plt.ylabel("Reward")
    plt.title("Episode Rewards")
    plt.grid(True)
    plt.savefig(os.path.join(episode_dir, "reward_plot.png"))
    plt.close()
    
    #Reward components plots
    '''
    for k, v in reward_components.items():
        plt.figure()
        plt.plot(v)
        plt.xlabel("Timestep")
        plt.ylabel(k)
        plt.title(f"{k} Transition")
        plt.grid(True)
        plt.savefig(os.path.join(episode_dir, f"{k}_plot.png"))
        plt.close()
    '''
    
    states = np.array(states)
    
    # Load the observation space from the JSON file
    with open(obs_path, "r") as f:
        obs_data = json.load(f)
    for state_data in obs_data['Observation Space']:
        idx = state_data['num']
        state = state_data['observation']
        title = state + " transition"
        plt.figure()
        plt.plot(states[:, idx])
        plt.xlabel("Timestep")
        plt.ylabel(state)
        plt.title(title)
        plt.grid(True)
        plt.savefig(os.path.join(episode_dir, f"{title}_plot.png"))
        plt.close()

    # Create text file with episode summary
    with open(os.path.join(episode_dir, "summary.txt"), "w") as f:
        ep_len = len(episode_rewards)
        ep_reward_sum = sum(episode_rewards)
        ep_reward_max = max(episode_rewards)
        ep_reward_min = min(episode_rewards)
        ep_reward_avg = np.mean(episode_rewards)

        f.write(f"Episode Length: {ep_len}\n")
        f.write(f"Total Reward: {ep_reward_sum}\n")
        f.write(f"Max Reward: {ep_reward_max}\n")
        f.write(f"Min Reward: {ep_reward_min}\n")
        f.write(f"Average Reward: {ep_reward_avg}\n")
        
        #Reward components summary
        f.write("Reward Components:\n")
        for k, v in reward_components.items():
            reward_component_sum = sum(v)
            reward_component_max = max(v)
            reward_component_min = min(v)
            reward_component_avg = np.mean(v)
            f.write(f"Total {k}: {reward_component_sum}\n")
            f.write(f"Max {k}: {reward_component_max}\n")   
            f.write(f"Min {k}: {reward_component_min}\n")
            f.write(f"Average {k}: {reward_component_avg}\n")
            
        f.write("States:\n")  # Add states if desired (consider potential file size)        
        for state_data in obs_data['Observation Space']:
            idx = state_data['num']
            state = state_data['observation']
            state_max = max(states[:, idx])
            state_min = min(states[:, idx])
            state_avg = np.mean(states[:, idx])
            state_var = np.var(states[:, idx])
            
            f.write(f"Max {state}: {state_max}\n")
            f.write(f"Min {state}: {state_min}\n")
            f.write(f"Mean {state}: {state_avg}\n")
            f.write(f"Variance of {state}: {state_var}\n")
    pass

def _test_env(
    env: gym.Env,
    agent: str = "DQN", 
    model_path: str = "./tmp/test",
    trajectory_dir: str = "/tmp/test-trajectory/",
    obs_path: str = "./../envs/acrobot_obs.json",
    render: bool = True,
    n_episodes: int = 1
):
    # Define the agent
    if agent == "DQN":
        agent = DQN.load(model_path)
    elif agent == "PPO":
        agent = PPO.load(model_path)
    elif agent == "A2C":
        agent = A2C.load(model_path)
    else:
        raise ValueError(f"Unsupported agent: {agent}")

    # Store the metrics
    ep_rewards = np.zeros(n_episodes)
    if not os.path.exists(trajectory_dir):
        os.makedirs(trajectory_dir)

    # Begin Testing
    quitted = False
    for i in range(n_episodes):
        if quitted:
            break
        obs, _ = env.reset()        
        rewards, states, actions = [], [], []
        reward_components_metrics = defaultdict(list)
        terminated = False
        truncated = False
        base_name = os.path.basename(model_path)      
        model_name, _ = os.path.splitext(base_name) 
        eval_dir = os.path.join(trajectory_dir, model_name)
        curr_ep_summary_dir = os.path.join(eval_dir, f"Ep_{i+1}_Summary")
        curr_trajectory_dir = os.path.join(eval_dir, f"Ep_{i+1}_Trajectory")
        os.makedirs(curr_trajectory_dir, exist_ok=True)
        os.makedirs(curr_ep_summary_dir, exist_ok=True)
        
        ind = 0
        if render:
            img_arr = env.render()
            img_pil = Image.fromarray(img_arr, 'RGB')
            img_pil.save(f"{curr_trajectory_dir}/{ind}.png")
            ind += 1
        while not (terminated or truncated):
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            reward_components = env.env.reward_components
            for k in reward_components:
                reward_components_metrics[k].append(reward_components[k])
            rewards.append(reward)
            states.append(obs)
            actions.append(action)
            if render:
                img_arr = env.render()
                img_pil = Image.fromarray(img_arr, 'RGB')
                img_pil.save(f"{curr_trajectory_dir}/{ind}.png")

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.display.quit()
                        pygame.quit()
                        return ep_rewards
        
            ep_rewards[i] += reward
            ind += 1
        plot_transitions_and_save_prompt(obs_path, rewards, reward_components_metrics, states, curr_ep_summary_dir)

    env.close()
    return ep_rewards

    

def main(_):
    # Set the random seed for reproducibility
    set_random_seed(FLAGS.seed)
    
    # Import the environment module dynamically.
    module_name = FLAGS.env_module_name  # Name of the module (without .py extension)
    module_path = FLAGS.env_module_path # Path to the module file (e.g., "path/to/module.py")
    obs_path = os.path.join(os.path.dirname(module_path), f"{module_name.split('_')[0]}_obs.json")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    logging.info("Loading module from path: %s", module_path)
    logging.info("Importing module: %s", module_name)
    custom_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(custom_module)  # <-- Execute the module!

    # Get the environment class from the module.
    class_name = FLAGS.env_class_name  # Name of the class you want to use
    logging.info("Loading class: %s", class_name)
    Env = getattr(custom_module, class_name)  # Assumes that the class is defined in the module

    # Initialize the environment.
    env_instance = Env(render_mode='rgb_array')
    logging.info("Environment initialized: %s", env_instance)

    # Optionally wrap the environment (assuming wrap_env is defined elsewhere)
    env_instance = wrap_env(env_instance, render_wrap=False)
    logging.info("Environment wrapped: %s", env_instance)
    
    #Start evaluating the environment.
    logging.info("Starting evaluation with agent %s", FLAGS.model_name)
    agent = FLAGS.model_name.split("_")[0]
    ep_rewards = _test_env(
        env=env_instance, 
        agent=agent, 
        model_path=os.path.join(FLAGS.model_path, FLAGS.model_name), 
        trajectory_dir=FLAGS.trajectory_dir,
        obs_path=obs_path,
        render=FLAGS.render,
        n_episodes=FLAGS.testing_episodes
    )
    logging.info("Evaluation completed.") 
    logging.info("Episode rewards: %s", ep_rewards)

if __name__ == "__main__":
    app.run(main)