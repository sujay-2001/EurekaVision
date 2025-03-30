"""Training an RL Agent using stable_baselines3 
"""

from absl import app
from absl import flags
import gymnasium as gym
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.utils import set_random_seed
import importlib
import os
from tqdm import tqdm
import logging

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

flags.DEFINE_bool(
    "save_model",
    True, 
    "Set to save the model after training. No effect if no training is done."
)

flags.DEFINE_string(
    "save_path", 
    "./models", 
    "Path to save trained weights of the model."
)

flags.DEFINE_integer(
    "training_steps", 
    int(1e5), 
    "Number of training steps."
)

flags.DEFINE_integer(
    "seed", 
    0, 
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
    env_check: bool = True, 
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

def _train_env(
    env: gym.Env, 
    agent: str = "DQN", 
    save_model: bool = True, 
    save_path: str = "./models", 
    total_timesteps: int = 1e4
):
    # Define the agent
    if agent == "DQN":
        agent = DQN("MlpPolicy", env, verbose=1)
    if agent == "PPO":
        agent = PPO("MlpPolicy", env, verbose=1)
    if agent == "A2C":
        agent = A2C("MlpPolicy", env, verbose=1)
    
    # Train the agent
    agent.learn(total_timesteps=total_timesteps)

    # Save the model (if required)
    if save_model:
        agent.save(save_path)
    
    del agent
    env.close()


def main(_):
    
    # Set the random seed for reproducibility
    set_random_seed(FLAGS.seed)
    
    # Import the environment module dynamically.
    module_name = FLAGS.env_module_name  # Name of the module (without .py extension)
    module_path = FLAGS.env_module_path # Path to the module file (e.g., "path/to/module.py")
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
    env_instance = Env()
    logging.info("Environment initialized: %s", env_instance)

    # Optionally wrap the environment (assuming wrap_env is defined elsewhere)
    env_instance = wrap_env(env_instance, render_wrap=False)
    logging.info("Environment wrapped: %s", env_instance)

    # Start training the environment.
    logging.info("Starting training with agent %s", FLAGS.model_name)
    _train_env(
        env=env_instance, 
        agent=FLAGS.model_name, 
        save_model=FLAGS.save_model, 
        save_path=os.path.join(FLAGS.save_path, FLAGS.model_name), 
        total_timesteps=FLAGS.training_steps
    )
    logging.info("Training completed.")


if __name__ == "__main__":
    app.run(main)
    

