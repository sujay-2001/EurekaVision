import numpy as np 
import json
import logging 
import matplotlib.pyplot as plt
import os
import openai
import re
import subprocess
from pathlib import Path
import shutil
import time 
import requests
import yaml
from types import SimpleNamespace
from utils.misc import *
import sys

EUREKA_ROOT_DIR = os.getcwd()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(cfg):
    workspace_dir = Path.cwd()
    logging.info(f"Workspace: {workspace_dir}")
    logging.info(f"Project Root: {EUREKA_ROOT_DIR}")

    openai.api_key = os.getenv("OPENAI_API_KEY")

    task = cfg.task
    env_name = cfg.env
    env_details_path = EUREKA_ROOT_DIR + f"/envs/{env_name}_obs.json"
    env_details = json.load(open(env_details_path, 'r'))
    task_description = env_details["Description"]
    suffix = cfg.models.coder_config.suffix
    model = cfg.models.coder_agent
    logging.info(f"Using LLM: {model}")
    logging.info("Task: " + task)
    logging.info("Task description: " + task_description)

    task_file = f'{EUREKA_ROOT_DIR}/envs/{env_name}.py'
    task_obs_file = f'{EUREKA_ROOT_DIR}/envs/{env_name}_obs.json'
    shutil.copy(task_obs_file, f"env_init_obs.json")
    task_code_string  = file_to_string(task_file)
    task_obs_code_string  = file_to_string(task_obs_file)
    output_file = f"{EUREKA_ROOT_DIR}/envs/{env_name}_{suffix.lower()}.py"

    # Loading all text prompts
    prompt_dir = f'{EUREKA_ROOT_DIR}/utils/prompts'
    initial_system = file_to_string(f'{prompt_dir}/initial_system.txt')
    code_output_tip = file_to_string(f'{prompt_dir}/code_output_tip.txt')
    code_feedback = file_to_string(f'{prompt_dir}/code_feedback.txt')
    initial_user = file_to_string(f'{prompt_dir}/initial_user.txt')
    reward_signature = file_to_string(f'{prompt_dir}/reward_signature.txt')
    policy_feedback = file_to_string(f'{prompt_dir}/policy_feedback.txt')
    execution_error_feedback = file_to_string(f'{prompt_dir}/execution_error_feedback.txt')

    initial_system = initial_system.format(task_reward_signature_string=reward_signature) + code_output_tip
    initial_user = initial_user.format(task_obs_code_string=task_obs_code_string, task_description=task_description)
    messages = [{"role": "system", "content": initial_system}, {"role": "user", "content": initial_user}]

    #task_code_string = task_code_string.replace(task, task+suffix)
    # Create Task YAML files
    #create_task(ISAAC_ROOT_DIR, cfg.env.task, cfg.env.env_name, suffix)

    DUMMY_FAILURE = -10000.
    max_successes = []
    max_successes_reward_correlation = []
    execute_rates = []
    best_code_paths = []
    max_success_overall = DUMMY_FAILURE
    max_success_reward_correlation_overall = DUMMY_FAILURE
    max_reward_code_path = None 
    
    # Eureka generation loop
    url = "http://localhost:11434/api/chat"
    for iter in range(cfg.iteration):
        # Get Eureka response
        responses = []
        response_cur = None
        total_samples = 0
        total_token = 0
        total_completion_token = 0
        chunk_size = 1

        logging.info(f"Iteration {iter}: Generating {cfg.sample} samples with {cfg.models.coder_agent}")

        while True:
            if total_samples >= cfg.sample:
                break
            for attempt in range(1000):
                try:
                    # Build the payload.
                    # Note: If Ollama supported multiple completions with "n", you could include it here.
                    payload = {
                        "model": model,
                        "messages": messages,
                        "temperature": cfg.models.coder_config.temperature,
                        "stream": False
                    }
                    logging.info(f"Attempt {attempt+1}: Sending request with chunk size {chunk_size}")
                    # Send the POST request (make sure the endpoint URL is correct for your installation)
                    response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
                    response.raise_for_status()  # Raise an exception for HTTP errors
                    response_cur = response.json()
                    total_samples += chunk_size  # Increase the sample count by the number returned (here we assume 1 per call)
                    break  # Exit the retry loop on success
                except Exception as e:
                    if attempt >= 10:
                        # Reduce chunk size if multiple failures occur (simulate backoff)
                        chunk_size = max(int(chunk_size / 2), 1)
                        logging.info(f"Reducing chunk size to {chunk_size}")
                    logging.info(f"Attempt {attempt+1} failed with error: {e}")
                    time.sleep(1)
            if response_cur is None:
                logging.info("Terminating due to too many failed attempts!")
                exit()

            # Ollama returns a JSON with a "message" key. We add this message to our list.
            responses.append(response_cur.get("message", {}))

            # If the response includes token usage info, update accumulators.
            # (This assumes your model provides a similar "usage" field as in OpenAI's API.)
            if "usage" in response_cur:
                prompt_tokens += response_cur["usage"].get("prompt_tokens", 0)
                total_completion_tokens += response_cur["usage"].get("completion_tokens", 0)
                total_tokens += response_cur["usage"].get("total_tokens", 0)
            else:
                logging.info("No usage information provided in the response.")

        # Logging the final outputs
        if cfg.sample == 1:
            # If only one sample was requested, log its content.
            logging.info("Ollama Output:\n" + responses[0].get("content", ""))
            
        #logging.info(f"Final Token Usage: Prompt Tokens: {prompt_tokens}, Completion Tokens: {total_completion_tokens}, Total Tokens: {total_tokens}")
        
        code_runs = [] 
        rl_runs = []
        for response_id in range(cfg.sample):
            response_cur = responses[response_id]["content"]
            logging.info(f"Iteration {iter}: Processing Code Run {response_id}")

            # Regex patterns to extract python code enclosed in GPT response
            patterns = [
                r'```python(.*?)```',
                r'```(.*?)```',
                r'"""(.*?)"""',
                r'""(.*?)""',
                r'"(.*?)"',
            ]
            for pattern in patterns:
                code_string = re.search(pattern, response_cur, re.DOTALL)
                if code_string is not None:
                    code_string = code_string.group(1).strip()
                    break
            code_string = response_cur if not code_string else code_string

            # Remove unnecessary imports
            lines = code_string.split("\n")
            for i, line in enumerate(lines):
                if line.strip().startswith("def "):
                    code_string = "\n".join(lines[i:])
            if 'self' not in code_string:
                code_string = code_string.replace('def compute_reward(', 'def compute_reward(self, ')
            code_runs.append(code_string)
            
            indent = " " * 4

            # Save the new environment code when the output contains valid code string!
            with open(output_file, 'w') as file:
                extras = task_code_string.split('return reward')[1]
                task_code_string = task_code_string.split('def compute_reward')[0] # Remove the old reward function
                code_string_lines = code_string.strip().splitlines()
                code_string = '\n'.join([indent + line for line in code_string_lines]) # Indent the code to fit in the environment class
                file.writelines(task_code_string + '\n')
                file.writelines(code_string + '\n')
                file.writelines(extras + '\n')
                

            with open(f"env_iter{iter}_response{response_id}_rewardonly.py", 'w') as file:
                file.writelines(code_string + '\n')

            # Copy the generated environment code to hydra output directory for bookkeeping
            shutil.copy(output_file, f"env_iter{iter}_response{response_id}.py")

            # Find the freest GPU to run GPU-accelerated RL
            #set_freest_gpu()
            
            # Execute the python file with flags
            
            rl_filepath = f"env_iter{iter}_response{response_id}.txt"
            logging.info("Launching train.py; output will be saved to %s", rl_filepath)

            # Build the command line using the flags from train.py:
            # train.py expects:
            #   --model_name
            #   --env_module_name
            #   --env_class_name
            #   --save_model
            #   --save_path
            #   --training_steps
            #   --seed
            command = [
                sys.executable, '-u', f'{EUREKA_ROOT_DIR}/utils/train.py',
                f'--model_name={cfg.rl.train_type}',
                f'--env_module_path={EUREKA_ROOT_DIR}\envs\{env_name}_{suffix.lower()}.py',
                f'--env_module_name={env_name}_{suffix.lower()}',
                f'--env_class_name={cfg.task}Env',
                f'--save_model={cfg.rl.save_model}',
                f'--save_path={cfg.rl.save_path}',
                f'--training_steps={cfg.rl.training_steps}',
                f'--seed={cfg.rl.seed}'
            ]

            # Launch train.py as a subprocess.
            with open(rl_filepath, 'w') as f:
                process = subprocess.Popen(command, stdout=f, stderr=f, env=os.environ)
                logging.info("train.py process launched with PID: %s", process.pid)
                process.wait()
                logging.info("train.py process completed with return code: %s", process.returncode)
            rl_runs.append(process)

def dict_to_namespace(d):
    """Recursively converts a dict into a SimpleNamespace."""
    return SimpleNamespace(**{
        k: dict_to_namespace(v) if isinstance(v, dict) else v 
        for k, v in d.items()
    })
    
if __name__ == "__main__":
    
    with open(f"{EUREKA_ROOT_DIR}/configs/config.yaml", 'r') as f:
        cfg = yaml.safe_load(f)
    cfg = dict_to_namespace(cfg)  # Convert to an object for easier attribute access
    main(cfg)
        
    