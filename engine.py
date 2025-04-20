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
from utils.agents import load_blip_model, compute_vision_alignment_score
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
    best_response_id = -1
    max_successes = []
    max_successes_reward_correlation = []
    execute_rates = []
    best_code_paths = []
    max_success_overall = DUMMY_FAILURE
    max_success_reward_correlation_overall = DUMMY_FAILURE
    max_reward_code_path = None 
    scores = []
    vision_text_model, processor = load_blip_model()
    
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
        max_score = float('-inf')

        logging.info(f"Iteration {iter}: Generating {cfg.sample} samples with {cfg.models.coder_agent}")       
        code_runs = [] 
        rl_runs = []

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
                    break  # Exit the retry loop on success
                except Exception as e:
                    if attempt >= 10:
                        # Reduce chunk size if multiple failures occur (simulate backoff)
                        chunk_size = max(int(chunk_size / 2), 1)
                        logging.info(f"Reducing chunk size to {chunk_size}")
                    logging.info(f"Attempt {attempt+1} failed with error: {e}")
                    time.sleep(1)

            while True:
                if response_cur is None:
                    logging.info("Terminating due to too many failed attempts!")
                    exit()

                # If the response includes token usage info, update accumulators.
                # (This assumes your model provides a similar "usage" field as in OpenAI's API.)
                if "usage" in response_cur:
                    prompt_tokens += response_cur["usage"].get("prompt_tokens", 0)
                    total_completion_tokens += response_cur["usage"].get("completion_tokens", 0)
                    total_tokens += response_cur["usage"].get("total_tokens", 0)
                else:
                    logging.info("No usage information provided in the response.")

                response_cur = response_cur.get("message", {})
                response_content = response_cur['content']
                
                logging.info(f"Iteration {iter}: Processing Code Run {total_samples}")

                # Regex patterns to extract python code enclosed in GPT response
                patterns = [
                    r'```python(.*?)```',
                    r'```(.*?)```',
                    r'"""(.*?)"""',
                    r'""(.*?)""',
                    r'"(.*?)"',
                ]
                for pattern in patterns:
                    code_string = re.search(pattern, response_content, re.DOTALL)
                    if code_string is not None:
                        code_string = code_string.group(1).strip()
                        break
                code_string = response_content if not code_string else code_string

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
                    pattern = re.compile(
                        r"return reward,\s*.*\r?\n"   # match the return‑line (anything after the comma)
                        r"([\s\S]*)"                  # capture everything else, including newlines
                    )
                    match = pattern.search(task_code_string)
                    if match:
                        extras = match.group(1)
                    task_code_string_without_rf = task_code_string.split('def compute_reward')[0] # Remove the old reward function
                    code_string_lines = code_string.strip().splitlines()
                    code_string = '\n'.join([indent + line for line in code_string_lines]) # Indent the code to fit in the environment class
                    file.writelines(task_code_string_without_rf + '\n')
                    file.writelines(code_string + '\n')
                    if match:
                        file.writelines(extras + '\n')
                    

                with open(f"env_iter{iter}_response{total_samples}_rewardonly.py", 'w') as file:
                    file.writelines(code_string + '\n')

                # Copy the generated environment code to hydra output directory for bookkeeping
                shutil.copy(output_file, f"env_iter{iter}_response{total_samples}.py")

                # Find the freest GPU to run GPU-accelerated RL
                #set_freest_gpu()
                
                # Execute the python file with flags
                
                rl_filepath = f"env_iter{iter}_train_response{total_samples}.txt"
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
                    f'--model_name={cfg.rl.train_type}_{total_samples}',
                    f'--env_module_path={EUREKA_ROOT_DIR}/env_iter{iter}_response{total_samples}.py',
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
                                
                if process.returncode == 0:
                    break

                logging.info("Error in train.py process. Skipping this response.")
                traceback_msg = file_to_string(rl_filepath).split("Traceback (most recent call last):")[-1]
                execution_error_feedback = execution_error_feedback.format(traceback_msg=traceback_msg)
                new_messages = messages + [{"role": "assistant", "content": response_content}, {"role": "user", "content": execution_error_feedback}]
                for attempt in range(1000):
                    try:
                        # Build the payload.
                        # Note: If Ollama supported multiple completions with "n", you could include it here.
                        payload = {
                            "model": model,
                            "messages": new_messages,
                            "temperature": cfg.models.coder_config.temperature,
                            "stream": False
                        }
                        logging.info(f"Attempt {attempt+1}: Sending request with chunk size {chunk_size}")
                        # Send the POST request (make sure the endpoint URL is correct for your installation)
                        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
                        response.raise_for_status()  # Raise an exception for HTTP errors
                        response_cur = response.json()
                        break  # Exit the retry loop on success
                    except Exception as e:
                        if attempt >= 10:
                            # Reduce chunk size if multiple failures occur (simulate backoff)
                            chunk_size = max(int(chunk_size / 2), 1)
                            logging.info(f"Reducing chunk size to {chunk_size}")
                        logging.info(f"Attempt {attempt+1} failed with error: {e}")
                        time.sleep(1)
                
            total_samples += chunk_size  # Increase the sample count by the number returned (here we assume 1 per call)

            # Ollama returns a JSON with a "message" key. We add this message to our list.
            responses.append(response_cur)

        # Logging the final outputs
        if cfg.sample == 1:
            # If only one sample was requested, log its content.
            logging.info("Ollama Output:\n" + responses[0].get("content", ""))
            
        #logging.info(f"Final Token Usage: Prompt Tokens: {prompt_tokens}, Completion Tokens: {total_completion_tokens}, Total Tokens: {total_tokens}")

        for response_id in range(cfg.sample):
            response_cur = responses[response_id]["content"]
            
            rl_filepath = f"env_iter{iter}_eval_response{response_id}.txt"
            logging.info("Launching eval.py; output will be saved to %s", rl_filepath)
            
            #Build the command line using the flags from eval.py:
            # eval.py expects:
            #   --model_name
            #   --env_module_path
            #   --env_module_name
            #   --env_class_name
            #   --model_path
            #   --trajectory_dir
            #   --testing_episodes
            #   --render
            #   --seed
            command = [
                sys.executable, '-u', f'{EUREKA_ROOT_DIR}/utils/eval.py',
                f'--model_name={cfg.rl.train_type}_{response_id}',
                f'--env_module_path={EUREKA_ROOT_DIR}/env_iter{iter}_response{response_id}.py',
                f'--obs_path={EUREKA_ROOT_DIR}/envs/{env_name}_obs.json',
                f'--env_module_name={env_name}_{suffix.lower()}',
                f'--env_class_name={cfg.task}Env',
                f'--model_path={cfg.rl.save_path}',
                f'--trajectory_dir={cfg.rl.trajectory_dir}',
                f'--testing_episodes={cfg.rl.testing_episodes}',
                f'--render={cfg.rl.render}',
                f'--seed={cfg.rl.seed}'
            ]
            
            # Launch eval.py as a subprocess.
            with open(rl_filepath, 'w') as f:
                process = subprocess.Popen(command, stdout=f, stderr=f, env=os.environ)
                logging.info("eval.py process launched with PID: %s", process.pid)
                process.wait()
                logging.info("eval.py process completed with return code: %s", process.returncode)
            rl_runs.append(process)
            score = 0.0
            # Compute the vision alignment score for the generated RL
            for j in range(cfg.rl.testing_episodes):
                cur_trajectory_dir = f"{cfg.rl.trajectory_dir}/{cfg.rl.train_type}_{response_id}/Ep_{j+1}_Trajectory"
                s, _, _ = compute_vision_alignment_score(vision_text_model, processor, cur_trajectory_dir, env_name, cfg.models.scorer_batch_size)
                score += s
            score /= cfg.rl.testing_episodes
            if score > max_score:
                max_score = score
                best_response_id = response_id
            logging.info(f"Score for response {response_id}: {score}")
        
        logging.info(f"Best response id: {best_response_id} with score {max_score}")
        #Reward reflection
        # We will use the best response to generate a new reward function
        # and then use that reward function to train a new agent.
        # This will be done in the next iteration.
        shutil.copy(f"env_iter{iter}_response{best_response_id}.py", output_file)
        feedback_agent = cfg.models.feedback_agent
        feedback_agent_system = file_to_string(f'{prompt_dir}/feedback_agent_system.txt')
        feedback_prompt = file_to_string(f'{prompt_dir}/feedback_agent_prompt.txt')
        summary = ''
        for j in range(cfg.rl.testing_episodes):
            summary_path = f"{cfg.rl.trajectory_dir}/{cfg.rl.train_type}_{best_response_id}/Ep_{j+1}_Summary/summary.txt"
            summary = f"Episode {j+1} evaluation metrics:\n" + file_to_string(summary_path)
        
        cur_reward_function = file_to_string(filename=f"env_iter{iter}_response{best_response_id}_rewardonly.py")
        feedback_prompt = feedback_prompt.format(env=env_name, task_description=task_description, summary=summary, reward_function=cur_reward_function)
        images_dir = f"{cfg.rl.trajectory_dir}/{cfg.rl.train_type}_{best_response_id}/Ep_1_Summary"
        images = [encode_image(os.path.join(images_dir,image_path)) for image_path in os.listdir(images_dir) if image_path.endswith('.png')]
        feedback_messages = [{"role": "system", "content": feedback_agent_system}, {"role": "user", "content": feedback_prompt, "images": images}]
        for attempt in range(1000):
            try:
                # Build the payload.
                # Note: If Ollama supported multiple completions with "n", you could include it here.
                payload = {
                    "model": feedback_agent,
                    "messages": feedback_messages,
                    "temperature": cfg.models.feedback_config.temperature,
                    "stream": False
                }
                logging.info(f"Attempt {attempt+1}: Sending request with chunk size {chunk_size}")
                # Send the POST request (make sure the endpoint URL is correct for your installation)
                response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
                response.raise_for_status()  # Raise an exception for HTTP errors
                response_cur = response.json()
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
        response_cur = response_cur.get("message", {})
        feedback = response_cur["content"]

        with open(f"env_iter{iter}_response{best_response_id}_feedback.txt", 'w') as file:
            file.writelines(feedback + '\n')
        
        # Feedback to coding LLM
        p_feedback = policy_feedback.format(reward_function=cur_reward_function, feedback=feedback, score=max_score)
        coder_feedback = p_feedback + '\n' + code_feedback
        
        messages = [{"role": "system", "content": initial_system}, {"role": "user", "content": coder_feedback}]
        
        scores.append(max_score)

        if os.path.exists(cfg.rl.trajectory_dir):
            shutil.rmtree(cfg.rl.trajectory_dir)
        if os.path.exists(cfg.rl.save_path):
            shutil.rmtree(cfg.rl.save_path)
    
    logging.info("Eureka process completed.")
    logging.info("Scores: " + str(scores))
    plot_result(scores)

        
        
            

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
        
    