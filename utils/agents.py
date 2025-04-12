import torch
import os
from PIL import Image
import requests
import math
from transformers import AutoProcessor, Blip2ForImageTextRetrieval

SCORE_CONSTANT = 1e3
# Define the environment descriptions for different environments
# Each environment has a goal and a baseline description
env_description= {
"cartpole": {
    "goal": "pole vertically upright on top of the cart",
    "baseline": "pole and cart"
},
"acrobot": {
    "goal": "double pendulum with free end touching horizontal line",
    "baseline": "double pendulum"
},
"pendulum": {
    "goal": "balanced inverted pendulum",
    "baseline": "pendulum"
},
"mountaincar": {
    "goal": "car at the top of the mountain",
    "baseline": "car on the mountain"
}
}



def get_frames(frames_dir):    
    """
    Get a list of image frames from the specified directory.
    """
    frames = [Image.open(os.path.join(frames_dir, frame)).convert("RGB") for frame in os.listdir(frames_dir) if frame.endswith(('.png', '.jpg', '.jpeg'))]

    return frames

def load_blip_model():
    """
    Load the BLIP model and processor for image-text retrieval.
    """
    # Set the device to GPU if available, otherwise use CPU
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use the first GPU
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Blip2ForImageTextRetrieval.from_pretrained("Salesforce/blip2-itm-vit-g", torch_dtype=torch.float16)
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-itm-vit-g")
    model.to(device)
    return model, processor

def compute_vision_alignment_score(model, processor, frames_dir, env, batch_size=128):
    """
    Compute the vision alignment score for the given frames using the BLIP model.
    
    This version assumes that the model is in float16 precision. 
    The inference is wrapped with torch.cuda.amp.autocast() to enforce half precision computation.
    """
    # Load the frames from the specified directory.
    frames = get_frames(frames_dir)
    env = env.lower()
    texts = [env_description[env]["goal"], env_description[env]["baseline"]]
    
    # Initialize accumulators for probabilities.
    total_goal_prob = 0.0
    total_baseline_prob = 0.0

    # Calculate number of batches needed.
    num_batches = math.ceil(len(frames) / batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for i in range(num_batches):
            # Get the batch of images for this iteration.
            batch_images = frames[i * batch_size : (i + 1) * batch_size]

            # Process the batch.
            inputs = processor(text=texts, images=batch_images, return_tensors="pt", padding=True).to(device)
            
            # Use AMP autocast context manager to ensure float16 precision computations.
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                itm_out = model(**inputs)
                logits_per_image = itm_out.logits_per_image

                # Calculate probabilities.
                probs = logits_per_image.softmax(dim=1)

            # Accumulate results.
            total_goal_prob += probs[:, 0].sum().item()
            total_baseline_prob += probs[:, 1].sum().item()

            # Free up GPU memory used by intermediate variables.
            del itm_out, logits_per_image, probs, inputs
            torch.cuda.empty_cache()

    # Compute the averages by dividing by the total number of images.
    goal_prob_avg = total_goal_prob / len(frames)
    baseline_prob_avg = total_baseline_prob / len(frames)
    
    # Compute the final score.
    score = math.log(SCORE_CONSTANT * (goal_prob_avg - baseline_prob_avg)) if goal_prob_avg > baseline_prob_avg else 0.0
    
    return score, goal_prob_avg, baseline_prob_avg

if __name__ == "__main__":
    # Example usage
    frames_dir = "../trajectories/PPO_0/Ep_1_Trajectory"  # Replace with your frames directory
    env = "cartpole"  # Replace with your environment name
    model, processor = load_blip_model()
    score, goal_prob_avg, baseline_prob_avg = compute_vision_alignment_score(model, processor, frames_dir, env, batch_size=4)
    print(f"Score: {score}, Goal Probability Average: {goal_prob_avg}, Baseline Probability Average: {baseline_prob_avg}")