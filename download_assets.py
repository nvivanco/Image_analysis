import os
import torch
from huggingface_hub import hf_hub_download

# --- Configuration ---
HF_REPO_ID = "nvivanco/DuMM_bacteria_track"
HF_FILENAME = "best_link_prediction_model.pt"
LOCAL_MODEL_DIR = "models"
LOCAL_MODEL_PATH = os.path.join(LOCAL_MODEL_DIR, HF_FILENAME)

def download_model():
    """Downloads the trained model file from the Hugging Face Hub."""

    # 1. Create the local directory if it doesn't exist
    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

    # 2. Check if the model already exists locally
    if os.path.exists(LOCAL_MODEL_PATH):
        print(f"Model already exists at {LOCAL_MODEL_PATH}. Skipping download.")
        return

    print(f"Downloading model from Hugging Face Hub: {HF_REPO_ID}/{HF_FILENAME}...")

    # 3. Download the file
    # This function automatically handles large files, progress, and caching.
    try:
        downloaded_file = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            local_dir=LOCAL_MODEL_DIR,
            local_dir_use_symlinks=False
        )
        print(f"Download complete. Model saved to: {downloaded_file}")

    except Exception as e:
        print(f"Error downloading model: {e}")
        print("Please ensure you have read access to the repository.")

if __name__ == "__main__":
    download_model()