import torch

class Config:
    DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RETRIEVAL_MODEL_NAME="clip"
    N_K=5
    RETRIEVAL_DIR="retri_result_clip"