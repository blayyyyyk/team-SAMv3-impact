import os
from dotenv import load_dotenv, find_dotenv
from huggingface_hub import login


load_dotenv(find_dotenv(".env.local"))
hf_token = os.getenv("HF_TOKEN")
login(token=hf_token, add_to_git_credential=True)

from ._inference_medsam3 import predict
__all__ = ["predict"]

