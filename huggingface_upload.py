from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HUGGINGFACE_TOKEN"))
api.upload_large_folder(
    folder_path="./data",
    repo_id="PypDeveloper/AircraftDetection_Dataset",
    repo_type="dataset",
)
