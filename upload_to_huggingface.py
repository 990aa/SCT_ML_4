import os
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN_HGR')

if not HF_TOKEN:
    raise ValueError("HF_TOKEN_HGR not found in .env file")

REPO_NAME = "hand-gesture-recognition"  
REPO_TYPE = "model"

FILES_TO_UPLOAD = [
    "hand_gesture_lstm_model.h5",
    "gesture_mapping.json",
    "README.md",
    "MODEL_CARD.md"
]

print("=" * 70)
print("Hugging Face Model Upload Script")
print("=" * 70)
print(f"\nRepository (name): {REPO_NAME}")
print(f"Type: {REPO_TYPE}\n")

api = HfApi(token=HF_TOKEN)

user_info = api.whoami(token=HF_TOKEN)
username = user_info.get('name') or user_info.get('user', {}).get('name')
repo_id = f"{username}/{REPO_NAME}"

print(f"Username: {username}")
print(f"Full Repository ID: {repo_id}\n")

print("Checking required files...")
missing_files = [f for f in FILES_TO_UPLOAD if not os.path.exists(f)]
if missing_files:
    print(f"❌ Missing files: {', '.join(missing_files)}")
    if "hand_gesture_lstm_model.h5" in missing_files or "gesture_mapping.json" in missing_files:
        print("Please ensure the model and gesture mapping are present before uploading.")
    raise SystemExit(1)

print("✓ All required files found\n")

print(f"Creating/Verifying Hugging Face model repo: {repo_id}...")
try:
    create_repo(repo_id=repo_id, repo_type=REPO_TYPE, token=HF_TOKEN, exist_ok=True)
    print(f"✓ Repository created/verified: {repo_id}\n")
except Exception as e:
    print(f"⚠️ Repository creation/verification warning: {e}\n")

print("Uploading files to Hugging Face model repo...")
print(f"{'=' * 70}\n")
for filename in FILES_TO_UPLOAD:
    try:
        print(f"Uploading {filename}...")
        api.upload_file(
            path_or_fileobj=filename,
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type=REPO_TYPE,
            token=HF_TOKEN,
            commit_message=f"Upload {filename}"
        )
        print(f"✓ Uploaded {filename}\n")
    except Exception as file_error:
        print(f"❌ Failed to upload {filename}: {file_error}\n")

print(f"\n{'=' * 70}")
print("MODEL UPLOAD COMPLETE!")
print(f"{'=' * 70}\n")

model_url = f"https://huggingface.co/{repo_id}"
print(f"Model repo is available at: {model_url}")
print("\nNote: model files are now stored in the repository root. Download them via the Hub API or `huggingface_hub.hf_hub_download`.")

print(f"\n{'=' * 70}")
print("✅ PROCESS COMPLETE!")
print(f"{'=' * 70}\n")
