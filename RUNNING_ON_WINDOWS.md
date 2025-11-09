# Running Scripts on Windows

### Option 1: Use Batch Files (Easiest)
```cmd
# Run inference with webcam
run_inference.bat

# Upload model to Hugging Face
run_upload.bat
```

### Option 2: Use Virtual Environment Directly
```cmd
# Activate virtual environment
.venv\Scripts\activate

# Run inference
python inference.py --repo a-01a/hand-gesture-recognition

# Run upload
python upload_to_huggingface.py

# Deactivate when done
deactivate
```

### Option 3: Run Without Activating
```cmd
# Run inference
.venv\Scripts\python.exe inference.py --repo a-01a/hand-gesture-recognition

# Run upload
.venv\Scripts\python.exe upload_to_huggingface.py
```

## Failing `uv run`

The `uv run` command has compatibility issues with TensorFlow on Windows due to the `tensorflow-io-gcs-filesystem` package not having Windows wheels. So use the virtual environment directly with Python 3.12.

## Setup (First Time Only)

1. Ensure you have the virtual environment:
   ```cmd
   uv venv --python 3.12 --clear
   ```

2. Install dependencies:
   ```cmd
   uv pip install -r requirements.txt
   ```