# MBD-Project
Two-Stage Knowledge Distillation between Two Adaptive LMS-FIR Filters for Denoising MEMS-IMU's Triaxial Gyroscope

## Python environment

This project includes a small Python helper script `py2csv.py` that converts PyTorch `.pt` tensor files to CSV. To run it locally on Windows (cmd.exe), create and activate a virtual environment, install dependencies, and run the script.

Quick steps (Windows cmd.exe):

1. Create a virtual environment (named `.venv`):

	python -m venv .venv

2. Activate it (cmd.exe):

	.venv\Scripts\activate

3. Install dependencies:

	pip install --upgrade pip
	pip install -r requirements.txt

Note on PyTorch: On Windows, installing `torch` from PyPI may fetch CPU-only builds. If you need a specific CUDA build, follow the install instructions from https://pytorch.org and replace the `pip install torch` step with the recommended command for your CUDA version.

4. Example usage:

	python py2csv.py datasets\\X_train_GT.pt X_train_GT.csv

Files added/updated:

- `requirements.txt` — minimal dependency list for the helper script.
- `.gitignore` — added Python-related ignores to avoid committing virtual environments and caches.
