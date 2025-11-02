# Knowledge Distillation from an LMS-FIR Filter to its Compressed Representation

In real-time applications, MEMS-IMU accelerometers are challenging to denoise due to the unknown gravity projection on the accelerometer axes. Classical time domain FIR filters such as HPF, LPF, BPF, and BSF cannot match the optimal frequency response. Manually designing a Hybrid Bandpass Filter is virtually impossible for real-time systems with complex evolving frequency spectra. Adaptive filters that solve this problem, often as pre-filters in fusion algorithms, can cost speed and memory because of their long filter lengths required to learn features.

<img width="915" height="723" alt="image" src="https://github.com/user-attachments/assets/50f79854-46b3-4eef-9c8e-4a2d65bcb7b8" />


The denoising of MEMS-IMU triaxial accelerometer signals represents a significant challenge in inertial navigation and motion sensing applications due to the inherently noisy characteristics of micro-electromechanical systems. This project proposes an innovative two-stage knowledge distillation framework that transfers knowledge from a sophisticated but computationally complex "teacher" filter to a compact "student" filter through a shallow MLP (Multi-Layer Perceptron) Encoder like the one commonly found in autoencoders.

## Python environment

This project includes a small Python helper script `pt2csv.py` that converts PyTorch `.pt` tensor files to CSV. To run it locally on Windows (cmd.exe), create and activate a virtual environment, install dependencies, and run the script.

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
