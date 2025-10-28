ALS In‑Silico Gene Perturbation + Geneformer V2

Overview
This repository provides a reproducible workflow to simulate gene perturbations in ALS single‑cell/snRNA‑seq data and compute Geneformer V2 embeddings for analysis and prioritization.

What’s inside
- Notebooks: end‑to‑end perturbation, embedding, interpretation, and optional prioritization
- Utilities: helpers for data I/O, batching, plotting
- Local Geneformer v2 clone (editable install)

Requirements
- Python 3.10
- NVIDIA GPU; driver supporting CUDA ≥ 12.1 (12.x drivers OK)
- PyTorch (CUDA build) compatible with your driver
- Packages in `requirements.txt`

Quickstart
1) Create a Python 3.10 virtual environment
```bash
# If system python3.10 is not available, provision it with conda:
conda create -y -p /cs/student/projects1/aibh/2024/rmaheswa/Helical_Task/.conda-py310 python=3.10
conda activate /cs/student/projects1/aibh/2024/rmaheswa/Helical_Task/.conda-py310
python -m venv /cs/student/projects1/aibh/2024/rmaheswa/Helical_Task/.venv-gf310
conda deactivate

# Activate the venv
source /cs/student/projects1/aibh/2024/rmaheswa/Helical_Task/.venv-gf310/bin/activate
python -V   # 3.10.x
```

2) Install CUDA PyTorch (pick one)
Check your driver: `nvidia-smi`. With CUDA 12.x drivers, use cu121 wheels:
```bash
python -m pip install -U pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

3) Install dependencies and local Geneformer v2
```bash
pip install -r /cs/student/projects1/aibh/2024/rmaheswa/Helical_Task/als-perturb-geneformer/requirements.txt
pip install -e /cs/student/projects1/aibh/2024/rmaheswa/Helical_Task/als-perturb-geneformer/Geneformer
```


Run notebooks
Open notebooks under `notebooks/` in order:
- `01_perturbation_workflow.ipynb`
- `02_apply_to_ALS_genes_and_embed.ipynb`
- `03_interpret_embedding.ipynb`
- `04_prioritize_targets_optional.ipynb`


License
MIT (see LICENSE).

References
- Geneformer: `https://huggingface.co/ctheodoris/Geneformer`
