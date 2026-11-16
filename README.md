# Humanizing a Local LLM with DPO + LoRA

Fine-tuning **Qwen3-1.7B** with **Direct Preference Optimization (DPO)** and **LoRA** so it answers in a more natural, human-like tone. Everything runs locally on an Apple Silicon Mac (M1, 16 GB) using [MLX](https://github.com/ml-explore/mlx).

A small Flask app shows the base model and the DPO model side by side, so you can compare them.

## Why

LLMs often give correct answers that sound robotic. DPO trains on preference pairs (a *chosen* human-like reply and a *rejected* robotic one), which pushes the model toward the preferred style. It doesn't need a separate reward model, and LoRA keeps memory use low enough for a laptop.

## Setup

| | |
|---|---|
| Base / reference model | `Qwen/Qwen3-1.7B` |
| Dataset | [`HumanLLMs/Human-Like-DPO-Dataset`](https://huggingface.co/datasets/HumanLLMs/Human-Like-DPO-Dataset) |
| Method | DPO (sigmoid loss, β = 0.1) + LoRA (rank 8) |
| Optimizer | Muon, lr 5e-5, 1 epoch |
| Hardware | Apple M1, 16 GB unified memory |

## Project structure

```
├── dpo_minimal.ipynb   # Training: load data → LoRA → DPO → evaluate → fuse
├── app.py              # Flask app: base vs. DPO model comparison
├── templates/          # Web UI
└── docs/report.pdf     # Write-up of the approach and findings
```

## Quick start

Requires macOS on Apple Silicon and Python 3.9+.

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

1. **Train:** run `dpo_minimal.ipynb`. It saves the fused model to `./tests_dpo`.
2. **Compare:**
   ```bash
   python app.py            # open http://localhost:5001
   ```
   To use a model hosted on Hugging Face instead of the local one:
   ```bash
   DPO_MODEL=<hf-user>/<repo> python app.py
   ```

## Results

See [docs/report.pdf](docs/report.pdf). Even with a small dataset, the DPO model's replies are clearer and more conversational than the base model's.

## Acknowledgements

Training uses [mlx-lm-lora](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora).
