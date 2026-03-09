# LayerSkip-LLaDA (`skipllada`)

This directory contains the custom implementation of **LayerSkip** applied to **LLaDA** (a Masked Diffusion Language Model). The codebase provides tools for continual pre-training, evaluation, and specialized early-exit inference algorithms.

---

## 📖 The Training Recipe

The core idea is to train LLaDA such that intermediate transformer layers can make accurate predictions without running the full 32-layer depth. Since LLaDA is a diffusion model, the "noise level" (timestep $t$) dictates how hard the prediction task is. We condition the early-exit training on this timestep.

### Objective Function

We augment the standard LLaDA masked cross-entropy loss ($L_{base}$) with early-exit losses ($L_{exit}^e$) from a subset of layers $E \subset \{1, \dots, L\}$. 

$$ L_{total} = L_{base} + \frac{\epsilon_{scale}}{|E|} \sum_{e \in E} w(e, t) L_{exit}^e $$

- $L_{base}$ is computed at the final layer ($L=32$) and ensures the model retains its full-depth capability.
- $L_{exit}^e$ is the standard diffusion loss computed at layer $e$, but we cap the $1/t$ reweighting to prevent gradient explosions from early, highly uncertain layers.
- $w(e, t)$ is a curriculum weight that activates/deactivates specific early exits depending on the training phase and noise level.

### Curricula & Layer Dropout

The training relies on three mechanisms to ensure stability:
1. **Gradual Curriculum:** We start by training the deepest exit (e.g., layer 24), then progressively activate shallower exits (layer 16, then layer 8).
2. **Timestep Annealing:** Early exits are initially only penalized on "easy" (high $t$, highly masked) steps. As training progresses, they are forced to predict on "hard" (low $t$, mostly unmasked) steps.
3. **Timestep-Conditioned Layer Dropout:** To decouple the layers and force early layers to be self-sufficient, we drop out layers with probability $p_l(t)$. Shallow layers are rarely dropped; deep layers are dropped more often, and dropout is highest when $t$ is large (high noise).

---

## 🧠 Codebase Structure

- `model/layerskip_llada.py`: The custom model wrapper. Adds `forward_early_exit()` and `forward_remainder()` capabilities and implements layer dropout.
- `trainer.py`: Custom HuggingFace Trainer. Computes the combined loss, applies the curriculum, handles gradient capping, and logs metrics.
- `curriculum.py`: Manages the gradual and timestep-annealed activation of early exits.
- `data.py`: `StreamingPretrainingDataset` to stream the Pile directly from HF without downloading.
- `inference.py`: Implements *Depth-Scheduled* and *Self-Speculative* decoding algorithms.
- `eval_early_exit.py` & `eval_speculative.py`: Evaluation scripts for measuring accuracy and speedup.

---

## 🚀 How to Run the Scripts

*Note: All commands assume you are running from the root repository directory (`/hdd1/ay8757/SkipDLM`).*

### 1. Training

To launch a continual pre-training run on the Pile using a single GPU (with 8-bit AdamW and gradient checkpointing to fit an 8B model):

```bash
python -m skipllada.train \
    --mode pretrain \
    --model_name_or_path GSAI-ML/LLaDA-8B-Base \
    --output_dir checkpoints/skipllada_100M \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --max_seq_length 2048 \
    --max_steps 1526 \
    --save_steps 500 \
    --learning_rate 1e-5 \
    --curriculum_mode gradual \
    --eps_scale 1.0 \
    --gradient_checkpointing \
    --bf16
```
*(1526 steps × 32 accum × 2048 seq_len ≈ 100M tokens).*

### 2. Converting Checkpoints

The custom Trainer saves weights with a `base_model.model.` prefix. To evaluate them using standard HuggingFace tools or `lm-eval`, you must convert them to the standard format:

```bash
python -m skipllada.convert_checkpoint_to_hf \
    --checkpoint_dir checkpoints/skipllada_100M/checkpoint-1526 \
    --output_dir checkpoints/skipllada_100M/checkpoint-1526-hf \
    --base_model GSAI-ML/LLaDA-8B-Base
```

### 3. Evaluation: Early Exit (Loglikelihood)

Measures accuracy if the model is forced to exit at a specific layer (e.g., Layer 16). Uses LLaDA's Monte-Carlo loglikelihood scoring on MMLU.

```bash
CKPT=checkpoints/skipllada_100M
PYTHONPATH=$CKPT/checkpoint-1526-hf \
python -m skipllada.eval_early_exit \
    --models GSAI-ML/LLaDA-8B-Base $CKPT/checkpoint-1526-hf \
    --model_names baseline ckpt-1526 \
    --exit_layers 8,16,24 \
    --subset abstract_algebra \
    --limit 100 \
    --mc_num 32 \
    --batch_size 4 \
    --output_json eval_results/early_exit/results.json
```
*(This script will also automatically generate bar charts and scatter plots in the output directory).*

### 4. Evaluation: Self-Speculative Decoding

Measures real-world generation accuracy, wall-clock speedup, and the "draft skip rate" using the draft-then-verify decoding algorithm.

```bash
CKPT=checkpoints/skipllada_100M
PYTHONPATH=$CKPT/checkpoint-1526-hf \
python -m skipllada.eval_speculative \
    --models GSAI-ML/LLaDA-8B-Base $CKPT/checkpoint-1526-hf \
    --model_names baseline ckpt-1526 \
    --draft_exit 16 \
    --gamma_base 0.85 \
    --gamma_low_t 0.95 \
    --steps 16 \
    --gen_length 4 \
    --subset abstract_algebra \
    --limit 100 \
    --output_json eval_results/speculative/results.json
```

### 5. Diagnostics

During or after training, you can generate heatmaps to see the exact perplexity and layer agreement at different noise buckets:

```bash
python -m skipllada.evaluate_baseline \
    --model_name_or_path checkpoints/skipllada_100M/checkpoint-1526-hf \
    --batch_size 1
```
*(Saves PNG heatmaps to `checkpoints/skipllada_100M/checkpoint-1526-hf/diagnostics/`)*
