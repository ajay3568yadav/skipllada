"""
Convert a Trainer checkpoint (LayerSkipLLaDA wrapper state_dict with
base_model.model.* keys) to standard HuggingFace format (model.* keys)
so LLaDAModelLM.from_pretrained() and eval scripts can load it.

Usage:
    python -m skipllada.convert_checkpoint_to_hf \\
        --checkpoint_dir ./checkpoints/skipllada_100M/checkpoint-500 \\
        --output_dir ./checkpoints/skipllada_100M/checkpoint-500-hf \\
        [--base_model GSAI-ML/LLaDA-8B-Base]
"""

import argparse
import os
import shutil

from safetensors.torch import load_file, save_file
from transformers import AutoConfig


def parse_args():
    p = argparse.ArgumentParser(description="Convert Trainer checkpoint to HF format")
    p.add_argument("--checkpoint_dir", type=str, required=True,
                   help="Path to checkpoint-500 or checkpoint-1000")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Where to write the converted model (model.* keys + config)")
    p.add_argument("--base_model", type=str, default="GSAI-ML/LLaDA-8B-Base",
                   help="Base model name for config and tokenizer")
    return p.parse_args()


def main():
    args = parse_args()
    ckpt_dir = args.checkpoint_dir.rstrip("/")
    out_dir = args.output_dir.rstrip("/")
    os.makedirs(out_dir, exist_ok=True)

    # Load state_dict from checkpoint (safetensors)
    st_path = os.path.join(ckpt_dir, "model.safetensors")
    if not os.path.isfile(st_path):
        raise FileNotFoundError(f"No model.safetensors in {ckpt_dir}")
    state = load_file(st_path, device="cpu")

    # Remap base_model. -> (strip prefix) so LLaDAModelLM sees model.*
    prefix = "base_model."
    new_state = {}
    for k, v in state.items():
        if k.startswith(prefix):
            new_state[k[len(prefix):]] = v
        else:
            new_state[k] = v
    if not new_state:
        raise RuntimeError("No keys started with base_model.; is this a Trainer checkpoint?")

    # Save remapped state
    save_file(new_state, os.path.join(out_dir, "model.safetensors"))

    # Save config from base model
    config = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True)
    config.save_pretrained(out_dir)

    # Copy tokenizer files from checkpoint so from_pretrained(out_dir) works
    for name in ["tokenizer_config.json", "tokenizer.json", "special_tokens_map.json"]:
        src = os.path.join(ckpt_dir, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(out_dir, name))

    # Copy model code so AutoModel.from_pretrained(out_dir, trust_remote_code=True) can load
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    for name, src_name in [("modeling_llada.py", "modeling_llada.py"), ("configuration_llada.py", "configuration_llada.py")]:
        src = os.path.join(pkg_dir, "model", src_name)
        if os.path.isfile(src):
            with open(src) as f:
                code = f.read()
            code = code.replace("from .configuration_llada import", "from configuration_llada import")
            with open(os.path.join(out_dir, name), "w") as f:
                f.write(code)

    print(f"Converted {ckpt_dir} -> {out_dir} ({len(new_state)} keys).")
    print(f"Load with: LayerSkipLLaDA.from_pretrained('{out_dir}') or AutoModel.from_pretrained('{out_dir}', trust_remote_code=True)")


if __name__ == "__main__":
    main()
