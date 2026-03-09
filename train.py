"""
Entry point for LayerSkip-LLaDA training.

Usage:
  # Single GPU
  python -m skipllada.train --mode pretrain

  # Multi-GPU with FSDP via accelerate
  accelerate launch --config_file accelerate_config.yaml -m skipllada.train --mode pretrain
"""

import argparse
import os
import sys
import logging

import torch
from transformers import TrainingArguments, AutoTokenizer, DataCollatorWithPadding

from .config import TrainingConfig, LayerSkipConfig, CurriculumConfig
from .curriculum import CurriculumScheduler
from .data import create_dummy_dataset, create_dummy_sft_dataset, StreamingPretrainingDataset
from .model.layerskip_llada import LayerSkipLLaDA
from .trainer import LayerSkipTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="LayerSkip-LLaDA Training")
    parser.add_argument("--mode", type=str, default="pretrain", choices=["pretrain", "sft"])
    parser.add_argument("--model_name_or_path", type=str, default="GSAI-ML/LLaDA-8B-Base")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/skipllada")
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--dataset_name", type=str, default="monology/pile-uncopyrighted")
    parser.add_argument("--metrics_every_n_steps", type=int, default=50)
    parser.add_argument("--dashboard_every_n_steps", type=int, default=500)
    parser.add_argument("--smoke_test", action="store_true", default=False)
    parser.add_argument("--curriculum_mode", type=str, default="gradual",
                        choices=["rotational", "gradual", "combined"])
    parser.add_argument("--eps_scale", type=float, default=0.2)
    parser.add_argument("--p_max", type=float, default=0.1)
    parser.add_argument("--c_cap", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def collate_fn(batch):
    """Simple collator that stacks input_ids and optional prompt_length."""
    input_ids = torch.stack([b["input_ids"] for b in batch])
    result = {"input_ids": input_ids}
    if "prompt_length" in batch[0]:
        result["prompt_length"] = torch.stack([b["prompt_length"] for b in batch])
    return result


def main():
    args = parse_args()

    tc = TrainingConfig(
        model_name_or_path=args.model_name_or_path,
        mode=args.mode,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        max_steps=3 if args.smoke_test else args.max_steps,
        num_train_epochs=1 if args.smoke_test else args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=1 if args.smoke_test else args.gradient_accumulation_steps,
        max_seq_length=512 if args.smoke_test else args.max_seq_length,
        logging_steps=1 if args.smoke_test else args.logging_steps,
        save_steps=args.save_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        seed=args.seed,
    )

    if args.learning_rate is not None:
        if args.mode == "sft":
            tc.sft_learning_rate = args.learning_rate
        else:
            tc.learning_rate = args.learning_rate

    tc.layerskip = LayerSkipConfig(
        eps_scale=args.eps_scale,
        p_max=args.p_max,
        c_cap=args.c_cap,
    )

    # Scale curriculum total_tokens to match actual run length so phase
    # boundaries land within the run (not at 2.3T default).
    tokens_per_step = (
        tc.per_device_train_batch_size
        * tc.gradient_accumulation_steps
        * tc.max_seq_length
    )
    actual_total_tokens = float(
        tokens_per_step * tc.max_steps if tc.max_steps > 0 else 1e8
    )
    tc.curriculum = CurriculumConfig(
        mode=args.curriculum_mode,
        total_tokens=actual_total_tokens,
    )

    logger.info(f"Loading tokenizer from {tc.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(tc.model_name_or_path, trust_remote_code=True)

    logger.info(f"Loading model from {tc.model_name_or_path}")
    ls_model = LayerSkipLLaDA.from_pretrained(
        tc.model_name_or_path,
        dtype=torch.bfloat16 if tc.bf16 else torch.float32,
    )
    ls_model.p_max = tc.layerskip.p_max
    ls_model.beta = tc.layerskip.beta
    ls_model.exit_layers = tc.layerskip.exit_layers

    curriculum = CurriculumScheduler(
        config=tc.curriculum,
        n_layers=ls_model.n_layers,
        rotation_period=tc.layerskip.rotation_period,
    )

    if args.smoke_test:
        logger.info("Creating dummy dataset for smoke test")
        if tc.mode == "sft":
            train_dataset = create_dummy_sft_dataset(tokenizer, num_samples=16, max_seq_length=tc.max_seq_length)
        else:
            train_dataset = create_dummy_dataset(tokenizer, num_samples=16, max_seq_length=tc.max_seq_length)
    elif tc.mode == "pretrain":
        dataset_name = tc.dataset_name
        logger.info(f"Creating streaming dataset from {dataset_name}")
        train_dataset = StreamingPretrainingDataset(
            tokenizer=tokenizer,
            dataset_name=dataset_name,
            dataset_split=tc.dataset_split,
            max_seq_length=tc.max_seq_length,
            seed=tc.seed,
        )
    else:
        raise NotImplementedError(
            "Full SFT dataset loading not yet implemented. Use --smoke_test for now."
        )

    multi_gpu = torch.cuda.device_count() > 1

    training_args = TrainingArguments(
        output_dir=tc.output_dir,
        num_train_epochs=tc.num_train_epochs,
        max_steps=tc.max_steps,
        per_device_train_batch_size=tc.per_device_train_batch_size,
        gradient_accumulation_steps=tc.gradient_accumulation_steps,
        learning_rate=tc.effective_lr,
        weight_decay=tc.weight_decay,
        warmup_ratio=tc.warmup_ratio,
        lr_scheduler_type=tc.lr_scheduler_type,
        bf16=tc.bf16,
        gradient_checkpointing=tc.gradient_checkpointing if not multi_gpu else False,
        optim="adamw_bnb_8bit" if not multi_gpu else "adamw_torch",
        logging_steps=tc.logging_steps,
        save_steps=tc.save_steps,
        save_total_limit=tc.save_total_limit,
        seed=tc.seed,
        remove_unused_columns=False,
        report_to="none" if args.smoke_test else "tensorboard",
        dataloader_pin_memory=True,
        fsdp="full_shard auto_wrap" if multi_gpu else "",
        fsdp_config={
            "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            "fsdp_transformer_layer_cls_to_wrap": "LLaDALlamaBlock",
            "fsdp_backward_prefetch": "backward_pre",
            "fsdp_forward_prefetch": True,
            "fsdp_use_orig_params": True,
            "activation_checkpointing": tc.gradient_checkpointing,
        } if multi_gpu else None,
    )

    trainer = LayerSkipTrainer(
        ls_model=ls_model,
        training_config=tc,
        curriculum=curriculum,
        metrics_every_n_steps=1 if args.smoke_test else args.metrics_every_n_steps,
        dashboard_every_n_steps=1 if args.smoke_test else args.dashboard_every_n_steps,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
        tokenizer=tokenizer,
    )

    logger.info("Starting training...")
    trainer.train()

    if not args.smoke_test:
        logger.info(f"Saving model to {tc.output_dir}")
        trainer.save_model(tc.output_dir)

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
