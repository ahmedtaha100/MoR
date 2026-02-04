#!/usr/bin/env python

import argparse
import sys
import yaml

sys.path.insert(0, 'src')

import torch
from torch.utils.data import DataLoader

from model.config import (
    MoRConfig,
    get_mor_135m_config,
    get_mor_360m_config,
    get_vanilla_360m_config,
)
from model.mor_model import MoRForCausalLM
from model.vanilla_model import VanillaForCausalLM
from data.dataset import create_random_dataset, collate_fn
from training.trainer import TrainingConfig, MoRTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train MoR models")

    parser.add_argument(
        "--model",
        type=str,
        default="mor_135m",
        choices=["mor_135m", "mor_360m", "vanilla_360m"],
        help="Model configuration to use",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (overrides --model)",
    )

    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="trapezoid",
        choices=["cosine", "trapezoid"],
    )
    parser.add_argument("--warmup_ratio", type=float, default=0.02)
    parser.add_argument("--cooldown_ratio", type=float, default=0.2)

    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["fp32", "fp16", "bf16"],
    )

    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--num_train_samples", type=int, default=10000)
    parser.add_argument("--num_eval_samples", type=int, default=1000)

    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)

    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def get_model_config(args) -> MoRConfig:
    if args.config:
        with open(args.config, "r") as f:
            config_dict = yaml.safe_load(f)
        return MoRConfig.from_dict(config_dict.get("model", config_dict))

    if args.model == "mor_135m":
        return get_mor_135m_config()
    elif args.model == "mor_360m":
        return get_mor_360m_config()
    elif args.model == "vanilla_360m":
        return get_vanilla_360m_config()
    else:
        raise ValueError(f"Unknown model: {args.model}")


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model_config = get_model_config(args)
    model_config.max_seq_len = args.seq_len

    print(f"Model config: {model_config}")

    if args.model.startswith("vanilla"):
        model = VanillaForCausalLM(model_config)
    else:
        model = MoRForCausalLM(model_config)

    num_params = model.get_num_parameters()
    print(f"Model parameters: {num_params:,}")

    print("Creating datasets...")
    train_dataset = create_random_dataset(
        args.num_train_samples, args.seq_len, model_config.vocab_size
    )
    eval_dataset = create_random_dataset(
        args.num_eval_samples, args.seq_len, model_config.vocab_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    training_config = TrainingConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler=args.lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        cooldown_ratio=args.cooldown_ratio,
        mixed_precision=args.mixed_precision,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        seed=args.seed,
    )

    trainer = MoRTrainer(
        model=model,
        config=training_config,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
    )

    print("Starting training...")
    metrics = trainer.train()
    print(f"Training complete. Final metrics: {metrics}")


if __name__ == "__main__":
    main()
