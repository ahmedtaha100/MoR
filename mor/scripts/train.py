import argparse
import sys
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from model.config import MoRConfig, get_mor_135m_config, get_mor_360m_config, get_vanilla_360m_config
from model.mor_model import MoRForCausalLM
from model.vanilla_model import VanillaForCausalLM
from data.dataset import create_random_dataset, collate_fn, create_hf_dataloader
from training.trainer import TrainingConfig, MoRTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mor_135m", choices=["mor_135m", "mor_360m", "vanilla_360m"])
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--lr_scheduler", type=str, default=None, choices=["cosine", "trapezoid"])
    parser.add_argument("--warmup_ratio", type=float, default=None)
    parser.add_argument("--cooldown_ratio", type=float, default=None)
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--num_train_samples", type=int, default=None)
    parser.add_argument("--num_eval_samples", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=None)
    parser.add_argument("--eval_interval", type=int, default=None)
    parser.add_argument("--save_interval", type=int, default=None)
    parser.add_argument("--save_steps", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--train_split", type=str, default=None)
    parser.add_argument("--eval_split", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--text_field", type=str, default=None)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--shuffle_buffer", type=int, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--use_random_data", action="store_true")
    return parser.parse_args()


def get_model_config(args, config_dict) -> MoRConfig:
    if config_dict and "model" in config_dict:
        return MoRConfig.from_dict(config_dict["model"])
    if args.model == "mor_135m":
        return get_mor_135m_config()
    if args.model == "mor_360m":
        return get_mor_360m_config()
    if args.model == "vanilla_360m":
        return get_vanilla_360m_config()
    raise ValueError("Unknown model")


def get_training_config(args, config_dict) -> TrainingConfig:
    if config_dict and "training" in config_dict:
        cfg = TrainingConfig(**config_dict["training"])
    else:
        cfg = TrainingConfig()
    for name in [
        "learning_rate",
        "batch_size",
        "gradient_accumulation_steps",
        "max_steps",
        "warmup_steps",
        "max_grad_norm",
        "lr_scheduler",
        "warmup_ratio",
        "cooldown_ratio",
        "mixed_precision",
        "log_interval",
        "eval_interval",
        "save_interval",
        "resume_from_checkpoint",
        "output_dir",
        "wandb_project",
        "wandb_run_name",
    ]:
        val = getattr(args, name)
        if val is not None:
            setattr(cfg, name, val)
    if args.save_steps:
        cfg.save_steps = [int(x) for x in args.save_steps.split(",") if x.strip()]
    return cfg


def get_data_config(args, config_dict):
    data_cfg = (config_dict or {}).get("data", {})
    merged = dict(data_cfg)
    for key, val in {
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "tokenizer": args.tokenizer,
        "text_field": args.text_field,
        "streaming": True if args.streaming else None,
        "shuffle_buffer": args.shuffle_buffer,
        "max_train_samples": args.max_train_samples,
        "max_eval_samples": args.max_eval_samples,
        "num_workers": args.num_workers,
    }.items():
        if val is not None:
            merged[key] = val
    return merged


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config_dict = None
    if args.config:
        with open(args.config, "r") as f:
            config_dict = yaml.safe_load(f)

    model_config = get_model_config(args, config_dict)
    if args.seq_len is not None:
        model_config.max_seq_len = args.seq_len

    if args.model.startswith("vanilla"):
        model = VanillaForCausalLM(model_config)
    else:
        model = MoRForCausalLM(model_config)

    training_config = get_training_config(args, config_dict)
    data_cfg = get_data_config(args, config_dict)

    if args.use_random_data or not data_cfg.get("dataset_name"):
        num_train = args.num_train_samples or 10000
        num_eval = args.num_eval_samples or 1000
        seq_len = model_config.max_seq_len
        train_dataset = create_random_dataset(num_train, seq_len, model_config.vocab_size)
        eval_dataset = create_random_dataset(num_eval, seq_len, model_config.vocab_size)
        train_loader = DataLoader(train_dataset, batch_size=training_config.batch_size, shuffle=True, collate_fn=collate_fn)
        eval_loader = DataLoader(eval_dataset, batch_size=training_config.batch_size, shuffle=False, collate_fn=collate_fn)
    else:
        seq_len = model_config.max_seq_len
        dataset_name = data_cfg.get("dataset_name")
        dataset_config = data_cfg.get("dataset_config")
        train_split = data_cfg.get("train_split", "train")
        eval_split = data_cfg.get("eval_split", "validation")
        tokenizer_name = data_cfg.get("tokenizer", data_cfg.get("tokenizer_name", "HuggingFaceH4/SmolLM-135M"))
        text_field = data_cfg.get("text_field", "text")
        streaming = bool(data_cfg.get("streaming", True))
        shuffle_buffer = data_cfg.get("shuffle_buffer", 10000)
        max_train = data_cfg.get("max_train_samples")
        max_eval = data_cfg.get("max_eval_samples")
        num_workers = data_cfg.get("num_workers", 0)
        train_loader, tokenizer = create_hf_dataloader(
            dataset_name,
            dataset_config,
            train_split,
            tokenizer_name,
            seq_len,
            training_config.batch_size,
            text_field=text_field,
            streaming=streaming,
            shuffle_buffer=shuffle_buffer,
            seed=args.seed,
            max_samples=max_train,
            num_workers=num_workers,
        )
        eval_loader, _ = create_hf_dataloader(
            dataset_name,
            dataset_config,
            eval_split,
            tokenizer_name,
            seq_len,
            training_config.batch_size,
            text_field=text_field,
            streaming=streaming,
            shuffle_buffer=0,
            seed=args.seed,
            max_samples=max_eval,
            num_workers=num_workers,
        )
        model_config.vocab_size = len(tokenizer)

    trainer = MoRTrainer(
        model=model,
        config=training_config,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
    )

    trainer.train()


if __name__ == "__main__":
    main()
