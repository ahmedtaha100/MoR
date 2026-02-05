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
    parser.add_argument("--budgets", type=str, required=True)
    parser.add_argument("--warmup_ratio", type=float, default=None)
    parser.add_argument("--cooldown_ratio", type=float, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
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
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.warmup_ratio is not None:
        cfg.warmup_ratio = args.warmup_ratio
    if args.cooldown_ratio is not None:
        cfg.cooldown_ratio = args.cooldown_ratio
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


def build_dataloaders(args, model_config, training_config, data_cfg):
    if args.use_random_data or not data_cfg.get("dataset_name"):
        num_train = data_cfg.get("max_train_samples", 10000)
        num_eval = data_cfg.get("max_eval_samples", 1000)
        seq_len = model_config.max_seq_len
        train_dataset = create_random_dataset(num_train, seq_len, model_config.vocab_size)
        eval_dataset = create_random_dataset(num_eval, seq_len, model_config.vocab_size)
        train_loader = DataLoader(train_dataset, batch_size=training_config.batch_size, shuffle=True, collate_fn=collate_fn)
        eval_loader = DataLoader(eval_dataset, batch_size=training_config.batch_size, shuffle=False, collate_fn=collate_fn)
        return train_loader, eval_loader, None
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
    return train_loader, eval_loader, tokenizer


def make_model(args, model_config):
    if args.model.startswith("vanilla"):
        return VanillaForCausalLM(model_config)
    return MoRForCausalLM(model_config)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config_dict = None
    if args.config:
        with open(args.config, "r") as f:
            config_dict = yaml.safe_load(f)

    budgets = [int(x) for x in args.budgets.split(",") if x.strip()]
    if not budgets:
        raise ValueError("budgets is empty")
    budgets = sorted(budgets)
    min_budget = budgets[0]
    max_budget = budgets[-1]

    model_config = get_model_config(args, config_dict)
    training_config = get_training_config(args, config_dict)
    data_cfg = get_data_config(args, config_dict)

    warmup_ratio_base = training_config.warmup_ratio
    cooldown_ratio_base = training_config.cooldown_ratio
    warmup_steps = int(min_budget * warmup_ratio_base)

    plateau_steps = {}
    for b in budgets:
        cooldown_steps = int(b * cooldown_ratio_base)
        plateau_steps[b] = b - cooldown_steps

    base_config = TrainingConfig(**vars(training_config))
    base_config.max_steps = max_budget
    base_config.lr_scheduler = "trapezoid"
    base_config.warmup_ratio = warmup_steps / max_budget
    base_config.cooldown_ratio = cooldown_ratio_base
    base_config.save_steps = [plateau_steps[b] for b in budgets]

    train_loader, eval_loader, tokenizer = build_dataloaders(args, model_config, base_config, data_cfg)
    if tokenizer is not None:
        model_config.vocab_size = len(tokenizer)

    base_model = make_model(args, model_config)
    base_trainer = MoRTrainer(
        model=base_model,
        config=base_config,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
    )
    base_trainer.train()

    for b in budgets:
        if b == max_budget:
            continue
        plateau_step = plateau_steps[b]
        ckpt_path = Path(base_config.output_dir) / f"checkpoint-{plateau_step}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(str(ckpt_path))
        budget_config = TrainingConfig(**vars(training_config))
        budget_config.max_steps = b
        budget_config.lr_scheduler = "trapezoid"
        budget_config.warmup_ratio = warmup_steps / b
        budget_config.cooldown_ratio = cooldown_ratio_base
        budget_config.output_dir = str(Path(base_config.output_dir) / f"budget_{b}")
        train_loader, eval_loader, _ = build_dataloaders(args, model_config, budget_config, data_cfg)
        budget_model = make_model(args, model_config)
        budget_trainer = MoRTrainer(
            model=budget_model,
            config=budget_config,
            train_dataloader=train_loader,
            eval_dataloader=eval_loader,
        )
        budget_trainer.load_checkpoint(str(ckpt_path), load_scheduler=False)
        budget_trainer.scheduler = budget_trainer._create_scheduler()
        budget_trainer.scheduler.last_epoch = budget_trainer.global_step - 1
        budget_trainer.scheduler.step()
        budget_trainer.train()


if __name__ == "__main__":
    main()
