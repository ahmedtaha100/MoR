import argparse
import sys
from pathlib import Path
import json
import yaml
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from model.config import MoRConfig, get_mor_135m_config, get_mor_360m_config, get_vanilla_360m_config
from model.mor_model import MoRForCausalLM
from model.vanilla_model import VanillaForCausalLM
from evaluation.throughput import run_throughput


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mor_360m", choices=["mor_135m", "mor_360m", "vanilla_360m"])
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default="HuggingFaceH4/SmolLM-360M")
    parser.add_argument("--num_queries", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--mean_len", type=float, default=256.0)
    parser.add_argument("--std_len", type=float, default=32.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--track_depth", action="store_true")
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


def main():
    args = parse_args()
    config_dict = None
    if args.config:
        with open(args.config, "r") as f:
            config_dict = yaml.safe_load(f)
    model_config = get_model_config(args, config_dict)
    if args.model.startswith("vanilla"):
        model = VanillaForCausalLM(model_config)
    else:
        model = MoRForCausalLM(model_config)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"], strict=False)
        else:
            model.load_state_dict(state, strict=False)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    result = run_throughput(
        model,
        args.tokenizer,
        num_queries=args.num_queries,
        batch_size=args.batch_size,
        mean_len=args.mean_len,
        std_len=args.std_len,
        device=device,
        track_depth=args.track_depth,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
