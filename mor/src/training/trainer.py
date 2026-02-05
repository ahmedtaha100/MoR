import os

import math

import time

from typing import Optional, Dict, Any

from dataclasses import dataclass



import torch

import torch.nn as nn

from torch.utils.data import DataLoader

from torch.optim import AdamW

from torch.optim.lr_scheduler import LambdaLR





@dataclass

class TrainingConfig:

    learning_rate: float = 3e-4

    weight_decay: float = 0.1

    beta1: float = 0.9

    beta2: float = 0.95

    eps: float = 1e-8



    warmup_steps: int = 2000

    max_steps: int = 100000

    min_lr_ratio: float = 0.1



    lr_scheduler: str = "cosine"

    warmup_ratio: float = 0.02

    cooldown_ratio: float = 0.2



    batch_size: int = 32

    gradient_accumulation_steps: int = 1

    max_grad_norm: float = 1.0



    mixed_precision: str = "bf16"



    log_interval: int = 100

    eval_interval: int = 1000

    save_interval: int = 5000



    output_dir: str = "outputs"

    wandb_project: Optional[str] = None

    wandb_run_name: Optional[str] = None



    seed: int = 42





class MoRTrainer:

    def __init__(

        self,

        model: nn.Module,

        config: TrainingConfig,

        train_dataloader: DataLoader,

        eval_dataloader: Optional[DataLoader] = None,

    ):

        self.model = model

        self.config = config

        self.train_dataloader = train_dataloader

        self.eval_dataloader = eval_dataloader



        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self.model.to(self.device)



        self.optimizer = self._create_optimizer()



        self.scheduler = self._create_scheduler()



        self.scaler = self._create_scaler()

        self.autocast_dtype = self._get_autocast_dtype()



        self.global_step = 0

        self.epoch = 0



        self.wandb_run = None

        if config.wandb_project:

            self._setup_wandb()



    def _create_optimizer(self) -> AdamW:

        decay_params = []

        no_decay_params = []



        for name, param in self.model.named_parameters():

            if not param.requires_grad:

                continue

            if "bias" in name or "norm" in name or "layernorm" in name:

                no_decay_params.append(param)

            else:

                decay_params.append(param)



        optimizer_groups = [

            {"params": decay_params, "weight_decay": self.config.weight_decay},

            {"params": no_decay_params, "weight_decay": 0.0},

        ]



        return AdamW(

            optimizer_groups,

            lr=self.config.learning_rate,

            betas=(self.config.beta1, self.config.beta2),

            eps=self.config.eps,

        )



    def _create_scheduler(self) -> LambdaLR:

        if self.config.lr_scheduler == "trapezoid":

            return self._create_trapezoid_scheduler()

        else:

            return self._create_cosine_scheduler()



    def _create_cosine_scheduler(self) -> LambdaLR:

        def lr_lambda(step: int) -> float:

            if step < self.config.warmup_steps:

                return step / max(1, self.config.warmup_steps)



            progress = (step - self.config.warmup_steps) / max(

                1, self.config.max_steps - self.config.warmup_steps

            )

            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))



            return self.config.min_lr_ratio + (1 - self.config.min_lr_ratio) * cosine_decay



        return LambdaLR(self.optimizer, lr_lambda)



    def _create_trapezoid_scheduler(self) -> LambdaLR:

        warmup_steps = int(self.config.max_steps * self.config.warmup_ratio)

        cooldown_steps = int(self.config.max_steps * self.config.cooldown_ratio)

        stable_steps = self.config.max_steps - warmup_steps - cooldown_steps



        def lr_lambda(step: int) -> float:

            if step < warmup_steps:

                return step / max(1, warmup_steps)

            elif step < warmup_steps + stable_steps:

                return 1.0

            else:

                cooldown_progress = (step - warmup_steps - stable_steps) / max(1, cooldown_steps)

                return max(self.config.min_lr_ratio, 1.0 - cooldown_progress * (1.0 - self.config.min_lr_ratio))



        return LambdaLR(self.optimizer, lr_lambda)



    def _create_scaler(self) -> Optional[torch.amp.GradScaler]:

        if self.config.mixed_precision == "fp16":

            return torch.amp.GradScaler('cuda')

        return None



    def _get_autocast_dtype(self) -> torch.dtype:

        if self.config.mixed_precision == "bf16":

            return torch.bfloat16

        elif self.config.mixed_precision == "fp16":

            return torch.float16

        return torch.float32



    def _setup_wandb(self):

        try:

            import wandb



            self.wandb_run = wandb.init(

                project=self.config.wandb_project,

                name=self.config.wandb_run_name,

                config=vars(self.config),

            )

        except ImportError:

            print("WandB not installed. Skipping WandB logging.")



    def train(self) -> Dict[str, float]:

        self.model.train()

        total_loss = 0.0

        total_aux_loss = 0.0

        num_steps = 0



        data_iter = iter(self.train_dataloader)

        start_time = time.time()



        while self.global_step < self.config.max_steps:

            try:

                batch = next(data_iter)

            except StopIteration:

                data_iter = iter(self.train_dataloader)

                batch = next(data_iter)

                self.epoch += 1



            batch = {k: v.to(self.device) for k, v in batch.items()}



            if hasattr(self.model, 'model') and hasattr(self.model.model, 'recursive_block'):

                router = self.model.model.recursive_block.router

                if hasattr(router, 'set_training_step'):

                    router.set_training_step(self.global_step)



            with torch.amp.autocast('cuda', dtype=self.autocast_dtype, enabled=self.device.type == "cuda"):

                outputs = self.model(

                    input_ids=batch["input_ids"],

                    attention_mask=batch.get("attention_mask"),

                    labels=batch["labels"],

                )

                loss = outputs.loss / self.config.gradient_accumulation_steps



            if self.scaler is not None:

                self.scaler.scale(loss).backward()

            else:

                loss.backward()



            total_loss += loss.item() * self.config.gradient_accumulation_steps

            if hasattr(outputs, "aux_loss") and outputs.aux_loss is not None:

                total_aux_loss += outputs.aux_loss.item()

            num_steps += 1



            if num_steps % self.config.gradient_accumulation_steps == 0:

                if self.scaler is not None:

                    self.scaler.unscale_(self.optimizer)



                grad_norm = torch.nn.utils.clip_grad_norm_(

                    self.model.parameters(), self.config.max_grad_norm

                )



                if self.scaler is not None:

                    self.scaler.step(self.optimizer)

                    self.scaler.update()

                else:

                    self.optimizer.step()



                self.scheduler.step()

                self.optimizer.zero_grad()

                self.global_step += 1



                if self.global_step % self.config.log_interval == 0:

                    avg_loss = total_loss / num_steps

                    avg_aux_loss = total_aux_loss / num_steps if total_aux_loss > 0 else 0

                    lr = self.scheduler.get_last_lr()[0]

                    elapsed = time.time() - start_time

                    tokens_per_sec = (

                        num_steps

                        * self.config.batch_size

                        * batch["input_ids"].shape[1]

                        / elapsed

                    )



                    print(

                        f"Step {self.global_step:6d} | "

                        f"Loss: {avg_loss:.4f} | "

                        f"Aux Loss: {avg_aux_loss:.4f} | "

                        f"LR: {lr:.2e} | "

                        f"Grad Norm: {grad_norm:.2f} | "

                        f"Tok/s: {tokens_per_sec:.0f}"

                    )



                    if self.wandb_run:

                        import wandb



                        wandb.log(

                            {

                                "train/loss": avg_loss,

                                "train/aux_loss": avg_aux_loss,

                                "train/learning_rate": lr,

                                "train/grad_norm": grad_norm,

                                "train/tokens_per_sec": tokens_per_sec,

                            },

                            step=self.global_step,

                        )



                    total_loss = 0.0

                    total_aux_loss = 0.0

                    num_steps = 0

                    start_time = time.time()



                if (

                    self.eval_dataloader is not None

                    and self.global_step % self.config.eval_interval == 0

                ):

                    eval_metrics = self.evaluate()

                    print(

                        f"Eval Step {self.global_step:6d} | "

                        f"Loss: {eval_metrics['eval_loss']:.4f} | "

                        f"PPL: {eval_metrics['perplexity']:.2f}"

                    )



                    if self.wandb_run:

                        import wandb



                        wandb.log(eval_metrics, step=self.global_step)



                    self.model.train()



                if self.global_step % self.config.save_interval == 0:

                    self.save_checkpoint()



        return {"final_loss": total_loss / max(1, num_steps)}



    @torch.no_grad()

    def evaluate(self) -> Dict[str, float]:

        self.model.eval()

        total_loss = 0.0

        total_tokens = 0



        for batch in self.eval_dataloader:

            batch = {k: v.to(self.device) for k, v in batch.items()}



            with torch.amp.autocast('cuda', dtype=self.autocast_dtype, enabled=self.device.type == "cuda"):

                outputs = self.model(

                    input_ids=batch["input_ids"],

                    attention_mask=batch.get("attention_mask"),

                    labels=batch["labels"],

                )



            num_tokens = (batch["labels"] != -100).sum().item()

            total_loss += outputs.loss.item() * num_tokens

            total_tokens += num_tokens



        avg_loss = total_loss / max(1, total_tokens)

        perplexity = math.exp(min(avg_loss, 100))



        return {

            "eval_loss": avg_loss,

            "perplexity": perplexity,

        }



    def save_checkpoint(self, path: Optional[str] = None):

        if path is None:

            os.makedirs(self.config.output_dir, exist_ok=True)

            path = os.path.join(

                self.config.output_dir, f"checkpoint-{self.global_step}.pt"

            )



        checkpoint = {

            "model_state_dict": self.model.state_dict(),

            "optimizer_state_dict": self.optimizer.state_dict(),

            "scheduler_state_dict": self.scheduler.state_dict(),

            "global_step": self.global_step,

            "epoch": self.epoch,

            "config": vars(self.config),

        }



        if self.scaler is not None:

            checkpoint["scaler_state_dict"] = self.scaler.state_dict()



        torch.save(checkpoint, path)

        print(f"Saved checkpoint to {path}")



    def load_checkpoint(self, path: str):

        checkpoint = torch.load(path, map_location=self.device)



        self.model.load_state_dict(checkpoint["model_state_dict"])

        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.global_step = checkpoint["global_step"]

        self.epoch = checkpoint["epoch"]



        if self.scaler is not None and "scaler_state_dict" in checkpoint:

            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])



        print(f"Loaded checkpoint from {path}")

