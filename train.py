from __future__ import annotations

import argparse
from collections.abc import Callable
import json
from pathlib import Path
import random
import re
from typing import Any, Iterator, Optional
import wandb
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    LlamaForCausalLM,
    GenerationConfig,
)
from loss import approx_kl_divergence, GRPOLoss
from replay_buffer import ReplayBuffer, Experience, join_experience_batch
import platform
import time


def load_model(
    model_name_or_path: str,
    trust_remote_code: bool = False,
    bf16: bool = True,
    device_map=None,
) -> tuple[LlamaForCausalLM, PreTrainedTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = LlamaForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16 if bf16 else "auto",
        device_map=device_map,
    )
    return model, tokenizer


# DeepSeek Zero system prompt
system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>
"""


@torch.no_grad()
def rollout(
    model: LlamaForCausalLM,
    tokenizer: PreTrainedTokenizer,
    task: str,
    oracle_answer: str,
    num_rollouts: int,
    max_length: int = 1024,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:

    model.eval()

    # 1. format prompt
    chat_messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": task,
        },
    ]
    chat_prompt = tokenizer.apply_chat_template(
        chat_messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer(
        [chat_prompt],
        return_tensors="pt",
        padding=True,
        padding_side="left",
        return_attention_mask=True,
    ).to("cuda")

    # duplicate prompt num_rollouts times
    model_inputs["attention_mask"] = model_inputs["attention_mask"].repeat(
        num_rollouts, 1
    )

    input_ids = model_inputs["input_ids"].repeat(num_rollouts, 1)
    model_inputs["input_ids"] = input_ids

    # 2. sample completions
    pad_token_id = tokenizer.eos_token_id
    generation_config = GenerationConfig(
        do_sample=True,
        top_p=top_p,
        temperature=temperature,
        max_length=max_length,
        pad_token_id=pad_token_id,
    )
    sequence_ids = model.generate(**model_inputs, generation_config=generation_config)
    completions = tokenizer.batch_decode(
        sequence_ids[:, input_ids.shape[1] :], skip_special_tokens=True
    )

    action_mask = torch.zeros_like(sequence_ids, dtype=torch.bool)
    action_mask[:, input_ids.shape[1] :] = True
    action_mask[sequence_ids == pad_token_id] = False
    action_mask = action_mask[:, 1:]

    # 3. determine rewards, rollout 12 times
    returns = torch.zeros(num_rollouts, 1, dtype=torch.float)
    for i, completion in enumerate(completions):
        # search answer tag
        answer_match = re.search(
            r"<answer>(.*?)</answer>",
            completion,
            flags=re.DOTALL,
        )

        think_match = re.search(
            r"<think>(.*?)</think>",
            completion,
            flags=re.DOTALL,
        )

        answer = answer_match.group(1) if answer_match else None
        reward = 0
        if answer is not None:
            if answer == oracle_answer:
                reward = 1.0
            elif oracle_answer in answer:
                reward = 0.5
            else:
                reward = 0.01

        returns[i] = reward

    return sequence_ids, returns.to(sequence_ids.device), action_mask, completions, answer, think_match


def init_rng(seed: int) -> torch.Generator:
    random.seed(seed)
    return torch.manual_seed(seed)


# baseline
def group_advantages_baseline(returns: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (returns - returns.mean()) / (returns.std() + eps)


class KalmanFilter:
    def __init__(self,
                 process_var: float = 1e-5,
                 meas_var: float = 1e-2,
                 init_mean: float = 0.0,
                 init_var: float = 1.0):
        self.Q = process_var
        self.R = meas_var
        # State: mean estimate (x) and variance estimate (P)
        self.x = torch.tensor(init_mean)
        self.P = torch.tensor(init_var)

    def update(self, z: torch.Tensor):
        # Predict step
        P_pred = self.P + self.Q
        # Kalman gain
        K = P_pred / (P_pred + self.R)
        # Update step
        self.x = self.x + K * (z - self.x)
        self.P = (1 - K) * P_pred
        return self.x, self.P


def group_advantages(returns: torch.Tensor,
                     process_var: float = 1e-5,
                     meas_var: float = 1e-2,
                     eps: float = 1e-8) -> torch.Tensor:
    # Initialize the filter
    kf = KalmanFilter(process_var=process_var, meas_var=meas_var)
    flat_returns = returns.flatten()
    advantages = []

    for r in flat_returns:
        mean_est, var_est = kf.update(r)
        # compute advantage and normalize by estimated std
        adv = (r - mean_est) / (torch.sqrt(var_est) + eps)
        advantages.append(adv)

    # Restore original shape
    return torch.stack(advantages).view_as(returns)

def sequence_log_probs_from_logits(
    logits: torch.tensor, output_ids: torch.tensor
) -> torch.Tensor:
    log_prob = F.log_softmax(logits, dim=-1)
    return log_prob.gather(dim=-1, index=output_ids.unsqueeze(-1)).squeeze(-1)


def sequences_log_probs(
    model: LlamaForCausalLM,
    sequence_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(mask=(attention_mask == 0), value=1)
    output = model.forward(
        input_ids=sequence_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    )
    logits = output["logits"]
    log_probs = sequence_log_probs_from_logits(
        logits=logits[:, :-1].to(torch.float32),
        output_ids=sequence_ids[:, 1:],
    )
    return log_probs


def read_jsonl(file_name: str | Path) -> Iterator:
    file_path = Path(file_name)
    with file_path.open(mode="r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def read_prompts(
    file_name: str,
    predicate: Optional[Callable[[Any], bool]] = None,
    max_rows: Optional[int] = None,
) -> list:
    rows = []
    for x in read_jsonl(file_name):
        if predicate is None or predicate(x):
            rows.append(x)
        if max_rows is not None and len(rows) >= max_rows:
            break
    return rows


def get_arguments():
    """
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_name", type=str, default='debug')
    return parser


def main():
    parser = get_arguments()
    print(parser)
    args = parser.parse_args()

    seed = 42
    device_index = 0
    process_var = 1e-5
    meas_var = 1e-2
    question_prefix = 'Please calculate the following expression: '

    # dataset related
    dataset_path = "data/MathTasks/MathTasks_train.jsonl"
    data_predicate = lambda x: len(x["question"]) < 128 and x["num_terms"] <= 5 and x["num_digits"] <= 5

    wandb_project = "krpo"
    model_name = "../../data/meta-llama/Llama-3.2-1B-Instruct"

    checkpoint_path = Path("./output/" + args.save_name)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    checkpoint_interval = 100
    train_batch_size = 16
    lr = 5e-6
    kl_weight = 0.01
    clip_eps = 0.2

    group_size = 12
    rollouts_per_step = 32
    epochs_per_step = 1
    max_norm = 1.0  # gradient clipping

    # rollout params
    max_length = 1024
    top_p = 1.0
    temperature = 1.0

    device = torch.device("cuda", device_index)
    cpu_device = torch.device("cpu")
    init_rng(seed)

    reference_model, _ = load_model(model_name, device_map=device)  # Llama model
    model, tokenizer = load_model(model_name, device_map=device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    reference_model.eval()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    pad_token_id = tokenizer.eos_token_id

    prompts = read_prompts(
        dataset_path,
        predicate=data_predicate,
        max_rows=64 * 1024,
    )
    print(f"found {len(prompts)} matching prompts")
    prompt_loader = DataLoader(
        prompts,
        batch_size=rollouts_per_step,
        shuffle=True,
        drop_last=True,
        pin_memory=False,
    )

    replay_buffer = ReplayBuffer()
    objective = GRPOLoss(clip_eps=clip_eps, kl_weight=kl_weight)

    if wandb_project is None:
        wandb.init(mode="disabled")
    else:
        wandb.init(project=wandb_project)

    num_epochs = 1
    for epoch in range(num_epochs):
        for k, prompt_batch in enumerate(prompt_loader):
            rollout_returns = []

            replay_buffer.clear()

            questions = prompt_batch["question"]
            answers = prompt_batch["answer"]

            with torch.no_grad():
                for q, a in zip(questions, answers):
                    # q = question_prefix + q
                    sequence_ids, returns, action_mask, completions, model_answer, model_think \
                        = rollout(model, tokenizer, q, a, num_rollouts=group_size, max_length=max_length, temperature=temperature, top_p=top_p)

                    print(
                        f"rollout q='{q}', a='{a}', returns={returns.sum().item():.2f}, replay_buffer_size={len(replay_buffer)}, sequence_ids={sequence_ids.shape}"
                    )
                    rollout_returns.append(returns.cpu())

                    # get advantages via group policy
                    advantages = group_advantages(returns, process_var=process_var, meas_var=meas_var)
                    attention_mask = sequence_ids != pad_token_id

                    log_probs = sequences_log_probs(
                        model=model,
                        sequence_ids=sequence_ids,
                        attention_mask=attention_mask,
                    )
                    log_probs_ref = sequences_log_probs(
                        model=reference_model,
                        sequence_ids=sequence_ids,
                        attention_mask=attention_mask,
                    )
                    kl = approx_kl_divergence(
                        log_probs=log_probs,
                        log_probs_ref=log_probs_ref,
                        action_mask=action_mask,
                    )

                    experience = Experience(
                        sequences=sequence_ids,
                        action_log_probs=log_probs,
                        log_probs_ref=log_probs_ref,
                        returns=returns,
                        advantages=advantages,
                        attention_mask=attention_mask,
                        action_mask=action_mask,
                        kl=kl,
                    )
                    replay_buffer.append(experience.to(cpu_device))

            torch.cuda.empty_cache()
            episode_return_sum = torch.stack(rollout_returns).sum()
            print(f"returns of step {k}: {episode_return_sum:.4f}")
            wandb.log({"returns": episode_return_sum})

            experience_sampler = DataLoader(
                replay_buffer,
                batch_size=train_batch_size,
                shuffle=True,
                drop_last=True,
                collate_fn=join_experience_batch,
            )

            for step_epoch in range(epochs_per_step):
                model.train()

                for exp in experience_sampler:
                    exp: Experience

                    exp = exp.to(device)

                    optimizer.zero_grad()

                    log_probs = sequences_log_probs(
                        model, sequence_ids=exp.sequences, attention_mask=exp.attention_mask
                    )

                    loss, kl = objective(log_probs=log_probs, experience=exp)

                    if not loss.isfinite():
                        print(f"Loss not finite, skipping backward, loss={loss}")
                        print(f"experience.advantages={experience.advantages}")
                        continue

                    loss.backward()
                    grad_norm = clip_grad_norm_(model.parameters(), max_norm=max_norm)
                    print(f"{step_epoch}: kl={kl: .4f}, grad_norm={grad_norm: .4f}")
                    wandb.log({"kl": kl, "grad_norm": grad_norm})

                    optimizer.step()

            if (
                checkpoint_path is not None
                and checkpoint_interval is not None
                and (epoch*k + 1) % checkpoint_interval == 0
            ):
                model.save_pretrained(checkpoint_path / f"step_{epoch*k + 1}")

    if checkpoint_path is not None:
        model.save_pretrained(checkpoint_path / f"step_{k}")


if __name__ == "__main__":
    main()
