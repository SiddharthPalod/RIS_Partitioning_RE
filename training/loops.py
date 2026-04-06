"""DQN / DDPG training loops and saving artifacts under ``results/``."""

from __future__ import annotations

import json
import random
from collections import deque
from typing import Any

import numpy as np
import torch
import torch.nn.utils as nn_utils

from agents.ddpg import Actor, Critic, ReplayBuffer
from agents.dqn import DQN
from config.paths import (
    DEFAULT_LOG_PREFIX_DDPG,
    DEFAULT_LOG_PREFIX_DQN,
    RESULTS_DIR,
    ensure_results_dir,
)
from env.simple_isac_env import SimpleISACRISEnv
from env.isac_env import ISACRISEnv

_SUMMARY_TAIL = 500


def _save_numpy_logs(prefix: str, **arrays: np.ndarray) -> None:
    ensure_results_dir()
    base = RESULTS_DIR
    for name, arr in arrays.items():
        np.save(base / f"{prefix}_{name}.npy", arr)


def _env_meta(env_kwargs: dict[str, Any] | None) -> dict[str, Any]:
    env_kwargs = env_kwargs or {}
    return {
        "L": env_kwargs.get("L"),
        "max_steps": env_kwargs.get("max_steps"),
    }


def _write_summary_json(
    prefix: str,
    algorithm: str,
    total_steps: int,
    env_kwargs: dict[str, Any],
    rewards: np.ndarray,
    losses: np.ndarray,
    jfi: np.ndarray,
    asir: np.ndarray,
    partitions: np.ndarray,
    r1: np.ndarray,
    r2: np.ndarray,
    rsum: np.ndarray,
    ravg: np.ndarray,
    rsum_passive: np.ndarray | None = None,
) -> None:
    ensure_results_dir()
    n = len(rewards)
    N = min(_SUMMARY_TAIL, n)
    tail = slice(-N, None)

    summary: dict[str, Any] = {
        "algorithm": algorithm,
        "log_prefix": prefix,
        "total_steps": total_steps,
        "w_c": env_kwargs.get("w_c", env_kwargs.get("lambda_1")),
        "w_r": env_kwargs.get("w_r", env_kwargs.get("lambda_2")),
        "w_f": env_kwargs.get("w_f"),
        "w_qos_f": env_kwargs.get("w_qos_f"),
        "w_qos_s": env_kwargs.get("w_qos_s"),
        "rf_min": env_kwargs.get("rf_min"),
        "asir_min": env_kwargs.get("asir_min"),
        "max_rsum": env_kwargs.get("max_rsum"),
        "max_asir": env_kwargs.get("max_asir"),
        **_env_meta(env_kwargs),
        "reward_formula": "w_c * normalize(R_n + R_f) + w_r * normalize(ASIR) + w_f * JFI(R_n, R_f) - w_qos_f*max(0, rf_min-R_f) - w_qos_s*max(0, asir_min-ASIR)",
        "summary_window": N,
        "reward_mean": float(rewards[tail].mean()),
        "reward_std": float(rewards[tail].std()),
        "jfi_mean": float(jfi[tail].mean()),
        "jfi_std": float(jfi[tail].std()),
        "asir_mean": float(asir[tail].mean()),
        "asir_std": float(asir[tail].std()),
        "r_n_mean": float(r1[tail].mean()),
        "r_f_mean": float(r2[tail].mean()),
        "rsum_mean": float(rsum[tail].mean()),
        "ravg_mean": float(ravg[tail].mean()),
        "loss_mean": float(losses[tail].mean()),
        "a_n_mean": float(partitions[tail, 0].mean()),
        "a_f_mean": float(partitions[tail, 1].mean()),
        "a_t_mean": float(partitions[tail, 2].mean()),
    }
    if rsum_passive is not None and len(rsum_passive) == n:
        summary["rsum_passive_mean"] = float(rsum_passive[tail].mean())
        summary["rsum_passive_std"] = float(rsum_passive[tail].std())
    if n >= 1000:
        summary["reward_first_500"] = float(rewards[:500].mean())
        summary["reward_last_500"] = float(rewards[-500:].mean())

    out = RESULTS_DIR / f"{prefix}_summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def _finalize_run(
    log_prefix: str,
    algorithm: str,
    total_steps: int,
    env_kwargs: dict[str, Any] | None,
    rewards_hist: list,
    losses_hist: list,
    actions_hist: list,
    jfi_hist: list,
    asir_hist: list,
    partition_hist: list,
    r1_hist: list,
    r2_hist: list,
    rsum_hist: list,
    ravg_hist: list,
    r1_passive_hist: list | None = None,
    r2_passive_hist: list | None = None,
    rsum_passive_hist: list | None = None,
) -> None:
    env_kwargs = env_kwargs or {}
    r = np.array(rewards_hist, dtype=np.float32)
    lo = np.array(losses_hist, dtype=np.float32)
    j = np.array(jfi_hist, dtype=np.float32)
    a = np.array(asir_hist, dtype=np.float32)
    p = np.array(partition_hist, dtype=np.float32)
    r1 = np.array(r1_hist, dtype=np.float32)
    r2 = np.array(r2_hist, dtype=np.float32)
    rs = np.array(rsum_hist, dtype=np.float32)
    ra = np.array(ravg_hist, dtype=np.float32)
    act = np.array(actions_hist)

    logs: dict[str, np.ndarray] = {
        "rewards": r,
        "losses": lo,
        "actions": act,
        "jfi": j,
        "asir": a,
        "partitions": p,
        "r1": r1,
        "r2": r2,
        "rsum": rs,
        "ravg": ra,
    }
    rsp: np.ndarray | None = None
    if rsum_passive_hist is not None and len(rsum_passive_hist) == len(r):
        rsp = np.array(rsum_passive_hist, dtype=np.float32)
        logs["rsum_passive"] = rsp
    if r1_passive_hist is not None and len(r1_passive_hist) == len(r):
        logs["r1_passive"] = np.array(r1_passive_hist, dtype=np.float32)
    if r2_passive_hist is not None and len(r2_passive_hist) == len(r):
        logs["r2_passive"] = np.array(r2_passive_hist, dtype=np.float32)

    _save_numpy_logs(log_prefix, **logs)
    _write_summary_json(
        log_prefix,
        algorithm,
        total_steps,
        env_kwargs,
        r,
        lo,
        j,
        a,
        p,
        r1,
        r2,
        rs,
        ra,
        rsp,
    )


def run_dqn(
    total_steps: int = 8000,
    log_prefix: str = DEFAULT_LOG_PREFIX_DQN,
    env_kwargs: dict[str, Any] | None = None,
) -> None:
    # env = SimpleISACRISEnv(**(env_kwargs or {}))
    env = ISACRISEnv(**(env_kwargs or {}))
    state_dim = env.state_dim
    n_actions = env.n_actions

    model = DQN(state_dim=state_dim, n_actions=n_actions)
    target_model = DQN(state_dim=state_dim, n_actions=n_actions)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.05
    warmup_steps = 500 

    replay_buffer: deque = deque(maxlen=10_000)
    batch_size = 64
    target_update_freq = 100

    def preprocess(state: np.ndarray) -> torch.Tensor:
        return torch.tensor(state, dtype=torch.float32)

    state = preprocess(env.reset())
    episode = 0
    global_step = 0
    last_loss = torch.tensor(0.0)

    rewards_hist: list[float] = []
    losses_hist: list[float] = []
    actions_hist: list[int] = []
    jfi_hist: list[float] = []
    asir_hist: list[float] = []
    partition_hist: list[list[float]] = []
    r1_hist: list[float] = []
    r2_hist: list[float] = []
    rsum_hist: list[float] = []
    ravg_hist: list[float] = []
    r1_passive_hist: list[float] = []
    r2_passive_hist: list[float] = []
    rsum_passive_hist: list[float] = []

    for step in range(total_steps):
        q_values = model(state)

        if random.random() < epsilon:
            action = random.randrange(n_actions)
        else:
            action = int(torch.argmax(q_values).item())

        next_state_np, reward, done, info = env.step(action)
        next_state = preprocess(next_state_np)

        replay_buffer.append(
            (state.detach(), action, float(reward), next_state.detach(), bool(done))
        )
        state = next_state

        if len(replay_buffer) >= batch_size and global_step > warmup_steps:
            batch = random.sample(replay_buffer, batch_size)
            states_b, actions_b, rewards_b, next_states_b, dones_b = zip(*batch)

            states_b = torch.stack(states_b)
            next_states_b = torch.stack(next_states_b)
            actions_b = torch.tensor(actions_b, dtype=torch.long)
            rewards_b = torch.tensor(rewards_b, dtype=torch.float32)
            dones_b = torch.tensor(dones_b, dtype=torch.bool)

            q_vals = model(states_b)
            q_sa = q_vals.gather(1, actions_b.view(-1, 1)).squeeze(1)

            # Double-DQN target: action from online net, value from target net.
            with torch.no_grad():
                next_actions = model(next_states_b).argmax(dim=1, keepdim=True)
                next_q = target_model(next_states_b).gather(1, next_actions).squeeze(1)
                target_q = rewards_b + gamma * next_q * (~dones_b)

            loss = torch.nn.functional.smooth_l1_loss(q_sa, target_q)
            optimizer.zero_grad()
            loss.backward()
            nn_utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            last_loss = loss.detach()

            if global_step > 0 and global_step % target_update_freq == 0:
                target_model.load_state_dict(model.state_dict())

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        rewards_hist.append(float(reward))
        losses_hist.append(float(last_loss))
        actions_hist.append(int(action))
        jfi_hist.append(float(info["jfi"]))
        asir_hist.append(float(info["asir"]))
        partition_hist.append([float(info["a_n"]), float(info["a_f"]), float(info["a_t"])])
        r1 = float(info["r_n"])
        r2 = float(info["r_f"])
        rsum = r1 + r2
        ravg = 0.5 * rsum
        r1_hist.append(r1)
        r2_hist.append(r2)
        rsum_hist.append(rsum)
        ravg_hist.append(ravg)
        r1_passive_hist.append(float(info["r_n_passive"]))
        r2_passive_hist.append(float(info["r_f_passive"]))
        rsum_passive_hist.append(float(info["rsum_passive"]))

        if step % 200 == 0:
            print(
                f"Step {step:05d} | ep={episode:03d} | eps={epsilon:.3f} | "
                f"action={action} | reward={reward:.4f} | jfi={info['jfi']:.4f} | "
                f"asir={info['asir']:.2f} | "
                f"a=[{info['a_n']:.2f},{info['a_f']:.2f},{info['a_t']:.2f}] | "
                f"loss={float(last_loss):.6f}"
            )

        global_step += 1

        if done:
            episode += 1
            state = preprocess(env.reset())

    _ek = {**(env_kwargs or {}), "L": env.L, "max_steps": env.max_steps}
    _finalize_run(
        log_prefix,
        "dqn",
        total_steps,
        _ek,
        rewards_hist,
        losses_hist,
        actions_hist,
        jfi_hist,
        asir_hist,
        partition_hist,
        r1_hist,
        r2_hist,
        rsum_hist,
        ravg_hist,
        r1_passive_hist,
        r2_passive_hist,
        rsum_passive_hist,
    )

    print(
        f"Training complete ({log_prefix}). Artifacts in {RESULTS_DIR}: "
        f"{log_prefix}_*.npy, {log_prefix}_summary.json"
    )


def run_ddpg(
    total_steps: int = 8000,
    log_prefix: str = DEFAULT_LOG_PREFIX_DDPG,
    env_kwargs: dict[str, Any] | None = None,
    *,
    entropy_coef: float = 0.01,
    noise_scale_start: float = 0.12,
    noise_scale_end: float = 0.02,
    actor_lr: float = 1.5e-4,
    grad_clip_critic: float = 1.0,
    grad_clip_actor: float = 0.5,
) -> None:
    """DDPG on the RIS partition simplex (actor outputs softmax).

    Entropy regularization fights collapse to a corner (e.g. all mass on one user).
    Exploration noise decays so early steps probe the simplex, later steps trust the policy.
    """
    # env = SimpleISACRISEnv(**(env_kwargs or {}))
    env = ISACRISEnv(**(env_kwargs or {}))
    state_dim = env.state_dim
    action_dim = 3

    actor = Actor(state_dim=state_dim, action_dim=action_dim)
    actor_target = Actor(state_dim=state_dim, action_dim=action_dim)
    actor_target.load_state_dict(actor.state_dict())

    critic = Critic(state_dim=state_dim, action_dim=action_dim)
    critic_target = Critic(state_dim=state_dim, action_dim=action_dim)
    critic_target.load_state_dict(critic.state_dict())

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)

    replay_buffer = ReplayBuffer(capacity=20_000)
    batch_size = 64
    gamma = 0.99
    tau = 0.01

    state = env.reset().astype(np.float32)
    episode = 0
    last_actor_loss = 0.0
    last_critic_loss = 0.0
    last_policy_entropy = 0.0

    rewards_hist: list[float] = []
    losses_hist: list[float] = []
    actions_hist: list[np.ndarray] = []
    jfi_hist: list[float] = []
    asir_hist: list[float] = []
    partition_hist: list[list[float]] = []
    r1_hist: list[float] = []
    r2_hist: list[float] = []
    rsum_hist: list[float] = []
    ravg_hist: list[float] = []
    r1_passive_hist: list[float] = []
    r2_passive_hist: list[float] = []
    rsum_passive_hist: list[float] = []

    for step in range(total_steps):
        denom = max(total_steps - 1, 1)
        noise_scale = noise_scale_start + (noise_scale_end - noise_scale_start) * (step / denom)

        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = actor(state_t).squeeze(0).cpu().numpy()

        noise = np.random.dirichlet(np.ones(action_dim)).astype(np.float32)
        action = (1.0 - noise_scale) * action + noise_scale * noise
        action = np.maximum(action, env.min_partition)
        action = action / np.sum(action)

        next_state, reward, done, info = env.step_continuous(action)
        replay_buffer.push(state, action, float(reward), next_state, bool(done))
        state = next_state.astype(np.float32)

        if len(replay_buffer) >= batch_size:
            states_b, actions_b, rewards_b, next_states_b, dones_b = replay_buffer.sample(batch_size)
            states_b = torch.tensor(states_b, dtype=torch.float32)
            actions_b = torch.tensor(actions_b, dtype=torch.float32)
            rewards_b = torch.tensor(rewards_b, dtype=torch.float32).unsqueeze(1)
            next_states_b = torch.tensor(next_states_b, dtype=torch.float32)
            dones_b = torch.tensor(dones_b, dtype=torch.float32).unsqueeze(1)

            with torch.no_grad():
                next_actions_b = actor_target(next_states_b)
                target_q = rewards_b + gamma * (1.0 - dones_b) * critic_target(next_states_b, next_actions_b)

            current_q = critic(states_b, actions_b)
            critic_loss = torch.mean((current_q - target_q) ** 2)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            if grad_clip_critic > 0:
                nn_utils.clip_grad_norm_(critic.parameters(), grad_clip_critic)
            critic_optimizer.step()
            last_critic_loss = float(critic_loss.detach())

            probs = actor(states_b)
            policy_entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
            q_pi = critic(states_b, probs).mean()
            # Maximize Q + entropy_coef * H(pi)  <=>  minimize -(Q + coef * H)
            actor_loss = -(q_pi + entropy_coef * policy_entropy)
            actor_optimizer.zero_grad()
            actor_loss.backward()
            if grad_clip_actor > 0:
                nn_utils.clip_grad_norm_(actor.parameters(), grad_clip_actor)
            actor_optimizer.step()
            last_actor_loss = float(actor_loss.detach())
            last_policy_entropy = float(policy_entropy.detach())

            with torch.no_grad():
                for tgt, src in zip(actor_target.parameters(), actor.parameters()):
                    tgt.data.copy_(tau * src.data + (1.0 - tau) * tgt.data)
                for tgt, src in zip(critic_target.parameters(), critic.parameters()):
                    tgt.data.copy_(tau * src.data + (1.0 - tau) * tgt.data)

        rewards_hist.append(float(reward))
        losses_hist.append(float(last_critic_loss))
        actions_hist.append(action.astype(np.float32))
        jfi_hist.append(float(info["jfi"]))
        asir_hist.append(float(info["asir"]))
        partition_hist.append([float(info["a_n"]), float(info["a_f"]), float(info["a_t"])])
        r1 = float(info["r_n"])
        r2 = float(info["r_f"])
        rsum = r1 + r2
        ravg = 0.5 * rsum
        r1_hist.append(r1)
        r2_hist.append(r2)
        rsum_hist.append(rsum)
        ravg_hist.append(ravg)
        r1_passive_hist.append(float(info["r_n_passive"]))
        r2_passive_hist.append(float(info["r_f_passive"]))
        rsum_passive_hist.append(float(info["rsum_passive"]))

        if step % 20 == 0:
            print(
                f"Step {step:05d} | ep={episode:03d} | ns={noise_scale:.3f} | reward={reward:.4f} | "
                f"jfi={info['jfi']:.4f} | asir={info['asir']:.2f} | "
                f"rsn={info.get('rsum_norm', 0.0):.3f} | asn={info.get('asir_norm', 0.0):.3f} | "
                f"a=[{info['a_n']:.2f},{info['a_f']:.2f},{info['a_t']:.2f}] | "
                f"critic_loss={last_critic_loss:.6f} | actor_loss={last_actor_loss:.6f} | "
                f"H_pi={last_policy_entropy:.3f}"
            )

        if done:
            episode += 1
            state = env.reset().astype(np.float32)

    _ek = {**(env_kwargs or {}), "L": env.L, "max_steps": env.max_steps}
    _finalize_run(
        log_prefix,
        "ddpg",
        total_steps,
        _ek,
        rewards_hist,
        losses_hist,
        actions_hist,
        jfi_hist,
        asir_hist,
        partition_hist,
        r1_hist,
        r2_hist,
        rsum_hist,
        ravg_hist,
        r1_passive_hist,
        r2_passive_hist,
        rsum_passive_hist,
    )

    print(
        f"Training complete ({log_prefix}). Artifacts in {RESULTS_DIR}: "
        f"{log_prefix}_*.npy, {log_prefix}_summary.json"
    )
