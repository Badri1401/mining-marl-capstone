"""
MADDPG V3 - REWARD+SMOOTH TRAINING
File: train_maddpg_v3.py

Canonical mode: --reward-mode v2_baseline --curriculum-mode v3_smooth
  - V3 smooth curriculum (8 phases, unified 0.75m threshold)
    - Goal-first reward hierarchy with sparse shaping
  - 90% mastery (ind + team), no episode cap
  - 40% of catastrophe (100+ collision) episodes kept in replay buffer
    - One-goal and zero-goal timeout penalties injected into replay buffer terminal reward
  - Live Med1 gate metrics logged every 500 episodes
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
from typing import Dict, Tuple, Optional, List
import time
import os
from datetime import datetime
import sys
import json
import csv
import matplotlib.pyplot as plt
import copy
import random
from time import perf_counter
from functools import partial

from environment_maddpg_v3 import MiningEnvironmentMADDPG_V3
from parallel_env_manager import ParallelEnvManager, EpisodeCollector


# ==================== RESULTS DIRECTORY ====================
def setup_results_directory(timestamp: str, tag: str = "") -> str:
    suffix = f"_{tag}" if tag else ""
    results_dir = f"results/maddpg_v3_{timestamp}{suffix}"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(f"{results_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{results_dir}/graphs", exist_ok=True)
    return results_dir


# ==================== PHASE TABLES ====================
# These must stay in sync with environment_maddpg_v3.py curriculum dicts

PHASE_LIST_V2 = ["Easy1", "Easy2", "Easy3", "Med1", "Med2a", "Med2b", "Hard1", "Hard2", "VHard"]
PHASE_LIST_V3 = ["Easy1", "Easy2", "Easy3", "Med1", "Med2", "Hard1", "Hard2", "VHard"]

# Advancement configs
PHASE_CONFIG_V2 = {
    # 90% mastery, NO episode cap — mastery-only advancement
    "Easy1":  {"min": 400,  "threshold": 0.90, "max_episodes": 999999},
    "Easy2":  {"min": 500,  "threshold": 0.90, "max_episodes": 999999},
    "Easy3":  {"min": 600,  "threshold": 0.90, "max_episodes": 999999},
    "Med1":   {"min": 800,  "threshold": 0.90, "max_episodes": 999999},
    "Med2a":  {"min": 1000, "threshold": 0.90, "max_episodes": 999999},
    "Med2b":  {"min": 1000, "threshold": 0.90, "max_episodes": 999999},
    "Hard1":  {"min": 1500, "threshold": 0.90, "max_episodes": 999999},
    "Hard2":  {"min": 2000, "threshold": 0.90, "max_episodes": 999999},
    "VHard":  {"min": 2500, "threshold": 0.90, "max_episodes": 999999},
}

PHASE_CONFIG_V3 = {
    # 90% mastery, NO episode cap — mastery-only advancement
    "Easy1": {"min": 300,  "threshold": 0.90, "max_episodes": 999999},
    "Easy2": {"min": 400,  "threshold": 0.90, "max_episodes": 999999},
    "Easy3": {"min": 500,  "threshold": 0.90, "max_episodes": 999999},
    "Med1":  {"min": 600,  "threshold": 0.90, "max_episodes": 999999},
    "Med2":  {"min": 800,  "threshold": 0.90, "max_episodes": 999999},
    "Hard1": {"min": 1000, "threshold": 0.90, "max_episodes": 999999},
    "Hard2": {"min": 1500, "threshold": 0.90, "max_episodes": 999999},
    "VHard": {"min": 2000, "threshold": 0.90, "max_episodes": 999999},
}

# Noise schedule per phase
PHASE_NOISE_V2 = {
    "Easy1": 0.20, "Easy2": 0.18, "Easy3": 0.16,
    "Med1": 0.14, "Med2a": 0.12, "Med2b": 0.10,
    "Hard1": 0.08, "Hard2": 0.06, "VHard": 0.05,
}
PHASE_NOISE_V3 = {
    "Easy1": 0.20, "Easy2": 0.18, "Easy3": 0.16,
    "Med1": 0.13, "Med2": 0.10,
    "Hard1": 0.08, "Hard2": 0.06, "VHard": 0.05,
}

# Replay buffer phase-group sampling
PHASE_GROUPS_V2 = {
    "Easy1": ["Easy1", "Easy2", "Easy3"],
    "Easy2": ["Easy1", "Easy2", "Easy3"],
    "Easy3": ["Easy1", "Easy2", "Easy3"],
    "Med1": ["Med1", "Med2a", "Med2b"],
    "Med2a": ["Med1", "Med2a", "Med2b"],
    "Med2b": ["Med1", "Med2a", "Med2b"],
    "Hard1": ["Hard1", "Hard2", "VHard"],
    "Hard2": ["Hard1", "Hard2", "VHard"],
    "VHard": ["Hard1", "Hard2", "VHard"],
}
PHASE_GROUPS_V3 = {
    "Easy1": ["Easy1", "Easy2", "Easy3"],
    "Easy2": ["Easy1", "Easy2", "Easy3"],
    "Easy3": ["Easy1", "Easy2", "Easy3"],
    "Med1": ["Med1", "Med2"],
    "Med2": ["Med1", "Med2"],
    "Hard1": ["Hard1", "Hard2", "VHard"],
    "Hard2": ["Hard1", "Hard2", "VHard"],
    "VHard": ["Hard1", "Hard2", "VHard"],
}

# Phase colors for plotting
PHASE_COLORS_V2 = {
    "Easy1": "#b3e5fc", "Easy2": "#81d4fa", "Easy3": "#4fc3f7",
    "Med1": "#c8e6c9", "Med2a": "#a5d6a7", "Med2b": "#81c784",
    "Hard1": "#ffe0b2", "Hard2": "#ffcc80", "VHard": "#ef9a9a",
}
PHASE_COLORS_V3 = {
    "Easy1": "#b3e5fc", "Easy2": "#81d4fa", "Easy3": "#4fc3f7",
    "Med1": "#c8e6c9", "Med2": "#a5d6a7",
    "Hard1": "#ffe0b2", "Hard2": "#ffcc80", "VHard": "#ef9a9a",
}


# ==================== REPLAY BUFFER (OPTIMIZED) ====================
class ReplayBuffer:
    """Success-weighted replay buffer with O(1) phase-indexed sampling.

    Uses list-backed circular buffer + phase-index dict instead of
    deque + linear scan.  Eliminates the O(200K) list() conversion
    and O(n) phase filtering that was the #1 training bottleneck.
    """

    def __init__(self, max_size: int = 200000, success_pool_ratio: float = 0.2,
                 success_phase_filter: bool = False, phase_groups: Optional[dict] = None):
        self.max_size = max_size
        self.success_pool_size = int(max_size * success_pool_ratio)
        self.main_pool_size = max_size - self.success_pool_size

        # List-backed circular buffers (O(1) random access)
        self._main_buf: List[Dict] = []
        self._main_ptr: int = 0
        self._main_full: bool = False

        self._succ_buf: List[Dict] = []
        self._succ_ptr: int = 0
        self._succ_full: bool = False

        # Phase index: phase_name -> set of valid indices  (O(1) lookup)
        self._main_phase_idx: Dict[str, set] = {}
        self._succ_phase_idx: Dict[str, set] = {}

        self.success_phase_filter = bool(success_phase_filter)
        self.phase_groups = phase_groups or PHASE_GROUPS_V3
        self.current_phase = "Easy1"
        self.rng = np.random.default_rng()

    # ── push helpers ──────────────────────────────────────────────
    def _push_to(self, buf, ptr, capacity, full_flag, phase_idx, transition):
        phase = transition.get('_phase', self.current_phase)
        if full_flag:
            old_phase = buf[ptr].get('_phase', '')
            if old_phase in phase_idx:
                phase_idx[old_phase].discard(ptr)
            buf[ptr] = transition
        else:
            buf.append(transition)
        phase_idx.setdefault(phase, set()).add(ptr)
        ptr += 1
        if ptr >= capacity:
            ptr = 0
            full_flag = True
        return ptr, full_flag

    def push(self, transition: Dict, phase: Optional[str] = None, success: bool = False):
        if phase:
            transition['_phase'] = phase
            self.current_phase = phase
        else:
            transition['_phase'] = self.current_phase
        if success:
            self._succ_ptr, self._succ_full = self._push_to(
                self._succ_buf, self._succ_ptr, self.success_pool_size,
                self._succ_full, self._succ_phase_idx, transition)
        else:
            self._main_ptr, self._main_full = self._push_to(
                self._main_buf, self._main_ptr, self.main_pool_size,
                self._main_full, self._main_phase_idx, transition)

    def set_phase(self, phase: str):
        self.current_phase = phase
        print(f"[BUFFER] Phase: {phase}, main: {len(self._main_buf)}, success: {len(self._succ_buf)}")

    # ── phase-indexed lookup (O(|phases|) not O(buffer_size)) ────
    @staticmethod
    def _get_phase_indices(phase_idx: dict, phases: set) -> list:
        out = []
        for ph in phases:
            s = phase_idx.get(ph)
            if s:
                out.extend(s)
        return out

    def sample(self, batch_size: int, current_phase: Optional[str] = None) -> list:
        if current_phase is None:
            current_phase = self.current_phase
        same_phases = set(self.phase_groups.get(current_phase, [current_phase]))
        samples = []
        succ_len = len(self._succ_buf)
        main_len = len(self._main_buf)

        n_success = min(int(batch_size * 0.3), succ_len)
        if n_success > 0:
            if self.success_phase_filter:
                cands = self._get_phase_indices(self._succ_phase_idx, same_phases)
                if len(cands) >= n_success:
                    chosen = self.rng.choice(len(cands), n_success, replace=False)
                    samples.extend(self._succ_buf[cands[i]] for i in chosen)
                else:
                    idx = self.rng.choice(succ_len, n_success, replace=False)
                    samples.extend(self._succ_buf[i] for i in idx)
            else:
                idx = self.rng.choice(succ_len, n_success, replace=False)
                samples.extend(self._succ_buf[i] for i in idx)

        n_main = batch_size - len(samples)
        if n_main > 0 and main_len > 0:
            same_idx = self._get_phase_indices(self._main_phase_idx, same_phases)
            n_same = min(n_main // 2, len(same_idx))
            n_uniform = n_main - n_same
            if n_same > 0:
                chosen = self.rng.choice(len(same_idx), n_same, replace=False)
                samples.extend(self._main_buf[same_idx[i]] for i in chosen)
            if n_uniform > 0:
                idx = self.rng.choice(main_len, min(n_uniform, main_len), replace=False)
                samples.extend(self._main_buf[i] for i in idx)
        return samples[:batch_size]

    def refresh_success_buffer(self, keep_ratio: float = 0.5):
        if len(self._succ_buf) > 0:
            n_keep = max(1, int(len(self._succ_buf) * keep_ratio))
            if self._succ_full:
                ordered = self._succ_buf[self._succ_ptr:] + self._succ_buf[:self._succ_ptr]
            else:
                ordered = self._succ_buf[:]
            recent = ordered[-n_keep:]
            self._succ_buf = list(recent)
            self._succ_ptr = len(self._succ_buf)
            self._succ_full = False
            self._succ_phase_idx.clear()
            for i, t in enumerate(self._succ_buf):
                self._succ_phase_idx.setdefault(t.get('_phase', ''), set()).add(i)
            print(f"[BUFFER REFRESH] Success buffer: kept {n_keep} recent entries")

    def __len__(self) -> int:
        return len(self._main_buf) + len(self._succ_buf)


# ==================== NETWORKS ====================
class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim), nn.Tanh(),
        )
        # Match V2: small-weight init on final layer to prevent action saturation
        with torch.no_grad():
            last_linear = self.net[-2]
            assert isinstance(last_linear, nn.Linear)
            last_linear.weight.uniform_(-0.003, 0.003)
            last_linear.bias.uniform_(-0.003, 0.003)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class TwinCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, num_agents: int, hidden_dim: int = 384):
        super().__init__()
        input_dim = (obs_dim + action_dim) * num_agents
        self.q1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs_list, actions_list):
        x = torch.cat([torch.cat(obs_list, dim=-1), torch.cat(actions_list, dim=-1)], dim=-1)
        return torch.clamp(self.q1(x), -100, 100), torch.clamp(self.q2(x), -100, 100)

    def q1_forward(self, obs_list, actions_list):
        x = torch.cat([torch.cat(obs_list, dim=-1), torch.cat(actions_list, dim=-1)], dim=-1)
        return torch.clamp(self.q1(x), -100, 100)


Critic = TwinCritic  # alias


class MADDPGAgent:
    def __init__(self, agent_id, obs_dim, action_dim, num_agents,
                 hidden_dim=384, actor_lr=1e-4, critic_lr=3e-4, device='cpu'):
        self.agent_id = agent_id
        self.device = device
        self.actor = Actor(obs_dim, action_dim, hidden_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic = TwinCritic(obs_dim, action_dim, num_agents, hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # torch.compile() — ONLY on GPU. On CPU it adds massive overhead
        # (compilation time + per-call overhead) for small 384-dim models.
        if device == 'cuda':
            try:
                self.actor = torch.compile(self.actor, mode='reduce-overhead')
                self.actor_target = torch.compile(self.actor_target, mode='reduce-overhead')
                self.critic = torch.compile(self.critic, mode='reduce-overhead')
                self.critic_target = torch.compile(self.critic_target, mode='reduce-overhead')
            except Exception:
                pass  # older PyTorch — skip silently

        # Pre-allocate observation buffer for get_action to avoid
        # creating a new tensor every step (~200-500 steps/episode)
        self._obs_buffer = torch.zeros(1, obs_dim, device=device)

    def get_action(self, obs, noise_scale=0.0):
        with torch.no_grad():
            self._obs_buffer[0].copy_(torch.as_tensor(obs, dtype=torch.float32))
            action = self.actor(self._obs_buffer).squeeze(0).cpu().numpy()
        if noise_scale > 0:
            action += np.random.normal(0, noise_scale, size=action.shape)
        return np.clip(action, -1.0, 1.0)

    def soft_update(self, tau=0.005):
        for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)


class MADDPG_V3:
    """MADDPG trainer — identical to V2 except takes phase_groups for buffer."""

    def __init__(self, num_agents, obs_dim, action_dim,
                 hidden_dim=384, actor_lr=1e-4, critic_lr=3e-4,
                 gamma=0.99, tau=0.005,
                 policy_noise=0.2, noise_clip=0.5, policy_delay=2,
                 device='cpu',
                 treat_truncations_as_terminals=False,
                 mask_actor_on_terminated=False,
                 replay_success_phase_filter=False,
                 skip_frozen_transitions=False,
                 phase_groups=None):

        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = max(1, int(policy_delay))
        self.device = device
        self.treat_truncations_as_terminals = bool(treat_truncations_as_terminals)
        self.mask_actor_on_terminated = bool(mask_actor_on_terminated)
        self.skip_frozen_transitions = bool(skip_frozen_transitions)

        self.agents = [
            MADDPGAgent(i, obs_dim, action_dim, num_agents, hidden_dim,
                        actor_lr, critic_lr, device)
            for i in range(num_agents)
        ]
        self.replay_buffer = ReplayBuffer(
            max_size=200000,
            success_phase_filter=replay_success_phase_filter,
            phase_groups=phase_groups or PHASE_GROUPS_V3,
        )
        self.train_steps = 0
        self.losses = []
        self.last_train_diagnostics = {
            'q_target_mean': 0.0,
            'q_pred_mean': 0.0,
            'td_abs_mean': 0.0,
            'actor_saturation': 0.0,
            'actor_grad_norm': 0.0,
            'critic_grad_norm': 0.0,
        }

    def get_actions(self, observations, noise_scale=0.0):
        actions = {}
        for i, name in enumerate(sorted(observations.keys())):
            actions[name] = self.agents[i].get_action(observations[name], noise_scale)
        return actions

    def store_transition(self, obs, actions, rewards, next_obs,
                         terminations, truncations, phase=None, success=False):
        transition = {
            'obs': obs, 'actions': actions, 'rewards': rewards,
            'next_obs': next_obs, 'terminations': terminations, 'truncations': truncations,
        }
        self.replay_buffer.push(transition, phase=phase, success=success)

    def train_step(self, batch_size=128):
        if len(self.replay_buffer) < batch_size:
            return None
        batch = self.replay_buffer.sample(batch_size)
        agent_names = sorted(batch[0]['obs'].keys())
        n = len(batch)

        obs_b = [np.empty((n, self.obs_dim), dtype=np.float32) for _ in range(self.num_agents)]
        act_b = [np.empty((n, self.action_dim), dtype=np.float32) for _ in range(self.num_agents)]
        rew_b = [np.empty((n, 1), dtype=np.float32) for _ in range(self.num_agents)]
        nxt_b = [np.empty((n, self.obs_dim), dtype=np.float32) for _ in range(self.num_agents)]
        ter_b = [np.empty((n, 1), dtype=np.float32) for _ in range(self.num_agents)]
        tru_b = [np.empty((n, 1), dtype=np.float32) for _ in range(self.num_agents)]

        for j, tr in enumerate(batch):
            for i, name in enumerate(agent_names):
                obs_b[i][j] = tr['obs'][name]
                act_b[i][j] = tr['actions'][name]
                rew_b[i][j, 0] = tr['rewards'][name]
                nxt_b[i][j] = tr['next_obs'][name]
                ter_b[i][j, 0] = float(tr['terminations'][name])
                tru_b[i][j, 0] = float(tr['truncations'][name])

        obs_b = [torch.from_numpy(o).to(self.device) for o in obs_b]
        act_b = [torch.from_numpy(a).to(self.device) for a in act_b]
        rew_b = [torch.from_numpy(r).to(self.device) for r in rew_b]
        nxt_b = [torch.from_numpy(o).to(self.device) for o in nxt_b]
        ter_b = [torch.from_numpy(d).to(self.device) for d in ter_b]
        tru_b = [torch.from_numpy(d).to(self.device) for d in tru_b]

        update_actor = (self.train_steps % self.policy_delay == 0)

        with torch.no_grad():
            next_actions = []
            for i in range(self.num_agents):
                a = self.agents[i].actor_target(nxt_b[i])
                noise = torch.clamp(torch.randn_like(a) * self.policy_noise,
                                    -self.noise_clip, self.noise_clip)
                next_actions.append(torch.clamp(a + noise, -1.0, 1.0))

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_q_target_mean = 0.0
        total_q_pred_mean = 0.0
        total_td_abs_mean = 0.0
        total_actor_saturation = 0.0
        total_actor_grad_norm = 0.0
        total_critic_grad_norm = 0.0

        for i in range(self.num_agents):
            agent = self.agents[i]
            if self.treat_truncations_as_terminals:
                done_i = torch.maximum(ter_b[i], tru_b[i])
            else:
                done_i = ter_b[i]

            with torch.no_grad():
                tq1, tq2 = agent.critic_target(nxt_b, next_actions)
                target_q = torch.min(tq1, tq2)
                q_target = rew_b[i] + self.gamma * target_q * (1 - done_i)

            q1, q2 = agent.critic(obs_b, act_b)
            critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_grad_norm = float(torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), max_norm=1.0))
            agent.critic_optimizer.step()

            td_abs_mean = 0.5 * (
                torch.mean(torch.abs(q1.detach() - q_target)).item() +
                torch.mean(torch.abs(q2.detach() - q_target)).item()
            )
            total_q_target_mean += float(q_target.mean().item())
            total_q_pred_mean += float((0.5 * (q1.detach().mean() + q2.detach().mean())).item())
            total_td_abs_mean += float(td_abs_mean)
            total_critic_grad_norm += critic_grad_norm

            if update_actor:
                actions_for_critic = [
                    self.agents[j].actor(obs_b[j]).detach() if j != i
                    else agent.actor(obs_b[i])
                    for j in range(self.num_agents)
                ]
                q_pi = agent.critic.q1_forward(obs_b, actions_for_critic)
                if self.mask_actor_on_terminated:
                    mask = (1.0 - ter_b[i]).detach()
                    denom = mask.sum().clamp(min=1.0)
                    actor_loss = -(q_pi * mask).sum() / denom
                else:
                    actor_loss = -q_pi.mean()
                actor_saturation = float((torch.abs(actions_for_critic[i].detach()) > 0.95).float().mean().item())
                agent.actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_grad_norm = float(torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), max_norm=1.0))
                agent.actor_optimizer.step()
                agent.soft_update(tau=self.tau)
                total_actor_loss += actor_loss.item()
                total_actor_saturation += actor_saturation
                total_actor_grad_norm += actor_grad_norm
            total_critic_loss += critic_loss.item()

        self.train_steps += 1
        avg_al = (total_actor_loss / self.num_agents) if update_actor else 0.0
        avg_cl = total_critic_loss / self.num_agents
        self.last_train_diagnostics = {
            'q_target_mean': total_q_target_mean / self.num_agents,
            'q_pred_mean': total_q_pred_mean / self.num_agents,
            'td_abs_mean': total_td_abs_mean / self.num_agents,
            'actor_saturation': (total_actor_saturation / self.num_agents) if update_actor else 0.0,
            'actor_grad_norm': (total_actor_grad_norm / self.num_agents) if update_actor else 0.0,
            'critic_grad_norm': total_critic_grad_norm / self.num_agents,
        }
        self.losses.append(avg_al + avg_cl)
        return {
            'actor_loss': avg_al,
            'critic_loss': avg_cl,
            'total_loss': avg_al + avg_cl,
            **self.last_train_diagnostics,
        }

    def recover_dead_actors(self, per_robot_success, threshold=0.05):
        vals = list(per_robot_success.values())
        best_idx = np.argmax(vals)
        for i, (name, sr) in enumerate(per_robot_success.items()):
            if sr < threshold and vals[best_idx] > threshold * 3:
                print(f"[RECOVERY] Robot {i} dead ({sr:.1%}), copying from Robot {best_idx}")
                self.agents[i].actor.load_state_dict(self.agents[best_idx].actor.state_dict())
                self.agents[i].actor_target.load_state_dict(self.agents[best_idx].actor_target.state_dict())
                with torch.no_grad():
                    for param in self.agents[i].actor.parameters():
                        param.add_(torch.randn_like(param) * 0.01)

    def save_checkpoint(self, path, episode):
        ckpt = {'episode': episode, 'train_steps': self.train_steps}
        for i, ag in enumerate(self.agents):
            ckpt[f'agent_{i}_actor'] = ag.actor.state_dict()
            ckpt[f'agent_{i}_actor_target'] = ag.actor_target.state_dict()
            ckpt[f'agent_{i}_critic'] = ag.critic.state_dict()
            ckpt[f'agent_{i}_critic_target'] = ag.critic_target.state_dict()
        torch.save(ckpt, path)
        print(f"[SAVE] Checkpoint saved: {path}")

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        for i, ag in enumerate(self.agents):
            ag.actor.load_state_dict(ckpt[f'agent_{i}_actor'])
            ag.actor_target.load_state_dict(ckpt[f'agent_{i}_actor_target'])
            ag.critic.load_state_dict(ckpt[f'agent_{i}_critic'])
            ag.critic_target.load_state_dict(ckpt[f'agent_{i}_critic_target'])
        self.train_steps = ckpt.get('train_steps', 0)
        print(f"[LOAD] Checkpoint loaded from episode {ckpt.get('episode', '?')}")


# ==================== NOISE ====================
def compute_adaptive_noise(episode, phase_success_history, phase,
                           phase_episode_count=0,
                           use_plateau_logic=True,
                           phase_team_success_history=None,
                           phase_noise_table=None):
    """Noise schedule — identical logic to V2, but takes an explicit noise table."""
    if phase_noise_table is None:
        phase_noise_table = PHASE_NOISE_V3
    base_noise = phase_noise_table.get(phase, 0.15)

    # Mastery clamp
    if phase_team_success_history and len(phase_team_success_history) >= 100:
        if float(np.mean(phase_team_success_history[-100:])) >= 0.90:
            return float(np.clip(base_noise * 0.3, 0.03, 0.07))

    # Warmup
    if phase_episode_count < 100:
        return min(0.30, base_noise * 1.5)

    # Stuck detection — if team_success is below 30% for 2000+ episodes,
    # the agent is trapped. Boost noise to full base_noise for exploration.
    if phase_team_success_history and phase_episode_count > 2000:
        recent_team = np.mean(phase_team_success_history[-min(500, len(phase_team_success_history)):])
        if recent_team < 0.30:
            return float(np.clip(base_noise * 1.0, 0.10, 0.25))

    # Monotone schedule — floor raised from 0.4 to 0.6 so minimum noise
    # is ~0.078 in Med1 (was 0.052), providing enough exploration.
    if not use_plateau_logic:
        decay = max(0.0, 1.0 - (phase_episode_count - 100) / 3000.0)
        noise = base_noise * (0.6 + 0.4 * decay)
        return float(np.clip(noise, 0.08, base_noise))

    # Adaptive — use team_success for low-success boost (not individual success_fraction)
    if phase_team_success_history and len(phase_team_success_history) >= 50:
        recent_team = np.mean(phase_team_success_history[-50:])
        if recent_team < 0.20:
            return min(0.30, base_noise * 1.3)

    if len(phase_success_history) >= 50:
        recent = np.mean(phase_success_history[-50:])
        if recent >= 0.5:
            mult = 1.0 - (recent - 0.5) * 1.5
        else:
            mult = 1.5 - recent * 1.0
        if len(phase_success_history) >= 400:
            r400 = phase_success_history[-400:]
            if np.std(r400) < 0.10 and 0.55 <= np.mean(r400) < 0.84:
                mult = min(mult * 1.25, 1.5)
        mult = np.clip(mult, 0.4, 1.5)
        return float(np.clip(base_noise * mult, 0.10, 0.25))

    return base_noise


# Module-level factory so multiprocessing.spawn can pickle it
def _make_env_factory(*, enable_waiting_reward, enable_frozen_bonus,
                      reward_mode, curriculum_mode):
    return MiningEnvironmentMADDPG_V3(
        num_robots=1, visualize=False,
        enable_waiting_reward=enable_waiting_reward,
        enable_frozen_bonus=enable_frozen_bonus,
        reward_mode=reward_mode, curriculum_mode=curriculum_mode,
    )


# ==================== MAIN TRAINING LOOP ====================
def train_maddpg_v3(
    n_episodes=100000,
    n_training_steps=20,
    batch_size=128,
    resume_from=None,
    *,
    reward_mode='v2_baseline',
    curriculum_mode='v3_smooth',
    treat_truncations_as_terminals=False,
    mask_actor_on_terminated=False,
    success_buffer_phase_filter=False,
    success_tail_fraction=1.0,
    enable_waiting_reward=False,
    skip_frozen_transitions=False,
    rolling_mastery_window=500,
    disable_plateau_noise=True,
    disable_waiting_shaping=True,
    partial_timeout_penalty=10.0,
    zero_timeout_penalty=6.0,
    num_parallel_envs=1,
    torch_threads=None,
    torch_interop_threads=1,
    results_tag=None,
):
    success_tail_fraction = float(np.clip(float(success_tail_fraction), 0.0, 1.0))

    # Select tables based on mode
    if curriculum_mode == 'v2_original':
        phase_list = PHASE_LIST_V2
        phase_config = PHASE_CONFIG_V2
        phase_noise_table = PHASE_NOISE_V2
        phase_groups = PHASE_GROUPS_V2
        phase_colors = PHASE_COLORS_V2
    else:
        phase_list = PHASE_LIST_V3
        phase_config = PHASE_CONFIG_V3
        phase_noise_table = PHASE_NOISE_V3
        phase_groups = PHASE_GROUPS_V3
        phase_colors = PHASE_COLORS_V3

    if results_tag:
        mode_tag = str(results_tag)
    elif reward_mode == 'v2_baseline' and curriculum_mode == 'v3_smooth':
        mode_tag = "reward+smooth"
    else:
        mode_tag = f"{reward_mode}__{curriculum_mode}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = setup_results_directory(timestamp, tag=mode_tag)

    print("\n" + "="*100)
    print(f"MADDPG V3 TRAINING — reward_mode={reward_mode}, curriculum_mode={curriculum_mode}")
    print("="*100)
    print(f"  Phases: {phase_list}")
    print(f"  Mastery threshold: {phase_config[phase_list[0]]['threshold']:.0%}")
    print(f"  Episode cap: NONE (mastery-only advancement)")
    print(f"  Parallel envs: {num_parallel_envs}")
    print(f"  Results: {results_dir}")
    print("="*100)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n[SETUP] Device: {device}")

    # CPU thread budget: env_workers + torch_threads <= cpu_count
    n_cpus = os.cpu_count() or 4
    if torch_threads is None:
        # Auto: leave 1 core per env worker, use the rest for torch
        torch_threads = max(1, n_cpus - num_parallel_envs)
    if device == 'cpu':
        torch.set_num_threads(torch_threads)
        torch.set_num_interop_threads(torch_interop_threads)
        print(f"[SETUP] torch.set_num_threads({torch_threads}), "
              f"set_num_interop_threads({torch_interop_threads})")
    total_budget = num_parallel_envs + torch_threads
    if total_budget > n_cpus:
        print(f"[WARNING] CPU oversubscription: {num_parallel_envs} env workers + "
              f"{torch_threads} torch threads = {total_budget} > {n_cpus} cores")
    else:
        print(f"[SETUP] CPU budget: {num_parallel_envs} env workers + "
              f"{torch_threads} torch threads = {total_budget} / {n_cpus} cores")

    _wr = enable_waiting_reward and (not disable_waiting_shaping)
    _fb = not disable_waiting_shaping
    
    # Environment factory — must be picklable for multiprocessing.spawn
    make_env = partial(_make_env_factory,
                       enable_waiting_reward=_wr, enable_frozen_bonus=_fb,
                       reward_mode=reward_mode, curriculum_mode=curriculum_mode)
    
    # Initialize environment(s)
    if num_parallel_envs > 1:
        print(f"[SETUP] Initializing {num_parallel_envs} parallel workers...")
        env_manager = ParallelEnvManager(make_env, num_workers=num_parallel_envs)
        # Get reference env for dimensions
        temp_env = make_env()
        assert temp_env.observation_space.shape is not None
        assert temp_env.action_space.shape is not None
        obs_dim = temp_env.observation_space.shape[0]
        action_dim = temp_env.action_space.shape[0]
        n_agents = temp_env.num_robots
        max_episode_steps = temp_env.max_episode_steps
        temp_env.close()
        use_parallel = True
        env = None  # No single env in parallel mode
        print(f"[SETUP] Parallel workers ready")
    else:
        env = make_env()
        assert env.observation_space.shape is not None
        assert env.action_space.shape is not None
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        n_agents = env.num_robots
        max_episode_steps = env.max_episode_steps
        env_manager = None
        use_parallel = False

    trainer = MADDPG_V3(
        num_agents=n_agents, obs_dim=obs_dim, action_dim=action_dim,
        hidden_dim=384, actor_lr=1e-4, critic_lr=3e-4,
        gamma=0.99, tau=0.005,
        policy_noise=0.2, noise_clip=0.5, policy_delay=2,
        device=device,
        treat_truncations_as_terminals=treat_truncations_as_terminals,
        mask_actor_on_terminated=mask_actor_on_terminated,
        replay_success_phase_filter=success_buffer_phase_filter,
        skip_frozen_transitions=skip_frozen_transitions,
        phase_groups=phase_groups,
    )

    # LR decay schedule
    LR_DECAY_SCHEDULE = [(50_000, 0.5), (150_000, 0.25)]
    _lr_decay_applied = set()

    if resume_from and os.path.exists(resume_from):
        trainer.load_checkpoint(resume_from)

    current_phase_idx = 0
    current_phase = phase_list[current_phase_idx]
    
    # Set phase on environment(s)
    if use_parallel:
        assert env_manager is not None
        env_manager.call_method_all('set_phase', current_phase)
    else:
        assert env is not None
        env.set_phase(current_phase)

    all_episodes = []
    success_history = []
    phase_episode_count = {p: 0 for p in phase_list}
    phase_success_sum = {p: 0.0 for p in phase_list}
    phase_team_success_sum = {p: 0.0 for p in phase_list}
    phase_step_count = {p: 0 for p in phase_list}
    phase_frozen = {p: False for p in phase_list}
    phase_collision_skips = {p: 0 for p in phase_list}
    phase_replay_stored = {p: 0 for p in phase_list}
    phase_regression_cooldown = {p: 0 for p in phase_list}
    FREEZE_MASTERY_WINDOW = 500

    # ── CACHED PER-PHASE HISTORY (avoids O(n) scan of all_episodes) ──
    _phase_success_cache = {p: deque(maxlen=2000) for p in phase_list}
    _phase_team_cache = {p: deque(maxlen=2000) for p in phase_list}
    _phase_onetimeout_cache = {p: deque(maxlen=2000) for p in phase_list}
    _phase_catastrophe_cache = {p: deque(maxlen=2000) for p in phase_list}

    robot_success_count = {f'robot_{i}': 0 for i in range(n_agents)}
    robot_episode_count = 0

    csv_file = f"{results_dir}/training_log.csv"
    last_written_idx = 0
    csv_header_written = False
    reward_term_keys = [
        'progress_reward',
        'team_progress_reward',
        'time_penalty',
        'near_goal_entry_bonus',
        'milestone_bonus',
        'goal_bonus',
        'partial_team_bonus',
        'full_team_bonus',
        'waiting_reward',
        'teammate_spacing_penalty',
        'collision_penalty',
        'partial_timeout_penalty',
        'zero_timeout_penalty',
    ]

    def zero_reward_terms():
        return {key: 0.0 for key in reward_term_keys}

    print(f"\n[TRAIN] Starting {n_episodes} episodes ...")
    print(f"{'Ep':>5} | {'Frac':>6} | {'Team':>6} | {'Reward':>10} | {'ActorL':>8} | {'CritL':>8} | {'Steps':>6} | {'Coll':>5} | {'Phase':>8}")
    print("-" * 95)

    start_time = time.time()
    timing_acc = {'action': 0.0, 'env': 0.0, 'store': 0.0, 'train': 0.0}
    timing_counts = {'steps': 0, 'store_calls': 0, 'train_calls': 0}

    # ═══════════════════════════════════════════════════════════
    # PARALLEL ROLLOUT MODE
    # ═══════════════════════════════════════════════════════════
    if use_parallel:
        assert env_manager is not None
        episodes_completed = 0
        
        while episodes_completed < n_episodes:
            # Get current phase and noise scales for all workers
            phases = env_manager.get_attr_all('current_phase')
            noise_scales = []
            for worker_phase in phases:
                phase_success_hist = list(_phase_success_cache[worker_phase])
                phase_team_hist = list(_phase_team_cache[worker_phase])
                noise = compute_adaptive_noise(
                    episodes_completed, phase_success_hist, worker_phase,
                    phase_episode_count[worker_phase],
                    use_plateau_logic=(not disable_plateau_noise),
                    phase_team_success_history=phase_team_hist,
                    phase_noise_table=phase_noise_table,
                )
                noise_scales.append(noise)
            
            # Reset all workers
            observations = env_manager.reset_all()
            worker_episode_rewards = [0.0] * num_parallel_envs
            worker_episode_transitions = [[] for _ in range(num_parallel_envs)]
            worker_cumulative_rewards = [{name: 0.0 for name in obs.keys()} for obs in observations]
            worker_cumulative_reward_terms = [zero_reward_terms() for _ in range(num_parallel_envs)]
            worker_steps = [0] * num_parallel_envs
            
            # Collect episodes from all workers
            for step in range(max_episode_steps):
                # Get actions for all workers
                actions_list = []
                t0 = perf_counter()
                for i, obs in enumerate(observations):
                    actions = trainer.get_actions(obs, noise_scale=noise_scales[i])
                    actions_list.append(actions)
                timing_acc['action'] += perf_counter() - t0
                
                # Step all environments
                t0 = perf_counter()
                transitions = env_manager.step_all(actions_list)
                timing_acc['env'] += perf_counter() - t0
                
                # Process transitions — index by worker_id, not list position
                for trans in transitions:
                    wid, obs_t, actions_t, rewards_t, next_obs_t, term_t, trunc_t, infos_t = trans
                    worker_episode_transitions[wid].append(trans)
                    worker_episode_rewards[wid] += sum(rewards_t.values())
                    for name in rewards_t:
                        worker_cumulative_rewards[wid][name] += float(rewards_t[name])
                        for term_key, term_value in infos_t[name].get('reward_terms', {}).items():
                            worker_cumulative_reward_terms[wid][term_key] += float(term_value)
                    worker_steps[wid] += 1
                    observations[wid] = next_obs_t
                
                # Check if any workers finished
                active_count = sum(env_manager.worker_active)
                if active_count == 0:
                    break
                
                timing_counts['steps'] += active_count
            
            # Process completed episodes
            robot_states_all = env_manager.get_attr_all('robot_states')
            
            for worker_id in range(num_parallel_envs):
                robot_states = robot_states_all[worker_id]
                episode_transitions = worker_episode_transitions[worker_id]
                episode_reward = worker_episode_rewards[worker_id]
                cumulative_robot_rewards = worker_cumulative_rewards[worker_id]
                cumulative_reward_terms = worker_cumulative_reward_terms[worker_id]
                episode_steps = worker_steps[worker_id]
                phase = phases[worker_id]
                
                # Calculate metrics
                goals_reached = sum(1 for s in robot_states.values() if s.get('reached', False))
                success_fraction = goals_reached / n_agents
                team_success = 1.0 if goals_reached == n_agents else 0.0
                episode_collisions = sum(s.get('collision_count', 0) for s in robot_states.values())
                
                # Determine end reason
                all_frozen = all(s.get('frozen', False) for s in robot_states.values())
                end_reason = "team_success" if all_frozen else "time_limit"
                
                # Inject one-goal timeout penalty into replay buffer
                # (not just logged total — critic must see this signal)
                if partial_timeout_penalty > 0 and goals_reached == max(1, n_agents - 1) and goals_reached < n_agents and end_reason == "time_limit" and episode_transitions:
                    last_trans = episode_transitions[-1]
                    wid_t, obs_t, act_t, rew_t, nxt_t, term_t, trunc_t, info_t = last_trans
                    penalized_rew = {name: r - partial_timeout_penalty for name, r in rew_t.items()}
                    episode_transitions[-1] = (wid_t, obs_t, act_t, penalized_rew, nxt_t, term_t, trunc_t, info_t)
                    for name in cumulative_robot_rewards:
                        cumulative_robot_rewards[name] -= partial_timeout_penalty
                    cumulative_reward_terms['partial_timeout_penalty'] -= partial_timeout_penalty * n_agents
                    episode_reward -= partial_timeout_penalty * n_agents
                elif zero_timeout_penalty > 0 and goals_reached == 0 and end_reason == "time_limit" and episode_transitions:
                    last_trans = episode_transitions[-1]
                    wid_t, obs_t, act_t, rew_t, nxt_t, term_t, trunc_t, info_t = last_trans
                    penalized_rew = {name: r - zero_timeout_penalty for name, r in rew_t.items()}
                    episode_transitions[-1] = (wid_t, obs_t, act_t, penalized_rew, nxt_t, term_t, trunc_t, info_t)
                    for name in cumulative_robot_rewards:
                        cumulative_robot_rewards[name] -= zero_timeout_penalty
                    cumulative_reward_terms['zero_timeout_penalty'] -= zero_timeout_penalty * n_agents
                    episode_reward -= zero_timeout_penalty * n_agents
                
                # Collision filtering — keep 40% of catastrophe episodes
                is_catastrophe = (episode_collisions > 100)
                skip_store = is_catastrophe and (random.random() > 0.40)
                if skip_store:
                    phase_collision_skips[phase] += 1
                
                # Store transitions
                is_success = (goals_reached == n_agents)
                if is_success and success_tail_fraction < 1.0:
                    tail_start = int(len(episode_transitions) * (1.0 - success_tail_fraction))
                else:
                    tail_start = 0
                
                if not skip_store:
                    t0 = perf_counter()
                    for idx, trans in enumerate(episode_transitions):
                        # Unpack: (worker_id, obs, actions, rewards, next_obs, term, trunc, infos)
                        _, obs_t, act_t, rew_t, nxt_t, term_t, tru_t, _ = trans
                        if trainer.skip_frozen_transitions and idx > 0:
                            _, _, _, _, _, prev_term, _, _ = episode_transitions[idx - 1]
                            if all(prev_term.get(a, False) for a in term_t.keys()):
                                continue
                            if any(prev_term.get(a, False) for a in term_t.keys()):
                                continue
                        route_success = bool(is_success and idx >= tail_start)
                        trainer.store_transition(obs_t, act_t, rew_t, nxt_t, term_t, tru_t,
                                                 phase=phase, success=route_success)
                        phase_replay_stored[phase] += 1
                        timing_counts['store_calls'] += 1
                    timing_acc['store'] += perf_counter() - t0
                
                # Training
                if phase_frozen.get(phase, False):
                    effective_train_steps = 0
                elif goals_reached == n_agents and episode_steps < max_episode_steps * 0.5:
                    effective_train_steps = max(1, n_training_steps // 4)
                else:
                    effective_train_steps = n_training_steps
                
                actor_loss_sum = 0.0
                critic_loss_sum = 0.0
                train_count = 0
                if len(trainer.replay_buffer) >= 512:
                    for _ in range(effective_train_steps):
                        t0 = perf_counter()
                        loss_dict = trainer.train_step(batch_size=batch_size)
                        timing_acc['train'] += perf_counter() - t0
                        timing_counts['train_calls'] += 1
                        if loss_dict is not None:
                            actor_loss_sum += loss_dict['actor_loss']
                            critic_loss_sum += loss_dict['critic_loss']
                            train_count += 1
                
                avg_actor_loss = actor_loss_sum / max(1, train_count)
                avg_critic_loss = critic_loss_sum / max(1, train_count)
                
                # Update statistics
                phase_episode_count[phase] += 1
                phase_success_sum[phase] += success_fraction
                phase_team_success_sum[phase] += team_success
                phase_step_count[phase] += episode_steps
                _phase_success_cache[phase].append(success_fraction)
                _phase_team_cache[phase].append(team_success)
                _phase_onetimeout_cache[phase].append(1.0 if (success_fraction == 0.5 and end_reason == "time_limit") else 0.0)
                _phase_catastrophe_cache[phase].append(1.0 if is_catastrophe else 0.0)
                success_history.append(success_fraction)
                
                # Per-robot tracking
                robot_episode_count += 1
                for i, (name, state) in enumerate(sorted(robot_states.items())):
                    if state.get('reached', False):
                        robot_success_count[f'robot_{i}'] += 1
                
                avg_final_dist = np.mean([s.get('distance_to_goal', 0.0) for s in robot_states.values()])
                robot_rewards = {f'robot_{i}_reward': cumulative_robot_rewards[name]
                                 for i, name in enumerate(sorted(cumulative_robot_rewards.keys()))}
                
                # Build episode data
                episode_data = {
                    'episode': episodes_completed + 1,
                    'reward': episode_reward,
                    'success_fraction': success_fraction,
                    'team_success': team_success,
                    'partial_success': 1.0 if goals_reached >= max(1, n_agents - 1) else 0.0,
                    'goals_reached': goals_reached,
                    'actor_loss': avg_actor_loss,
                    'critic_loss': avg_critic_loss,
                    'steps': episode_steps,
                    'collisions': episode_collisions,
                    'avg_final_distance': avg_final_dist,
                    'noise': noise_scales[worker_id],
                    'phase': phase,
                    'end_reason': end_reason,
                    'replay_skipped_collision': int(skip_store),
                    'is_catastrophe': int(is_catastrophe),
                    'phase_collision_skips': phase_collision_skips[phase],
                    'phase_episode_count': phase_episode_count[phase],
                    'phase_replay_stored': phase_replay_stored[phase],
                }
                episode_data.update({f'term_{key}': value for key, value in cumulative_reward_terms.items()})
                episode_data.update(robot_rewards)
                all_episodes.append(episode_data)
                
                episodes_completed += 1
                
                # Print progress
                if episodes_completed % 10 == 0 or episodes_completed <= 10:
                    print(f"{episodes_completed:>5} | {success_fraction:>6.2f} | {team_success:>6.2f} | "
                          f"{episode_reward:>10.2f} | {avg_actor_loss:>8.4f} | {avg_critic_loss:>8.4f} | "
                          f"{episode_steps:>6} | {episode_collisions:>5} | {phase:>8}")

                    if episodes_completed % 100 == 0:
                        blk = all_episodes[-100:]
                        bf = np.mean([e['success_fraction'] for e in blk])
                        bt = np.mean([e['team_success'] for e in blk])
                        br = np.mean([e['reward'] for e in blk])
                        bc = np.mean([e['collisions'] for e in blk])
                        bs = np.mean([e['steps'] for e in blk])
                        bskip = np.mean([e['replay_skipped_collision'] for e in blk])
                        pe = phase_episode_count[phase]
                        pa = (phase_success_sum[phase] / pe) if pe > 0 else 0.0
                        pt = (phase_team_success_sum[phase] / pe) if pe > 0 else 0.0
                        elapsed = time.time() - start_time
                        eps_sec = episodes_completed / elapsed if elapsed > 0 else 0.0
                        print(f"\n  ╔═══════════════════════════════════════════════════════════════════════════════╗")
                        print(f"  ║ BLOCK {episodes_completed // 100:3d}: Ep {episodes_completed - 99:5d}-{episodes_completed:<5d}  │  Speed: {eps_sec:.1f} eps/sec           ║")
                        print(f"  ╠═══════════════════════════════════════════════════════════════════════════════╣")
                        print(f"  ║  Success: {bf:6.1%}  │  Team: {bt:6.1%}  │  Noise: {noise_scales[worker_id]:.3f}                 ║")
                        print(f"  ║  Reward: {br:9.1f}  │  Steps: {bs:6.0f}    │  Phase: {phase:>8}                ║")
                        print(f"  ║  Coll:   {bc:6.1f}  │  Buffer: {len(trainer.replay_buffer):6d}                                   ║")
                        print(f"  ║  Skip:   {bskip:6.1%}  │  Stored: {phase_replay_stored[phase]:6d}  │  SkipCnt: {phase_collision_skips[phase]:4d}         ║")
                        print(f"  ║  Phase Overall: {pa:6.1%}  │  Phase Team: {pt:6.1%}  │  Phase Eps: {pe:5d}     ║")
                        print(f"  ╚═══════════════════════════════════════════════════════════════════════════════╝\n")
                        sys.stdout.flush()
                
                # LR decay
                for milestone, factor in LR_DECAY_SCHEDULE:
                    if trainer.train_steps >= milestone and milestone not in _lr_decay_applied:
                        _lr_decay_applied.add(milestone)
                        for ag in trainer.agents:
                            for pg in ag.actor_optimizer.param_groups:
                                pg['lr'] = 1e-4 * factor
                            for pg in ag.critic_optimizer.param_groups:
                                pg['lr'] = 3e-4 * factor
                        print(f"  [LR DECAY] train_steps={trainer.train_steps:,}, "
                              f"actor_lr -> {1e-4*factor:.2e}, critic_lr -> {3e-4*factor:.2e}")
                
                # Periodic eval
                if episodes_completed % 2000 == 0 and episodes_completed > 0:
                    eval_env = make_env()
                    eval_env.set_phase(phase)
                    eval_team = 0
                    n_eval = 20
                    for _ in range(n_eval):
                        eo, _ = eval_env.reset()
                        for _ in range(max_episode_steps):
                            ea = trainer.get_actions(eo, noise_scale=0.0)
                            eo, _, et, etr, _ = eval_env.step(ea)
                            if all(et.values()) or any(etr.values()):
                                break
                        if all(s.get('frozen', False) for s in eval_env.robot_states.values()):
                            eval_team += 1
                    eval_env.close()
                    print(f"  [EVAL] Ep {episodes_completed}: team_success={eval_team/n_eval:.0%}, phase={phase}")
                
                # Checkpoint + periodic plots
                if episodes_completed % 10000 == 0:
                    ckpt_path = f"{results_dir}/checkpoints/episode_{episodes_completed}.pt"
                    trainer.save_checkpoint(ckpt_path, episodes_completed)
                if episodes_completed % 5000 == 0 and len(all_episodes) > 100:
                    generate_plots(all_episodes, results_dir, phase_episode_count,
                                   phase_success_sum, phase_list, phase_colors)
                
                # CSV write
                if episodes_completed % 500 == 0:
                    if not csv_header_written:
                        with open(csv_file, 'w', newline='') as f:
                            writer = csv.DictWriter(f, fieldnames=all_episodes[0].keys())
                            writer.writeheader()
                            csv_header_written = True
                    with open(csv_file, 'a', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=all_episodes[0].keys())
                        writer.writerows(all_episodes[last_written_idx:])
                    last_written_idx = len(all_episodes)
                
                # Phase advancement check
                if episodes_completed % 500 == 0:
                    # ── Live gate metrics for current phase ──
                    _sc_cp = _phase_success_cache[current_phase]
                    _tc_cp = _phase_team_cache[current_phase]
                    _ot_cp = _phase_onetimeout_cache[current_phase]
                    _cat_cp = _phase_catastrophe_cache[current_phase]
                    w = rolling_mastery_window
                    if len(_sc_cp) >= w:
                        r500_ind = float(np.mean(list(_sc_cp)[-w:]))
                        r500_team = float(np.mean(list(_tc_cp)[-w:]))
                        r500_onetimeout = float(np.mean(list(_ot_cp)[-w:]))
                        r500_catastrophe = float(np.mean(list(_cat_cp)[-w:]))
                        pe_cp = phase_episode_count[current_phase]
                        r500_skip = phase_collision_skips[current_phase] / max(1, pe_cp)
                        print(f"\n  ┌─ GATE [{current_phase}] ep {episodes_completed} ─────────────────────────────────┐")
                        print(f"  │  Ind success  (500w): {r500_ind:6.1%}  │  threshold: {phase_config[current_phase]['threshold']:.0%}")
                        print(f"  │  Team success (500w): {r500_team:6.1%}  │")
                        print(f"  │  1-goal timeout (500w): {r500_onetimeout:6.1%}  │")
                        print(f"  │  Catastrophe 100+ (500w): {r500_catastrophe:6.1%}  │")
                        print(f"  │  Replay-skipped (phase): {r500_skip:6.1%}  │  phase eps: {pe_cp}")
                        print(f"  └───────────────────────────────────────────────────────┘")

                    for phase_name in phase_list:
                        count = phase_episode_count[phase_name]
                        if count >= phase_config[phase_name]['min']:
                            _sc_pn = _phase_success_cache[phase_name]
                            _tc_pn = _phase_team_cache[phase_name]
                            if len(_sc_pn) >= rolling_mastery_window and len(_tc_pn) >= rolling_mastery_window:
                                recent_ind = float(np.mean(list(_sc_pn)[-rolling_mastery_window:]))
                                recent_team = float(np.mean(list(_tc_pn)[-rolling_mastery_window:]))
                            else:
                                recent_ind = 0.0
                                recent_team = 0.0
                            mastery_thresh = phase_config[phase_name]['threshold']
                            if recent_ind >= mastery_thresh and recent_team >= mastery_thresh and phase_name == current_phase:
                                next_idx = current_phase_idx + 1
                                if next_idx < len(phase_list):
                                    current_phase_idx = next_idx
                                    current_phase = phase_list[current_phase_idx]
                                    env_manager.call_method_all('set_phase', current_phase)
                                    trainer.replay_buffer.set_phase(current_phase)
                                    max_episode_steps = env_manager.get_attr_all('max_episode_steps')[0]
                                    print(f"\n{'='*80}\n  ADVANCED: {phase_name} → {current_phase} "
                                          f"(ind={recent_ind:.1%}, team={recent_team:.1%}, steps={max_episode_steps})\n{'='*80}\n")
                
                if episodes_completed >= n_episodes:
                    break
    
    # ═══════════════════════════════════════════════════════════
    # SINGLE ENVIRONMENT MODE (original)
    # ═══════════════════════════════════════════════════════════
    else:
        assert env is not None
        for episode in range(n_episodes):
            obs, _ = env.reset()
            episode_reward = 0.0
            phase = env.current_phase

            # O(1) cached lookups instead of O(n) all_episodes scan
            phase_success_hist = list(_phase_success_cache[phase])
            phase_team_hist = list(_phase_team_cache[phase])
            noise_scale = compute_adaptive_noise(
                episode, phase_success_hist, phase, phase_episode_count[phase],
                use_plateau_logic=(not disable_plateau_noise),
                phase_team_success_history=phase_team_hist,
                phase_noise_table=phase_noise_table,
            )

            episode_transitions = []
            one_frozen_steps = 0
            partial_frozen_steps = 0
            frozen_reward_sum = 0.0
            waiting_reward_base_sum = 0.0
            other_frozen_bonus_est_sum = 0.0
            end_reason = "unknown"
            cumulative_robot_rewards = {name: 0.0 for name in obs.keys()}
            episode_reward_terms = zero_reward_terms()
            episode_action_time = 0.0
            episode_env_time = 0.0
            episode_store_time = 0.0
            episode_train_time = 0.0

            for step in range(env.max_episode_steps):
                t0 = perf_counter()
                actions = trainer.get_actions(obs, noise_scale=noise_scale)
                episode_action_time += perf_counter() - t0

                t0 = perf_counter()
                next_obs, rewards, terminations, truncations, infos = env.step(actions)
                episode_env_time += perf_counter() - t0
                episode_transitions.append((obs, actions, rewards, next_obs, terminations, truncations))
                episode_reward += sum(rewards.values())
                for name in rewards:
                    cumulative_robot_rewards[name] += float(rewards[name])
                    for term_key, term_value in infos[name].get('reward_terms', {}).items():
                        episode_reward_terms[term_key] += float(term_value)

                frozen_count = sum(1 for s in env.robot_states.values() if s.get('frozen', False))
                if frozen_count == 1:
                    one_frozen_steps += 1
                if 0 < frozen_count < n_agents:
                    partial_frozen_steps += 1
                for name, state in env.robot_states.items():
                    if state.get('frozen', False):
                        frozen_reward_sum += float(rewards.get(name, 0.0))
                if frozen_count > 0 and getattr(env, 'enable_frozen_bonus', False):
                    for name, state in env.robot_states.items():
                        if not state.get('frozen', False):
                            other_frozen_bonus_est_sum += 0.2 * max(0, frozen_count)
                obs = next_obs

                all_frozen = all(terminations.values())
                any_truncated = any(truncations.values())
                if all_frozen or any_truncated:
                    end_reason = "team_success" if all_frozen else "time_limit"
                    break

            episode_steps = step + 1
            phase_step_count[phase] += episode_steps
            episode_collisions = sum(s.get('collision_count', 0) for s in env.robot_states.values())

            goals_reached = sum(1 for s in env.robot_states.values() if s.get('reached', False))
            success_fraction = goals_reached / n_agents
            partial_target = max(1, n_agents - 1)
            partial_success = 1.0 if goals_reached >= partial_target else 0.0
            team_success = 1.0 if goals_reached == n_agents else 0.0

            # Inject one-goal timeout penalty into replay buffer
            # (not just logged total — critic must see this signal)
            if partial_timeout_penalty > 0 and goals_reached == max(1, n_agents - 1) and goals_reached < n_agents and end_reason == "time_limit" and episode_transitions:
                last_idx = len(episode_transitions) - 1
                obs_t, act_t, rew_t, nxt_t, term_t, tru_t = episode_transitions[last_idx]
                penalized_rew = {name: r - partial_timeout_penalty for name, r in rew_t.items()}
                episode_transitions[last_idx] = (obs_t, act_t, penalized_rew, nxt_t, term_t, tru_t)
                for name in cumulative_robot_rewards:
                    cumulative_robot_rewards[name] -= partial_timeout_penalty
                episode_reward_terms['partial_timeout_penalty'] -= partial_timeout_penalty * n_agents
                episode_reward -= partial_timeout_penalty * n_agents
            elif zero_timeout_penalty > 0 and goals_reached == 0 and end_reason == "time_limit" and episode_transitions:
                last_idx = len(episode_transitions) - 1
                obs_t, act_t, rew_t, nxt_t, term_t, tru_t = episode_transitions[last_idx]
                penalized_rew = {name: r - zero_timeout_penalty for name, r in rew_t.items()}
                episode_transitions[last_idx] = (obs_t, act_t, penalized_rew, nxt_t, term_t, tru_t)
                for name in cumulative_robot_rewards:
                    cumulative_robot_rewards[name] -= zero_timeout_penalty
                episode_reward_terms['zero_timeout_penalty'] -= zero_timeout_penalty * n_agents
                episode_reward -= zero_timeout_penalty * n_agents

            # Collision filter — keep 40% of catastrophe episodes for critic learning
            is_catastrophe = (episode_collisions > 100)
            skip_store = is_catastrophe and (random.random() > 0.40)
            if skip_store:
                phase_collision_skips[phase] += 1

            is_success = (goals_reached == n_agents)
            if is_success and success_tail_fraction < 1.0:
                tail_start = int(len(episode_transitions) * (1.0 - success_tail_fraction))
            else:
                tail_start = 0

            if not skip_store:
                t0 = perf_counter()
                for idx, (obs_t, act_t, rew_t, nxt_t, term_t, tru_t) in enumerate(episode_transitions):
                    if trainer.skip_frozen_transitions and idx > 0:
                        prev_term = episode_transitions[idx - 1][4]
                        if all(prev_term.get(a, False) for a in term_t.keys()):
                            continue
                        if any(prev_term.get(a, False) for a in term_t.keys()):
                            continue
                    route_success = bool(is_success and idx >= tail_start)
                    trainer.store_transition(obs_t, act_t, rew_t, nxt_t, term_t, tru_t,
                                             phase=phase, success=route_success)
                    phase_replay_stored[phase] += 1
                    timing_counts['store_calls'] += 1
                episode_store_time += perf_counter() - t0

            # Training
            if phase_frozen.get(phase, False):
                effective_train_steps = 0
            elif goals_reached == n_agents and episode_steps < env.max_episode_steps * 0.5:
                effective_train_steps = max(1, n_training_steps // 4)
            else:
                effective_train_steps = n_training_steps

            actor_loss_sum = 0.0
            critic_loss_sum = 0.0
            train_count = 0
            if len(trainer.replay_buffer) >= 512:
                for _ in range(effective_train_steps):
                    t0 = perf_counter()
                    loss_dict = trainer.train_step(batch_size=batch_size)
                    episode_train_time += perf_counter() - t0
                    timing_counts['train_calls'] += 1
                    if loss_dict is not None:
                        actor_loss_sum += loss_dict['actor_loss']
                        critic_loss_sum += loss_dict['critic_loss']
                        train_count += 1

            avg_actor_loss = actor_loss_sum / max(1, train_count)
            avg_critic_loss = critic_loss_sum / max(1, train_count)
            train_diag = dict(trainer.last_train_diagnostics)
            timing_acc['action'] += episode_action_time
            timing_acc['env'] += episode_env_time
            timing_acc['store'] += episode_store_time
            timing_acc['train'] += episode_train_time
            timing_counts['steps'] += episode_steps

            # LR decay
            for milestone, factor in LR_DECAY_SCHEDULE:
                if trainer.train_steps >= milestone and milestone not in _lr_decay_applied:
                    _lr_decay_applied.add(milestone)
                    for ag in trainer.agents:
                        for pg in ag.actor_optimizer.param_groups:
                            pg['lr'] = 1e-4 * factor
                        for pg in ag.critic_optimizer.param_groups:
                            pg['lr'] = 3e-4 * factor
                    print(f"  [LR DECAY] train_steps={trainer.train_steps:,}, "
                          f"actor_lr -> {1e-4*factor:.2e}, critic_lr -> {3e-4*factor:.2e}")

            # Periodic eval (reduced: 20 eps every 2000 to cut overhead)
            if (episode + 1) % 2000 == 0 and episode > 0:
                eval_team = 0
                n_eval = 20
                for _ in range(n_eval):
                    eo, _ = env.reset()
                    for _ in range(env.max_episode_steps):
                        ea = trainer.get_actions(eo, noise_scale=0.0)
                        eo, _, et, etr, _ = env.step(ea)
                        if all(et.values()) or any(etr.values()):
                            break
                    if all(s.get('frozen', False) for s in env.robot_states.values()):
                        eval_team += 1
                print(f"  [EVAL] Ep {episode+1}: team_success={eval_team/n_eval:.0%}, phase={phase}")

            # Per-robot tracking
            robot_episode_count += 1
            for i, (name, state) in enumerate(sorted(env.robot_states.items())):
                if state.get('reached', False):
                    robot_success_count[f'robot_{i}'] += 1

            avg_final_dist = np.mean([s.get('distance_to_goal', 0.0) for s in env.robot_states.values()])
            robot_rewards = {f'robot_{i}_reward': cumulative_robot_rewards[name]
                             for i, name in enumerate(sorted(cumulative_robot_rewards.keys()))}

            success_history.append(success_fraction)
            phase_episode_count[phase] += 1
            phase_success_sum[phase] += success_fraction
            phase_team_success_sum[phase] += team_success
            _phase_success_cache[phase].append(success_fraction)
            _phase_team_cache[phase].append(team_success)
            _phase_onetimeout_cache[phase].append(1.0 if (success_fraction == 0.5 and end_reason == "time_limit") else 0.0)
            _phase_catastrophe_cache[phase].append(1.0 if is_catastrophe else 0.0)

            episode_data = {
                'episode': episode + 1,
                'reward': episode_reward,
                'success_fraction': success_fraction,
                'team_success': team_success,
                'partial_success': partial_success,
                'goals_reached': goals_reached,
                'actor_loss': avg_actor_loss,
                'critic_loss': avg_critic_loss,
                'steps': episode_steps,
                'collisions': episode_collisions,
                'avg_final_distance': avg_final_dist,
                'noise': noise_scale,
                'phase': phase,
                'end_reason': end_reason,
                'replay_skipped_collision': int(skip_store),
                'is_catastrophe': int(is_catastrophe),
                'phase_collision_skips': phase_collision_skips[phase],
                'phase_replay_stored': phase_replay_stored[phase],
                'q_target_mean': train_diag.get('q_target_mean', 0.0),
                'q_pred_mean': train_diag.get('q_pred_mean', 0.0),
                'td_abs_mean': train_diag.get('td_abs_mean', 0.0),
                'actor_saturation': train_diag.get('actor_saturation', 0.0),
                'actor_grad_norm': train_diag.get('actor_grad_norm', 0.0),
                'critic_grad_norm': train_diag.get('critic_grad_norm', 0.0),
                'action_time_s': episode_action_time,
                'env_time_s': episode_env_time,
                'store_time_s': episode_store_time,
                'train_time_s': episode_train_time,
                'one_frozen_steps': one_frozen_steps,
                'partial_frozen_steps': partial_frozen_steps,
                'frozen_reward_sum': frozen_reward_sum,
                'waiting_reward_base_sum': waiting_reward_base_sum,
                'other_frozen_bonus_est_sum': other_frozen_bonus_est_sum,
                **robot_rewards,
            }
            episode_data.update({f'term_{key}': value for key, value in episode_reward_terms.items()})
            all_episodes.append(episode_data)

            # Print every 10
            if (episode + 1) % 10 == 0:
                print(f"{episode+1:5d} | {success_fraction:6.2f} | {team_success:6.0f} | "
                      f"{episode_reward:10.2f} | {avg_actor_loss:8.4f} | {avg_critic_loss:8.4f} | "
                      f"{episode_steps:6d} | {episode_collisions:5d} | {phase:>8}")

            # Block summary every 100
            if (episode + 1) % 100 == 0:
                blk = all_episodes[-100:]
                bf = np.mean([e['success_fraction'] for e in blk])
                bt = np.mean([e['team_success'] for e in blk])
                br = np.mean([e['reward'] for e in blk])
                bc = np.mean([e['collisions'] for e in blk])
                bs = np.mean([e['steps'] for e in blk])
                bskip = np.mean([e['replay_skipped_collision'] for e in blk])
                btd = np.mean([e['td_abs_mean'] for e in blk])
                bsat = np.mean([e['actor_saturation'] for e in blk])
                pe = phase_episode_count[phase]
                pa = (phase_success_sum[phase] / pe) if pe > 0 else 0.0
                pt = (phase_team_success_sum[phase] / pe) if pe > 0 else 0.0
                elapsed = time.time() - start_time
                eps_sec = (episode + 1) / elapsed if elapsed > 0 else 0
                print(f"\n  ╔═══════════════════════════════════════════════════════════════════════════════╗")
                print(f"  ║ BLOCK {(episode+1)//100:3d}: Ep {episode-98:5d}-{episode+1:<5d}  │  Speed: {eps_sec:.1f} eps/sec           ║")
                print(f"  ╠═══════════════════════════════════════════════════════════════════════════════╣")
                print(f"  ║  Success: {bf:6.1%}  │  Team: {bt:6.1%}  │  Noise: {noise_scale:.3f}                 ║")
                print(f"  ║  Reward: {br:9.1f}  │  Steps: {bs:6.0f}    │  Phase: {phase:>8}                ║")
                print(f"  ║  Coll:   {bc:6.1f}  │  Buffer: {len(trainer.replay_buffer):6d}                                   ║")
                print(f"  ║  Skip:   {bskip:6.1%}  │  Stored: {phase_replay_stored[phase]:6d}  │  SkipCnt: {phase_collision_skips[phase]:4d}         ║")
                print(f"  ║  TD|err|:{btd:7.3f}  │  Sat: {bsat:6.1%}  │  Qμ: {train_diag.get('q_pred_mean', 0.0):7.3f}           ║")
                print(f"  ║  Phase Overall: {pa:6.1%}  │  Phase Team: {pt:6.1%}  │  Phase Eps: {pe:5d}     ║")
                print(f"  ╚═══════════════════════════════════════════════════════════════════════════════╝\n")
                sys.stdout.flush()

            if (episode + 1) % 500 == 0 and timing_counts['steps'] > 0:
                avg_action_ms = 1000.0 * timing_acc['action'] / max(1, timing_counts['steps'])
                avg_env_ms = 1000.0 * timing_acc['env'] / max(1, timing_counts['steps'])
                avg_store_ms = 1000.0 * timing_acc['store'] / max(1, timing_counts['store_calls']) if timing_counts['store_calls'] > 0 else 0.0
                avg_train_ms = 1000.0 * timing_acc['train'] / max(1, timing_counts['train_calls']) if timing_counts['train_calls'] > 0 else 0.0
                print(
                    f"  [TIMING] Ep {episode+1}: action={avg_action_ms:.3f}ms/step, "
                    f"env={avg_env_ms:.3f}ms/step, store={avg_store_ms:.3f}ms/call, train={avg_train_ms:.3f}ms/call"
                )

            # ── Live gate metrics every 500 episodes ──
            if (episode + 1) % 500 == 0:
                _sc_cp = _phase_success_cache[phase]
                _tc_cp = _phase_team_cache[phase]
                _ot_cp = _phase_onetimeout_cache[phase]
                _cat_cp = _phase_catastrophe_cache[phase]
                w = rolling_mastery_window
                if len(_sc_cp) >= w:
                    r500_ind = float(np.mean(list(_sc_cp)[-w:]))
                    r500_team = float(np.mean(list(_tc_cp)[-w:]))
                    r500_onetimeout = float(np.mean(list(_ot_cp)[-w:]))
                    r500_catastrophe = float(np.mean(list(_cat_cp)[-w:]))
                    pe_cp = phase_episode_count[phase]
                    r500_skip = phase_collision_skips[phase] / max(1, pe_cp)
                    print(f"\n  ┌─ GATE [{phase}] ep {episode+1} ─────────────────────────────────┐")
                    print(f"  │  Ind success  (500w): {r500_ind:6.1%}  │  threshold: {phase_config[phase]['threshold']:.0%}")
                    print(f"  │  Team success (500w): {r500_team:6.1%}  │")
                    print(f"  │  1-goal timeout (500w): {r500_onetimeout:6.1%}  │")
                    print(f"  │  Catastrophe 100+ (500w): {r500_catastrophe:6.1%}  │")
                    print(f"  │  Replay-skipped (phase): {r500_skip:6.1%}  │  phase eps: {pe_cp}")
                    print(f"  └───────────────────────────────────────────────────────┘")

            # Dead actor recovery
            if (episode + 1) % 300 == 0 and robot_episode_count >= 300:
                pr = {}
                recent = all_episodes[-300:]
                for i in range(n_agents):
                    key = f'robot_{i}_reward'
                    pr[f'robot_{i}'] = sum(1 for e in recent if e.get(key, 0) > 0.1) / 300
                trainer.recover_dead_actors(pr, threshold=0.10)

            # Refresh success buffer
            if (episode + 1) % 2000 == 0:
                trainer.replay_buffer.refresh_success_buffer(keep_ratio=0.5)

            # ── CURRICULUM ADVANCEMENT ──────────────────────────────────────
            config = phase_config[phase]
            max_ep = config.get("max_episodes", 999999)
            pe = phase_episode_count[phase]

            # Mastery evaluation (O(1) from cached deques, not O(n) scan)
            _sc = _phase_success_cache[phase]
            _tc = _phase_team_cache[phase]
            if rolling_mastery_window > 0 and len(_sc) >= rolling_mastery_window:
                mastery_ind = float(np.mean(list(_sc)[-rolling_mastery_window:]))
                mastery_team = float(np.mean(list(_tc)[-rolling_mastery_window:]))
            else:
                mastery_ind = (phase_success_sum[phase] / pe) if pe > 0 else 0.0
                mastery_team = (phase_team_success_sum[phase] / pe) if pe > 0 else 0.0

            phase_avg_success = (phase_success_sum[phase] / pe) if pe > 0 else 0.0
            phase_avg_team = (phase_team_success_sum[phase] / pe) if pe > 0 else 0.0

            # Freeze-on-mastery (O(1) from cached deques)
            if not phase_frozen[phase] and len(_sc) >= FREEZE_MASTERY_WINDOW:
                fi = float(np.mean(list(_sc)[-FREEZE_MASTERY_WINDOW:]))
                ft = float(np.mean(list(_tc)[-FREEZE_MASTERY_WINDOW:]))
                if fi >= 0.90 and ft >= 0.90:
                    phase_frozen[phase] = True
                    print(f"\n  ╔ FREEZE-ON-MASTERY: Phase {phase} LOCKED (Ind={fi:.1%} Team={ft:.1%}) ╗\n")

            should_advance = False
            advance_reason = ""

            if pe >= config["min"]:
                thr = config["threshold"]
                if mastery_ind >= thr and mastery_team >= thr:
                    should_advance = True
                    advance_reason = f"MASTERY: Ind {mastery_ind:.1%} + Team {mastery_team:.1%} >= {thr:.0%}"

            # No episode cap — mastery-only advancement (no forced skipping)

            # STUCK WARNING: warn every 10K eps if stuck in any phase
            if pe > 0 and pe % 10000 == 0 and not should_advance:
                print(f"\n  ⚠ STUCK WARNING: Phase {phase}, {pe:,} eps, "
                      f"Ind {phase_avg_success:.1%}, Team {phase_avg_team:.1%} — not advancing\n")

            if phase_regression_cooldown[phase] > 0:
                phase_regression_cooldown[phase] -= 1

            if (
                current_phase_idx > 0 and
                pe >= max(config['min'], 4000) and
                not should_advance and
                len(_sc) >= min(rolling_mastery_window, 500) and
                len(_tc) >= min(rolling_mastery_window, 500) and
                phase_regression_cooldown[phase] == 0
            ):
                recent_ind = float(np.mean(list(_sc)[-min(rolling_mastery_window, 500):]))
                recent_team = float(np.mean(list(_tc)[-min(rolling_mastery_window, 500):]))
                if recent_ind < 0.45 and recent_team < 0.25:
                    prev_phase = phase_list[current_phase_idx - 1]
                    phase_regression_cooldown[phase] = 1500
                    current_phase_idx -= 1
                    print(
                        f"\n[PHASE REGRESS] {phase} stalled after {pe:,} eps: "
                        f"recent_ind={recent_ind:.1%}, recent_team={recent_team:.1%}. "
                        f"Returning to {prev_phase}.\n"
                    )
                    env.set_phase(prev_phase)
                    trainer.replay_buffer.set_phase(prev_phase)
                    continue

            if should_advance and current_phase_idx < len(phase_list) - 1:
                print(f"\n[PHASE ADVANCE] {phase}: {pe:,} eps, {phase_step_count[phase]:,} steps")
                print(f"                {advance_reason}")
                current_phase_idx += 1
                next_phase = phase_list[current_phase_idx]
                env.set_phase(next_phase)
                trainer.replay_buffer.set_phase(next_phase)
                print(f"[PHASE START] {next_phase}\n")

            # CSV every 50 episodes
            if (episode + 1) % 50 == 0:
                with open(csv_file, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=episode_data.keys())
                    if not csv_header_written:
                        writer.writeheader()
                        csv_header_written = True
                    for ed in all_episodes[last_written_idx:]:
                        writer.writerow(ed)
                    last_written_idx = len(all_episodes)

            # Checkpoint + plots every 5000
            if (episode + 1) % 5000 == 0:
                ckpt_path = f"{results_dir}/checkpoints/checkpoint_ep{episode+1}.pth"
                trainer.save_checkpoint(ckpt_path, episode + 1)
                gd = f"{results_dir}/graphs"
                if os.path.exists(gd):
                    for old in os.listdir(gd):
                        if old.endswith('.png'):
                            os.remove(os.path.join(gd, old))
                if len(all_episodes) > 100:
                    generate_plots(all_episodes, results_dir, phase_episode_count,
                                   phase_success_sum, phase_list, phase_colors)

    # Final CSV flush
    if last_written_idx < len(all_episodes):
        with open(csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_episodes[0].keys())
            if not csv_header_written:
                writer.writeheader()
            for ed in all_episodes[last_written_idx:]:
                writer.writerow(ed)

    # Close environment(s)
    if use_parallel:
        if env_manager is not None:
            env_manager.close()
    else:
        if env is not None:
            env.close()
    
    elapsed = time.time() - start_time

    print("\n" + "="*75)
    print(f"[COMPLETE] {n_episodes} episodes in {elapsed/3600:.2f} hours")

    final_200 = all_episodes[-200:] if len(all_episodes) >= 200 else all_episodes
    print(f"[RESULT] Final 200: Ind {np.mean([e['success_fraction'] for e in final_200]):.1%}, "
          f"Team {np.mean([e['team_success'] for e in final_200]):.1%}")

    print(f"\n[PHASE STATS]")
    print(f"{'Phase':>8} | {'Episodes':>10} | {'Avg Success':>12}")
    print("-" * 40)
    for ph in phase_list:
        if phase_episode_count[ph] > 0:
            avg = phase_success_sum[ph] / phase_episode_count[ph]
            print(f"{ph:>8} | {phase_episode_count[ph]:>10} | {avg:>11.1%}")

    print(f"\n[PER-ROBOT SUCCESS]")
    for i in range(n_agents):
        ravg = robot_success_count[f'robot_{i}'] / max(1, robot_episode_count)
        print(f"  Robot {i}: {ravg:.1%}")

    json_file = f"{results_dir}/training_data.json"
    with open(json_file, 'w') as f:
        json.dump(all_episodes, f, indent=2)

    generate_plots(all_episodes, results_dir, phase_episode_count,
                   phase_success_sum, phase_list, phase_colors)

    print(f"\n[SAVE] All results: {results_dir}/")
    return all_episodes


# ==================== PLOTTING ====================
def generate_plots(episodes, results_dir, phase_episode_count, phase_success_sum,
                   phase_list, phase_colors):
    """Generate 6 comprehensive plots."""
    gd = f"{results_dir}/graphs"
    if len(episodes) < 2:
        return

    ep_nums = np.array([e['episode'] for e in episodes])
    rewards = np.array([e['reward'] for e in episodes])
    success = np.array([e['success_fraction'] for e in episodes])
    team_success = np.array([e['team_success'] for e in episodes])
    actor_losses = np.array([e['actor_loss'] for e in episodes])
    critic_losses = np.array([e['critic_loss'] for e in episodes])
    steps_arr = np.array([e['steps'] for e in episodes])
    collisions = np.array([e['collisions'] for e in episodes])
    phases = np.array([e['phase'] for e in episodes])
    window = min(50, len(episodes))

    def shade(ax, ep_nums, phases):
        cur = phases[0]; si = 0
        for i, ph in enumerate(phases):
            if ph != cur or i == len(phases) - 1:
                ei = i if i < len(phases) - 1 else len(phases)
                c = phase_colors.get(cur, '#ffffff')
                ax.axvspan(ep_nums[si], ep_nums[min(ei, len(ep_nums)-1)], alpha=0.3, color=c)
                cur = ph; si = i

    # 1. Success rate
    fig, ax = plt.subplots(figsize=(12, 6))
    shade(ax, ep_nums, phases)
    ax.plot(ep_nums, success, alpha=0.2, color='green', label='Raw')
    if len(success) >= window >= 2:
        ma = np.convolve(success, np.ones(window)/window, mode='valid')
        ax.plot(ep_nums[window-1:], ma, lw=2, color='darkgreen', label=f'MA-{window}')
        mt = np.convolve(team_success, np.ones(window)/window, mode='valid')
        ax.plot(ep_nums[window-1:], mt, lw=2, color='purple', ls='--', label=f'Team MA-{window}')
    ax.axhline(0.8, color='r', ls='--', alpha=0.7, lw=1.5, label='Target 80%')
    ax.set_xlabel('Episode'); ax.set_ylabel('Success Fraction')
    ax.set_title('Robot Success Rate Over Training', fontweight='bold')
    ax.set_ylim(0, 1.05); ax.legend(loc='lower right'); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{gd}/01_success_rate.png", dpi=150, bbox_inches='tight'); plt.close()

    # 2. Reward
    fig, ax = plt.subplots(figsize=(12, 6))
    shade(ax, ep_nums, phases)
    ax.plot(ep_nums, rewards, alpha=0.15, color='blue', label='Raw')
    if len(rewards) >= window >= 2:
        ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(ep_nums[window-1:], ma, lw=2, color='darkblue', label=f'MA-{window}')
    ax.set_xlabel('Episode'); ax.set_ylabel('Episode Reward')
    ax.set_title('Reward Convergence', fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{gd}/02_reward_convergence.png", dpi=150, bbox_inches='tight'); plt.close()

    # 3. Actor / Critic loss
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    vm = (actor_losses != 0) | (critic_losses != 0)
    ve, va, vc = ep_nums[vm], actor_losses[vm], critic_losses[vm]
    if len(ve) >= window >= 2:
        ma_a = np.convolve(va, np.ones(window)/window, mode='valid')
        ma_c = np.convolve(vc, np.ones(window)/window, mode='valid')
        a1.plot(ve, va, alpha=0.2, color='orange')
        a1.plot(ve[window-1:], ma_a, lw=2, color='darkorange', label=f'Actor MA-{window}')
        a1.set_ylabel('Actor Loss'); a1.legend(); a1.grid(True, alpha=0.3)
        a1.set_title('Actor & Critic Loss', fontweight='bold')
        a2.plot(ve, vc, alpha=0.2, color='red')
        a2.plot(ve[window-1:], ma_c, lw=2, color='darkred', label=f'Critic MA-{window}')
        a2.set_xlabel('Episode'); a2.set_ylabel('Critic Loss'); a2.legend(); a2.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{gd}/03_actor_critic_loss.png", dpi=150, bbox_inches='tight'); plt.close()

    # 4. Phase bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    avgs, counts, cols = [], [], []
    for ph in phase_list:
        cnt = phase_episode_count.get(ph, 0)
        avgs.append((phase_success_sum[ph] / cnt) if cnt > 0 else 0)
        counts.append(cnt)
        cols.append(phase_colors.get(ph, '#e0e0e0'))
    x = np.arange(len(phase_list))
    bars = ax.bar(x, avgs, color=cols, edgecolor='black', lw=1)
    for i, (bar, cnt) in enumerate(zip(bars, counts)):
        if cnt > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{cnt} eps', ha='center', va='bottom', fontsize=9)
    ax.axhline(0.8, color='r', ls='--', alpha=0.7, label='Target 80%')
    ax.set_xlabel('Phase'); ax.set_ylabel('Avg Success')
    ax.set_title('Performance by Phase', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(phase_list, rotation=45)
    ax.set_ylim(0, 1.15); ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout(); plt.savefig(f"{gd}/04_phase_performance.png", dpi=150, bbox_inches='tight'); plt.close()

    # 5. Collisions + steps
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    a1.plot(ep_nums, collisions, alpha=0.15, color='red')
    if len(collisions) >= window >= 2:
        mc = np.convolve(collisions, np.ones(window)/window, mode='valid')
        a1.plot(ep_nums[window-1:], mc, lw=2, color='darkred', label=f'Coll MA-{window}')
    a1.set_ylabel('Collisions'); a1.set_title('Safety & Efficiency', fontweight='bold')
    a1.legend(); a1.grid(True, alpha=0.3)
    a2.plot(ep_nums, steps_arr, alpha=0.15, color='teal')
    if len(steps_arr) >= window >= 2:
        ms = np.convolve(steps_arr, np.ones(window)/window, mode='valid')
        a2.plot(ep_nums[window-1:], ms, lw=2, color='darkcyan', label=f'Steps MA-{window}')
    a2.set_xlabel('Episode'); a2.set_ylabel('Steps'); a2.legend(); a2.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{gd}/05_safety_efficiency.png", dpi=150, bbox_inches='tight'); plt.close()

    # 6. Per-robot reward trend
    fig, ax = plt.subplots(figsize=(12, 6))
    rk = [k for k in episodes[0] if k.startswith('robot_') and k.endswith('_reward')]
    ri = sorted({int(k.split('_')[1]) for k in rk if k.split('_')[1].isdigit()})
    rc = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
    for pos, i in enumerate(ri):
        key = f'robot_{i}_reward'
        rr = np.array([e.get(key, 0.0) for e in episodes])
        if len(rr) >= window >= 2:
            mr = np.convolve(rr, np.ones(window)/window, mode='valid')
            ax.plot(ep_nums[window-1:], mr, lw=2, color=rc[pos % len(rc)],
                    label=f'Robot {i}', alpha=0.8)
    ax.set_xlabel('Episode'); ax.set_ylabel('Per-Episode Robot Reward')
    ax.set_title('Per-Robot Reward Trend', fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{gd}/06_per_robot_success.png", dpi=150, bbox_inches='tight'); plt.close()

    print(f"\n[PLOTS] 6 graphs saved to: {gd}/")


# ==================== MAIN ====================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='MADDPG V3 — Dual-Mode Training')
    parser.add_argument('--episodes', type=int, default=999999)
    parser.add_argument('--training-steps', type=int, default=4,
                        help='Training steps per episode (V2 CLI default: 4)')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--results-tag', type=str, default=None,
                        help='Optional results folder suffix override')

    # ── MODE SWITCHES (the key new flags) ──
    parser.add_argument('--reward-mode', type=str, default='v2_baseline',
                        choices=['v2_baseline'],
                        help='Reward function mode: v2_baseline (goal-first reward hierarchy)')
    parser.add_argument('--curriculum-mode', type=str, default='v3_smooth',
                        choices=['v2_original', 'v3_smooth'],
                        help='Curriculum mode: v2_original (9 phases, mixed thresholds, 15K cap) '
                             'or v3_smooth (8 phases, 0.75m unified, 70%% gate, no cap)')

    # ── EXISTING ABLATION FLAGS (carried from V2) ──
    parser.add_argument('--treat-truncations-as-terminals', action='store_true')
    parser.add_argument('--mask-actor-on-terminated', action='store_true')
    parser.add_argument('--success-buffer-phase-filter', action='store_true')
    parser.add_argument('--success-tail-fraction', type=float, default=1.0)
    parser.add_argument('--enable-waiting-reward', action='store_true')
    parser.add_argument('--skip-frozen-transitions', action='store_true')
    parser.add_argument('--rolling-mastery-window', type=int, default=500)
    parser.add_argument('--enable-plateau-noise', action='store_true')
    parser.add_argument('--enable-waiting-shaping', action='store_true')
    parser.add_argument('--partial-timeout-penalty', type=float, default=10.0,
                        help='Per-agent terminal penalty when exactly one robot reaches and the episode times out')
    parser.add_argument('--zero-timeout-penalty', type=float, default=6.0,
                        help='Per-agent terminal penalty when no robots reach and the episode times out')
    
    # ── PARALLEL ROLLOUT & CPU BUDGET ──
    parser.add_argument('--num-parallel-envs', type=int, default=1,
                        help='Number of parallel environment workers (default: 1 = single env)')
    parser.add_argument('--torch-threads', type=int, default=None,
                        help='PyTorch intra-op CPU threads (default: auto = cpu_count - num_parallel_envs)')
    parser.add_argument('--torch-interop-threads', type=int, default=1,
                        help='PyTorch inter-op CPU threads (default: 1)')

    args = parser.parse_args()

    print("\n" + "="*70)
    print(f"MADDPG V3 — {args.reward_mode.upper()} + {args.curriculum_mode.upper()}")
    print(f"           {args.episodes:,} EPISODES")
    print("="*70)

    train_maddpg_v3(
        n_episodes=args.episodes,
        n_training_steps=args.training_steps,
        batch_size=args.batch_size,
        resume_from=args.resume,
        reward_mode=args.reward_mode,
        curriculum_mode=args.curriculum_mode,
        treat_truncations_as_terminals=args.treat_truncations_as_terminals,
        mask_actor_on_terminated=args.mask_actor_on_terminated,
        success_buffer_phase_filter=args.success_buffer_phase_filter,
        success_tail_fraction=args.success_tail_fraction,
        enable_waiting_reward=args.enable_waiting_reward,
        skip_frozen_transitions=args.skip_frozen_transitions,
        rolling_mastery_window=args.rolling_mastery_window,
        disable_plateau_noise=(not args.enable_plateau_noise),
        disable_waiting_shaping=(not args.enable_waiting_shaping),
        partial_timeout_penalty=args.partial_timeout_penalty,
        zero_timeout_penalty=args.zero_timeout_penalty,
        num_parallel_envs=args.num_parallel_envs,
        torch_threads=args.torch_threads,
        torch_interop_threads=args.torch_interop_threads,
        results_tag=args.results_tag,
    )

    print(f"\n✓ MADDPG V3 Training Complete!")
