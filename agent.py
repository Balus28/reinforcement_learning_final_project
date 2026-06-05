# =============================================================================
# YOUR FILE — this is the only file you submit.
# Implement TradingEnv and Agent below. Do not modify anything in src/.
# =============================================================================

"""
agent.py

Primera propuesta deimplementación para el proyecto de asignación de portafolio
con aprendizaje por refuerzo.

Adiciones:
    - TradingEnv: subclase de BaseTradingEnv
    - Agent: agente Double DQN con espacio de acciones discreto

Diseño metodológico:
    - Estado: ventana histórica de retornos logarítmicos + pesos actuales.
    - Acción: menú discreto e interpretable de portafolios.
    - Recompensa: log-retorno neto del portafolio, con penalización opcional por turnover.
    - Algoritmo: Double DQN con replay buffer y target network.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gymnasium import spaces


try:
    from src.base import BaseAgent
    from src.env import BaseTradingEnv
except ModuleNotFoundError:
    from base import BaseAgent
    from env import BaseTradingEnv



# 1. Entorno de trading: primera implementación específica para el agente.



class TradingEnv(BaseTradingEnv):
    """
    Entorno específico del agente.

    La clase base ya gestiona:
        - avance temporal;
        - valor del portafolio;
        - costos de transacción;
        - turnover;
        - restricciones sobre pesos.

    Esta subclase define:
        - observación del agente;
        - menú discreto de acciones;
        - función de recompensa.
    """

    ACTIONS = np.array(
        [
            [0.00, 0.00, 0.00, 1.00],        # 0: 100% cash
            [1.00, 0.00, 0.00, 0.00],        # 1: 100% asset_0
            [0.00, 1.00, 0.00, 0.00],        # 2: 100% asset_1
            [0.00, 0.00, 1.00, 0.00],        # 3: 100% asset_2
            [1/3, 1/3, 1/3, 0.00],           # 4: equal weight risky

            [0.50, 0.50, 0.00, 0.00],        # 5: long asset_0 + asset_1
            [0.50, 0.00, 0.50, 0.00],        # 6: long asset_0 + asset_2
            [0.00, 0.50, 0.50, 0.00],        # 7: long asset_1 + asset_2

            [0.25, 0.25, 0.25, 0.25],        # 8: balanced risky/cash
            [0.50, 0.00, 0.00, 0.50],        # 9: defensive asset_0
            [0.00, 0.50, 0.00, 0.50],        # 10: defensive asset_1
            [0.00, 0.00, 0.50, 0.50],        # 11: defensive asset_2

            [-0.25, 0.50, 0.25, 0.50],       # 12: short asset_0, long asset_1/2, cash buffer
            [0.50, -0.25, 0.25, 0.50],       # 13: short asset_1, long asset_0/2, cash buffer
            [0.50, 0.25, -0.25, 0.50],       # 14: short asset_2, long asset_0/1, cash buffer
        ],
        dtype=np.float32,
    )

    def __init__(
        self,
        prices,
        transaction_cost_bps: float = 10.0,
        initial_cash: float = 10_000.0,
        lookback: int = 20,
        turnover_penalty: float = 0.0,
    ):
        self._lookback = lookback
        self.turnover_penalty = turnover_penalty
        self._last_turnover = 0.0

        super().__init__(
            prices=prices,
            transaction_cost_bps=transaction_cost_bps,
            initial_cash=initial_cash,
        )

        close = self.prices[:, :3].astype(np.float32)
        log_returns = np.zeros_like(close, dtype=np.float32)
        log_returns[1:] = np.log((close[1:] + 1e-8) / (close[:-1] + 1e-8))

        self._log_returns = log_returns

        obs_dim = self._lookback * 3 + 4
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(len(self.ACTIONS))

    @property
    def n_actions(self) -> int:
        return len(self.ACTIONS)

    def _obs(self) -> np.ndarray:
        start = self._t - self._lookback
        end = self._t

        ret_window = self._log_returns[start:end].reshape(-1)
        obs = np.concatenate([ret_window, self._weights], axis=0)

        return obs.astype(np.float32)

    def _weights_from_action(self, action: int) -> np.ndarray:
        action_idx = int(action)
        if action_idx < 0 or action_idx >= len(self.ACTIONS):
            raise ValueError(f"Invalid action index: {action_idx}")

        weights = self.ACTIONS[action_idx].copy()
        self._last_turnover = float(np.abs(weights - self._weights).sum())

        return weights

    def _reward(self, prev_value: float, curr_value: float) -> float:
        log_ret = float(np.log((curr_value + 1e-8) / (prev_value + 1e-8)))
        return log_ret - self.turnover_penalty * self._last_turnover



# 2. Red neuronal Q: arquitectura simple de MLP con heads separados para valor y ventaja (Dueling DQN).



class QNetwork(nn.Module):
    """
    Red neuronal para aproximar Q(s, a).

    Entrada:
        vector de observación.

    Salida:
        un Q-value por cada acción discreta del menú de portafolios.
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()

        self.feature_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.value_head = nn.Linear(hidden_dim, 1)
        self.advantage_head = nn.Linear(hidden_dim, n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.feature_net(obs)

        value = self.value_head(features)
        advantage = self.advantage_head(features)

        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values


# 3. Replay buffer: historial de transacciones para entrenamiento off-policy.


@dataclass
class Transition:
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool


class ReplayBuffer:
    """
    Replay buffer simple para DQN.

    Guarda transiciones históricas y permite muestrear minibatches aleatorios.
    Esto reduce correlación temporal entre pasos consecutivos.
    """

    def __init__(self, capacity: int = 50_000):
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append(
            Transition(
                obs=obs.astype(np.float32),
                action=int(action),
                reward=float(reward),
                next_obs=next_obs.astype(np.float32),
                done=bool(done),
            )
        )

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


# 4. Agente Double DQN


class Agent(BaseAgent):
    """
    Agente Double DQN.

    Este agente aprende una función Q sobre un menú discreto de portafolios.
    La política final selecciona la acción con mayor Q-value estimado.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        seed: int = 42,
        lr: float = 1e-3,
        gamma: float = 0.99,
        batch_size: int = 64,
        buffer_capacity: int = 50_000,
        min_replay_size: int = 1_000,
        target_update_freq: int = 500,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 50_000,
    ):
        super().__init__(obs_dim=obs_dim, n_actions=n_actions)

        self.seed = seed
        self.gamma = gamma
        self.batch_size = batch_size
        self.min_replay_size = min_replay_size
        self.target_update_freq = target_update_freq

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net = QNetwork(obs_dim, n_actions).to(self.device)
        self.target_net = QNetwork(obs_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay = ReplayBuffer(capacity=buffer_capacity)

        self.total_steps = 0

    def _epsilon(self) -> float:
        frac = min(1.0, self.total_steps / self.epsilon_decay_steps)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def _select_action(self, obs: np.ndarray, explore: bool = True) -> int:
        if explore and np.random.rand() < self._epsilon():
            return int(np.random.randint(self.n_actions))

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            q_values = self.q_net(obs_t)

        return int(torch.argmax(q_values, dim=1).item())

    def act(self, obs: np.ndarray) -> int:
        return self._select_action(obs, explore=False)

    def train(self, env, n_steps: int = 200_000) -> None:
        obs, _ = env.reset(seed=self.seed)

        for step in range(1, n_steps + 1):
            self.total_steps += 1

            action = self._select_action(obs, explore=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            self.replay.push(obs, action, reward, next_obs, done)

            obs = next_obs

            if done:
                obs, _ = env.reset()

            if len(self.replay) >= self.min_replay_size:
                self._learn_step()

            if step % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())

    def _learn_step(self) -> None:
        batch = self.replay.sample(self.batch_size)

        obs = torch.as_tensor(
            np.stack([t.obs for t in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.as_tensor(
            [t.action for t in batch],
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(1)
        rewards = torch.as_tensor(
            [t.reward for t in batch],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(1)
        next_obs = torch.as_tensor(
            np.stack([t.next_obs for t in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        dones = torch.as_tensor(
            [t.done for t in batch],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(1)

        q_current = self.q_net(obs).gather(1, actions)

        with torch.no_grad():
            next_actions = self.q_net(next_obs).argmax(dim=1, keepdim=True)
            q_next = self.target_net(next_obs).gather(1, next_actions)
            q_target = rewards + (1.0 - dones) * self.gamma * q_next

        loss = F.smooth_l1_loss(q_current, q_target)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()