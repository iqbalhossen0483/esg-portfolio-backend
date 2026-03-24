"""DRL training loop: trains 3 PPO agents + CNN fusion."""

import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

from .environment import PortfolioEnv
from .models.mappo import MAPPO

from config import settings


def train_mappo(
    features: np.ndarray,
    returns: np.ndarray,
    sector_ids: np.ndarray,
    esg_scores: np.ndarray,
    esg_lambda: float = 0.5,
    episodes: int = 500,
    lr: float = 1e-3,
    gamma: float = 0.99,
    clip_eps: float = 0.2,
    train_ratio: float = 0.7,
) -> dict:
    """Train the MAPPO model.

    Args:
        features: Shape [T, N, F].
        returns: Shape [T, N].
        sector_ids: Shape [N].
        esg_scores: Shape [N].
        esg_lambda: ESG importance weight.
        episodes: Number of training episodes.
        lr: Learning rate.
        gamma: Discount factor.
        clip_eps: PPO clip range.
        train_ratio: Fraction of data for training.

    Returns:
        Dict with model_path, training metrics, etc.
    """
    T, N, F = features.shape
    train_T = int(T * train_ratio)

    train_features = features[:train_T]
    train_returns = returns[:train_T]

    model = MAPPO(n_assets=N, feat_dim=F, lr=lr)
    training_history = []

    for episode in range(episodes):
        env = PortfolioEnv(
            features=train_features,
            returns=train_returns,
            sector_ids=sector_ids,
            esg_scores=esg_scores,
            esg_lambda=esg_lambda,
        )

        obs = env.reset()
        episode_rewards = {"combined": [], "return": [], "sharpe": [], "drawdown": []}
        states, actions, rewards_list, values_list = [], [], [], []

        done = False
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32)

            # Get actions from each agent
            logits1, val1 = model.agent1(obs_tensor)
            logits2, val2 = model.agent2(obs_tensor)
            logits3, val3 = model.agent3(obs_tensor)

            # Convert logits to weights
            w1 = torch.softmax(logits1, dim=-1).detach().numpy()
            w2 = torch.softmax(logits2, dim=-1).detach().numpy()
            w3 = torch.softmax(logits3, dim=-1).detach().numpy()

            # Fuse weights
            stacked = torch.stack([
                torch.tensor(w1), torch.tensor(w2), torch.tensor(w3)
            ], dim=-1).unsqueeze(0)
            final_w = model.fusion(stacked).squeeze(0).detach().numpy()

            # Step environment
            next_obs, reward_dict, done = env.step(final_w)

            if not done:
                states.append(obs_tensor)
                actions.append(final_w)
                for key in episode_rewards:
                    episode_rewards[key].append(reward_dict.get(key, 0))

            obs = next_obs

        if not states:
            continue

        # PPO update for each agent with its own reward signal
        reward_keys = ["return", "sharpe", "drawdown"]
        agents = [model.agent1, model.agent2, model.agent3]
        optimizers = [model.opt1, model.opt2, model.opt3]

        for agent, optimizer, rkey in zip(agents, optimizers, reward_keys):
            agent_rewards = episode_rewards[rkey]
            if not agent_rewards:
                continue

            # Compute discounted returns
            G = []
            g = 0
            for r in reversed(agent_rewards):
                g = r + gamma * g
                G.insert(0, g)
            G = torch.tensor(G, dtype=torch.float32)

            # Normalize returns
            if G.std() > 0:
                G = (G - G.mean()) / (G.std() + 1e-8)

            # Forward pass
            state_batch = torch.stack(states[:len(G)])
            logits_batch, values_batch = agent(state_batch)
            probs = torch.softmax(logits_batch, dim=-1)

            # Simple policy gradient (PPO-simplified)
            advantages = G - values_batch.detach()
            log_probs = torch.log(probs + 1e-8).sum(dim=-1)
            actor_loss = -(log_probs * advantages).mean()
            critic_loss = nn.functional.mse_loss(values_batch, G)

            loss = actor_loss + 0.5 * critic_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5)
            optimizer.step()

        # Train fusion layer
        if states:
            state_batch = torch.stack(states)
            with torch.no_grad():
                w1_batch = model.agent1.get_weights(state_batch)
                w2_batch = model.agent2.get_weights(state_batch)
                w3_batch = model.agent3.get_weights(state_batch)
            stacked_batch = torch.stack([w1_batch, w2_batch, w3_batch], dim=-1)

            fused = model.fusion(stacked_batch)
            combined_returns = torch.tensor(
                episode_rewards["combined"][:len(fused)], dtype=torch.float32
            )

            # Maximize combined reward via fused weights
            returns_tensor = torch.tensor(
                train_returns[1:len(fused) + 1], dtype=torch.float32
            )
            port_returns = (fused * returns_tensor).sum(dim=-1)
            fusion_loss = -port_returns.mean()

            model.opt_fusion.zero_grad()
            fusion_loss.backward()
            model.opt_fusion.step()

        # Record metrics
        avg_combined = np.mean(episode_rewards["combined"]) if episode_rewards["combined"] else 0
        training_history.append({
            "episode": episode + 1,
            "avg_reward": avg_combined,
            "portfolio_value": env.portfolio_value,
        })

        if (episode + 1) % 50 == 0:
            print(
                f"Episode {episode + 1}/{episodes} | "
                f"Avg Reward: {avg_combined:.6f} | "
                f"Portfolio: {env.portfolio_value:.4f}"
            )

    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(settings.MODEL_CHECKPOINT_DIR, f"mappo_{timestamp}.pt")
    os.makedirs(settings.MODEL_CHECKPOINT_DIR, exist_ok=True)
    model.save(model_path)

    return {
        "model_path": model_path,
        "episodes": episodes,
        "training_history": training_history,
        "final_portfolio_value": env.portfolio_value if training_history else 1.0,
    }
