import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


# =========================================================
# 2) Actor-Critic with separate encoders
# =========================================================
class GraphREINFORCE(nn.Module):
    """
    Actor:
      - actor 전용 node encoder + intra self-attention
      - representative waypoint embedding 추출
      - actor query와 rep embedding을 concat
      - MLP로 action logit 산출

    Critic:
      - critic 전용 node encoder + intra self-attention
      - waypoint-conditioned critic query로 전체 waypoint attention
      - pooled context -> scalar value
    """
    def __init__(
        self,
        node_input_dim: int = 5,
        actor_query_dim: int = 8,
        d_model: int = 64,
        num_actor_layers: int = 3,
        num_critic_layers: int = 4,

    ):
        super().__init__()

        # --------------------------------------------------
        # Actor encoder (separate)
        # --------------------------------------------------
        self.act_node_encoder = nn.Linear(node_input_dim, d_model)
        self.critic_node_encoder =nn.Linear(node_input_dim, d_model)

        self.actor_query_proj = nn.Sequential(
            nn.Linear(actor_query_dim, d_model//2),
            nn.ELU(),
            nn.Linear(d_model//2, d_model),
        )

        # concat([Q,K]) -> scalar logit
        self.actor_score_mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ELU(),
            nn.Linear(d_model, d_model//2),
            nn.ELU(),
            nn.Linear(d_model//2, 1),
        )
        self.critic_query_proj = nn.Sequential(
            nn.Linear(actor_query_dim, d_model // 2),
            nn.ELU(),
            nn.Linear(d_model // 2, d_model),
        )

        # qk -> gate logit
        self.critic_gate_mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ELU(),
            nn.Linear(d_model, d_model // 2),
            nn.ELU(),
            nn.Linear(d_model // 2, 1),
        )

        # qk -> candidate value vector
        self.critic_candidate_value_mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ELU(),
            nn.Linear(d_model, d_model),
            nn.ELU(),
        )

        # sum된 value vector -> scalar V(s)
        self.critic_value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ELU(),
            nn.Linear(d_model // 2, 1),
        )
    # ------------------------------------------------------
    # Actor encoding
    # ------------------------------------------------------
    def encode_waypoints(
            self,
            usv_node_features: torch.Tensor,  # (B, U, W, F)

    ):

        B, U, W, F = usv_node_features.shape

        N = U * W  # original waypoint 개수
        K = F - 3  # x,y,due_date 제외하고 time 개수라고 가정
        M = N * K  # expanded node 개수 per batch

        # --------------------------------------------------
        # 1. waypoint flatten
        # --------------------------------------------------
        act_node=self.act_node_encoder(usv_node_features)
        critic_node=self.critic_node_encoder(usv_node_features)

        # --------------------------------------------------
        # 2. graph는 한 번만 생성
        # --------------------------------------------------

        return act_node, critic_node


    # ------------------------------------------------------
    # Actor forward
    # ------------------------------------------------------
    def actor_forward(
            self,
            actor_encoded_nodes: torch.Tensor,  # (B, U,  W, F)
            rep_node_indices: torch.Tensor,  # (B, U), W index
            actor_query_features: torch.Tensor,  # (B, U, Qdim)
            alive_mask: torch.Tensor,  # (B, U)
            usv_node_features: torch.Tensor,  # (B, U, W, F), F = x,y,t0,t1,t2,due_date
    ):
        """
        actor_encoded_nodes:
            (B, dead_U, node_U, W, K, H)

            actor_encoded_nodes[b, dead_u]:
                dead_u번 USV가 죽었다고 가정한 graph embedding.

            actor_encoded_nodes[b, dead_u, node_u]:
                dead_u번 USV가 죽은 상황에서 node_u번 USV의 waypoint embedding.

        rep_node_indices:
            (B, U)
            각 action/candidate USV별 representative waypoint index.

        actor_query_features:
            (B, U, Qdim)

        alive_mask:
            (B, U)
        """

        B, U, W, D = actor_encoded_nodes.shape

        # --------------------------------------------------
        # 1. rep waypoint의 t0,t1,t2 가져오기

        gather_idx = rep_node_indices.unsqueeze(-1).unsqueeze(-1).expand(B, U, 1, D)
        rep_embed = torch.gather(actor_encoded_nodes, dim=2, index=gather_idx).squeeze(2)  # (B,U,D)

        K = rep_embed  # (B,U,D)
        Q = self.actor_query_proj(actor_query_features)  # (B,U,D)

        # concat-based score
        qk = torch.cat([Q, K], dim=-1)  # (B,U,2D)

        logits = self.actor_score_mlp(qk).squeeze(-1)  # (B,U)

        logits = logits.masked_fill(~alive_mask.bool(), -1e9)
        probs = F.softmax(logits, dim=-1)

        return {
            "logits": logits,
            "probs": probs,
            "actor_rep_embed": rep_embed,
            "actor_Q": Q,
            "actor_K": K,
        }
    # ------------------------------------------------------
    # Critic forward
    # ------------------------------------------------------
    def critic_forward(
            self,
            critic_encoded_nodes: torch.Tensor,  # (B, U, U, W, K, H)
            rep_node_indices: torch.Tensor,  # (B, U), W index
            actor_query_features: torch.Tensor,  # (B, U, Qdim)
            alive_mask: torch.Tensor,  # (B, U)

    ):
        """
        critic_encoded_nodes:
            (B, dead_U, node_U, W, K, H)

            critic_encoded_nodes[b, dead_u]:
                dead_u번 USV가 죽었다고 가정한 graph embedding.

        critic_query_features:
            (B, U, Qdim)

        Returns:
            value:
                (B,)
        """

        B, U, W, D = critic_encoded_nodes.shape

        # --------------------------------------------------
        # 1. rep waypoint의 t0,t1,t2 가져오기

        gather_idx = rep_node_indices.unsqueeze(-1).unsqueeze(-1).expand(B, U, 1, D)
        rep_embed = torch.gather(critic_encoded_nodes, dim=2, index=gather_idx).squeeze(2)  # (B,U,D)

        K = rep_embed  # (B,U,D)

        # --------------------------------------------------
        # 4. critic query embedding
        # --------------------------------------------------
        Q_embed = self.critic_query_proj(actor_query_features)  # (B,U,H)

        qk = torch.cat([Q_embed, K], dim=-1)  # (B,U,2H)

        # --------------------------------------------------
        # 5. 후보별 gate
        # --------------------------------------------------
        gate_logit = self.critic_gate_mlp(qk).squeeze(-1)  # (B,U)

        gate_logit = gate_logit.masked_fill(
            ~alive_mask.bool(),
            -1e9
        )

        gate = torch.sigmoid(gate_logit)  # (B,U)

        # 확실하게 dead 후보는 0
        gate = gate * alive_mask

        # --------------------------------------------------
        # 6. 후보별 value vector
        # --------------------------------------------------
        candidate_value_vec = self.critic_candidate_value_mlp(qk)  # (B,U,H)

        # gate 적용
        gated_value_vec = candidate_value_vec * gate.unsqueeze(-1)  # (B,U,H)

        # --------------------------------------------------
        # 7. 모든 후보의 value vector를 합산
        # --------------------------------------------------
        critic_context = gated_value_vec.sum().unsqueeze(-1)  # (B,H)

        # --------------------------------------------------
        # 8. 최종 scalar value
        # --------------------------------------------------
        value = critic_context

        return {
            "value": value,
            "critic_context": critic_context,
            "critic_gate": gate,
            "critic_gate_logit": gate_logit,
            "critic_candidate_value_vec": candidate_value_vec,
            "critic_gated_value_vec": gated_value_vec,
            "critic_rep_embed": rep_embed,
            "critic_Q": Q_embed,
        }
    # ------------------------------------------------------
    # Full forward
    # ------------------------------------------------------
    def forward(
            self,
            *,
            usv_node_features: torch.Tensor,
            rep_node_indices: torch.Tensor,
            actor_query_features: torch.Tensor,
            critic_query_features: torch.Tensor,
            alive_mask: torch.Tensor,
            attn_soft_bias: Optional[torch.Tensor] = None,
            attn_hard_bias: Optional[torch.Tensor] = None,
            print_gate: Optional[bool] = False,

    ):
        actor_encoded_nodes,critic_encoded_nodes = self.encode_waypoints(
            usv_node_features,

        )

        actor_out = self.actor_forward(
            actor_encoded_nodes=actor_encoded_nodes,
            rep_node_indices=rep_node_indices,
            actor_query_features=actor_query_features,
            alive_mask=alive_mask,
            usv_node_features=usv_node_features,
        )

        critic_out = self.critic_forward(
            critic_encoded_nodes=critic_encoded_nodes,
            rep_node_indices=rep_node_indices,
            actor_query_features=actor_query_features,
            alive_mask=alive_mask,
        )

        return {
            **actor_out,
            **critic_out,
            "actor_encoded_nodes": actor_encoded_nodes,
            "critic_encoded_nodes": critic_encoded_nodes,
        }

@dataclass
class PPORolloutBatch:
    usv_node_features: torch.Tensor
    rep_node_indices: torch.Tensor
    actor_query_features: torch.Tensor
    critic_query_features: torch.Tensor
    alive_mask: torch.Tensor
    attn_soft_bias: torch.Tensor
    attn_hard_bias: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor
    returns: Optional[torch.Tensor] = None
    advantages: Optional[torch.Tensor] = None


class PPORolloutBuffer:
    def __init__(self):
        self.storage = []

    def add(
            self,
            *,
            usv_node_features,
            rep_node_indices,
            actor_query_features,
            critic_query_features,
            alive_mask,
            action,
            log_prob,
            reward,
            done,
            attn_hard_bias,
            attn_soft_bias,
            value,

    ):
        self.storage.append({
            "usv_node_features": usv_node_features.detach().cpu(),
            "rep_node_indices": rep_node_indices.detach().cpu(),
            "actor_query_features": actor_query_features.detach().cpu(),
            "critic_query_features": critic_query_features.detach().cpu(),
            "alive_mask": alive_mask.detach().cpu(),
            "action": torch.as_tensor(action).cpu(),
            "log_prob": log_prob.detach().cpu(),
            "reward": torch.as_tensor(reward, dtype=torch.float32).cpu(),
            "done": torch.as_tensor(done, dtype=torch.float32).cpu(),
            "attn_soft_bias": attn_soft_bias.detach().cpu(),
            "attn_hard_bias": attn_hard_bias.detach().cpu(),
            "value": value.detach().cpu(),
        })

    def clear(self):
        self.storage = []

    def __len__(self):
        return len(self.storage)

    def compute_returns_and_advantages(self, last_value: float, gamma: float = 0.99, gae_lambda: float = 0.95):
        rewards = [x["reward"].item() for x in self.storage]
        dones = [x["done"].item() for x in self.storage]
        values = [x["value"].item() for x in self.storage]

        advantages = []
        gae = 0.0
        next_value = last_value

        for t in reversed(range(len(self.storage))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_value * mask - values[t]
            gae = delta + gamma * gae_lambda * mask * gae
            advantages.append(gae)
            next_value = values[t]

        advantages.reverse()
        returns = [adv + val for adv, val in zip(advantages, values)]

        for i in range(len(self.storage)):
            self.storage[i]["advantage"] = torch.tensor(advantages[i], dtype=torch.float32)
            self.storage[i]["return"] = torch.tensor(returns[i], dtype=torch.float32)

    def to_batch(self, device: str = "cpu") -> PPORolloutBatch:
        return PPORolloutBatch(
            usv_node_features=torch.stack([x["usv_node_features"] for x in self.storage]).to(device),
            rep_node_indices=torch.stack([x["rep_node_indices"] for x in self.storage]).to(device),
            actor_query_features=torch.stack([x["actor_query_features"] for x in self.storage]).to(device),
            critic_query_features=torch.stack([x["critic_query_features"] for x in self.storage]).to(device),
            alive_mask=torch.stack([x["alive_mask"] for x in self.storage]).to(device),
            attn_soft_bias=torch.stack([x["attn_soft_bias"] for x in self.storage]).to(device),
            attn_hard_bias=torch.stack([x["attn_hard_bias"] for x in self.storage]).to(device),
            actions=torch.stack([x["action"] for x in self.storage]).long().to(device),
            log_probs=torch.stack([x["log_prob"] for x in self.storage]).to(device),
            rewards=torch.stack([x["reward"] for x in self.storage]).to(device),
            dones=torch.stack([x["done"] for x in self.storage]).to(device),
            advantages=torch.stack([x["advantage"] for x in self.storage]).to(device),
            values=torch.stack([x["value"] for x in self.storage]).to(device),
            returns=torch.stack([x["return"] for x in self.storage]).to(device),
        )
def ppo_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: PPORolloutBatch,
    *,
    clip_eps: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    normalize_advantage: bool = False,
):
    advantages = batch.advantages
    if normalize_advantage:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    out = model(
        usv_node_features=batch.usv_node_features,
        rep_node_indices=batch.rep_node_indices,
        actor_query_features=batch.actor_query_features,
        critic_query_features=batch.critic_query_features,
        alive_mask=batch.alive_mask,
        attn_soft_bias=batch.attn_soft_bias,
        attn_hard_bias=batch.attn_hard_bias,
    )

    logits = out["logits"]          # (T,U)
    values = out["value"]
    dist = torch.distributions.Categorical(logits=logits)

    new_log_probs = dist.log_prob(batch.actions)   # (T,)
    entropy = dist.entropy().mean()

    ratio = torch.exp(new_log_probs - batch.log_probs)

    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    value_loss = F.mse_loss(values, batch.returns)

    loss = policy_loss +value_loss*value_coef - entropy_coef * entropy

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()

    info = {
        "loss": loss.item(),
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": entropy.item(),
        "mean_ratio": ratio.mean().item(),
    }
    return info
def ppo_update_minibatch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: PPORolloutBatch,
    *,
    device: str = "cuda",
    mini_batch_size: int = 4,
    clip_eps: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    normalize_advantage: bool = False,
):
    """
    PPO update mini-batch version.

    전체 batch를 한 번에 forward하지 않고,
    mini_batch_size 단위로 나눠서 forward/backward 한다.

    loss는 전체 평균과 같도록 각 mini-batch loss에 n / T를 곱해 backward한다.
    optimizer.step()은 모든 mini-batch backward가 끝난 뒤 한 번만 수행한다.
    """

    model.train()
    optimizer.zero_grad(set_to_none=True)

    T = int(batch.actions.shape[0])
    if T == 0:
        raise ValueError("Empty PPO batch.")

    # --------------------------------------------------
    # 1. advantage 준비
    # --------------------------------------------------
    advantages_all = batch.advantages.detach().float()

    if normalize_advantage:
        advantages_all = (advantages_all - advantages_all.mean()) / (
            advantages_all.std(unbiased=False) + 1e-8
        )

    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_ratio = 0.0
    total_count = 0

    # --------------------------------------------------
    # 2. mini-batch loop
    # --------------------------------------------------
    mini_batch_size=T//mini_batch_size
    for start in range(0, T, mini_batch_size):
        end = min(start + mini_batch_size, T)
        n = end - start

        # 전체 평균 loss가 되도록 가중치
        weight = float(n) / float(T)

        # --------------------------------------------------
        # 3. mini-batch만 device로 이동
        # --------------------------------------------------
        usv_node_features = batch.usv_node_features[start:end].to(device, non_blocking=True)
        rep_node_indices = batch.rep_node_indices[start:end].to(device, non_blocking=True)
        actor_query_features = batch.actor_query_features[start:end].to(device, non_blocking=True)
        critic_query_features = batch.critic_query_features[start:end].to(device, non_blocking=True)
        alive_mask = batch.alive_mask[start:end].to(device, non_blocking=True)

        attn_soft_bias = batch.attn_soft_bias[start:end].to(device, non_blocking=True)
        attn_hard_bias = batch.attn_hard_bias[start:end].to(device, non_blocking=True)

        actions = batch.actions[start:end].to(device, non_blocking=True).view(-1)
        old_log_probs = batch.log_probs[start:end].to(device, non_blocking=True).view(-1)
        advantages = advantages_all[start:end].to(device, non_blocking=True).view(-1)
        returns = batch.returns[start:end].to(device, non_blocking=True).view(-1)

        # --------------------------------------------------
        # 4. forward
        # --------------------------------------------------
        out = model(
            usv_node_features=usv_node_features,
            rep_node_indices=rep_node_indices,
            actor_query_features=actor_query_features,
            critic_query_features=critic_query_features,
            alive_mask=alive_mask,
            attn_soft_bias=attn_soft_bias,
            attn_hard_bias=attn_hard_bias,
        )

        logits = out["logits"]          # (n, U)
        values = out["value"].view(-1)  # (n,)
        returns = returns.view(-1)
        dist = torch.distributions.Categorical(logits=logits)

        new_log_probs = dist.log_prob(actions)  # (n,)
        entropy = dist.entropy().mean()

        ratio = torch.exp(new_log_probs - old_log_probs)

        # --------------------------------------------------
        # 5. PPO clipped policy loss
        # --------------------------------------------------
        surr1 = ratio * advantages
        surr2 = torch.clamp(
            ratio,
            1.0 - clip_eps,
            1.0 + clip_eps
        ) * advantages

        policy_loss = -torch.min(surr1, surr2).mean()

        # --------------------------------------------------
        # 6. value loss
        # --------------------------------------------------
        value_loss = F.mse_loss(values, returns)

        # --------------------------------------------------
        # 7. total loss
        # --------------------------------------------------
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

        # 전체 batch 평균 기준으로 gradient 누적
        (loss * weight).backward()

        # --------------------------------------------------
        # 8. logging
        # --------------------------------------------------
        with torch.no_grad():
            total_loss += float(loss.detach().cpu()) * n
            total_policy_loss += float(policy_loss.detach().cpu()) * n
            total_value_loss += float(value_loss.detach().cpu()) * n
            total_entropy += float(entropy.detach().cpu()) * n
            total_ratio += float(ratio.mean().detach().cpu()) * n
            total_count += n

        # --------------------------------------------------
        # 9. 메모리 참조 제거
        # --------------------------------------------------
        del (
            usv_node_features,
            rep_node_indices,
            actor_query_features,
            critic_query_features,
            alive_mask,
            attn_soft_bias,
            attn_hard_bias,
            actions,
            old_log_probs,
            advantages,
            returns,
            out,
            logits,
            values,
            dist,
            new_log_probs,
            entropy,
            ratio,
            surr1,
            surr2,
            policy_loss,
            value_loss,
            loss,
        )

    # --------------------------------------------------
    # 10. optimizer step은 한 번만
    # --------------------------------------------------
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()

    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    denom = max(total_count, 1)

    info = {
        "loss": total_loss / denom,
        "policy_loss": total_policy_loss / denom,
        "value_loss": total_value_loss / denom,
        "entropy": total_entropy / denom,
        "mean_ratio": total_ratio / denom,
        "mini_batch_size": int(mini_batch_size),
        "num_transitions": int(T),
    }

    return info


class MultiPPORolloutBuffer:
    def __init__(self):
        self.storage = []

    def extend_from_buffer(self, buffer: PPORolloutBuffer):
        self.storage.extend(buffer.storage)

    def clear(self):
        self.storage = []

    def __len__(self):
        return len(self.storage)

    def compute_returns_and_advantages(self, *args, **kwargs):
        raise RuntimeError("Use per-episode buffers first, then merge. This buffer assumes returns are already computed.")

    def to_batch(self, device: str = "cpu") -> PPORolloutBatch:
        return PPORolloutBatch(
            usv_node_features=torch.stack([x["usv_node_features"] for x in self.storage]).to(device),
            rep_node_indices=torch.stack([x["rep_node_indices"] for x in self.storage]).to(device),
            actor_query_features=torch.stack([x["actor_query_features"] for x in self.storage]).to(device),
            critic_query_features=torch.stack([x["critic_query_features"] for x in self.storage]).to(device),
            alive_mask=torch.stack([x["alive_mask"] for x in self.storage]).to(device),
            attn_soft_bias=torch.stack([x["attn_soft_bias"] for x in self.storage]).to(device),
            attn_hard_bias=torch.stack([x["attn_hard_bias"] for x in self.storage]).to(device),
            actions=torch.stack([x["action"] for x in self.storage]).long().to(device),
            log_probs=torch.stack([x["log_prob"] for x in self.storage]).to(device),
            rewards=torch.stack([x["reward"] for x in self.storage]).to(device),
            dones=torch.stack([x["done"] for x in self.storage]).to(device),
            advantages=torch.stack([x["advantage"] for x in self.storage]).to(device),
            returns=torch.stack([x["return"] for x in self.storage]).to(device),
            values=torch.stack([x["value"] for x in self.storage]).to(device),
        )