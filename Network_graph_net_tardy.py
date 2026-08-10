import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

# =========================================================
# 1) Intra-USV self-attention block
# =========================================================
@torch.no_grad()
def build_batched_earliest_meeting_graph_per_usv_torch(
    node_feat: torch.Tensor,       # (B, W, F), F = x,y,t0,t1,t2,due_date
    travel_time: torch.Tensor,     # (B, W, W)
    *,
    num_usv: int = 12,
    time_dim: int = 3,
):
    """
    node_feat:
        (B, W, F)
        F >= 2 + time_dim + 1
        예: [x, y, t0, t1, t2, due_date]

    travel_time:
        (B, W, W)

    핵심:
        - pair_bias_mask를 사용하지 않음.
        - 자기 자신 USV로 향하는 edge만 제거.
        - 각 src expanded node마다,
          각 target USV에 대해 가장 빠른 meeting time의 dst node 하나를 선택.
        - 차단해야 하는 pair/node는 이후 graph convolution gate에서 마스킹하는 구조.

    Returns:
        expanded_xyt:
            (B, M, 4) = [x, y, selected_time, due_date]

        edge_src, edge_dst:
            global expanded node index.
            index 기준은 b * M + local_node_idx.

        edge_cost:
            이동 시간 travel_time 기준.

        edge_dist:
            유클리디안 거리.

        edge_duedate:
            edge가 향하는 dst node의 due_date.

        edge_batch:
            각 edge가 어느 batch에 속하는지.
    """

    device = node_feat.device
    dtype = node_feat.dtype

    B, W, F = node_feat.shape

    assert F >= 2 + time_dim + 1, (
        f"node_feat feature dim F={F} is too small. "
        f"Expected at least {2 + time_dim + 1} for [x,y,t0..tK,due_date]."
    )
    assert travel_time.shape == (B, W, W), (
        f"travel_time shape mismatch: got {travel_time.shape}, expected {(B, W, W)}"
    )
    assert W % num_usv == 0, (
        f"W={W} must be divisible by num_usv={num_usv}."
    )

    waypoints_per_usv = W // num_usv

    # --------------------------------------------------
    # 0. base feature split
    # --------------------------------------------------
    xy = node_feat[..., :2]                         # (B, W, 2)
    times = node_feat[..., 2:2 + time_dim]          # (B, W, K)
    due = node_feat[..., 2 + time_dim]              # (B, W)

    K = time_dim
    M = W * K

    # --------------------------------------------------
    # 1. expanded node index
    # expanded_idx = base_waypoint_idx * K + time_idx
    # --------------------------------------------------
    expanded_base = torch.arange(W, device=device).repeat_interleave(K)  # (M,)
    expanded_time_idx = torch.arange(K, device=device).repeat(W)         # (M,)

    waypoint_usv = torch.arange(W, device=device) // waypoints_per_usv   # (W,)
    expanded_usv = waypoint_usv[expanded_base]                           # (M,)

    # --------------------------------------------------
    # 2. expanded node feature
    # --------------------------------------------------
    expanded_xy = xy.index_select(dim=1, index=expanded_base)            # (B, M, 2)
    expanded_t = times.reshape(B, M)                                     # (B, M)
    expanded_due = due.index_select(dim=1, index=expanded_base)          # (B, M)

    expanded_xyt_due = torch.cat(
        [
            expanded_xy,                    # (B, M, 2)
            expanded_t.unsqueeze(-1),        # (B, M, 1)
            expanded_due.unsqueeze(-1),      # (B, M, 1)
        ],
        dim=-1
    )  # (B, M, 4) = [x, y, t, due_date]

    # --------------------------------------------------
    # 3. expanded move_time
    # --------------------------------------------------
    move_time = travel_time.index_select(1, expanded_base)
    move_time = move_time.index_select(2, expanded_base)  # (B, M, M)

    # --------------------------------------------------
    # 4. expanded distance matrix
    # --------------------------------------------------
    diff_xy = expanded_xy[:, :, None, :] - expanded_xy[:, None, :, :]  # (B, M, M, 2)
    dist_matrix = torch.linalg.norm(diff_xy, dim=-1)                   # (B, M, M)

    # --------------------------------------------------
    # 5. feasibility 계산
    # src 시간 + 이동시간 <= dst 시간 이면 feasible
    # --------------------------------------------------
    arrival_time = expanded_t[:, :, None] + move_time       # (B, M, M)
    time_feasible = arrival_time <= expanded_t[:, None, :]  # (B, M, M)

    # --------------------------------------------------
    # 6. finite mask
    # --------------------------------------------------
    finite_mask = (
        torch.isfinite(arrival_time)
        & torch.isfinite(expanded_t[:, :, None])
        & torch.isfinite(expanded_t[:, None, :])
        & torch.isfinite(move_time)
        & torch.isfinite(dist_matrix)
        & torch.isfinite(expanded_due[:, :, None])
        & torch.isfinite(expanded_due[:, None, :])
    )

    # --------------------------------------------------
    # 7. 자기 자신의 USV로 향하는 edge 제외
    # --------------------------------------------------
    different_usv = expanded_usv[:, None] != expanded_usv[None, :]  # (M, M)

    feasible_base = (
        time_feasible
        & finite_mask
        & different_usv[None, :, :]
    )  # (B, M, M)

    # --------------------------------------------------
    # 8. target USV별 earliest meeting node 선택
    # --------------------------------------------------
    edge_src_list = []
    edge_dst_list = []
    edge_cost_list = []
    edge_dist_list = []
    edge_duedate_list = []
    edge_batch_list = []

    inf = torch.tensor(float("inf"), device=device, dtype=dtype)

    for target_usv in range(num_usv):
        # dst가 target_usv에 속하는 expanded node인 경우만 허용
        target_mask = expanded_usv[None, None, :] == target_usv  # (1, 1, M)

        feasible = feasible_base & target_mask  # (B, M, M)

        # --------------------------------------------------
        # 1차 기준:
        # 가장 빠른 dst meeting time 선택
        # --------------------------------------------------
        meet_time_score = torch.where(
            feasible,
            expanded_t[:, None, :],
            inf
        )  # (B, M, M)

        best_meet_time = meet_time_score.min(dim=2).values  # (B, M)
        has_edge = torch.isfinite(best_meet_time)           # (B, M)

        # --------------------------------------------------
        # 2차 tie-break:
        # 같은 meeting time이면 이동 시간이 짧은 dst 선택
        # --------------------------------------------------
        same_best_time = (
            feasible
            & (expanded_t[:, None, :] == best_meet_time[:, :, None])
        )

        move_score = torch.where(
            same_best_time,
            move_time,
            inf
        )  # (B, M, M)

        best_dst = move_score.argmin(dim=2)  # (B, M)

        b_idx, src_local = torch.where(has_edge)

        if b_idx.numel() == 0:
            continue

        dst_local = best_dst[b_idx, src_local]

        cost = move_time[b_idx, src_local, dst_local]
        dist = dist_matrix[b_idx, src_local, dst_local]
        edge_duedate = expanded_due[b_idx, dst_local]

        src_global = b_idx * M + src_local
        dst_global = b_idx * M + dst_local

        edge_src_list.append(src_global)
        edge_dst_list.append(dst_global)
        edge_cost_list.append(cost)
        edge_dist_list.append(dist)
        edge_duedate_list.append(edge_duedate)
        edge_batch_list.append(b_idx)

    # --------------------------------------------------
    # 9. concat edge lists
    # --------------------------------------------------
    if len(edge_src_list) > 0:
        edge_src = torch.cat(edge_src_list, dim=0).long()
        edge_dst = torch.cat(edge_dst_list, dim=0).long()
        edge_cost = torch.cat(edge_cost_list, dim=0).to(dtype)
        edge_dist = torch.cat(edge_dist_list, dim=0).to(dtype)
        edge_duedate = torch.cat(edge_duedate_list, dim=0).to(dtype)
        edge_batch = torch.cat(edge_batch_list, dim=0).long()
    else:
        edge_src = torch.empty(0, device=device, dtype=torch.long)
        edge_dst = torch.empty(0, device=device, dtype=torch.long)
        edge_cost = torch.empty(0, device=device, dtype=dtype)
        edge_dist = torch.empty(0, device=device, dtype=dtype)
        edge_duedate = torch.empty(0, device=device, dtype=dtype)
        edge_batch = torch.empty(0, device=device, dtype=torch.long)

    return {
        # 기존 이름 유지: 마지막 dim이 4 = [x, y, t, due_date]
        "expanded_xyt": expanded_xyt_due,

        # 명시적 이름
        "expanded_xyt_due": expanded_xyt_due,

        "expanded_xy": expanded_xy,
        "expanded_t": expanded_t,
        "expanded_due": expanded_due,

        "expanded_usv": expanded_usv,
        "expanded_base": expanded_base,
        "expanded_time_idx": expanded_time_idx,

        "edge_src": edge_src,
        "edge_dst": edge_dst,
        "edge_cost": edge_cost,          # 이동 시간
        "edge_dist": edge_dist,          # 유클리디안 거리
        "edge_duedate": edge_duedate,    # dst node의 due_date
        "edge_batch": edge_batch,

        "num_expanded_nodes_per_batch": M,
        "num_total_expanded_nodes": B * M,
    }
# =========================================================
# 1) FGCN
# =========================================================
def expand_pair_bias_mask_to_edges(
    pair_bias_mask: torch.Tensor,  # (B, A, N, N), 0 or -1e9
    edge_src: torch.Tensor,        # (E,), global expanded node index = b * M + local
    edge_dst: torch.Tensor,        # (E,), global expanded node index = b * M + local
    edge_batch: torch.Tensor,      # (E,)
    expanded_base: torch.Tensor,   # (M,), expanded node -> original waypoint index
    *,
    num_expanded_nodes_per_batch: int,
):
    """
    Returns:
        edge_pair_bias: (A, E, 1)

    여기서 A는 보통 U.
    즉 A개의 dead-USV 상황별 edge mask.
    """

    device = pair_bias_mask.device
    dtype = pair_bias_mask.dtype

    B, A, N, N2 = pair_bias_mask.shape
    assert N == N2

    M = num_expanded_nodes_per_batch
    E = edge_src.numel()

    edge_src = edge_src.to(device=device, dtype=torch.long)
    edge_dst = edge_dst.to(device=device, dtype=torch.long)
    edge_batch = edge_batch.to(device=device, dtype=torch.long)
    expanded_base = expanded_base.to(device=device, dtype=torch.long)

    # global expanded index -> local expanded index
    src_local = edge_src % M  # (E,)
    dst_local = edge_dst % M  # (E,)

    # local expanded index -> original waypoint index
    src_base = expanded_base[src_local]  # (E,)
    dst_base = expanded_base[dst_local]  # (E,)

    # (E, A)
    edge_bias_ea = pair_bias_mask[
        edge_batch,
        :,
        src_base,
        dst_base,
    ]

    # (E, A) -> (A, E, 1)
    edge_pair_bias = edge_bias_ea.transpose(0, 1).unsqueeze(-1).contiguous()

    return edge_pair_bias.to(dtype)

class FGCN_layer(nn.Module):
    def __init__(self, hidden_dim, gate_dim):
        super().__init__()

        self.node_linear = nn.Linear(hidden_dim, hidden_dim)
        self.node_self_linear = nn.Linear(hidden_dim, hidden_dim)

        self.edge_filter = nn.Sequential(
            nn.Linear(2, gate_dim),
            nn.ELU(),
            nn.Linear(gate_dim, 1)
        )

        self.update = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ELU()
        )

    def forward(
        self,
        x: torch.Tensor,               # (N_total,H) or (A,N_total,H)
        edge_pair_bias: torch.Tensor,  # (A,E,1)
        edge_src: torch.Tensor,        # (E,)
        edge_dst: torch.Tensor,        # (E,)
        edge_time: torch.Tensor,       # (E,)
        edge_duedate: torch.Tensor,    # (E,)
    ):
        device = x.device
        dtype = x.dtype

        edge_src = edge_src.to(device=device, dtype=torch.long)
        edge_dst = edge_dst.to(device=device, dtype=torch.long)
        edge_time = edge_time.to(device=device, dtype=dtype)
        edge_duedate = edge_duedate.to(device=device, dtype=dtype)
        edge_pair_bias = edge_pair_bias.to(device=device, dtype=dtype)

        assert edge_pair_bias.dim() == 3
        assert edge_pair_bias.size(-1) == 1

        A, E, _ = edge_pair_bias.shape

        # --------------------------------------------------
        # 1. x shape 처리
        # --------------------------------------------------
        if x.dim() == 2:
            # 첫 layer
            N_total, H = x.shape

            dst_h = self.node_linear(x[edge_dst])  # (E,H)
            dst_h = F.elu(dst_h)
            dst_h = dst_h.unsqueeze(0).expand(A, E, H)  # (A,E,H)

            self_h = self.node_self_linear(x)  # (N_total,H)
            self_h = F.elu(self_h)
            self_h = self_h.unsqueeze(0).expand(A, N_total, H)  # (A,N,H)

        elif x.dim() == 3:
            # 두 번째 layer 이후
            A2, N_total, H = x.shape
            assert A2 == A, f"x A={A2}, edge_pair_bias A={A}"

            dst_h = self.node_linear(x[:, edge_dst, :])  # (A,E,H)
            dst_h = F.elu(dst_h)

            self_h = self.node_self_linear(x)  # (A,N,H)
            self_h = F.elu(self_h)

        else:
            raise ValueError(f"x must be (N,H) or (A,N,H), got {x.shape}")

        # --------------------------------------------------
        # 2. edge gate
        # --------------------------------------------------
        edge_feat = torch.stack(
            [edge_time, edge_duedate],
            dim=-1
        )  # (E,2)

        gate_logit = self.edge_filter(edge_feat)  # (E,1)

        valid_edge = (edge_pair_bias > -1e7).to(dtype)  # (A,E,1)

        gate = gate_logit.unsqueeze(0) + edge_pair_bias  # (A,E,1)
        gate = torch.sigmoid(gate)
        gate = gate * valid_edge

        # --------------------------------------------------
        # 3. message
        # --------------------------------------------------
        msg = dst_h * gate  # (A,E,H)

        # --------------------------------------------------
        # 4. aggregation
        # --------------------------------------------------
        a_offset = torch.arange(A, device=device).view(A, 1) * N_total
        flat_src = edge_src.view(1, E) + a_offset  # (A,E)
        flat_src = flat_src.reshape(-1)            # (A*E,)

        msg_flat = msg.reshape(A * E, H)

        agg_flat = torch.zeros(
            A * N_total,
            H,
            device=device,
            dtype=dtype
        )

        agg_flat.index_add_(0, flat_src, msg_flat)

        agg = agg_flat.view(A, N_total, H)

        # --------------------------------------------------
        # 5. degree
        # --------------------------------------------------
        deg_flat = torch.zeros(
            A * N_total,
            1,
            device=device,
            dtype=dtype
        )

        deg_flat.index_add_(
            0,
            flat_src,
            valid_edge.reshape(A * E, 1)
        )

        deg = deg_flat.view(A, N_total, 1)

        agg = agg / deg.clamp(min=1.0)

        # --------------------------------------------------
        # 6. update
        # --------------------------------------------------
        out = self.update(
            torch.cat([self_h, agg], dim=-1)
        )  # (A,N_total,H)

        return out

class FGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, gate_dim, num_layers=3):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.layers = nn.ModuleList([
            FGCN_layer(hidden_dim, gate_dim)
            for _ in range(num_layers)
        ])

    def forward(
            self,
            x: torch.Tensor,  # (N_total,input_dim)
            edge_pair_bias: torch.Tensor,  # (A,E,1)
            edge_src: torch.Tensor,
            edge_dst: torch.Tensor,
            edge_time: torch.Tensor,
            edge_duedate: torch.Tensor,
    ):
        h = self.input_proj(x)  # (N_total,H)

        for layer in self.layers:
            h = layer(
                h,
                edge_pair_bias,
                edge_src,
                edge_dst,
                edge_time,
                edge_duedate,
            )
            # 첫 layer 이후 h: (A,N_total,H)

        return h  # (A,N_total,H)

def expand_attn_hard_bias_dead_usv(
    attn_hard_bias: torch.Tensor,  # (B, N, N), 0 or -1e9
    *,
    W: int,  # waypoints per USV
    U: int,  # num USV
):
    """
    Returns:
        expanded_bias: (B, U, N, N)

    expanded_bias[:, u]:
        u번 USV가 죽은 상황.
        u번 USV에 해당하는 row/col을 모두 -1e9로 masking.
    """

    assert attn_hard_bias is not None
    assert attn_hard_bias.dim() == 3

    B, N, N2 = attn_hard_bias.shape
    assert N == N2
    assert N == W * U, f"N={N}, but W*U={W * U}"

    expanded_bias = attn_hard_bias[:, None, :, :].expand(B, U, N, N).clone()

    for u in range(U):
        s = u * W
        e = (u + 1) * W

        expanded_bias[:, u, :, s:e] = -1e9

    return expanded_bias

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
        self.act_node_encoder = FGCN(
            input_dim=4,      # x, y, t
            hidden_dim=d_model,
            gate_dim=d_model//4,
            num_layers=num_actor_layers
        )
        self.critic_node_encoder = FGCN(
            input_dim=4,  # x, y, t
            hidden_dim=d_model,
            gate_dim=d_model // 4,
            num_layers=num_critic_layers
        )

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
            attn_soft_bias: Optional[torch.Tensor] = None,  # (B, N, N)
            attn_hard_bias: Optional[torch.Tensor] = None,  # (B, N, N)
            print_gate: Optional[bool] = False,
    ):
        device = usv_node_features.device
        dtype = usv_node_features.dtype

        B, U, W, F = usv_node_features.shape

        N = U * W  # original waypoint 개수
        K = F - 3  # x,y,due_date 제외하고 time 개수라고 가정
        M = N * K  # expanded node 개수 per batch

        # --------------------------------------------------
        # 1. waypoint flatten
        # --------------------------------------------------
        x_wp = usv_node_features.reshape(B, N, F)  # (B,N,F)

        # --------------------------------------------------
        # 2. graph는 한 번만 생성
        # --------------------------------------------------
        graph = build_batched_earliest_meeting_graph_per_usv_torch(
            node_feat=x_wp,
            travel_time=attn_soft_bias,
            num_usv=U,
            time_dim=K,
        )

        # --------------------------------------------------
        # 3. actor query별 dead-USV mask 제거
        #    - 기존: attn_hard_bias_dead = (B,U,N,N)
        #    - 변경: 공통 hard mask 하나만 사용 = (B,1,N,N)
        #    - 따라서 FGCN 출력의 A축은 항상 1
        # --------------------------------------------------
        if attn_hard_bias is None:
            attn_hard_bias_single = torch.zeros(
                B, 1, N, N,
                device=device,
                dtype=dtype,
            )
        else:
            attn_hard_bias_single = attn_hard_bias[:, None, :, :].to(
                device=device,
                dtype=dtype,
            )  # (B,1,N,N)

        # --------------------------------------------------
        # 4. node-pair mask를 edge mask로 변환
        # --------------------------------------------------
        edge_pair_bias = expand_pair_bias_mask_to_edges(
            pair_bias_mask=attn_hard_bias_single,  # (B,1,N,N)
            edge_src=graph["edge_src"],
            edge_dst=graph["edge_dst"],
            edge_batch=graph["edge_batch"],
            expanded_base=graph["expanded_base"],
            num_expanded_nodes_per_batch=graph["num_expanded_nodes_per_batch"],
        )  # (1,E,1)

        # --------------------------------------------------
        # 5. FGCN 입력 node feature flatten
        # --------------------------------------------------
        x_graph = graph["expanded_xyt"].reshape(B * M, 4)  # (B*M,4)

        # --------------------------------------------------
        # 6. 같은 graph + 단일 mask로 단일 결과 생성
        # --------------------------------------------------
        h_graph = self.act_node_encoder(
            x=x_graph,
            edge_pair_bias=edge_pair_bias,  # (1,E,1)
            edge_src=graph["edge_src"],
            edge_dst=graph["edge_dst"],
            edge_time=graph["edge_cost"],
            edge_duedate=graph["edge_duedate"],
        )  # (1,B*M,H)

        h_graph_critic = self.critic_node_encoder(
            x=x_graph,
            edge_pair_bias=edge_pair_bias,  # (1,E,1)
            edge_src=graph["edge_src"],
            edge_dst=graph["edge_dst"],
            edge_time=graph["edge_cost"],
            edge_duedate=graph["edge_duedate"],
        )  # (1,B*M,H)

        H = h_graph.size(-1)

        # --------------------------------------------------
        # 7. reshape
        #    기존: (B, dead_usv, node_usv, W, K, H)
        #    변경: (B, node_usv, W, K, H)
        # --------------------------------------------------
        h_graph = h_graph.squeeze(0).view(B, N, K, H)
        h_graph = h_graph.view(B, U, W, K, H)

        h_graph_critic = h_graph_critic.squeeze(0).view(B, N, K, H)
        h_graph_critic = h_graph_critic.view(B, U, W, K, H)

        return h_graph, h_graph_critic


    # ------------------------------------------------------
    # Actor forward
    # ------------------------------------------------------
    def actor_forward(
            self,
            actor_encoded_nodes: torch.Tensor,  # (B, U, W, K, H)
            rep_node_indices: torch.Tensor,  # (B, U), W index
            actor_query_features: torch.Tensor,  # (B, U, Qdim)
            alive_mask: torch.Tensor,  # (B, U)
            usv_node_features: torch.Tensor,  # (B, U, W, F), F = x,y,t0,t1,t2,due_date
    ):
        """
        actor_encoded_nodes:
            (B, U, W, K, H)
            모든 actor query가 공유하는 단일 graph embedding.
            더 이상 dead_u별 embedding 축을 만들지 않는다.

        rep_node_indices:
            (B, U)
            각 action/candidate USV별 representative waypoint index.

        actor_query_features:
            (B, U, Qdim)

        alive_mask:
            (B, U)
        """

        B, U, W, K, H = actor_encoded_nodes.shape
        device = actor_encoded_nodes.device

        # --------------------------------------------------
        # 1. rep waypoint의 t0,t1,t2 가져오기
        # --------------------------------------------------
        times = usv_node_features[..., 2:2 + K]  # (B, U, W, K)

        batch_idx = torch.arange(B, device=device)[:, None]  # (B,1)
        usv_idx = torch.arange(U, device=device)[None, :]    # (1,U)

        rep_w = rep_node_indices.long()  # (B,U)

        # rep_times: (B,U,K)
        rep_times = times[batch_idx, usv_idx, rep_w]

        # --------------------------------------------------
        # 2. actor query의 시간 기준
        # --------------------------------------------------
        query_t = actor_query_features[..., 2]  # (B,U)

        # --------------------------------------------------
        # 3. query_t와 가장 가까운 t index 선택
        # --------------------------------------------------
        rep_k = torch.abs(rep_times - query_t[..., None]).argmin(dim=-1)  # (B,U)

        # --------------------------------------------------
        # 4. action u별 rep embedding 추출
        #    기존: actor_encoded_nodes[b, dead_u=u, node_u=u, rep_w, rep_k]
        #    변경: actor_encoded_nodes[b, node_u=u, rep_w, rep_k]
        # --------------------------------------------------
        rep_embed = actor_encoded_nodes[
            batch_idx,  # b
            usv_idx,    # node_u = action u
            rep_w,      # representative waypoint
            rep_k,      # selected time index
        ]  # (B,U,H)

        # --------------------------------------------------
        # 5. Actor score 계산
        # --------------------------------------------------
        Q_embed = self.actor_query_proj(actor_query_features)  # (B,U,H)

        qk = torch.cat([Q_embed, rep_embed], dim=-1)  # (B,U,2H)

        logits = self.actor_score_mlp(qk).squeeze(-1) / 1.5  # (B,U)

        logits = logits.masked_fill(~alive_mask.bool(), -1e9)

        probs = F.softmax(logits, dim=-1)

        return {
            "logits": logits,
            "probs": probs,
            "actor_rep_embed": rep_embed,
            "actor_Q": Q_embed,
            "rep_k": rep_k,
            "rep_times": rep_times,
            "query_t": query_t,
        }
    # ------------------------------------------------------
    # Critic forward
    # ------------------------------------------------------
    def critic_forward(
            self,
            critic_encoded_nodes: torch.Tensor,  # (B, U, W, K, H)
            rep_node_indices: torch.Tensor,  # (B, U), W index
            actor_query_features: torch.Tensor,  # (B, U, Qdim)
            alive_mask: torch.Tensor,  # (B, U)
            usv_node_features: torch.Tensor,  # (B, U, W, F), F = x,y,t0,t1,t2,due_date
    ):
        """
        critic_encoded_nodes:
            (B, U, W, K, H)
            모든 후보가 공유하는 단일 graph embedding.

        Returns:
            value:
                (B,)
        """

        B, U, W, K, H = critic_encoded_nodes.shape
        device = critic_encoded_nodes.device
        dtype = critic_encoded_nodes.dtype

        # --------------------------------------------------
        # 1. rep waypoint의 t0,t1,t2 가져오기
        # --------------------------------------------------
        times = usv_node_features[..., 2:2 + K]  # (B, U, W, K)

        batch_idx = torch.arange(B, device=device)[:, None]  # (B,1)
        usv_idx = torch.arange(U, device=device)[None, :]    # (1,U)

        rep_w = rep_node_indices.long()  # (B,U)

        rep_times = times[batch_idx, usv_idx, rep_w]  # (B,U,K)

        # --------------------------------------------------
        # 2. query time 기준으로 가장 가까운 k 선택
        # --------------------------------------------------
        query_t = actor_query_features[..., 2]  # (B,U)

        rep_k = torch.abs(
            rep_times - query_t[..., None]
        ).argmin(dim=-1)  # (B,U)

        # --------------------------------------------------
        # 3. 후보 u별 rep embedding 추출
        #    기존: critic_encoded_nodes[b, dead_u=u, node_u=u, rep_w, rep_k]
        #    변경: critic_encoded_nodes[b, node_u=u, rep_w, rep_k]
        # --------------------------------------------------
        rep_embed = critic_encoded_nodes[
            batch_idx,
            usv_idx,
            rep_w,
            rep_k,
        ]  # (B,U,H)

        # --------------------------------------------------
        # 4. critic query embedding
        # --------------------------------------------------
        Q_embed = self.critic_query_proj(actor_query_features)  # (B,U,H)

        qk = torch.cat([Q_embed, rep_embed], dim=-1)  # (B,U,2H)

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
        gate = gate * alive_mask.to(dtype)

        # --------------------------------------------------
        # 6. 후보별 value vector
        # --------------------------------------------------
        candidate_value_vec = self.critic_candidate_value_mlp(qk)  # (B,U,H)

        # gate 적용
        gated_value_vec = candidate_value_vec * gate.unsqueeze(-1)  # (B,U,H)

        # --------------------------------------------------
        # 7. 모든 후보의 value vector를 합산
        # --------------------------------------------------
        critic_context = gated_value_vec.sum(dim=1)  # (B,H)

        # --------------------------------------------------
        # 8. 최종 scalar value
        # --------------------------------------------------
        value = self.critic_value_head(critic_context).squeeze(-1)  # (B,)

        return {
            "value": value,
            "critic_context": critic_context,
            "critic_gate": gate,
            "critic_gate_logit": gate_logit,
            "critic_candidate_value_vec": candidate_value_vec,
            "critic_gated_value_vec": gated_value_vec,
            "critic_rep_embed": rep_embed,
            "critic_Q": Q_embed,
            "rep_k": rep_k,
            "rep_times": rep_times,
            "query_t": query_t,
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
            attn_hard_bias=attn_hard_bias,
            attn_soft_bias=attn_soft_bias,
            print_gate=print_gate,
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
            usv_node_features=usv_node_features,
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