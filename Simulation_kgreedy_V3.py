import math
from typing import Dict, List, Tuple, Literal,Optional,Sequence,Any
import numpy as np
import matplotlib.pyplot as plt
import heapq
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

mesh_mode = Literal["Square", "Hexa"]
NeighborMode = Literal["Edges_only", "Extended_edges","Extra_extended_edges"]
#인접한 mesh 분석
# mesh 생성 코드
def mesh(
    x_s: float, x_f: float,
    y_s: float, y_f: float,
    x_c: float, y_c: float,
    r: float, mod: str

) -> Tuple[np.ndarray, Dict[Tuple[int, int], int]]:
    """
    Equilateral triangular mesh (side length r) inside rectangle [x_s,x_f] x [y_s,y_f],
    anchored at (x_c, y_c).

    Returns:
      vertices: (N,2) float array
      idx_map: dict mapping (i,j) -> vertex_index (lattice coords)
    """
    if r <= 0:
        raise ValueError("r must be positive.")
    if x_s > x_f:
        x_s, x_f = x_f, x_s
    if y_s > y_f:
        y_s, y_f = y_f, y_s
    mod=mod.strip()
    if mod == "Square":
        i_min = math.ceil((x_s - x_c) / r)
        i_max = math.floor((x_f - x_c) / r)
        j_min = math.ceil((y_s - y_c) / r)
        j_max = math.floor((y_f - y_c) / r)

        vertices: List[Tuple[float, float]] = []
        idx_map: Dict[Tuple[int, int], int] = {}

        for j in range(j_min, j_max + 1):
            y = y_c + j * r
            for i in range(i_min, i_max + 1):
                x = x_c + i * r
                # boundary included
                if (x_s <= x <= x_f) and (y_s <= y <= y_f):
                    idx_map[(i, j)] = len(vertices)
                    vertices.append((x, y))

        return np.array(vertices, dtype=float), idx_map
    if mod=='Hexa':
        dy = r * math.sqrt(3) / 2.0  # row spacing

        j_min = math.ceil((y_s - y_c ) / dy)
        j_max = math.floor((y_f - y_c ) / dy)

        vertices: List[Tuple[float, float]] = []
        idx_map: Dict[Tuple[int, int], int] = {}

        for j in range(j_min, j_max + 1):
            y = y_c + j * dy

            # x = x_c + i*r + j*(r/2)
            i_min = math.ceil((x_s - x_c - (j * r / 2.0)) / r)
            i_max = math.floor((x_f - x_c - (j * r / 2.0) ) / r)

            for i in range(i_min, i_max + 1):
                x = x_c + i * r + j * (r / 2.0)
                if (x_s) <= x <= (x_f) and (y_s) <= y <= (y_f):
                    idx_map[(i, j)] = len(vertices)
                    vertices.append((x, y))
        return np.array(vertices, dtype=float),  idx_map


def vertex_index_from_xy(
    x: float, y: float,
    idx_map: Dict[Tuple[int, int], int],
    mod: Literal["Square", "Hexa"],
    x_c: float, y_c: float,
    r: float,
) -> int:
    """
    Return vertex index for the nearest lattice vertex to (x,y).
    Uses idx_map from mesh().

    Square:
      i = round((x-x_c)/r), j = round((y-y_c)/r)

    Hexa:
      dy = r*sqrt(3)/2
      j = round((y-y_c)/dy)
      i = round((x-x_c - j*(r/2))/r)
    """
    mod = mod.strip()
    if r <= 0:
        raise ValueError("r must be positive.")

    if mod == "Square":
        i = int(round((x - x_c) / r))
        j = int(round((y - y_c) / r))
    elif mod == "Hexa":
        dy = r * math.sqrt(3) / 2.0
        j = int(round((y - y_c) / dy))
        i = int(round((x - x_c - j * (r / 2.0)) / r))
    else:
        raise ValueError("mod must be 'Square' or 'Hexa'.")

    key = (i, j)
    if key not in idx_map:
        raise KeyError(f"(x,y)=({x},{y}) -> (i,j)=({i},{j}) not in idx_map (outside domain?)")
    return idx_map[key]


def build_vertex_adjacency(
    idx_map: Dict[Tuple[int, int], int],
    num_vertices: int,
    mod: mesh_mode = "Hexa",
    mode: NeighborMode = "Extended_edges",
    *,
    missing: int = -1,   # 없는 이웃 채우는 값
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Returns:
      adj: (N, K) int array, adj[v, k] = neighbor index or 'missing'
      offsets: length K, adjacency index k가 의미하는 (di, dj)
    """
    mod = mod.strip()
    mode = mode.strip()

    # ---- offsets depending on mesh type ----
    if mod == "Square":
        offsets = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        if mode in ("Extended_edges", "Extra_extended_edges"):
            offsets = [
                (0, 1), (1, 1), (1, 0), (1, -1),
                (0, -1), (-1, -1), (-1, 0), (-1, 1),
            ]
        if mode == "Extra_extended_edges":
            offsets = [
                (0, 1), (1, 2), (1, 1), (2, 1),
                (1, 0), (2, -1), (1, -1), (1, -2),
                (0, -1), (-1, -2), (-1, -1), (-2, -1),
                (-1, 0), (-2, 1), (-1, 1), (-1, 2),
            ]

    elif mod == "Hexa":
        offsets = [(0,1),(1,0),(1,-1),(0,-1),(-1,0),(-1,1)]
        if mode in ("Extended_edges", "Extra_extended_edges"):
            offsets = [
                (-1,2),(0,1),(1,1),(1,0),(2,-1),(1,-1),
                (1,-2),(0,-1),(-1,-1),(-1,0),(-2,1),(-1,1)
            ]
        if mode == "Extra_extended_edges":
            offsets = [
                (-1,2),(-1,3),(0,1),(1,2),(1,1),(2,1),
                (1,0),(3,-1),(2,-1),(3,-2),(1,-1),(2,-3),
                (1,-2),(1,-3),(0,-1),(-1,-2),(-1,-1),(-2,-1),
                (-1,0),(-3,1),(-2,1),(-3,2),(-1,1),(-2,3)
            ]
    else:
        raise ValueError("mod must be 'Square' or 'Hexa'.")

    K = len(offsets)
    adj = np.full((num_vertices, K), missing, dtype=np.int32)

    # idx_map의 (i,j)마다 K개 슬롯을 채운다
    for (i, j), v in idx_map.items():
        for k, (di, dj) in enumerate(offsets):
            u = idx_map.get((i + di, j + dj))
            if u is not None:
                adj[v, k] = u

    return adj
def angle_from_y_clockwise_deg(dx: float, dy: float) -> float:
    """0°=+y, 90°=+x, 180°=-y, 270°=-x"""
    ang = math.degrees(math.atan2(dx, dy))
    return (ang + 360.0) % 360.0

def build_adjacency_angles(
    vertices: np.ndarray,     # (N,2)
    adj_fixed: np.ndarray,    # (N,K)  (-1 for missing)
    *,
    missing_angle: float = float("nan"),
) -> np.ndarray:
    """
    Returns:
      ang: (N,K) float array, ang[v,k] = angle_deg or NaN if missing neighbor
    """
    N, K = adj_fixed.shape
    ang = np.full((N, K), missing_angle, dtype=np.float64)

    for v in range(N):
        xv, yv = vertices[v]
        for k in range(K):
            u = int(adj_fixed[v, k])
            if u < 0:
                continue
            xu, yu = vertices[u]
            dx, dy = (xu - xv), (yu - yv)
            ang[v, k] = angle_from_y_clockwise_deg(dx, dy)

    return ang

def fixed_to_ragged(adj_fixed: np.ndarray, missing: int = -1) -> List[List[int]]:
    return [[int(u) for u in row if int(u) != missing] for row in adj_fixed]
from dataclasses import dataclass

@dataclass
class Vortex:
    x0: float
    y0: float
    Gamma: float       # circulation strength
    core: float = 1.0  # core radius a (regularization)

def angle_from_y_clockwise_deg_vortex(ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """
    Angle from +y axis, clockwise, in degrees.
    0°=+y, 90°=+x, 180°=-y, 270°=-x
    """
    ang = np.degrees(np.arctan2(ux, uy))  # <-- IMPORTANT: atan2(ux, uy)
    return (ang + 360.0) % 360.0

def vortex_velocity(points: np.ndarray, vortex: Vortex, eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    """
    points: (N,2) array [[x,y],...]
    returns: ux, uy each (N,)
    Lamb-Oseen-like regularized point vortex
    """
    x = points[:, 0]
    y = points[:, 1]
    dx = x - vortex.x0
    dy = y - vortex.y0
    r2 = dx*dx + dy*dy

    # regularization factor alpha(r) = 1 - exp(-r^2/a^2)
    a2 = max(vortex.core, eps) ** 2
    alpha = 1.0 - np.exp(-r2 / a2)

    # avoid divide-by-zero at center
    denom = np.maximum(r2, eps)

    coef = vortex.Gamma / (2.0 * math.pi)
    ux = -coef * (dy / denom) * alpha
    uy =  coef * (dx / denom) * alpha
    return ux, uy

def composite_current(
    points: np.ndarray,
    vortices: List[Vortex],
    uniform: Tuple[float, float] = (0.0, 0.0),
    return_per_vortex: bool = False
) -> Dict[str, np.ndarray]:
    """
    returns dict with:
      total_ux, total_uy, speed, angle_deg
      (optional) per_vortex_ux, per_vortex_uy, per_vortex_speed, per_vortex_angle_deg
    """
    N = points.shape[0]
    total_ux = np.full(N, uniform[0], dtype=float)
    total_uy = np.full(N, uniform[1], dtype=float)

    if return_per_vortex:
        K = len(vortices)
        per_ux = np.zeros((K, N), dtype=float)
        per_uy = np.zeros((K, N), dtype=float)

    for k, vtx in enumerate(vortices):
        ux, uy = vortex_velocity(points, vtx)
        total_ux += ux
        total_uy += uy
        if return_per_vortex:
            per_ux[k] = ux
            per_uy[k] = uy

    speed = np.hypot(total_ux, total_uy)
    angle_deg = angle_from_y_clockwise_deg_vortex(total_ux, total_uy)

    out = {
        "total_ux": total_ux,
        "total_uy": total_uy,
        "speed": speed,
        "angle_deg": angle_deg,
    }
    return out
def Gamma_from_RV(R: float, V: float) -> float:
    return 2.0 * math.pi * R * V

def _rand_around(
    rng: np.random.Generator,
    base: float,
    rel_std: float,
    clip_lo: float,
    clip_hi: float
) -> float:
    """base 주변 랜덤(정규) 변동 + 클리핑."""
    v = base * (1.0 + rng.normal(0.0, rel_std))
    return float(np.clip(v, clip_lo * base, clip_hi * base))


def _sample_pos(
    rng: np.random.Generator,
    Lx: float,
    Ly: float,
    margin: float,
    existing_xy: List[Tuple[float, float]],
    min_dist: float,
    max_tries: int = 5000,
) -> Tuple[float, float]:
    """영역 내 좌표 샘플링 + 기존 점들과 최소거리 제약."""
    for _ in range(max_tries):
        x0 = float(rng.uniform(margin, Lx - margin))
        y0 = float(rng.uniform(margin, Ly - margin))
        ok = True
        for ex, ey in existing_xy:
            if (x0 - ex) ** 2 + (y0 - ey) ** 2 < min_dist ** 2:
                ok = False
                break
        if ok:
            return x0, y0

    # 제약이 빡세서 실패할 수 있음. 마지막 시도값 반환(원하면 raise로 변경 가능)
    return x0, y0
def _rotate_points_2d(points: np.ndarray, angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    R = np.array([[c, -s],
                  [s,  c]], dtype=float)
    return points @ R.T
def make_random_vortices(
    *,
    Lx: float = 3600.0,
    Ly: float = 3600.0,
    seed: Optional[int] = None,

    # 개수
    n_mid: int = 3,
    n_small: int = 6,

    # 강도 스케일
    big_R: float = 1800.0,
    big_V: float = 0.5,
    mid_R: float = 900.0,
    mid_V: float = 0.40,
    small_R: float = 250.0,
    small_V: float = 0.4,

    # 코어 반경
    big_core: float = 900.0,
    mid_core: float = 550.0,
    small_core: float = 350.0,

    # 랜덤 변동 폭
    gamma_rel_std_big: float = 0.06,
    gamma_rel_std_mid: float = 0.08,
    gamma_rel_std_small: float = 0.10,
    core_rel_std: float = 0.08,

    # 배치 제약
    margin: float = 120.0,
    min_dist_big: float = 900.0,
    min_dist_mid: float = 450.0,
    min_dist_small: float = 250.0,

    # big template 크기
    pair_dist_base_frac: float = 0.50,   # pair 거리의 기준: min(Lx, Ly)의 비율
    pair_dist_rel_std: float = 0.12,
    tri_scale_base_frac: float = 0.42,   # triangle 크기 기준
    tri_scale_rel_std: float = 0.12,

    # big 중심점 흔들기
    big_center_jitter_frac_x: float = 0.08,
    big_center_jitter_frac_y: float = 0.08,

    # big 회전 방향 / 부호 랜덤성
    randomize_big_signs: bool = True,
    rebalance_big_signs: bool = True,

    # mid / small 부호 랜덤성
    randomize_mid_signs: bool = True,
    randomize_small_signs: bool = True,
    rebalance_mid_small_signs: bool = True,

    # uniform flow 랜덤성
    randomize_uniform: bool = True,
    uniform_speed_min: float = 0.0,
    uniform_speed_max: float = 0.08,

    # fallback
    default_uniform: Tuple[float, float] = (0.0, 0.0),
):
    """
    Returns:
      vortices: List[Vortex]
      uniform:  (Ux, Uy)

    특징:
      - big vortices는 2개 또는 3개
      - 2개면 pair 구조, 3개면 triangle 구조
      - 단, pair / triangle 전체 orientation은 랜덤 회전
      - big vortex의 부호(회전 방향)도 랜덤
      - mid / small도 독립적인 랜덤 부호
      - uniform flow도 방향 및 세기 랜덤
    """
    rng = np.random.default_rng(seed)

    # --------------------------------------------------
    # base Gamma
    # --------------------------------------------------
    Gamma_big_base = Gamma_from_RV(big_R, big_V)
    Gamma_mid_base = Gamma_from_RV(mid_R, mid_V)
    Gamma_small_base = Gamma_from_RV(small_R, small_V)

    vortices: List[Vortex] = []
    existing_xy: List[Tuple[float, float]] = []

    # --------------------------------------------------
    # local helper: boundary 안으로 shift
    # --------------------------------------------------
    def _shift_points_into_domain(pts: np.ndarray, margin_local: float) -> np.ndarray:
        pts = pts.copy()

        min_x, max_x = np.min(pts[:, 0]), np.max(pts[:, 0])
        min_y, max_y = np.min(pts[:, 1]), np.max(pts[:, 1])

        shift_x = 0.0
        shift_y = 0.0

        if min_x < margin_local:
            shift_x += (margin_local - min_x)
        if max_x > Lx - margin_local:
            shift_x -= (max_x - (Lx - margin_local))
        if min_y < margin_local:
            shift_y += (margin_local - min_y)
        if max_y > Ly - margin_local:
            shift_y -= (max_y - (Ly - margin_local))

        pts[:, 0] += shift_x
        pts[:, 1] += shift_y
        return pts

    # --------------------------------------------------
    # 1) big vortices
    # --------------------------------------------------
    n_big = int(rng.integers(2, 4))   # 2 or 3

    # 중심점은 전체 domain 중심 근처에서 랜덤
    cx = float(np.clip(
        0.5 * Lx + rng.normal(0.0, big_center_jitter_frac_x * Lx),
        margin, Lx - margin
    ))
    cy = float(np.clip(
        0.5 * Ly + rng.normal(0.0, big_center_jitter_frac_y * Ly),
        margin, Ly - margin
    ))

    # orientation 랜덤
    rot = float(rng.uniform(0.0, 2.0 * math.pi))

    if n_big == 2:
        # -------------------------
        # pair template
        # -------------------------
        pair_dist_base = max(min_dist_big, pair_dist_base_frac * min(Lx, Ly))
        pair_dist = _rand_around(
            rng,
            base=pair_dist_base,
            rel_std=pair_dist_rel_std,
            clip_lo=0.7,
            clip_hi=1.3,
        )
        half = 0.5 * pair_dist

        local_pts = np.array([
            [-half, 0.0],
            [ half, 0.0],
        ], dtype=float)

        pts = _rotate_points_2d(local_pts, rot)
        pts[:, 0] += cx
        pts[:, 1] += cy
        pts = _shift_points_into_domain(pts, margin)

        # big sign randomization
        if randomize_big_signs:
            if rng.random() < 0.5:
                s1, s2 = +1.0, -1.0
            else:
                s1, s2 = -1.0, +1.0
        else:
            s1, s2 = +1.0, -1.0

        G1 = _rand_around(rng, Gamma_big_base, gamma_rel_std_big, 0.6, 1.4)
        G2 = _rand_around(rng, Gamma_big_base, gamma_rel_std_big, 0.6, 1.4)
        C1 = _rand_around(rng, big_core, core_rel_std, 0.6, 1.4)
        C2 = _rand_around(rng, big_core, core_rel_std, 0.6, 1.4)

        vortices.append(Vortex(float(pts[0, 0]), float(pts[0, 1]), s1 * G1, C1))
        vortices.append(Vortex(float(pts[1, 0]), float(pts[1, 1]), s2 * G2, C2))
        existing_xy += [
            (float(pts[0, 0]), float(pts[0, 1])),
            (float(pts[1, 0]), float(pts[1, 1])),
        ]

    else:
        # -------------------------
        # triangle template
        # local: top 1개 + bottom 2개
        # 이후 통째로 회전
        # -------------------------
        tri_scale_base = max(min_dist_big, tri_scale_base_frac * min(Lx, Ly))
        tri_scale = _rand_around(
            rng,
            base=tri_scale_base,
            rel_std=tri_scale_rel_std,
            clip_lo=0.75,
            clip_hi=1.25,
        )

        h = tri_scale
        w = 0.92 * tri_scale

        local_pts = np.array([
            [ 0.0,      +0.65 * h],   # top
            [-0.5 * w,  -0.35 * h],   # bottom-left
            [+0.5 * w,  -0.35 * h],   # bottom-right
        ], dtype=float)

        pts = _rotate_points_2d(local_pts, rot)
        pts[:, 0] += cx
        pts[:, 1] += cy
        pts = _shift_points_into_domain(pts, margin)

        if randomize_big_signs:
            big_signs = rng.choice([-1.0, 1.0], size=3)
            if rebalance_big_signs and abs(np.sum(big_signs)) == 3:
                big_signs[-1] *= -1.0
        else:
            big_signs = np.array([+1.0, -1.0, +1.0], dtype=float)

        Gs = np.array([
            _rand_around(rng, Gamma_big_base, gamma_rel_std_big, 0.6, 1.4),
            _rand_around(rng, Gamma_big_base, gamma_rel_std_big, 0.6, 1.4),
            _rand_around(rng, Gamma_big_base, gamma_rel_std_big, 0.6, 1.4),
        ], dtype=float)

        Cs = np.array([
            _rand_around(rng, big_core, core_rel_std, 0.6, 1.4),
            _rand_around(rng, big_core, core_rel_std, 0.6, 1.4),
            _rand_around(rng, big_core, core_rel_std, 0.6, 1.4),
        ], dtype=float)

        for k in range(3):
            vortices.append(
                Vortex(
                    float(pts[k, 0]),
                    float(pts[k, 1]),
                    float(big_signs[k] * Gs[k]),
                    float(Cs[k]),
                )
            )
            existing_xy.append((float(pts[k, 0]), float(pts[k, 1])))

    # --------------------------------------------------
    # 2) mid vortices
    # --------------------------------------------------
    if randomize_mid_signs:
        mid_signs = rng.choice([-1.0, 1.0], size=n_mid)
        if rebalance_mid_small_signs and n_mid >= 2 and abs(np.sum(mid_signs)) == n_mid:
            mid_signs[-1] *= -1.0
    else:
        mid_signs = np.array(
            [-1.0, +1.0] * ((n_mid + 1) // 2),
            dtype=float
        )[:n_mid]

    for k in range(n_mid):
        xm, ym = _sample_pos(
            rng,
            Lx, Ly,
            margin=max(margin, 0.6 * mid_core),
            existing_xy=existing_xy,
            min_dist=min_dist_mid,
        )
        Gmid = _rand_around(rng, Gamma_mid_base, gamma_rel_std_mid, 0.6, 1.4) * float(mid_signs[k])
        Cmid = _rand_around(rng, mid_core, core_rel_std, 0.6, 1.4)
        vortices.append(Vortex(xm, ym, Gmid, Cmid))
        existing_xy.append((xm, ym))

    # --------------------------------------------------
    # 3) small vortices
    # --------------------------------------------------
    if randomize_small_signs:
        small_signs = rng.choice([-1.0, 1.0], size=n_small)

        if rebalance_mid_small_signs and n_small >= 4:
            total_sign = int(np.sum(small_signs))
            if abs(total_sign) > max(2, n_small // 2):
                target_sign = np.sign(total_sign)
                idx_perm = rng.permutation(n_small)
                flip_needed = abs(total_sign) // 2
                flipped = 0
                for j in idx_perm:
                    if np.sign(small_signs[j]) == target_sign:
                        small_signs[j] *= -1.0
                        flipped += 1
                        if flipped >= flip_needed:
                            break
    else:
        small_signs = np.array(
            [+1.0, +1.0, -1.0, -1.0] * ((n_small + 3) // 4),
            dtype=float
        )[:n_small]

    for k in range(n_small):
        xs, ys = _sample_pos(
            rng,
            Lx, Ly,
            margin=max(margin, 0.6 * small_core),
            existing_xy=existing_xy,
            min_dist=min_dist_small,
        )
        Gs = _rand_around(rng, Gamma_small_base, gamma_rel_std_small, 0.6, 1.4) * float(small_signs[k])
        Cs = _rand_around(rng, small_core, core_rel_std, 0.6, 1.4)
        vortices.append(Vortex(xs, ys, Gs, Cs))
        existing_xy.append((xs, ys))

    # --------------------------------------------------
    # 4) uniform flow
    # --------------------------------------------------
    if randomize_uniform:
        u_speed = float(rng.uniform(uniform_speed_min, uniform_speed_max))
        u_ang = float(rng.uniform(0.0, 360.0))
        th = np.deg2rad(u_ang)

        # angle convention: 0=+y, 90=+x
        uniform = (
            float(u_speed * np.sin(th)),
            float(u_speed * np.cos(th)),
        )
    else:
        uniform = default_uniform

    return vortices, uniform



def representative_center(vortices: List[Vortex], method: str = "absGamma") -> Tuple[float, float]:
    """
    method:
      - 'Gamma'    : weighted by Gamma (can cancel out if signs mix)
      - 'absGamma' : weighted by |Gamma| (stable)
    """
    if len(vortices) == 0:
        raise ValueError("No vortices provided.")
    w = np.array([v.Gamma for v in vortices], dtype=float)
    if method == "absGamma":
        w = np.abs(w)
    denom = np.sum(w)
    if denom == 0:
        # fallback: simple average
        xs = np.array([v.x0 for v in vortices])
        ys = np.array([v.y0 for v in vortices])
        return float(xs.mean()), float(ys.mean())

    xs = np.array([v.x0 for v in vortices])
    ys = np.array([v.y0 for v in vortices])
    return float(np.sum(w * xs) / denom), float(np.sum(w * ys) / denom)
def debug_print_astar_failure_for_usv(
    env,
    state: Dict,
    *,
    usv_idx: int,
    candidate_pool: List[Dict],
    heuristic_mode: str,
    reason: str = "all_candidates_failed",
):
    current_time = float(state["command_time"])
    command_vid = int(state["command_vid"])
    command_xy = np.asarray(state["command_xy"], dtype=float)
    command_heading_deg = float(state["command_heading_deg"])

    # current statistics
    current_speed = np.hypot(env.total_ux, env.total_uy)
    current_mean = float(np.mean(current_speed))
    current_max = float(np.max(current_speed))
    current_min = float(np.min(current_speed))

    # graph feasibility
    inf_edge_ratio = float(np.mean(~np.isfinite(env.cost_e)))

    # command node outgoing edge stats
    row_eids = env.cache.edge_id[command_vid]
    cmd_valid_edge_count = 0
    cmd_total_edge_count = 0
    for e in row_eids:
        if int(e) >= 0:
            cmd_total_edge_count += 1
            if np.isfinite(env.cost_e[int(e)]):
                cmd_valid_edge_count += 1

    print("\n" + "=" * 100)
    print("[ASTAR SURROGATE FAILURE DEBUG]")
    print(f"reason                : {reason}")
    print(f"heuristic_mode        : {heuristic_mode}")
    print(f"usv_idx               : {usv_idx}")
    print(f"current_time          : {current_time:.6f}")
    print(f"command_vid           : {command_vid}")
    print(f"command_xy            : ({command_xy[0]:.3f}, {command_xy[1]:.3f})")
    print(f"command_heading_deg   : {command_heading_deg:.3f}")
    print(f"current_mean          : {current_mean:.6f}")
    print(f"current_max           : {current_max:.6f}")
    print(f"current_min           : {current_min:.6f}")
    print(f"usv_speed             : {float(env.config.usv_speed):.6f}")
    print(f"inf_edge_ratio        : {inf_edge_ratio:.6f}")
    print(f"cmd_valid_edges       : {cmd_valid_edge_count}/{cmd_total_edge_count}")
    print(f"num_candidates        : {len(candidate_pool)}")

    for rank, cand in enumerate(candidate_pool):
        goal_vid = int(cand["vertex_idx"])
        goal_xy = np.asarray(cand["xy"], dtype=float)

        # goal node outgoing edge stats
        row_eids_goal = env.cache.edge_id[goal_vid]
        goal_valid_edge_count = 0
        goal_total_edge_count = 0
        for e in row_eids_goal:
            if int(e) >= 0:
                goal_total_edge_count += 1
                if np.isfinite(env.cost_e[int(e)]):
                    goal_valid_edge_count += 1

        dist_euclid = float(np.linalg.norm(goal_xy - command_xy))
        future_residual = np.asarray(cand["future_residual"], dtype=float)
        finite_future = future_residual[np.isfinite(future_residual)]

        print("-" * 100)
        print(f"[candidate rank {rank}]")
        print(f"gidx                  : {int(cand['gidx'])}")
        print(f"local_wp_idx          : {int(cand['local_wp_idx'])}")
        print(f"goal_vid              : {goal_vid}")
        print(f"goal_xy               : ({goal_xy[0]:.3f}, {goal_xy[1]:.3f})")
        print(f"euclidean_dist        : {dist_euclid:.6f}")
        print(f"pred_travel_time_rel  : {float(cand['pred_travel_time_rel']):.6f}")
        print(f"goal_valid_edges      : {goal_valid_edge_count}/{goal_total_edge_count}")
        print(f"future_residual_all   : {future_residual.tolist()}")
        print(f"future_residual_finite: {finite_future.tolist()}")

    print("=" * 100 + "\n")

@dataclass(slots=True)
class EdgeCache:
    N: int
    K: int
    E: int
    missing: int
    adj: np.ndarray        # (N,K) int32  neighbor or -1
    edge_id: np.ndarray    # (N,K) int32  directed edge id or -1
    dst_k_in: np.ndarray   # (E,)   int32  for edge e=(v->u), k_in at u such that adj[u,k_in]==v
    in_edge_id: np.ndarray # (N,K) int32  incoming edge id for state (v,k_in): (adj[v,k_in]
    src: np.ndarray        # (E,) int32
    dst: np.ndarray        # (E,) int32
    dx: np.ndarray         # (E,) float64
    dy: np.ndarray         # (E,) float64
    L: np.ndarray          # (E,) float64
    theta0: np.ndarray     # (E,) float64 rad

def build_edge_cache(vertices: np.ndarray, adj_fixed: np.ndarray, missing: int = -1) -> EdgeCache:
    """
    Build fixed-slot edge cache.

    Requirements:
      - adj_fixed is (N,K) with neighbor index or missing(-1)
      - adj_fixed should be symmetric (if u is neighbor of v, v should appear in u's slots)
        so that dst_k_in / in_edge_id can be constructed.

    Returns EdgeCacheFixed with:
      adj, edge_id, src, dst, dx, dy, L, theta0,
      dst_k_in, in_edge_id
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    adj_fixed = np.asarray(adj_fixed, dtype=np.int32)

    if vertices.ndim != 2 or vertices.shape[1] != 2:
        raise ValueError("vertices must be (N,2).")
    N = vertices.shape[0]
    if adj_fixed.ndim != 2:
        raise ValueError("adj_fixed must be 2D (N,K).")
    if adj_fixed.shape[0] != N:
        raise ValueError("adj_fixed first dim must match vertices N.")
    K = int(adj_fixed.shape[1])

    valid = (adj_fixed != missing)
    E = int(valid.sum())

    src = np.empty(E, dtype=np.int32)
    dst = np.empty(E, dtype=np.int32)
    edge_id = np.full((N, K), missing, dtype=np.int32)

    # --- build directed edge list in (v, k) order ---
    e = 0
    for v in range(N):
        for k in range(K):
            u = int(adj_fixed[v, k])
            if u == missing:
                continue
            if not (0 <= u < N):
                raise ValueError(f"adj_fixed[{v},{k}]={u} out of range [0,{N}).")
            src[e] = v
            dst[e] = u
            edge_id[v, k] = e
            e += 1
    if e != E:
        raise RuntimeError("Edge count mismatch while building edge list.")

    # --- neighbor -> slot map (for fast reverse lookup) ---
    nbr2k: List[Dict[int, int]] = []
    for v in range(N):
        d: Dict[int, int] = {}
        for k in range(K):
            u = int(adj_fixed[v, k])
            if u != missing:
                d[u] = k
        nbr2k.append(d)

    # --- dst_k_in[e] : at destination u, which slot corresponds to coming from v? ---
    dst_k_in = np.full(E, missing, dtype=np.int32)
    for ee in range(E):
        v = int(src[ee])
        u = int(dst[ee])
        k_in = nbr2k[u].get(v, None)
        if k_in is None:
            raise RuntimeError(
                f"adj_fixed is not symmetric: node {u} has no slot for neighbor {v} "
                f"(needed for dst_k_in of edge {v}->{u})."
            )
        dst_k_in[ee] = int(k_in)

    # --- in_edge_id[v, k_in] : edge id of (adj_fixed[v,k_in] -> v) ---
    in_edge_id = np.full((N, K), missing, dtype=np.int32)
    for v in range(N):
        for k_in in range(K):
            pv = int(adj_fixed[v, k_in])
            if pv == missing:
                continue
            k_out = nbr2k[pv].get(v, None)
            if k_out is None:
                raise RuntimeError(
                    f"adj_fixed is not symmetric: node {pv} has no slot for neighbor {v} "
                    f"(needed for in_edge_id of state (v={v},k_in={k_in}))."
                )
            eid = int(edge_id[pv, int(k_out)])
            if eid == missing:
                raise RuntimeError("edge_id lookup failed unexpectedly (internal inconsistency).")
            in_edge_id[v, k_in] = eid

    # --- geometry ---
    dx = vertices[dst, 0] - vertices[src, 0]
    dy = vertices[dst, 1] - vertices[src, 1]
    L = np.hypot(dx, dy)
    if np.any(L == 0):
        raise ValueError("Zero-length edge found (duplicate vertex or self-loop).")
    theta0 = np.arctan2(dx, dy)  # rad, from +y clockwise

    return EdgeCache(
        N=N, K=K, E=E, missing=missing,
        adj=adj_fixed.astype(np.int32, copy=False),
        edge_id=edge_id,
        dst_k_in=dst_k_in,
        in_edge_id=in_edge_id,
        src=src, dst=dst,
        dx=dx, dy=dy, L=L, theta0=theta0
    )

def compute_theta_cost_from_cache(
    cache: EdgeCache,
    theta_f_deg: np.ndarray,   # (N,)
    v_f: np.ndarray,           # (N,)
    V: float,
    eps_denom: float = 1e-12,
    infeasible_to_nan: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized per-edge compute using cached geometry.

    Returns:
      theta_deg: (E,) deg (from +y clockwise), infeasible -> nan (optional)
      denom:     (E,)
      cost:      (E,) invalid -> inf
      feasible:  (E,) bool
      valid:     (E,) bool (feasible & denom>eps_denom)
    """
    if V <= 0:
        raise ValueError("V must be positive.")
    if theta_f_deg.shape[0] != cache.N or v_f.shape[0] != cache.N:
        raise ValueError("theta_f_deg and v_f must have shape (N,) matching cache.N.")

    theta_f = np.deg2rad(theta_f_deg[cache.src])  # per-edge source current angle
    vf_e = v_f[cache.src]

    delta = theta_f - cache.theta0
    s = (vf_e / float(V)) * np.sin(delta)

    feasible = np.abs(s) <= 1.0
    s_clip = np.clip(s, -1.0, 1.0)

    # theta = theta0 - asin(s)
    theta = cache.theta0 - np.arcsin(s_clip)
    if infeasible_to_nan:
        theta = np.where(feasible, theta, np.nan)

    # denom = vf*cos(delta) + V*cos(theta0-theta)
    # cos(theta0-theta) = cos(asin(s)) = sqrt(1-s^2)
    cos_term = np.sqrt(np.maximum(0.0, 1.0 - s_clip**2))
    denom = vf_e * np.cos(delta) + float(V) * cos_term

    valid = feasible & (denom > eps_denom)

    cost = np.full(cache.E, np.inf, dtype=float)
    cost[valid] = cache.L[valid] / denom[valid]

    theta_deg = (np.rad2deg(theta) + 360.0) % 360.0
    return theta_deg, denom, cost, feasible, valid

def get_edge_id(cache: EdgeCache, v: int, k: int) -> int:
    return int(cache.edge_id[v, k])  # -1이면 missing

def wrap180_deg(a: float) -> float:
    return (float(a) + 180.0) % 360.0 - 180.0

def ang_diff_deg(a_deg: float, b_deg: float) -> float:
    return wrap180_deg(float(a_deg) - float(b_deg))

def turn_penalty_deg(delta_deg: float,
                     th1: float, lam1: float,
                     th2: float, lam2: float) -> float:
    """
    2-stage proportional penalty (piecewise linear):
      - |delta| <= th1 : 0
      - th1 < |delta| <= th2 : lam1 * (|delta|-th1)
      - |delta| > th2 : lam1*(th2-th1) + lam2*(|delta|-th2)

    여기서 lam1, lam2는 'deg당 비용' (cost per degree) 의미.
    """
    ad = abs(float(delta_deg))
    if ad <= th1:
        return 0.0
    if ad <= th2:
        return float(lam1) * (ad - th1)
    return float(lam2)

def dijkstra_turn_state_core(
    cache: EdgeCache,
    base_cost_e: np.ndarray,     # (E,)
    theta_e_deg: np.ndarray,     # (E,) (+y clockwise, deg)
    start: int,
    goal: int,

    th1: float = 40.0, lam1: float = 10,
    th2: float = 50.0, lam2: float = np.inf,

    use_start_heading: bool = False,
    start_heading_deg: float = 0.0,

    termination: Literal["goal_best", "goal_allk", "all"] = "goal_best",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    """
    Fixed(K) Dijkstra over state (v, k_in).

    Uses:
      cache.adj (N,K), cache.edge_id (N,K),
      cache.dst_k_in (E,), cache.in_edge_id (N,K)
    """
    N = int(cache.N)
    K = int(cache.K)
    missing = int(cache.missing)

    if base_cost_e.shape[0] != cache.E or theta_e_deg.shape[0] != cache.E:
        raise ValueError("base_cost_e/theta_e_deg must have shape (E,) matching cache.E.")

    valid_in = (cache.adj != missing)  # (N,K)

    INF = float("inf")
    dist = np.full((N, K), INF, dtype=float)
    prev_node = np.full((N, K), -1, dtype=np.int32)
    prev_k    = np.full((N, K), -1, dtype=np.int32)

    if start == goal and termination !="all":
        return dist, prev_node, prev_k, 0.0, -1

    # goal_allk: 유효한 k만 세야 함
    goal_valid = valid_in[goal]
    goal_settled = np.zeros(K, dtype=bool)
    goal_settled_cnt = 0
    goal_valid_cnt = int(goal_valid.sum())

    h: list[tuple[float, int, int]] = []  # (d, v, k_in)

    # --- init from start (start -> u) ---
    for k_out in range(K):
        e = int(cache.edge_id[start, k_out])
        if e == missing:
            continue
        u = int(cache.adj[start, k_out])
        if u == missing:
            continue

        k_in_u = int(cache.dst_k_in[e])
        if not (0 <= k_in_u < K) or not bool(valid_in[u, k_in_u]):
            continue

        c = float(base_cost_e[e])
        th_out = float(theta_e_deg[e])
        if not np.isfinite(c) or not np.isfinite(th_out):
            continue

        cost = c
        if use_start_heading:
            delta0 = wrap180_deg(th_out - float(start_heading_deg))
            cost += turn_penalty_deg(delta0, th1, lam1, th2, lam2)

        if cost < dist[u, k_in_u]:
            dist[u, k_in_u] = cost
            prev_node[u, k_in_u] = int(start)
            prev_k[u, k_in_u] = -1
            heapq.heappush(h, (cost, u, k_in_u))

    best_goal = INF
    best_goal_k = -1

    pop_cnt = 0
    while h:
        d, v, k_in = heapq.heappop(h)
        if d != dist[v, k_in]:
            continue
        pop_cnt += 1

        if v == goal:
            if d < best_goal:
                best_goal = float(d)
                best_goal_k = int(k_in)

            if goal_valid[k_in] and not goal_settled[k_in]:
                goal_settled[k_in] = True
                goal_settled_cnt += 1

            if termination == "goal_best":
                #print("STOP at goal_best, pop_cnt =", pop_cnt)
                break
            if termination == "goal_allk" and goal_settled_cnt == goal_valid_cnt:
                break
            continue

        # incoming edge (prev -> v) for this state
        e_in = int(cache.in_edge_id[v, k_in])
        if e_in == missing:
            continue
        theta_in = float(theta_e_deg[e_in])
        if not np.isfinite(theta_in):
            continue

        # outgoing by fixed slots
        for k_out in range(K):
            e_out = int(cache.edge_id[v, k_out])
            if e_out == missing:
                continue
            u = int(cache.adj[v, k_out])
            if u == missing:
                continue

            k_in_u = int(cache.dst_k_in[e_out])
            if not (0 <= k_in_u < K) or not bool(valid_in[u, k_in_u]):
                continue

            c = float(base_cost_e[e_out])
            th_out = float(theta_e_deg[e_out])
            if not np.isfinite(c) or not np.isfinite(th_out):
                continue

            delta = wrap180_deg(th_out - theta_in)
            nd = float(d) + c + turn_penalty_deg(delta, th1, lam1, th2, lam2)

            if nd < dist[u, k_in_u]:
                dist[u, k_in_u] = nd
                prev_node[u, k_in_u] = int(v)
                prev_k[u, k_in_u] = int(k_in)
                heapq.heappush(h, (nd, u, k_in_u))

    return dist, prev_node, prev_k, float(best_goal), int(best_goal_k)
def astar_turn_state_core(
    cache: EdgeCache,
    base_cost_e: np.ndarray,         # (E,)
    theta_e_deg: np.ndarray,         # (E,)  +y clockwise, deg
    points: np.ndarray,              # (N,2)
    start: int,
    goal: int,

    heuristic_mode: str = "euclidean",   # "euclidean" or "rf"
    rf_model=None,
    rf_feature_names: Optional[List[str]] = None,
    blocked_mask: Optional[np.ndarray] = None,
    total_ux: Optional[np.ndarray] = None,
    total_uy: Optional[np.ndarray] = None,
    current_heading_deg: float = 0.0,
    inner_width: float = 90.0,
    outer_width: float = 180.0,
    usv_speed: float = 2.5,
    residual_mode: str = "percent",

    th1: float = 40.0,
    lam1: float = 10.0,
    th2: float = 50.0,
    lam2: float = np.inf,

    use_start_heading: bool = False,
    start_heading_deg: float = 0.0,

    termination: Literal["goal_best", "goal_allk", "all"] = "goal_best",
    heuristic_weight: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    """
    A* over state (v, k_in).

    Returns:
      dist:      (N,K) g-cost
      prev_node: (N,K)
      prev_k:    (N,K)
      best_goal: best g-cost to goal
      best_goal_k: best incoming slot at goal
    """
    N = int(cache.N)
    K = int(cache.K)
    missing = int(cache.missing)

    if base_cost_e.shape[0] != cache.E or theta_e_deg.shape[0] != cache.E:
        raise ValueError("base_cost_e/theta_e_deg must have shape (E,) matching cache.E.")
    if points.shape[0] != N:
        raise ValueError("points must have shape (N,2) matching cache.N.")
    if usv_speed <= 0:
        raise ValueError("usv_speed must be positive.")

    valid_in = (cache.adj != missing)

    INF = float("inf")
    dist = np.full((N, K), INF, dtype=float)
    prev_node = np.full((N, K), -1, dtype=np.int32)
    prev_k = np.full((N, K), -1, dtype=np.int32)

    if start == goal and termination != "all":
        return dist, prev_node, prev_k, 0.0, -1

    goal_valid = valid_in[goal]
    goal_settled = np.zeros(K, dtype=bool)
    goal_settled_cnt = 0
    goal_valid_cnt = int(goal_valid.sum())

    # ---------------------------------------
    # heuristic
    # ---------------------------------------
    goal_xy = np.asarray(points[goal], dtype=float)

    h_cache: Dict[int, float] = {}

    def heuristic(node_vid: int) -> float:
        node_vid = int(node_vid)
        if node_vid in h_cache:
            return h_cache[node_vid]

        if heuristic_mode == "euclidean":
            d = float(np.linalg.norm(points[node_vid] - goal_xy))
            h = d / float(usv_speed)

        elif heuristic_mode == "rf":
            if rf_model is None or rf_feature_names is None:
                raise ValueError("rf_model and rf_feature_names are required for heuristic_mode='rf'.")
            if blocked_mask is None or total_ux is None or total_uy is None:
                raise ValueError("blocked_mask, total_ux, total_uy are required for heuristic_mode='rf'.")

            heading_for_h = float(current_heading_deg)
            if node_vid != start:
                # 중간 state에서는 현재 노드->goal의 bearing을 임시 heading으로 사용
                dx = float(points[goal, 0] - points[node_vid, 0])
                dy = float(points[goal, 1] - points[node_vid, 1])
                if abs(dx) > 1e-12 or abs(dy) > 1e-12:
                    heading_for_h = angle_from_y_clockwise_deg(dx, dy)

            heading_diff_deg = compute_heading_goal_diff_deg(
                points=points,
                start_idx=node_vid,
                goal_idx=goal,
                current_heading_deg=heading_for_h,
            )

            feat_dict = extract_fixed_features(
                points=points,
                blocked_mask=blocked_mask,
                total_ux=total_ux,
                total_uy=total_uy,
                start_idx=node_vid,
                goal_idx=goal,
                inner_width=inner_width,
                outer_width=outer_width,
                heading_diff_deg=heading_diff_deg,
            )

            row = np.asarray([[feat_dict[name] for name in rf_feature_names]], dtype=float)
            base_time = float(feat_dict["distance"]) / float(usv_speed)
            pred_residual = float(rf_model.predict(row)[0])

            if residual_mode == "ratio":
                h = base_time * (1.0 + pred_residual)
            elif residual_mode == "percent":
                h = base_time * (1.0 + pred_residual / 100.0)
            else:
                raise ValueError("residual_mode must be 'ratio' or 'percent'.")

            h = max(h, 1e-6)

        else:
            raise ValueError(f"Unknown heuristic_mode: {heuristic_mode}")

        h_cache[node_vid] = float(heuristic_weight) * float(h)
        return h_cache[node_vid]

    # heap item: (f, g, v, k_in)
    heap: List[Tuple[float, float, int, int]] = []

    # ---------------------------------------
    # init from start
    # ---------------------------------------
    for k_out in range(K):
        e = int(cache.edge_id[start, k_out])
        if e == missing:
            continue

        u = int(cache.adj[start, k_out])
        if u == missing:
            continue

        k_in_u = int(cache.dst_k_in[e])
        if not (0 <= k_in_u < K) or not bool(valid_in[u, k_in_u]):
            continue

        c = float(base_cost_e[e])
        th_out = float(theta_e_deg[e])
        if not np.isfinite(c) or not np.isfinite(th_out):
            continue

        g = c
        if use_start_heading:
            delta0 = wrap180_deg(th_out - float(start_heading_deg))
            g += turn_penalty_deg(delta0, th1, lam1, th2, lam2)

        if g < dist[u, k_in_u]:
            dist[u, k_in_u] = g
            prev_node[u, k_in_u] = int(start)
            prev_k[u, k_in_u] = -1
            f = g + heuristic(u)
            heapq.heappush(heap, (f, g, u, k_in_u))

    best_goal = INF
    best_goal_k = -1

    while heap:
        f, g, v, k_in = heapq.heappop(heap)

        # stale entry check: g 기준으로 확인
        if g != dist[v, k_in]:
            continue

        if v == goal:
            if g < best_goal:
                best_goal = float(g)
                best_goal_k = int(k_in)

            if goal_valid[k_in] and not goal_settled[k_in]:
                goal_settled[k_in] = True
                goal_settled_cnt += 1

            if termination == "goal_best":
                break
            if termination == "goal_allk" and goal_settled_cnt == goal_valid_cnt:
                break
            if termination != "all":
                continue

        e_in = int(cache.in_edge_id[v, k_in])
        if e_in == missing:
            continue

        theta_in = float(theta_e_deg[e_in])
        if not np.isfinite(theta_in):
            continue

        for k_out in range(K):
            e_out = int(cache.edge_id[v, k_out])
            if e_out == missing:
                continue

            u = int(cache.adj[v, k_out])
            if u == missing:
                continue

            k_in_u = int(cache.dst_k_in[e_out])
            if not (0 <= k_in_u < K) or not bool(valid_in[u, k_in_u]):
                continue

            c = float(base_cost_e[e_out])
            th_out = float(theta_e_deg[e_out])
            if not np.isfinite(c) or not np.isfinite(th_out):
                continue

            delta = wrap180_deg(th_out - theta_in)
            ng = float(g) + c + turn_penalty_deg(delta, th1, lam1, th2, lam2)

            if ng < dist[u, k_in_u]:
                dist[u, k_in_u] = ng
                prev_node[u, k_in_u] = int(v)
                prev_k[u, k_in_u] = int(k_in)
                nf = ng + heuristic(u)
                heapq.heappush(heap, (nf, ng, u, k_in_u))

    return dist, prev_node, prev_k, float(best_goal), int(best_goal_k)
def reconstruct_path_from_prev(
    cache,
    prev_node: np.ndarray,   # (N,K) int32
    prev_k: np.ndarray,      # (N,K) int32
    start: int,
    goal: int,
    best_goal_k: int,
    *,
    use_k: bool = False,
    fixed_k: int = 0,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Reconstruct (node_path, edge_path, k_in_path) from FIXED (N,K) prev arrays.

    Assumed cache fields (fixed version):
      - N: int
      - K: int
      - missing: int (typically -1)
      - in_edge_id: (N,K) int32 : incoming edge id for state (v,k_in) meaning (adj[v,k_in] -> v), or missing
    """
    if start == goal:
        return [start], [], []

    N = int(cache.N)
    K = int(cache.K)
    missing = int(getattr(cache, "missing", -1))

    if best_goal_k < 0:
        return [], [], []

    v = int(goal)
    k = int(fixed_k) if use_k else int(best_goal_k)
    if not (0 <= k < K):
        return [], [], []

    node_path_rev: List[int] = []
    k_path_rev: List[int] = []
    edge_path_rev: List[int] = []

    while True:
        node_path_rev.append(v)
        k_path_rev.append(k)

        pv = int(prev_node[v, k])
        pk = int(prev_k[v, k])

        # Collect incoming edge id for current state (pv -> v)
        e_in = int(cache.in_edge_id[v, k]) if hasattr(cache, "in_edge_id") else missing
        if e_in != missing:
            edge_path_rev.append(e_in)

        if pv < 0:
            break

        # reached the first hop from start
        if pv == start and pk == -1:
            node_path_rev.append(int(start))
            break

        v, k = pv, pk
        if not (0 <= k < K):
            # corrupted prev pointers
            return [], [], []

    node_path = node_path_rev[::-1]

    # k_path: start 다음 노드부터의 k_in (각 노드에 "어디서 들어왔는지" 슬롯)
    k_path = k_path_rev[::-1]
    if len(k_path) == len(node_path):
        k_path = k_path[1:]

    # edge_path: reverse로 쌓인 (prev->cur) 들을 뒤집기
    edge_path = edge_path_rev[::-1]

    # sanity: edge_path length should match node_path-1 (allow mismatch if cache.in_edge_id missing)
    if edge_path and len(edge_path) != len(node_path) - 1:
        # if mismatch, return without edges rather than returning wrong ones
        edge_path = []

    return node_path, edge_path, k_path

def last_heading_from_edge_path(theta_e_deg: np.ndarray, edge_path: list[int]) -> float:
    if not edge_path:
        raise ValueError("edge_path is empty; cannot get last heading.")
    return float(theta_e_deg[int(edge_path[-1])])

def sample_random_points(
    n: int,
    start: Tuple[float, float] = (300.0, 300.0),
    goal: Tuple[float, float]  = (3300.0, 1500.0),
    x_range: Tuple[float, float] = (0.0, 3600.0),
    y_range: Tuple[float, float] = (0.0, 1800.0),
    min_dist: float = 0.0,
    seed: Optional[int] = 42,
    *,
    dtype=np.float64,
    max_tries: int = 300000,
    batch: int = 1024,
) -> np.ndarray:
    """
    Returns:
        points: (n,2) array
          - points[0]  = start
          - points[-1] = goal
          - all pairwise distances >= min_dist (if min_dist > 0)
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    if min_dist < 0:
        raise ValueError("min_dist must be >= 0")

    x0, x1 = x_range
    y0, y1 = y_range
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0

    rng = np.random.default_rng(seed)

    if n == 0:
        return np.empty((0, 2), dtype=dtype)
    if n == 1:
        # 정책: 1개면 start만
        return np.array([[float(start[0]), float(start[1])]], dtype=dtype)

    sx, sy = float(start[0]), float(start[1])
    gx, gy = float(goal[0]), float(goal[1])

    # start/goal 범위 체크
    if not (x0 <= sx <= x1 and y0 <= sy <= y1):
        raise ValueError(f"start={start} is outside x_range/y_range.")
    if not (x0 <= gx <= x1 and y0 <= gy <= y1):
        raise ValueError(f"goal={goal} is outside x_range/y_range.")

    # min_dist면 start-goal 거리도 만족해야 함
    if min_dist > 0.0:
        d2 = (sx - gx) ** 2 + (sy - gy) ** 2
        if d2 < min_dist ** 2:
            raise ValueError(
                f"start and goal are closer than min_dist. dist={d2**0.5:.3f}, min_dist={min_dist}"
            )

    # 결과 배열: start, (random...), goal
    points = np.empty((n, 2), dtype=np.float64)
    points[0] = (sx, sy)
    points[-1] = (gx, gy)

    if n == 2:
        return points.astype(dtype, copy=False)

    # min_dist 없으면 가운데만 그냥 샘플링
    if min_dist <= 0.0:
        xs = rng.uniform(x0, x1, size=n - 2)
        ys = rng.uniform(y0, y1, size=n - 2)
        points[1:-1, 0] = xs
        points[1:-1, 1] = ys
        return points.astype(dtype, copy=False)

    min2 = float(min_dist * min_dist)

    # 채워진 점들의 리스트(거리 체크용): start + goal은 이미 들어있다고 보고 시작
    # points_filled에는 현재까지 확정된 점들을 모아두고, 마지막에 points[1:-1]에 채운다.
    filled = np.empty((n, 2), dtype=np.float64)
    filled[0] = (sx, sy)
    filled[1] = (gx, gy)
    k = 2  # filled에 들어있는 개수 (start, goal)

    out_mid = np.empty((n - 2, 2), dtype=np.float64)
    mcount = 0  # mid points 개수

    tries = 0
    while mcount < (n - 2) and tries < max_tries:
        m = min(batch, (n - 2) - mcount)
        cand_x = rng.uniform(x0, x1, size=m)
        cand_y = rng.uniform(y0, y1, size=m)
        cand = np.column_stack([cand_x, cand_y])

        for i in range(m):
            # filled(= start, goal, 그리고 이미 채택된 mid들)과의 거리 체크
            dx = filled[:k, 0] - cand[i, 0]
            dy = filled[:k, 1] - cand[i, 1]
            if np.min(dx * dx + dy * dy) >= min2:
                out_mid[mcount] = cand[i]
                filled[k] = cand[i]
                k += 1
                mcount += 1
                if mcount >= (n - 2):
                    break

        tries += m

    if mcount < (n - 2):
        raise RuntimeError(
            f"Failed to sample {n} points with min_dist={min_dist} "
            f"within max_tries={max_tries}. Only sampled {mcount + 2} points total. "
            f"(Try lowering min_dist or n, or increasing max_tries.)"
        )

    points[1:-1] = out_mid
    return points.astype(dtype, copy=False)



def k_from_coordinate(sx: float, sy: float, fx: float, fy: float, mod,neigh_mod) -> int:
    dx = fx - sx
    dy = fy - sy

    if dx == 0 and dy == 0:
        raise ValueError("start와 finish가 동일하면 방향 각도를 정의할 수 없다.")

    # y축 기준 각도 (deg) in [0, 360)
    theta = math.degrees(math.atan2(dx, dy)) % 360.0
    if mod=='Hexa':
        if neigh_mod == 'Edges':
            return int(((theta-30)//60+1)%6)
        if neigh_mod == 'Extended_edges':
            return int(((theta-15)//30+1)%12)
        if neigh_mod == 'Extra_extended_edges':
            return int(((theta-7.5)//15+1)%24)
    if mod=='Square':
        if neigh_mod == 'Edges':
            return int(((theta-45)//90+1)%4)
        if neigh_mod == 'Extended_edges':
            return int(((theta-22.5)//45+1)%8)
        if neigh_mod == 'Extra_extended_edges':
            return int(((theta-11.25)//22.5+1)%16)



def current_uv_from_speed_angle(
    speed: np.ndarray,        # (N,)
    angle_deg: np.ndarray,    # (N,) 0=+y, 90=+x (clockwise from +y)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert (speed, angle_deg) to (u,v) for quiver.
    Here:
      angle=0 => +y, angle=90 => +x
    so:
      u = speed * sin(theta), v = speed * cos(theta)
    """
    th = np.deg2rad(angle_deg.astype(float))
    u = speed.astype(float) * np.sin(th)
    v = speed.astype(float) * np.cos(th)
    return u, v


def edge_path_to_xy(
    points: np.ndarray,   # (N,2)
    cache,                # EdgeCacheFixed (src,dst)
    edge_path: Sequence[int],
) -> np.ndarray:
    """
    Build polyline coordinates (M,2) following the edge_path.
    Returns a sequence of points along the path (node coords in order).
    """
    if len(edge_path) == 0:
        return np.zeros((0, 2), dtype=float)

    edge_path = [int(e) for e in edge_path]
    src0 = int(cache.src[edge_path[0]])
    coords = [points[src0]]

    for e in edge_path:
        t = int(cache.dst[int(e)])
        coords.append(points[t])

    return np.asarray(coords, dtype=float)

def _uv_from_speed_angle_ycw(speed: np.ndarray, angle_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    angle_deg: 0=+y, 90=+x (clockwise from +y)
    return: (ux, uy) for quiver (x,y plane)
    """
    th = np.deg2rad(angle_deg.astype(float))
    ux = speed.astype(float) * np.sin(th)
    uy = speed.astype(float) * np.cos(th)
    return ux, uy



# ---------------------------------------------------
# 1) 원형 장애물 정의
# ---------------------------------------------------
@dataclass
class CircleObstacle:
    x: float
    y: float
    r: float


# ---------------------------------------------------
# 2) 무작위 원형 장애물 생성
# ---------------------------------------------------
def make_random_circle_obstacles(
    *,
    x_s: float,
    x_f: float,
    y_s: float,
    y_f: float,
    num_obstacles: int = 5,
    radius_range: Tuple[float, float] = (120.0, 240.0),
    border_margin: float = 120.0,
    allow_overlap: bool = False,
    obstacle_clearance: float = 0.0,
    seed: Optional[int] = None,
    max_tries_per_obstacle: int = 3000,
) -> List[CircleObstacle]:
    """
    무작위 원형 장애물 생성

    Args:
      num_obstacles: 장애물 개수
      radius_range: (min_radius, max_radius)
      border_margin: 장애물 중심이 경계에서 최소 이만큼 떨어지도록 강제
      allow_overlap: 장애물끼리 겹침 허용 여부
      obstacle_clearance: 장애물끼리 추가 이격 거리
    """
    rng = np.random.default_rng(seed)

    rmin, rmax = radius_range
    if rmin <= 0 or rmax <= 0 or rmin > rmax:
        raise ValueError("radius_range must satisfy 0 < rmin <= rmax")

    obstacles: List[CircleObstacle] = []

    for _ in range(num_obstacles):
        placed = False

        for _try in range(max_tries_per_obstacle):
            rr = float(rng.uniform(rmin, rmax))

            xmin = x_s + border_margin + rr
            xmax = x_f - border_margin - rr
            ymin = y_s + border_margin + rr
            ymax = y_f - border_margin - rr

            if xmin >= xmax or ymin >= ymax:
                raise ValueError("border_margin / radius_range too large for the domain.")

            cx = float(rng.uniform(xmin, xmax))
            cy = float(rng.uniform(ymin, ymax))

            if not allow_overlap:
                ok = True
                for obs in obstacles:
                    d2 = (cx - obs.x) ** 2 + (cy - obs.y) ** 2
                    min_d = rr + obs.r + obstacle_clearance
                    if d2 < min_d ** 2:
                        ok = False
                        break
                if not ok:
                    continue

            obstacles.append(CircleObstacle(cx, cy, rr))
            placed = True
            break

        if not placed:
            raise RuntimeError(
                f"Failed to place obstacle after {max_tries_per_obstacle} tries. "
                f"Try fewer obstacles, smaller radius_range, or allow_overlap=True."
            )

    return obstacles


# ---------------------------------------------------
# 3) 각 vertex가 장애물 내부인지 판정
# ---------------------------------------------------
def build_blocked_node_mask_from_circles(
    points: np.ndarray,
    obstacles: List[CircleObstacle],
    *,
    inclusive: bool = True,
) -> np.ndarray:
    """
    Returns:
      blocked: (N,) bool
    """
    N = len(points)
    blocked = np.zeros(N, dtype=bool)

    if len(obstacles) == 0:
        return blocked

    px = points[:, 0]
    py = points[:, 1]

    for obs in obstacles:
        d2 = (px - obs.x) ** 2 + (py - obs.y) ** 2
        if inclusive:
            blocked |= (d2 <= obs.r ** 2)
        else:
            blocked |= (d2 < obs.r ** 2)

    return blocked


# ---------------------------------------------------
# 4) blocked node를 adjacency에서 제거
# ---------------------------------------------------
def prune_adjacency_by_blocked_nodes(
    adj: np.ndarray,
    blocked: np.ndarray,
    *,
    missing: int = -1,
) -> np.ndarray:
    """
    blocked node로 들어가거나, blocked node에서 나가는 연결 제거
    cache의 대칭성 가정이 깨지지 않도록 노드 자체를 그래프에서 제거하는 방식
    """
    adj2 = np.array(adj, copy=True)
    N, K = adj2.shape

    if blocked.shape[0] != N:
        raise ValueError("blocked mask length must match number of nodes")

    # 1) blocked node에서 나가는 edge 제거
    adj2[blocked, :] = missing

    # 2) blocked node로 들어가는 edge 제거
    for v in range(N):
        if blocked[v]:
            continue
        row = adj2[v]
        valid = (row != missing)
        nbrs = row[valid]
        if len(nbrs) == 0:
            continue
        bad = blocked[nbrs]
        row_idx = np.where(valid)[0][bad]
        adj2[v, row_idx] = missing

    return adj2


# ---------------------------------------------------
# 5) 장애물 적용 전체 래퍼
# ---------------------------------------------------
def apply_circle_obstacles_to_graph(
    points: np.ndarray,
    adj: np.ndarray,
    obstacles: List[CircleObstacle],
    *,
    missing: int = -1,
    inclusive: bool = True,
):
    """
    Returns:
      blocked_mask: (N,) bool
      adj_pruned: (N,K)
    """
    blocked_mask = build_blocked_node_mask_from_circles(
        points,
        obstacles,
        inclusive=inclusive,
    )
    adj_pruned = prune_adjacency_by_blocked_nodes(
        adj,
        blocked_mask,
        missing=missing,
    )
    return blocked_mask, adj_pruned

def compute_command_all_shortest_once(
    env,
):
    """
    현재 지휘함 상태에서 전체 노드까지 shortest path를 한 번만 계산
    """
    dist_cmd, prev_node_cmd, prev_k_cmd, _, _ = dijkstra_turn_state_core(
        cache=env.cache,
        base_cost_e=env.cost_e,
        theta_e_deg=env.theta_e_deg,
        start=env.command_vid,
        goal=env.command_vid,
        termination="all",
        use_start_heading=env.config.use_start_heading,
        start_heading_deg=env.command_heading_deg,
        th1=env.config.th1,
        lam1=env.config.lam1,
        th2=env.config.th2,
        lam2=env.config.lam2,
    )

    cmd_best_time_rel, cmd_best_k = reduce_state_dist_to_node_best(dist_cmd)
    cmd_best_time_rel[env.command_vid] = 0.0
    cmd_best_k[env.command_vid] = -1

    return {
        "dist_cmd": dist_cmd,
        "prev_node_cmd": prev_node_cmd,
        "prev_k_cmd": prev_k_cmd,
        "cmd_best_time_rel": cmd_best_time_rel,
        "cmd_best_k": cmd_best_k,
    }

### 해류 패트롤 시뮬레이션

@dataclass
class PatrolRegion:
    region_id: int
    x0: float
    x1: float
    y0: float
    y1: float


def make_8_regions(x_s: float, x_f: float, y_s: float, y_f: float) -> List[PatrolRegion]:
    """
    전체 영역을 4 x 2 = 8개로 분할
    각 영역 크기: 900 x 900 (현재 값 기준)
    """
    W = (x_f - x_s) / 4.0
    H = (y_f - y_s) / 2.0

    regions = []
    rid = 0
    for row in range(2):         # y direction
        for col in range(4):     # x direction
            rx0 = x_s + col * W
            rx1 = x_s + (col + 1) * W
            ry0 = y_s + row * H
            ry1 = y_s + (row + 1) * H
            regions.append(PatrolRegion(rid, rx0, rx1, ry0, ry1))
            rid += 1
    return regions

def make_16_regions(x_s: float, x_f: float, y_s: float, y_f: float) -> List[PatrolRegion]:
    """
    전체 영역을 4 x 2 = 8개로 분할
    각 영역 크기: 900 x 900 (현재 값 기준)
    """
    W = (x_f - x_s) / 4.0
    H = (y_f - y_s) / 4.0

    regions = []
    rid = 0
    for row in range(4):         # y direction
        for col in range(4):     # x direction
            rx0 = x_s + col * W
            rx1 = x_s + (col + 1) * W
            ry0 = y_s + row * H
            ry1 = y_s + (row + 1) * H
            regions.append(PatrolRegion(rid, rx0, rx1, ry0, ry1))
            rid += 1
    return regions

def make_12_type3_regions(x_s: float, x_f: float, y_s: float, y_f: float) -> List[PatrolRegion]:
    W = (x_f - x_s) / 5.0
    H = (y_f - y_s) / 4.0

    regions = []
    rid = 0
    for row in range(3):  # y direction
        for col in range(4):  # x direction
            rx0 = x_s + col * W
            rx1 = x_s + (col + 2) * W
            ry0 = y_s + row * H
            ry1 = y_s + (row + 2) * H
            regions.append(PatrolRegion(rid, rx0, rx1, ry0, ry1))
            rid += 1
    return regions

def make_12_regions(x_s: float, x_f: float, y_s: float, y_f: float) -> List[PatrolRegion]:
    """
    전체 영역을 4 x 2 = 8개로 분할
    각 영역 크기: 900 x 900 (현재 값 기준)
    """
    W = (x_f - x_s) / 4.0
    H = (y_f - y_s) / 3.0

    regions = []
    rid = 0
    for row in range(3):         # y direction
        for col in range(4):     # x direction
            rx0 = x_s + col * W
            rx1 = x_s + (col + 1) * W
            ry0 = y_s + row * H
            ry1 = y_s + (row + 1) * H
            regions.append(PatrolRegion(rid, rx0, rx1, ry0, ry1))
            rid += 1
    return regions
def make_24_regions(x_s: float, x_f: float, y_s: float, y_f: float) -> List[PatrolRegion]:
    """
    전체 영역을 4 x 2 = 8개로 분할
    각 영역 크기: 900 x 900 (현재 값 기준)
    """
    W = (x_f - x_s) / 4.0
    H = (y_f - y_s) / 3.0

    regions = []
    rid = 0
    for row in range(3):         # y direction
        for col in range(4):     # x direction
            rx0 = x_s + col * W
            rx1 = x_s + (col + 1) * W
            ry0 = y_s + row * H
            ry1 = y_s + (row + 1) * H
            regions.append(PatrolRegion(rid, rx0, rx1, ry0, ry1))
            rid += 1
            regions.append(PatrolRegion(rid, rx0, rx1, ry0, ry1))
            rid += 1
    return regions


def make_24_type2_regions(
    x_s: float,
    x_f: float,
    y_s: float,
    y_f: float
) -> List[PatrolRegion]:
    """
    총 12개 region 생성

    1. 가로로 4개 직사각형
       - 전체 영역을 y 방향으로 4등분
       - 각 region은 전체 x 범위를 가짐

    2. 세로로 4개 직사각형
       - 전체 영역을 x 방향으로 4등분
       - 각 region은 전체 y 범위를 가짐

    3. 전체를 2 x 2 = 4칸으로 나눈 region
       - 각 region은 전체 영역의 사분면
    """

    regions = []
    rid = 0

    # --------------------------------------------------
    # 1. 가로로 4개 직사각형
    # --------------------------------------------------
    H4 = (y_f - y_s) / 4.0

    for row in range(4):
        rx0 = x_s
        rx1 = x_f
        ry0 = y_s + row * H4
        ry1 = y_s + (row + 1) * H4

        regions.append(PatrolRegion(rid, rx0, rx1, ry0, ry1))
        rid += 1
        regions.append(PatrolRegion(rid, rx0, rx1, ry0, ry1))
        rid += 1

    # --------------------------------------------------
    # 2. 세로로 4개 직사각형
    # --------------------------------------------------
    W4 = (x_f - x_s) / 4.0

    for col in range(4):
        rx0 = x_s + col * W4
        rx1 = x_s + (col + 1) * W4
        ry0 = y_s
        ry1 = y_f

        regions.append(PatrolRegion(rid, rx0, rx1, ry0, ry1))
        rid += 1
        regions.append(PatrolRegion(rid, rx0, rx1, ry0, ry1))
        rid += 1

    # --------------------------------------------------
    # 3. 전체를 2 x 2 = 4칸으로 나눈 region
    # --------------------------------------------------
    W2 = (x_f - x_s) / 2.0
    H2 = (y_f - y_s) / 2.0

    for row in range(2):
        for col in range(2):
            rx0 = x_s + col * W2
            rx1 = x_s + (col + 1) * W2
            ry0 = y_s + row * H2
            ry1 = y_s + (row + 1) * H2

            regions.append(PatrolRegion(rid, rx0, rx1, ry0, ry1))
            rid += 1
            regions.append(PatrolRegion(rid, rx0, rx1, ry0, ry1))
            rid += 1

    return regions
def make_16_type2_regions(
    x_s: float,
    x_f: float,
    y_s: float,
    y_f: float
) -> List[PatrolRegion]:
    """
    총 12개 region 생성

    1. 가로로 4개 직사각형
       - 전체 영역을 y 방향으로 4등분
       - 각 region은 전체 x 범위를 가짐

    2. 세로로 4개 직사각형
       - 전체 영역을 x 방향으로 4등분
       - 각 region은 전체 y 범위를 가짐

    3. 전체를 2 x 2 = 4칸으로 나눈 region
       - 각 region은 전체 영역의 사분면
    """

    regions = []
    rid = 0

    # --------------------------------------------------
    # 1. 가로로 4개 직사각형
    # --------------------------------------------------
    H4 = (y_f - y_s) / 4.0

    for row in range(4):
        rx0 = x_s
        rx1 = x_f
        ry0 = y_s + row * H4
        ry1 = y_s + (row + 1) * H4

        regions.append(PatrolRegion(rid, rx0, rx1, ry0, ry1))
        rid += 1

    # --------------------------------------------------
    # 2. 세로로 4개 직사각형
    # --------------------------------------------------
    W4 = (x_f - x_s) / 4.0

    for col in range(4):
        rx0 = x_s + col * W4
        rx1 = x_s + (col + 1) * W4
        ry0 = y_s
        ry1 = y_f

        regions.append(PatrolRegion(rid, rx0, rx1, ry0, ry1))
        rid += 1

    # --------------------------------------------------
    # 3. 전체를 2 x 2 = 4칸으로 나눈 region
    # --------------------------------------------------
    W2 = (x_f - x_s) / 2.0
    H2 = (y_f - y_s) / 2.0

    for row in range(2):
        for col in range(2):
            rx0 = x_s + col * W2
            rx1 = x_s + (col + 1) * W2
            ry0 = y_s + row * H2
            ry1 = y_s + (row + 1) * H2

            regions.append(PatrolRegion(rid, rx0, rx1, ry0, ry1))
            rid +=1
            regions.append(PatrolRegion(rid, rx0, rx1, ry0, ry1))
            rid += 1


    return regions

def make_12_type2_regions(
    x_s: float,
    x_f: float,
    y_s: float,
    y_f: float
) -> List[PatrolRegion]:
    """
    총 12개 region 생성

    1. 가로로 4개 직사각형
       - 전체 영역을 y 방향으로 4등분
       - 각 region은 전체 x 범위를 가짐

    2. 세로로 4개 직사각형
       - 전체 영역을 x 방향으로 4등분
       - 각 region은 전체 y 범위를 가짐

    3. 전체를 2 x 2 = 4칸으로 나눈 region
       - 각 region은 전체 영역의 사분면
    """

    regions = []
    rid = 0

    # --------------------------------------------------
    # 1. 가로로 4개 직사각형
    # --------------------------------------------------
    H4 = (y_f - y_s) / 4.0

    for row in range(4):
        rx0 = x_s
        rx1 = x_f
        ry0 = y_s + row * H4
        ry1 = y_s + (row + 1) * H4

        regions.append(PatrolRegion(rid, rx0, rx1, ry0, ry1))
        rid += 1

    # --------------------------------------------------
    # 2. 세로로 4개 직사각형
    # --------------------------------------------------
    W4 = (x_f - x_s) / 4.0

    for col in range(4):
        rx0 = x_s + col * W4
        rx1 = x_s + (col + 1) * W4
        ry0 = y_s
        ry1 = y_f

        regions.append(PatrolRegion(rid, rx0, rx1, ry0, ry1))
        rid += 1

    # --------------------------------------------------
    # 3. 전체를 2 x 2 = 4칸으로 나눈 region
    # --------------------------------------------------
    W2 = (x_f - x_s) / 2.0
    H2 = (y_f - y_s) / 2.0

    for row in range(2):
        for col in range(2):
            rx0 = x_s + col * W2
            rx1 = x_s + (col + 1) * W2
            ry0 = y_s + row * H2
            ry1 = y_s + (row + 1) * H2

            regions.append(PatrolRegion(rid, rx0, rx1, ry0, ry1))
            rid += 1

    return regions


def make_new16_region(
    x_s: float,
    x_f: float,
    y_s: float,
    y_f: float
) -> List[PatrolRegion]:
    """
    총 12개 region 생성

    1. 가로로 4개 직사각형
       - 전체 영역을 y 방향으로 4등분
       - 각 region은 전체 x 범위를 가짐

    2. 세로로 4개 직사각형
       - 전체 영역을 x 방향으로 4등분
       - 각 region은 전체 y 범위를 가짐

    3. 전체를 2 x 2 = 4칸으로 나눈 region
       - 각 region은 전체 영역의 사분면
    """

    regions = []
    rid = 0

    # --------------------------------------------------
    # 1. 가로로 4개 직사각형
    # --------------------------------------------------
    H4 = (y_f - y_s) / 4.0

    for row in range(4):
        rx0 = x_s
        rx1 = x_f
        ry0 = y_s + row * H4
        ry1 = y_s + (row + 1) * H4

        regions.append(PatrolRegion(rid, rx0, rx1, ry0, ry1))
        rid += 1

    # --------------------------------------------------
    # 2. 세로로 4개 직사각형
    # --------------------------------------------------
    W4 = (x_f - x_s) / 4.0

    for col in range(4):
        rx0 = x_s + col * W4
        rx1 = x_s + (col + 1) * W4
        ry0 = y_s
        ry1 = y_f

        regions.append(PatrolRegion(rid, rx0, rx1, ry0, ry1))
        rid += 1

    # --------------------------------------------------
    # 3. 전체를 2 x 2 = 4칸으로 나눈 region
    # --------------------------------------------------
    W4 = (x_f - x_s) / 4.0
    H4 = (y_f - y_s) / 4.0

    for row in range(3):
        for col in range(3):
            rx0 = x_s + col * W4
            rx1 = x_s + (col + 2) * W4
            ry0 = y_s + row * H4
            ry1 = y_s + (row + 2) * H4
            if row!=1 or col!=1:
                regions.append(PatrolRegion(rid, rx0, rx1, ry0, ry1))
                rid += 1

    return regions


def angle_from_vec_y_clockwise_deg(vec: np.ndarray) -> float:
    dx, dy = float(vec[0]), float(vec[1])
    ang = np.degrees(np.arctan2(dx, dy))
    return float((ang + 360.0) % 360.0)
def compute_heading_goal_diff_deg(
    points: np.ndarray,
    start_idx: int,
    goal_idx: int,
    current_heading_deg: float,
) -> float:
    p0 = points[start_idx]
    p1 = points[goal_idx]
    dx = float(p1[0] - p0[0])
    dy = float(p1[1] - p0[1])

    goal_heading_deg = angle_from_y_clockwise_deg(dx, dy)
    diff = wrap180_deg(goal_heading_deg - float(current_heading_deg))
    return float(abs(diff))

def extract_fixed_features(
    *,
    points: np.ndarray,
    blocked_mask: np.ndarray,
    total_ux: np.ndarray,
    total_uy: np.ndarray,
    start_idx: int,
    goal_idx: int,
    inner_width: float = 90.0,
    outer_width: float = 180.0,
    heading_diff_deg: float = 0.0,
) -> Dict[str, float]:
    """
    feature:
      1) inner perpendicular current mean
      2) inner perpendicular current var
      3) inner parallel current mean
      4) inner parallel current var
      5) heading-goal difference
      6) distance
      7) blocked ratio (inner corridor 기준)
      8) parallel outer-inner mean gap
      9) perp outer-inner mean gap
    """

    p0 = points[start_idx]
    p1 = points[goal_idx]
    goal_vec = p1 - p0
    dist = float(np.linalg.norm(goal_vec))

    if dist < 1e-12:
        return {
            "perp_current_mean": 0.0,
            "perp_current_var": 0.0,
            "parallel_current_mean": 0.0,
            "parallel_current_var": 0.0,
            "heading_goal_diff": 0.0,
            "distance": 0.0,
            "blocked_ratio": 0.0,
            "parallel_outer_inner_gap_mean_abs": 0.0,
            "perp_outer_inner_gap_mean_abs": 0.0,
        }

    dir_u = goal_vec / dist

    # 각 점의 직선 기준 signed perpendicular distance 계산
    ap = points - p0[None, :]
    t_along = ap @ dir_u  # 직선 방향 투영 길이

    # signed perpendicular:
    # left/right 구분용
    # 2D cross product scalar = ap_x * dir_y - ap_y * dir_x
    signed_perp = ap[:, 0] * dir_u[1] - ap[:, 1] * dir_u[0]
    abs_perp = np.abs(signed_perp)

    # 선분 내부만 사용
    seg_mask = (t_along >= 0.0) & (t_along <= dist)

    inner_mask = seg_mask & (abs_perp <= inner_width)
    left_outer_mask = seg_mask & (signed_perp > inner_width) & (signed_perp <= outer_width)
    right_outer_mask = seg_mask & (signed_perp < -inner_width) & (signed_perp >= -outer_width)

    # 해류를 goal 방향 기준으로 분해
    parallel = total_ux * dir_u[0] + total_uy * dir_u[1]
    perpendicular = -total_ux * dir_u[1] + total_uy * dir_u[0]

    # blocked ratio는 inner 기준
    inner_total = int(np.sum(inner_mask))
    inner_blocked = int(np.sum(blocked_mask[inner_mask]))
    blocked_ratio = float(inner_blocked / max(inner_total, 1))

    # 통계는 free node 기준으로
    inner_stat_mask = inner_mask & (~blocked_mask)
    left_outer_stat_mask = left_outer_mask & (~blocked_mask)
    right_outer_stat_mask = right_outer_mask & (~blocked_mask)

    def safe_mean_var(arr: np.ndarray, mask: np.ndarray):
        vals = arr[mask]
        if vals.size == 0:
            return 0.0, 0.0
        return float(np.mean(vals)), float(np.var(vals))

    # inner stats
    par_in_mean, par_in_var = safe_mean_var(parallel, inner_stat_mask)
    perp_in_mean, perp_in_var = safe_mean_var(perpendicular, inner_stat_mask)

    # outer stats
    par_left_mean, _ = safe_mean_var(parallel, left_outer_stat_mask)
    par_right_mean, _ = safe_mean_var(parallel, right_outer_stat_mask)

    perp_left_mean, _ = safe_mean_var(perpendicular, left_outer_stat_mask)
    perp_right_mean, _ = safe_mean_var(perpendicular, right_outer_stat_mask)

    # 너가 원하는 gap feature
    parallel_outer_inner_gap_mean_abs = 0.5 * (
        abs(par_left_mean - par_in_mean) + abs(par_right_mean - par_in_mean)
    )

    perp_outer_inner_gap_mean_abs = 0.5 * (
        abs(perp_left_mean - perp_in_mean) + abs(perp_right_mean - perp_in_mean)
    )

    heading_goal_diff_deg = heading_diff_deg

    return {
        "perp_current_mean": perp_in_mean,
        "perp_current_var": perp_in_var,
        "parallel_current_mean": par_in_mean,
        "parallel_current_var": par_in_var,
        "heading_goal_diff": heading_goal_diff_deg,
        "distance": dist,
        "blocked_ratio": blocked_ratio,
        "parallel_outer_inner_gap_mean_abs": float(parallel_outer_inner_gap_mean_abs),
        "perp_outer_inner_gap_mean_abs": float(perp_outer_inner_gap_mean_abs),
    }
def load_rf_payload(path: str):
    payload = joblib.load(path)
    model = payload["model"]
    feature_names = payload["feature_names"]
    return model, feature_names

def nearest_vertex_index(points: np.ndarray, xy: np.ndarray) -> int:
    """
    단순 최근접 vertex index
    """
    diff = points - xy[None, :]
    d2 = np.sum(diff * diff, axis=1)
    return int(np.argmin(d2))



def nearest_free_vertex_index_in_region(
    points: np.ndarray,
    xy: np.ndarray,
    region: PatrolRegion,
    blocked_mask: np.ndarray,
) -> int:
    """
    region 내부이면서 blocked가 아닌 free vertex 중에서
    xy와 가장 가까운 vertex index 반환
    """
    mask = (
        (points[:, 0] >= region.x0) & (points[:, 0] <= region.x1) &
        (points[:, 1] >= region.y0) & (points[:, 1] <= region.y1) &
        (~blocked_mask)
    )

    idxs = np.where(mask)[0]
    if len(idxs) == 0:
        raise RuntimeError(f"No free vertex exists in region {region.region_id}")

    sub = points[idxs]
    diff = sub - xy[None, :]
    d2 = np.sum(diff * diff, axis=1)
    return int(idxs[np.argmin(d2)])

# =========================================================
# 1. free vertex 관련 유틸
# =========================================================
def nearest_free_vertex_index(points: np.ndarray, xy: np.ndarray, blocked_mask: np.ndarray) -> int:
    idxs = np.where(~blocked_mask)[0]
    if len(idxs) == 0:
        raise RuntimeError("No free vertex exists.")
    sub = points[idxs]
    d2 = np.sum((sub - xy[None, :]) ** 2, axis=1)
    return int(idxs[np.argmin(d2)])


# =========================================================
# 2. waypoint 생성 관련
# =========================================================
def make_region_waypoints(
    region: PatrolRegion,
    margin: float = 180.0,
    num_waypoints: int = 4,
) -> np.ndarray:
    """
    region 내부 사각형의 둘레를 따라 num_waypoints개를 균등 배치
    """
    if num_waypoints < 1:
        raise ValueError("num_waypoints must be >= 1")

    x0, x1, y0, y1 = region.x0, region.x1, region.y0, region.y1

    if (x1 - x0) <= 2 * margin or (y1 - y0) <= 2 * margin:
        raise ValueError("margin too large for region size.")

    left   = x0 + margin
    right  = x1 - margin
    bottom = y0 + margin
    top    = y1 - margin

    w = right - left
    h = top - bottom
    perimeter = 2.0 * (w + h)

    s_vals = np.linspace(0.0, perimeter, num_waypoints, endpoint=False)

    pts = []
    for s in s_vals:
        if s < w:
            x = left + s
            y = bottom
        elif s < w + h:
            x = right
            y = bottom + (s - w)
        elif s < 2 * w + h:
            x = right - (s - (w + h))
            y = top
        else:
            x = left
            y = top - (s - (2 * w + h))
        pts.append([x, y])

    return np.asarray(pts, dtype=float)
def make_region_anchor_points(
    region: PatrolRegion,
    margin: float = 180.0,
    num_anchor_points: int = 8,
    square_ratio_threshold: float = 1.35,
) -> np.ndarray:
    """
    region 내부 사각형 둘레를 따라 anchor point 생성.

    중요:
      - 정사각형/직사각형 판단은 margin 적용 전 원래 region 크기로 한다.
      - 실제 anchor 좌표 생성에는 margin을 적용한다.

    num_anchor_points=4:
      BL -> BR -> TR -> TL

    num_anchor_points=8:
      - 정사각형에 가까우면 8방향 anchor
      - 가로로 긴 직사각형이면 아래 4개 + 위 4개
      - 세로로 긴 직사각형이면 왼쪽 4개 + 오른쪽 4개
    """

    x0, x1, y0, y1 = region.x0, region.x1, region.y0, region.y1

    raw_w = float(x1 - x0)
    raw_h = float(y1 - y0)

    if raw_w <= 0 or raw_h <= 0:
        raise ValueError("Invalid region size.")

    if raw_w <= 2 * margin or raw_h <= 2 * margin:
        raise ValueError("margin too large for region size.")

    # 판단은 margin 적용 전 원래 region 크기로
    raw_aspect = max(raw_w, raw_h) / min(raw_w, raw_h)

    # 실제 anchor 좌표는 margin 적용 후 내부 사각형 기준
    left   = x0 + margin
    right  = x1 - margin
    bottom = y0 + margin
    top    = y1 - margin

    inner_w = right - left
    inner_h = top - bottom

    if inner_w <= 0 or inner_h <= 0:
        raise ValueError("Invalid inner region size.")

    if num_anchor_points == 4:
        pts = [
            [left,  bottom],
            [right, bottom],
            [right, top],
            [left,  top],
        ]
        return np.asarray(pts, dtype=float)

    if num_anchor_points != 8:
        raise ValueError("num_anchor_points must be 4 or 8.")

    # --------------------------------------------------
    # 1) 원래 region이 정사각형에 가까운 경우
    # --------------------------------------------------
    if raw_aspect <= square_ratio_threshold:
        mid_x = 0.5 * (left + right)
        mid_y = 0.5 * (bottom + top)

        pts = [
            [left,  bottom],
            [mid_x, bottom],
            [right, bottom],
            [right, mid_y],
            [right, top],
            [mid_x, top],
            [left,  top],
            [left,  mid_y],
        ]

    # --------------------------------------------------
    # 2) 원래 region이 가로로 긴 경우: 아래 4개 + 위 4개
    # --------------------------------------------------
    elif raw_w > raw_h:
        xs = np.linspace(left, right, 4)

        bottom_row = [[float(x), bottom] for x in xs]
        top_row    = [[float(x), top] for x in xs[::-1]]

        pts = bottom_row + top_row

    # --------------------------------------------------
    # 3) 원래 region이 세로로 긴 경우: 왼쪽 4개 + 오른쪽 4개
    # --------------------------------------------------
    else:
        ys = np.linspace(bottom, top, 4)

        left_col  = [[left, float(y)] for y in ys]
        right_col = [[right, float(y)] for y in ys[::-1]]

        pts = left_col + right_col

    return np.asarray(pts, dtype=float)
def make_region_corner_points(
    region: PatrolRegion,
    margin: float = 180.0,
) -> np.ndarray:
    """
    region 내부 사각형의 4개 꼭지점 근처 anchor point 생성.
    순서는 기본적으로 clockwise:
      bottom-left -> bottom-right -> top-right -> top-left
    """
    x0, x1, y0, y1 = region.x0, region.x1, region.y0, region.y1

    if (x1 - x0) <= 2 * margin or (y1 - y0) <= 2 * margin:
        raise ValueError("margin too large for region size.")

    left   = x0 + margin
    right  = x1 - margin
    bottom = y0 + margin
    top    = y1 - margin

    return np.asarray([
        [left,  bottom],
        [right, bottom],
        [right, top],
        [left,  top],
    ], dtype=float)


def _concat_leg_paths(legs: List[Dict]) -> Tuple[List[int], List[int]]:
    """
    여러 leg의 node_path / edge_path를 하나의 loop path로 연결.
    중복되는 연결 node는 제거.
    """
    full_node_path = []
    full_edge_path = []

    for i, leg in enumerate(legs):
        npth = leg.get("node_path", [])
        epth = leg.get("edge_path", [])

        if len(npth) == 0:
            continue

        if i == 0:
            full_node_path.extend(npth)
        else:
            full_node_path.extend(npth[1:])

        full_edge_path.extend(epth)

    return full_node_path, full_edge_path
def _sample_indices_along_path_by_cost(
    node_path: List[int],
    edge_path: List[int],
    cost_e: np.ndarray,
    num_waypoints: int,
) -> Tuple[List[int], List[int]]:
    """
    full loop path 위에서 누적 cost 기준으로 waypoint를 샘플링한다.

    return:
      sampled_vids
      sampled_node_indices

    sampled_node_indices는 node_path 안에서의 위치 index.
    이후 이 index를 이용해서 원래 full path를 그대로 잘라 leg를 만든다.
    """
    if num_waypoints < 1:
        raise ValueError("num_waypoints must be >= 1")

    if len(node_path) == 0:
        raise ValueError("node_path is empty.")

    if len(edge_path) == 0:
        return [int(node_path[0])] * num_waypoints, [0] * num_waypoints

    edge_costs = np.asarray([float(cost_e[eid]) for eid in edge_path], dtype=float)

    if not np.all(np.isfinite(edge_costs)):
        raise ValueError("edge_costs contain non-finite values.")

    total_cost = float(np.sum(edge_costs))

    if total_cost <= 0:
        return [int(node_path[0])] * num_waypoints, [0] * num_waypoints

    cum_cost = np.concatenate([[0.0], np.cumsum(edge_costs)])
    targets = np.linspace(0.0, total_cost, num_waypoints, endpoint=False)

    sampled_node_indices = []

    for t in targets:
        j = int(np.searchsorted(cum_cost, t, side="right") - 1)
        j = max(0, min(j, len(node_path) - 2))
        sampled_node_indices.append(j)

    # 중복 index 보정
    fixed_indices = []
    used = set()

    for idx in sampled_node_indices:
        if idx not in used:
            fixed_indices.append(idx)
            used.add(idx)
        else:
            replacement = None

            # 가까운 뒤쪽 index부터 찾기
            for cand in range(idx + 1, len(node_path) - 1):
                if cand not in used:
                    replacement = cand
                    break

            # 뒤쪽에 없으면 앞쪽에서 찾기
            if replacement is None:
                for cand in range(0, idx):
                    if cand not in used:
                        replacement = cand
                        break

            if replacement is None:
                replacement = idx

            fixed_indices.append(replacement)
            used.add(replacement)

    # loop 진행 순서를 유지하도록 정렬
    fixed_indices = sorted(fixed_indices)

    sampled_vids = [int(node_path[i]) for i in fixed_indices]

    return sampled_vids, fixed_indices
def _build_legs_from_sampled_indices_on_loop(
    *,
    full_node_path: List[int],
    full_edge_path: List[int],
    sampled_node_indices: List[int],
    cost_e: np.ndarray,
    theta_e_deg: np.ndarray,
    close_loop: bool = True,
    start_heading_deg: Optional[float] = None,
) -> List[Dict]:
    """
    full_node_path/full_edge_path를 기준으로 sampled waypoint 사이 구간을 그대로 잘라 leg를 만든다.
    Dijkstra를 다시 돌리지 않기 때문에 내부 shortcut이 생기지 않는다.
    """
    if len(full_node_path) == 0:
        raise ValueError("full_node_path is empty.")

    if len(full_edge_path) != len(full_node_path) - 1:
        raise ValueError("full_edge_path length must be len(full_node_path)-1.")

    sampled_node_indices = [int(i) for i in sampled_node_indices]

    M = len(sampled_node_indices)
    if M < 2:
        raise ValueError("Need at least 2 sampled waypoints.")

    legs = []
    current_heading_deg = start_heading_deg

    num_pairs = M if close_loop else M - 1

    for leg_idx in range(num_pairs):
        i0 = sampled_node_indices[leg_idx]
        i1 = sampled_node_indices[(leg_idx + 1) % M]

        # 일반 구간
        if i0 < i1:
            node_path = full_node_path[i0:i1 + 1]
            edge_path = full_edge_path[i0:i1]

        # 마지막 waypoint -> 첫 waypoint 구간
        else:
            node_path = full_node_path[i0:] + full_node_path[1:i1 + 1]
            edge_path = full_edge_path[i0:] + full_edge_path[:i1]

        if len(node_path) == 0:
            continue

        if len(edge_path) > 0:
            best_goal = float(np.sum([cost_e[eid] for eid in edge_path]))
            end_heading_deg = float(theta_e_deg[edge_path[-1]])
        else:
            best_goal = 0.0
            end_heading_deg = current_heading_deg

        legs.append({
            "leg_idx": leg_idx,
            "from_wp_idx": leg_idx,
            "to_wp_idx": (leg_idx + 1) % M,
            "start_vid": int(node_path[0]),
            "goal_vid": int(node_path[-1]),
            "start_heading_deg": current_heading_deg,
            "use_start_heading": current_heading_deg is not None,
            "end_heading_deg": end_heading_deg,
            "best_goal": best_goal,
            "best_goal_k": -1,
            "dist": None,
            "prev_node": None,
            "prev_k": None,
            "node_path": node_path,
            "edge_path": edge_path,
            "k_path": [],
        })

        current_heading_deg = end_heading_deg

    return legs
def _sample_vertices_along_path_by_cost(
    node_path: List[int],
    edge_path: List[int],
    cost_e: np.ndarray,
    num_waypoints: int,
) -> List[int]:
    """
    실제 loop path 위에서 누적 edge cost 기준으로 num_waypoints개 vertex를 균등 샘플링.

    반환:
      sampled_vids: path 위 vertex id들
    """
    if num_waypoints < 1:
        raise ValueError("num_waypoints must be >= 1")

    if len(node_path) == 0:
        raise ValueError("node_path is empty.")

    if len(edge_path) == 0:
        # 경로가 너무 짧은 경우 fallback
        return [int(node_path[0])] * num_waypoints

    edge_costs = np.asarray([float(cost_e[eid]) for eid in edge_path], dtype=float)

    if not np.all(np.isfinite(edge_costs)):
        raise ValueError("edge_costs contain non-finite values.")

    total_cost = float(np.sum(edge_costs))

    if total_cost <= 0:
        return [int(node_path[0])] * num_waypoints

    cum_cost = np.concatenate([[0.0], np.cumsum(edge_costs)])
    targets = np.linspace(0.0, total_cost, num_waypoints, endpoint=False)

    sampled_vids = []

    for t in targets:
        # cum_cost[j] <= t < cum_cost[j+1]인 edge를 찾음
        j = int(np.searchsorted(cum_cost, t, side="right") - 1)
        j = max(0, min(j, len(edge_path) - 1))

        # t가 해당 edge 구간에 있으면, 그 edge의 도착 node를 waypoint로 사용
        # node_path[j] --edge_path[j]--> node_path[j+1]
        vid = int(node_path[j])
        sampled_vids.append(vid)

    # 중복이 너무 많으면 인접한 다른 node로 약간 보정
    # 특히 path가 짧거나 num_waypoints가 큰 경우 중복 발생 가능
    fixed = []
    used = set()

    for vid in sampled_vids:
        if vid not in used:
            fixed.append(vid)
            used.add(vid)
        else:
            # path 안에서 아직 안 쓴 가까운 vertex를 찾음
            replacement = None
            for cand in node_path:
                cand = int(cand)
                if cand not in used:
                    replacement = cand
                    break

            if replacement is None:
                replacement = vid

            fixed.append(replacement)
            used.add(replacement)

    return fixed


def waypoint_vertices_for_region_corner_path(
    points: np.ndarray,
    cache: EdgeCache,
    cost_e: np.ndarray,
    theta_e_deg: np.ndarray,
    region: PatrolRegion,
    blocked_mask: np.ndarray,
    *,
    margin: float = 180.0,
    num_waypoints: int = 12,
    close_loop: bool = True,
    th1: float = 40.0,
    lam1: float = 0.0,
    th2: float = 50.0,
    lam2: float = np.inf,
    clockwise: bool = True,
    start_heading_deg: Optional[float] = None,
    num_anchor_points: int = 8,
) -> Tuple[np.ndarray, List[int], Dict]:
    """
    1) region의 4개 꼭지점 근처 anchor waypoint 생성
    2) anchor들을 실제 shortest path로 연결해서 loop 생성
    3) 그 loop path 위에서 누적 cost 기준으로 num_waypoints개를 균등 샘플링

    return:
      wp_xy
      wp_vids
      debug_info
    """

    if num_waypoints < 4:
        raise ValueError("corner_path 방식에서는 num_waypoints >= 4 권장.")

    # 1. 꼭지점 anchor 생성 후 free vertex로 snap
    corner_xy_target = make_region_anchor_points(
        region=region,
        margin=margin,
        num_anchor_points=num_anchor_points,
    )

    corner_vids = [
        nearest_free_vertex_index_in_region(points, xy, region, blocked_mask)
        for xy in corner_xy_target
    ]
    corner_xy = points[np.asarray(corner_vids, dtype=int)].copy()
    print(corner_xy)
    # 2. 방향 결정
    num_anchors = len(corner_vids)

    order = list(range(num_anchors))
    if not clockwise:
        order = order[::-1]

    if close_loop:
        pairs = [(order[i], order[(i + 1) % num_anchors]) for i in range(num_anchors)]
    else:
        pairs = [(order[i], order[i + 1]) for i in range(num_anchors - 1)]

    # 3. 꼭지점 anchor 간 shortest path loop 생성
    legs = []
    current_heading_deg = start_heading_deg

    for leg_idx, (a, b) in enumerate(pairs):
        s = int(corner_vids[a])
        g = int(corner_vids[b])

        use_heading = current_heading_deg is not None

        out = shortest_path_between_vertices(
            cache=cache,
            cost_e=cost_e,
            theta_e_deg=theta_e_deg,
            start_vid=s,
            goal_vid=g,
            th1=th1,
            lam1=lam1,
            th2=th2,
            lam2=lam2,
            use_start_heading=False,
            start_heading_deg=current_heading_deg,
            termination="goal_best",
        )

        edge_path = out.get("edge_path", [])

        if len(edge_path) > 0:
            last_eid = edge_path[-1]
            next_heading_deg = float(theta_e_deg[last_eid])

        else:
            next_heading_deg = current_heading_deg

        legs.append({
            "leg_idx": leg_idx,
            "from_corner_idx": a,
            "to_corner_idx": b,
            "start_vid": s,
            "goal_vid": g,
            "start_heading_deg": current_heading_deg,
            "use_start_heading": use_heading,
            "end_heading_deg": next_heading_deg,
            **out,
        })

        current_heading_deg = next_heading_deg

    full_node_path, full_edge_path = _concat_leg_paths(legs)

    if len(full_node_path) == 0:
        raise RuntimeError("corner anchor loop path generation failed.")

    # 4. 실제 loop path 위에서 균등 간격 waypoint 샘플링
    wp_vids, sampled_node_indices = _sample_indices_along_path_by_cost(
        node_path=full_node_path,
        edge_path=full_edge_path,
        cost_e=cost_e,
        num_waypoints=num_waypoints,
    )

    wp_xy = points[np.asarray(wp_vids, dtype=int)].copy()
    print(wp_xy)
    debug_info = {
        "corner_xy_target": corner_xy_target,
        "corner_vids": corner_vids,
        "corner_order": order,
        "corner_legs": legs,
        "corner_full_node_path": full_node_path,
        "corner_full_edge_path": full_edge_path,
        "sampled_node_indices": sampled_node_indices,
        "clockwise": clockwise,
        "final_heading_deg": current_heading_deg,
    }

    return wp_xy, wp_vids, debug_info

def waypoint_vertices_for_region(
    points: np.ndarray,
    region: PatrolRegion,
    blocked_mask: np.ndarray,
    margin: float = 180.0,
    num_waypoints: int = 4,
) -> Tuple[np.ndarray, List[int]]:
    """
    region 안의 num_waypoints개 이상적 waypoint를 만든 뒤,
    free vertex로 snap
    """
    wp_xy_target = make_region_waypoints(
        region=region,
        margin=margin,
        num_waypoints=num_waypoints,
    )

    wp_vids = [
        nearest_free_vertex_index_in_region(points, wp, region, blocked_mask)
        for wp in wp_xy_target
    ]

    # 실제 선택된 free vertex 좌표 사용
    wp_xy = points[np.array(wp_vids)].copy()
    return wp_xy, wp_vids

def flatten_waypoints_from_patrols(patrols: List[Dict]) -> List[Dict]:
    out = []
    gid = 0
    for usv_idx, patrol in enumerate(patrols):
        wp_vids = patrol["waypoint_vids"]
        wp_xy = patrol["waypoint_xy"]
        for local_wp_idx, (vid, xy) in enumerate(zip(wp_vids, wp_xy)):
            out.append({
                "global_wp_idx": gid,
                "usv_idx": usv_idx,
                "local_wp_idx": local_wp_idx,
                "vertex_idx": int(vid),
                "xy": np.asarray(xy, dtype=float),
            })
            gid += 1
    return out


# =========================================================
# 3. 두 정점 사이 shortest path
# =========================================================
def shortest_path_between_vertices(
    cache: EdgeCache,
    cost_e: np.ndarray,
    theta_e_deg: np.ndarray,
    start_vid: int,
    goal_vid: int,
    *,
    th1: float = 20.0,
    lam1: float = 1.0,
    th2: float = 30.0,
    lam2: float = 3.0,
    use_start_heading: bool = False,
    start_heading_deg: float = 0.0,
    termination: str = "goal_best",
):
    dist, prev_node, prev_k, best_goal, best_goal_k = dijkstra_turn_state_core(
        cache=cache,
        base_cost_e=cost_e,
        theta_e_deg=theta_e_deg,
        start=start_vid,
        goal=goal_vid,
        th1=th1, lam1=lam1,
        th2=th2, lam2=lam2,
        use_start_heading=use_start_heading,
        start_heading_deg=start_heading_deg,
        termination=termination,
    )

    node_path, edge_path, k_path = reconstruct_path_from_prev(
        cache=cache,
        prev_node=prev_node,
        prev_k=prev_k,
        start=start_vid,
        goal=goal_vid,
        best_goal_k=best_goal_k,
    )

    return {
        "dist": dist,
        "prev_node": prev_node,
        "prev_k": prev_k,
        "best_goal": best_goal,
        "best_goal_k": best_goal_k,
        "node_path": node_path,
        "edge_path": edge_path,
        "k_path": k_path,
    }


# =========================================================
# 4. region별 patrol loop 생성
# =========================================================
def build_patrol_loop_for_region(
    points: np.ndarray,
    cache: EdgeCache,
    cost_e: np.ndarray,
    theta_e_deg: np.ndarray,
    region: PatrolRegion,
    blocked_mask: np.ndarray,
    *,
    margin: float = 250.0,
    num_waypoints: int = 12,
    close_loop: bool = True,
    th1: float = 40.0,
    lam1: float = 0.0,
    th2: float = 50.0,
    lam2: float = np.inf,
    clockwise: Optional[bool] = None,
    rng: Optional[np.random.Generator] = None,
    start_heading_deg: Optional[float] = None,
) -> Dict:

    if clockwise is None:
        if rng is None:
            rng = np.random.default_rng()
        clockwise = bool(rng.integers(0, 2))

    # =====================================================
    # 처음 버전: region perimeter에서 waypoint 직접 생성
    # =====================================================
    wp_xy, wp_vids = waypoint_vertices_for_region(
        points=points,
        region=region,
        blocked_mask=blocked_mask,
        margin=margin,
        num_waypoints=num_waypoints,
    )

    M = len(wp_vids)
    if M < 2:
        raise ValueError("Need at least 2 waypoint vertices.")

    order = list(range(M))
    if not clockwise:
        order = order[::-1]

    if close_loop:
        pairs = [(order[i], order[(i + 1) % M]) for i in range(M)]
    else:
        pairs = [(order[i], order[i + 1]) for i in range(M - 1)]

    legs = []
    total_cost = 0.0
    full_node_path = []
    full_edge_path = []

    current_heading_deg = start_heading_deg

    for leg_idx, (a, b) in enumerate(pairs):
        s = int(wp_vids[a])
        g = int(wp_vids[b])

        use_heading = current_heading_deg is not None

        out = shortest_path_between_vertices(
            cache=cache,
            cost_e=cost_e,
            theta_e_deg=theta_e_deg,
            start_vid=s,
            goal_vid=g,
            th1=th1,
            lam1=lam1,
            th2=th2,
            lam2=lam2,
            use_start_heading=False,
            start_heading_deg=current_heading_deg,
            termination="goal_best",
        )

        edge_path = out.get("edge_path", [])
        node_path = out.get("node_path", [])

        if len(edge_path) > 0:
            current_heading_deg = float(theta_e_deg[int(edge_path[-1])])

        legs.append({
            "leg_idx": leg_idx,
            "from_wp_idx": a,
            "to_wp_idx": b,
            "start_vid": s,
            "goal_vid": g,
            "start_heading_deg": current_heading_deg,
            "use_start_heading": use_heading,
            **out,
        })

        total_cost += float(out.get("best_goal", 0.0))

        if len(node_path) == 0:
            continue

        if len(full_node_path) == 0:
            full_node_path.extend(node_path)
        else:
            full_node_path.extend(node_path[1:])

        full_edge_path.extend(edge_path)

    return {
        "region": region,
        "waypoint_xy": wp_xy,
        "waypoint_vids": wp_vids,
        "order": order,
        "legs": legs,
        "total_cost": total_cost,
        "full_node_path": full_node_path,
        "full_edge_path": full_edge_path,
        "clockwise": clockwise,
        "start_heading_deg": start_heading_deg,
        "final_heading_deg": current_heading_deg,
    }
def _heading_features_from_two_points(
    command_xy,
    goal_xy,
    command_heading_deg: float,
):
    """
    기존 코드와의 호환용.
    return:
      sin(command_heading), cos(command_heading),
      sin(goal_bearing),    cos(goal_bearing)
    """
    goal_bearing_deg = _bearing_deg_from_two_points(
        command_xy=command_xy,
        goal_xy=goal_xy,
        fallback_heading_deg=command_heading_deg,
    )

    sin_ch, cos_ch = _heading_feature_pair_from_angle_deg(command_heading_deg)
    sin_gb, cos_gb = _heading_feature_pair_from_angle_deg(goal_bearing_deg)

    return sin_ch, cos_ch, sin_gb, cos_gb
def build_all_patrols(
    points: np.ndarray,
    cache: EdgeCache,
    cost_e: np.ndarray,
    theta_e_deg: np.ndarray,
    blocked_mask: np.ndarray,
    *,
    x_s: float,
    x_f: float,
    y_s: float,
    y_f: float,
    margin: float = 180.0,
    num_waypoints: int = 4,
    close_loop: bool = True,
    th1: float = 20.0,
    lam1: float = 1.0,
    th2: float = 30.0,
    lam2: float = 3.0,
    rng: Optional[np.random.Generator] = None,
    region_mode: Optional[str] = None,
) -> List[Dict]:

    if rng is None:
        rng = np.random.default_rng()

    if region_mode is None:
        rg_mode=rng.random()
        if rg_mode<0.5:
            region_mode="type1"
        else:
            region_mode="type2"
    if region_mode =="16":
        rg_mode = rng.random()
        if rg_mode < 0.5:
            region_mode = "16_type1"
        else:
            region_mode = "16_type2"
    if region_mode == "type1":
        regions = make_12_regions(x_s, x_f, y_s, y_f)
        #print('1')
    elif region_mode == "16_type1":
        regions=make_16_regions(x_s, x_f, y_s, y_f)
    elif region_mode == "16_type2":
        regions=make_16_type2_regions(x_s, x_f, y_s, y_f)
    elif region_mode == "24_type1":
        regions=make_24_regions(x_s, x_f, y_s, y_f)
    elif region_mode == "24_type2":
        regions=make_24_type2_regions(x_s, x_f, y_s, y_f)
    else:
        regions = make_12_type2_regions(x_s, x_f, y_s, y_f)
        #print('2')

    patrols = []
    for region in regions:
        patrol = build_patrol_loop_for_region(
            points=points,
            cache=cache,
            cost_e=cost_e,
            theta_e_deg=theta_e_deg,
            region=region,
            blocked_mask=blocked_mask,
            margin=margin,
            num_waypoints=num_waypoints,
            close_loop=close_loop,
            th1=th1, lam1=lam1,
            th2=th2, lam2=lam2,
            rng=rng,
        )
        patrols.append(patrol)

    return patrols,region_mode


# =========================================================
# 5. 시작 waypoint를 기준으로 patrol 회전
# =========================================================
def rotate_patrol_legs_by_start_wp(patrol: Dict, start_wp_idx: int) -> List[Dict]:
    legs = patrol["legs"]
    M = len(legs)
    if M == 0:
        raise ValueError("patrol['legs'] is empty.")
    s = int(start_wp_idx) % M
    return legs[s:] + legs[:s]


def build_rotated_patrol_path(
    patrol: Dict,
    start_wp_idx: int,
) -> Tuple[List[int], List[int]]:
    legs_rot = rotate_patrol_legs_by_start_wp(patrol, start_wp_idx)

    full_node_path: List[int] = []
    full_edge_path: List[int] = []

    for i, leg in enumerate(legs_rot):
        npth = leg["node_path"]
        epth = leg["edge_path"]

        if len(npth) == 0:
            continue

        if i == 0:
            full_node_path.extend(npth)
        else:
            full_node_path.extend(npth[1:])

        full_edge_path.extend(epth)

    return full_node_path, full_edge_path


# =========================================================
# 6. USV 순찰 초기화
# =========================================================
def compute_arrival_times_on_path(
    num_points: int,
    node_path: List[int],
    edge_path: List[int],
    cost_e: np.ndarray,
    *,
    num_laps: int = 1,
) -> np.ndarray:
    arr = np.full(num_points, np.inf, dtype=float)

    if len(node_path) == 0:
        return arr
    if len(edge_path) != len(node_path) - 1:
        raise ValueError("edge_path length must be len(node_path)-1")

    t = 0.0
    arr[node_path[0]] = 0.0

    for lap in range(num_laps):
        for i, e in enumerate(edge_path):
            u = node_path[i + 1]
            t += float(cost_e[e])
            if t < arr[u]:
                arr[u] = t

    return arr


def region_point_mask(points: np.ndarray, region: PatrolRegion) -> np.ndarray:
    return (
        (points[:, 0] >= region.x0) & (points[:, 0] <= region.x1) &
        (points[:, 1] >= region.y0) & (points[:, 1] <= region.y1)
    )
def predict_command_to_waypoints_surrogate(env, state=None):
    if state is None:
        state = env.get_state()

    mode = getattr(env.config, "surrogate_mode", "rf_direct")

    if mode == "rf_direct":
        return predict_command_to_waypoints_rf_direct(env, state)
    elif mode == "euclidean":
        return predict_command_to_waypoints_euclidean(env, state)
    elif mode == "astar_euclidean":
        return predict_command_to_waypoints_rf_direct(env, state)
    elif mode == "astar_rf":
        return predict_command_to_waypoints_rf_direct(env, state)
    else:
        raise ValueError(f"Unknown surrogate_mode: {mode}")
def reconstruct_surrogate_astar_command_segment(
    env,
    state,
    cand,
    *,
    heuristic_mode: str = None,
    fallback_to_direct: bool = True,
):
    """
    Surrogate mode에서 현재 command 위치 -> candidate rep_vertex_idx까지
    A*로 실제 graph path를 복원한다.

    env.config.surrogate_mode가
      - "astar_euclidean"이면 heuristic_mode="euclidean"
      - "astar_rf"이면 heuristic_mode="rf"
    로 사용한다.

    실패하면 [start_vid, goal_vid]만 반환한다.
    """

    start_vid = int(state["command_vid"])
    goal_vid = int(cand["rep_vertex_idx"])

    if start_vid == goal_vid:
        return [start_vid]

    if heuristic_mode is None:
        if env.config.surrogate_mode == "astar_rf":
            heuristic_mode = "rf"
        else:
            heuristic_mode = "euclidean"

    try:
        dist, prev_node, prev_k, best_goal, best_goal_k = astar_turn_state_core(
            cache=env.cache,
            base_cost_e=env.cost_e,
            theta_e_deg=env.theta_e_deg,
            points=env.points,
            start=start_vid,
            goal=goal_vid,

            heuristic_mode=heuristic_mode,
            rf_model=env.rf_model,
            rf_feature_names=env.rf_feature_names,
            blocked_mask=env.blocked_mask,
            total_ux=env.total_ux,
            total_uy=env.total_uy,
            current_heading_deg=float(state["command_heading_deg"]),
            inner_width=float(env.config.inner_width),
            outer_width=float(env.config.outer_width),
            usv_speed=float(env.config.usv_speed),
            residual_mode=env.config.residual_mode,

            th1=float(env.config.th1),
            lam1=float(env.config.lam1),
            th2=float(env.config.th2),
            lam2=float(env.config.lam2),

            use_start_heading=bool(env.config.use_start_heading),
            start_heading_deg=float(state["command_heading_deg"]),

            termination="goal_best",
            heuristic_weight=1.0,
        )

        if (not np.isfinite(best_goal)) or best_goal_k < 0:
            if fallback_to_direct:
                return [start_vid, goal_vid]
            return [start_vid]

        node_path, edge_path, k_path = reconstruct_path_from_prev(
            cache=env.cache,
            prev_node=prev_node,
            prev_k=prev_k,
            start=start_vid,
            goal=goal_vid,
            best_goal_k=best_goal_k,
        )

        if node_path is None or len(node_path) == 0:
            if fallback_to_direct:
                return [start_vid, goal_vid]
            return [start_vid]

        return [int(v) for v in node_path]

    except Exception as e:
        print(f"[WARN] surrogate A* path reconstruction failed: {e}")
        if fallback_to_direct:
            return [start_vid, goal_vid]
        return [start_vid]


def build_surrogate_candidate_pools_from_direct(
    env,
    state: Dict,
    *,
    pred_time_to_wp: np.ndarray,
    top_k_per_usv: int = 5,
) -> List[List[Dict]]:
    """
    direct prediction을 이용해 각 USV별 waypoint 후보 pool 생성
    """
    wp_list = state["wp_list"]
    meeting_node_features = state["surrogate_meeting_node_features"]
    alive_mask = state["alive_mask"]
    current_time = float(state["command_time"])

    t_scale = float(env.config.normalize_t) if env.config.normalize_t is not None else 1.0

    num_usv = len(alive_mask)
    pools: List[List[Dict]] = [[] for _ in range(num_usv)]

    for gidx, meta in enumerate(wp_list):
        usv_idx = int(meta["usv_idx"])
        if not alive_mask[usv_idx]:
            continue

        travel_time_rel_pred = float(pred_time_to_wp[gidx])

        future_residual = np.asarray(meeting_node_features[gidx, 2:], dtype=float).copy()
        finite_mask = np.isfinite(future_residual)
        future_residual[finite_mask] *= t_scale

        feasible_future = future_residual[np.isfinite(future_residual) & (future_residual >= travel_time_rel_pred-50)]
        if feasible_future.size == 0:
            continue

        direct_usv_arrival_rel = float(feasible_future[0])
        direct_meet_time_abs = current_time + direct_usv_arrival_rel

        pools[usv_idx].append({
            "usv_idx": usv_idx,
            "gidx": int(gidx),
            "local_wp_idx": int(meta["local_wp_idx"]),
            "vertex_idx": int(meta["vertex_idx"]),
            "xy": np.asarray(meta["xy"], dtype=float).copy(),
            "pred_travel_time_rel": travel_time_rel_pred,
            "future_residual": future_residual,
            "direct_usv_arrival_rel": direct_usv_arrival_rel,
            "direct_meet_time_abs": direct_meet_time_abs,
        })

    for usv_idx in range(num_usv):

        pools[usv_idx].sort(key=lambda x: x["direct_meet_time_abs"])
        if top_k_per_usv > 0:
            pools[usv_idx] = pools[usv_idx][:top_k_per_usv]


    return pools
def refine_one_usv_candidate_pool_with_astar(
    env,
    state: Dict,
    *,
    usv_idx: int,
    candidate_pool: List[Dict],
    heuristic_mode: str,   # "euclidean" or "rf"
    astar_cache: Optional[Dict[int, Tuple[float, int]]] = None,
) -> Optional[Dict]:
    if astar_cache is None:
        astar_cache = {}

    current_time = float(state["command_time"])
    start_vid = int(state["command_vid"])

    for cand in candidate_pool:
        goal_vid = int(cand["vertex_idx"])

        if goal_vid not in astar_cache:
            #start = time.time()
            _, _, _, best_goal, best_goal_k = astar_turn_state_core(
                cache=env.cache,
                base_cost_e=env.cost_e,
                theta_e_deg=env.theta_e_deg,
                points=env.points,
                start=start_vid,
                goal=goal_vid,
                heuristic_mode=heuristic_mode,
                rf_model=env.rf_model if heuristic_mode == "rf" else None,
                rf_feature_names=env.rf_feature_names if heuristic_mode == "rf" else None,
                blocked_mask=env.blocked_mask,
                total_ux=env.total_ux,
                total_uy=env.total_uy,
                current_heading_deg=env.command_heading_deg,
                inner_width=env.config.inner_width,
                outer_width=env.config.outer_width,
                usv_speed=env.config.usv_speed,
                residual_mode=env.config.residual_mode,
                th1=env.config.th1,
                lam1=env.config.lam1,
                th2=env.config.th2,
                lam2=env.config.lam2,
                use_start_heading=env.config.use_start_heading,
                start_heading_deg=env.command_heading_deg,
                termination="goal_best",
            )
            #print(time.time() - start)
            astar_cache[goal_vid] = (float(best_goal), int(best_goal_k))

        astar_travel_time_rel, best_goal_k = astar_cache[goal_vid]

        if not np.isfinite(astar_travel_time_rel):
            print("inf")
            continue

        if cand['direct_usv_arrival_rel']<astar_travel_time_rel:
            continue
        usv_arrival_rel = cand['direct_usv_arrival_rel']
        meet_time_abs = current_time + usv_arrival_rel
        arrival_heading_deg = get_arrival_heading_deg_at_node(
            cache=env.cache,
            theta_e_deg=env.theta_e_deg,
            node_vid=int(goal_vid),
            best_k_in=int(best_goal_k),
            fallback_heading_deg=float(state["command_heading_deg"]),
        )
        return {
            "usv_idx": int(usv_idx),
            "rep_global_wp_idx": int(cand["gidx"]),
            "rep_local_wp_idx": int(cand["local_wp_idx"]),
            "rep_vertex_idx": int(cand["vertex_idx"]),
            "rep_xy": np.asarray(cand["xy"], dtype=float).copy(),
            "command_travel_time_rel": float(astar_travel_time_rel),
            "usv_arrival_time_rel": float(usv_arrival_rel),
            "meet_time_abs": float(meet_time_abs),
            "waiting_time": float(usv_arrival_rel - astar_travel_time_rel),
            "arrival_heading_deg": float(arrival_heading_deg),
        }
    print("error")
    debug_print_astar_failure_for_usv(
        env,
        state,
        usv_idx=usv_idx,
        candidate_pool=candidate_pool,
        heuristic_mode=heuristic_mode,
        reason="all_candidates_failed",
    )
    raise RuntimeError(f"A* refinement failed for usv_idx={usv_idx}: all candidates failed.")


def predict_command_to_waypoints_rf_direct(
    env,
    state: Optional[Dict] = None,
) -> Dict[str, np.ndarray]:
    if state is None:
        state = env.get_state()

    wp_list = state["wp_list"]
    start_vid = int(env.command_vid)

    rows = []
    base_times = []
    wp_vertex_idx = []

    for meta in wp_list:
        goal_vid = int(meta["vertex_idx"])

        heading_diff_deg = compute_heading_goal_diff_deg(
            points=env.points,
            start_idx=start_vid,
            goal_idx=goal_vid,
            current_heading_deg=env.command_heading_deg,
        )

        feat_dict = extract_fixed_features(
            points=env.points,
            blocked_mask=env.blocked_mask,
            total_ux=env.total_ux,
            total_uy=env.total_uy,
            start_idx=start_vid,
            goal_idx=goal_vid,
            inner_width=env.config.inner_width,
            outer_width=env.config.outer_width,
            heading_diff_deg=heading_diff_deg,   # 핵심
        )

        row = []
        for name in env.rf_feature_names:
            if name not in feat_dict:
                raise KeyError(f"RF feature '{name}' not found in extract_fixed_features output.")
            row.append(feat_dict[name])

        base_time = float(feat_dict["distance"]) / float(env.config.usv_speed)

        rows.append(row)
        base_times.append(base_time)
        wp_vertex_idx.append(goal_vid)

    X_rf = np.asarray(rows, dtype=float)
    base_times = np.asarray(base_times, dtype=float)
    wp_vertex_idx = np.asarray(wp_vertex_idx, dtype=np.int64)

    pred_residual = np.asarray(env.rf_model.predict(X_rf), dtype=float)

    if env.config.residual_mode == "ratio":
        pred_time = base_times * (1.0 + pred_residual)
    elif env.config.residual_mode == "percent":
        pred_time = base_times * (1.0 + pred_residual / 100.0)
    else:
        raise ValueError("residual_mode must be 'ratio' or 'percent'.")

    pred_time = np.maximum(pred_time, 1e-6)

    return {
        "pred_time_to_wp": pred_time,
        "wp_vertex_idx": wp_vertex_idx,
    }

def predict_command_to_waypoints_euclidean(env, state=None):
    if state is None:
        state = env.get_state()

    wp_list = state["wp_list"]
    start_vid = int(state["command_vid"])
    p0 = env.points[start_vid]

    pred_time = []
    wp_vertex_idx = []

    for meta in wp_list:
        goal_vid = int(meta["vertex_idx"])
        p1 = env.points[goal_vid]
        dist = float(np.linalg.norm(p1 - p0))
        pred_time.append(max(dist / float(env.config.usv_speed), 1e-6))
        wp_vertex_idx.append(goal_vid)

    return {
        "pred_time_to_wp": np.asarray(pred_time, dtype=float),
        "wp_vertex_idx": np.asarray(wp_vertex_idx, dtype=np.int64),
    }
def rf_time_heuristic(env, start_vid: int, goal_vid: int, current_heading_deg: float) -> float:
    heading_diff_deg = compute_heading_goal_diff_deg(
        points=env.points,
        start_idx=start_vid,
        goal_idx=goal_vid,
        current_heading_deg=current_heading_deg,
    )

    feat_dict = extract_fixed_features(
        points=env.points,
        blocked_mask=env.blocked_mask,
        total_ux=env.total_ux,
        total_uy=env.total_uy,
        start_idx=start_vid,
        goal_idx=goal_vid,
        inner_width=env.config.inner_width,
        outer_width=env.config.outer_width,
        heading_diff_deg=heading_diff_deg,
    )

    row = [feat_dict[name] for name in env.rf_feature_names]
    base_time = float(feat_dict["distance"]) / float(env.config.usv_speed)

    pred_residual = float(env.rf_model.predict(np.asarray([row], dtype=float))[0])

    if env.config.residual_mode == "ratio":
        pred_time = base_time * (1.0 + pred_residual)
    elif env.config.residual_mode == "percent":
        pred_time = base_time * (1.0 + pred_residual / 100.0)
    else:
        raise ValueError("residual_mode must be 'ratio' or 'percent'.")

    return max(pred_time, 1e-6)
def build_surrogate_intercept_candidates(
    env,
    state: Dict,
    *,
    pred_time_to_wp: np.ndarray,
) -> List[Optional[Dict]]:
    """
    mode에 따라
      - rf_direct / euclidean : 기존 direct 방식
      - astar_euclidean / astar_rf : direct 후보 추출 + A* refinement
    """
    mode = getattr(env.config, "surrogate_mode", "rf_direct")

    if mode in ("rf_direct", "euclidean"):
        return build_surrogate_intercept_candidates_direct(
            env,
            state,
            pred_time_to_wp=pred_time_to_wp,
        )

    if mode not in ("astar_euclidean", "astar_rf"):
        raise ValueError(f"Unknown surrogate_mode: {mode}")

    heuristic_mode = "euclidean" if mode == "astar_euclidean" else "rf"

    candidate_pools = build_surrogate_candidate_pools_from_direct(
        env,
        state,
        pred_time_to_wp=pred_time_to_wp,
        top_k_per_usv=int(getattr(env.config, "surrogate_topk_per_usv", 3)),
    )

    alive_mask = state["alive_mask"]
    num_usv = len(alive_mask)
    candidates: List[Optional[Dict]] = [None for _ in range(num_usv)]

    astar_cache: Dict[int, Tuple[float, int]] = {}

    for usv_idx in range(num_usv):
        if not alive_mask[usv_idx]:
            continue

        candidates[usv_idx] = refine_one_usv_candidate_pool_with_astar(
            env,
            state,
            usv_idx=usv_idx,
            candidate_pool=candidate_pools[usv_idx],
            heuristic_mode=heuristic_mode,
            astar_cache=astar_cache,
        )

    return candidates

def build_surrogate_intercept_candidates_direct(
    env,
    state: Dict,
    *,
    pred_time_to_wp: np.ndarray,
) -> List[Optional[Dict]]:
    """
    surrogate travel time을 사용해서 각 USV별 대표 waypoint 후보 선택

    Returns:
      candidates[u] = {
        "usv_idx": u,
        "rep_global_wp_idx": .,
        "rep_local_wp_idx": .,
        "rep_vertex_idx": .,
        "rep_xy": .,
        "command_travel_time_rel": .,
        "usv_arrival_time_rel": .,
        "meet_time_abs": .,
        "waiting_time": .,
        "arrival_heading_deg": .,
      }
      or None
    """
    wp_list = state["wp_list"]
    meeting_node_features = state["surrogate_meeting_node_features"]
    alive_mask = state["alive_mask"]
    current_time = float(state["command_time"])
    command_xy = np.asarray(state["command_xy"], dtype=np.float32)
    command_heading_deg = float(state["command_heading_deg"])

    t_scale = float(env.config.normalize_t) if env.config.normalize_t is not None else 1.0

    num_usv = len(alive_mask)
    candidates: List[Optional[Dict]] = [None for _ in range(num_usv)]

    for usv_idx in range(num_usv):
        if not alive_mask[usv_idx]:
            continue

        best = None
        best_meet_abs = float("inf")

        for gidx, meta in enumerate(wp_list):
            if int(meta["usv_idx"]) != int(usv_idx):
                continue

            rep_xy = np.asarray(meta["xy"], dtype=np.float32)
            pred_cmd_rel = float(pred_time_to_wp[gidx])
            if not np.isfinite(pred_cmd_rel):
                continue

            future_residual_norm = meeting_node_features[gidx, 2:]
            future_residual = np.asarray(future_residual_norm, dtype=float) * t_scale
            future_residual = future_residual[np.isfinite(future_residual)]

            if future_residual.size == 0:
                continue

            candidate_meet_abs = None
            for t_rel in future_residual:
                t_abs = current_time + float(t_rel)
                if current_time + pred_cmd_rel <= t_abs:
                    candidate_meet_abs = float(t_abs)
                    break

            if candidate_meet_abs is None:
                continue

            if candidate_meet_abs < best_meet_abs:
                bearing_deg = _bearing_deg_from_two_points(
                    command_xy=command_xy,
                    goal_xy=rep_xy,
                    fallback_heading_deg=command_heading_deg,
                )

                best_meet_abs = candidate_meet_abs
                best = {
                    "usv_idx": int(usv_idx),
                    "rep_global_wp_idx": int(gidx),
                    "rep_local_wp_idx": int(meta["local_wp_idx"]),
                    "rep_vertex_idx": int(meta["vertex_idx"]),
                    "rep_xy": rep_xy.copy(),
                    "command_travel_time_rel": float(pred_cmd_rel),
                    "usv_arrival_time_rel": float(candidate_meet_abs - current_time),
                    "meet_time_abs": float(candidate_meet_abs),
                    "waiting_time": float(candidate_meet_abs - (current_time + pred_cmd_rel)),
                    "arrival_heading_deg": float(bearing_deg),
                }

        candidates[usv_idx] = best

    return candidates

def simulate_multi_usv_patrol_arrival(
    points: np.ndarray,
    patrols: List[Dict],
    cost_e: np.ndarray,
    *,
    seed: Optional[int] = 0,
    num_laps: int = 1,
    assign_only_inside_region: bool = True,
    region_mode,
) -> Dict:
    rng = np.random.default_rng(seed)

    start_wp_indices: List[int] = []
    usv_node_paths: List[List[int]] = []
    usv_edge_paths: List[List[int]] = []
    usv_arrival_times: List[np.ndarray] = []

    N = len(points)

    for e,patrol in enumerate(patrols):

        num_wp = len(patrol["waypoint_vids"])
        if region_mode == "16_type2" and e > 7 and e%2==1:
            start_wp=(start_wp+num_wp//2)%num_wp
        elif region_mode == "24_type1" and e%2==1:
            start_wp=(start_wp+num_wp//2)%num_wp
        elif region_mode == "24_type2" and e%2==1:
            start_wp=(start_wp+num_wp//2)%num_wp
        else:
            start_wp = int(rng.integers(0, num_wp))

        start_wp_indices.append(start_wp)

        node_path, edge_path = build_rotated_patrol_path(patrol, start_wp)
        usv_node_paths.append(node_path)
        usv_edge_paths.append(edge_path)

        arr = compute_arrival_times_on_path(
            num_points=N,
            node_path=node_path,
            edge_path=edge_path,
            cost_e=cost_e,
            num_laps=num_laps,
        )

        if assign_only_inside_region:
            mask = region_point_mask(points, patrol["region"])
            arr = np.where(mask, arr, np.inf)

        usv_arrival_times.append(arr)

    return {
        "start_wp_indices": start_wp_indices,
        "usv_node_paths": usv_node_paths,
        "usv_edge_paths": usv_edge_paths,
        "usv_arrival_times": usv_arrival_times,
    }
def select_future_visit_times(
    times: List[float],
    current_time: float,
    num_future_times: int,
    *,
    as_residual: bool = True,
) -> np.ndarray:
    future = [t for t in times if t > current_time]

    out = np.full(num_future_times, np.inf, dtype=float)
    m = min(num_future_times, len(future))
    if m > 0:
        vals = np.asarray(future[:m], dtype=float)
        if as_residual:
            vals = vals - current_time
        out[:m] = vals
    return out



# =========================================================
# 7. USV 시간표
# =========================================================
def build_usv_path_timeline(
    node_path: List[int],
    edge_path: List[int],
    cost_e: np.ndarray,
    *,
    num_laps: int = 1,
) -> Tuple[List[int], np.ndarray]:
    if len(node_path) == 0:
        return [], np.array([], dtype=float)

    if len(edge_path) != len(node_path) - 1:
        raise ValueError("edge_path length must be len(node_path)-1")

    visit_nodes = [node_path[0]]
    visit_times = [0.0]

    t = 0.0
    for lap in range(num_laps):
        for i, e in enumerate(edge_path):
            t += float(cost_e[e])
            visit_nodes.append(node_path[i + 1])
            visit_times.append(t)

    return visit_nodes, np.array(visit_times, dtype=float)


def build_usv_node_visit_schedule(
    node_path: List[int],
    edge_path: List[int],
    cost_e: np.ndarray,
    *,
    num_laps: int = 5,
) -> Dict[int, List[float]]:
    visit_dict: Dict[int, List[float]] = {}

    if len(node_path) == 0:
        return visit_dict
    if len(edge_path) != len(node_path) - 1:
        raise ValueError("edge_path length must be len(node_path)-1")

    t = 0.0
    v0 = int(node_path[0])
    visit_dict.setdefault(v0, []).append(t)

    for lap in range(num_laps):
        for i, e in enumerate(edge_path):
            t += float(cost_e[e])
            v = int(node_path[i + 1])
            visit_dict.setdefault(v, []).append(t)

    return visit_dict
def select_future_visit_times(
    times: List[float],
    current_time: float,
    num_future_times: int,
    *,
    as_residual: bool = True,
) -> np.ndarray:
    """
    times 중 current_time보다 큰 값만 남기고,
    앞에서 num_future_times개를 반환.
    부족하면 inf로 패딩.
    """
    future = [t for t in times if t > current_time]

    out = np.full(num_future_times, np.inf, dtype=float)
    m = min(num_future_times, len(future))
    if m > 0:
        vals = np.asarray(future[:m], dtype=float)
        if as_residual:
            vals = vals - current_time
        out[:m] = vals
    return out

def build_usv_waypoint_time_features(
    patrols: List[Dict],
    sim_result: Dict,
    cost_e: np.ndarray,
    *,
    current_time: float,
    num_future_times: int = 5,
    num_laps_for_schedule: int = 20,
    as_residual: bool = True,
) -> Dict[int, np.ndarray]:
    """
    각 USV의 waypoint feature 생성

    Returns:
      features[usv_idx] = (num_waypoints, 2 + num_future_times)
      columns = [x, y, future_t0, future_t1, ..., future_t_{num_future_times-1}]

    여기서 future_tk는
      - as_residual=True  : current_time 기준 남은 시간
      - as_residual=False : 절대 시각
    """
    out: Dict[int, np.ndarray] = {}

    for usv_idx, patrol in enumerate(patrols):
        node_path = sim_result["usv_node_paths"][usv_idx]
        edge_path = sim_result["usv_edge_paths"][usv_idx]
        wp_vids = patrol["waypoint_vids"]
        wp_xy = patrol["waypoint_xy"]

        # 충분히 긴 horizon으로 방문 시각표 생성
        visit_schedule = build_usv_node_visit_schedule(
            node_path=node_path,
            edge_path=edge_path,
            cost_e=cost_e,
            num_laps=num_laps_for_schedule,
        )

        feat_rows = []
        for j, vid in enumerate(wp_vids):
            xy = wp_xy[j]
            times = visit_schedule.get(int(vid), [])

            future_arr = select_future_visit_times(
                times=times,
                current_time=current_time,
                num_future_times=num_future_times,
                as_residual=as_residual,
            )

            row = np.concatenate([np.asarray(xy, dtype=float), future_arr])
            feat_rows.append(row)

        out[usv_idx] = np.vstack(feat_rows)

    return out
def build_waypoint_node_features(
    patrols: List[Dict],
    sim_result: Dict,
    cost_e: np.ndarray,
    *,
    current_time: float,
    num_future_times: int = 5,
    num_laps_for_state_schedule: int = 20,
    as_residual: bool = True,
    normalize_xy: Optional[Tuple[float, float]] = None,
    normalize_t: Optional[float] = None,
) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
    """
    Returns:
      wp_list
      node_features: (N, 2 + num_future_times)
      usv_id_per_node: (N,)
    """
    wp_list = flatten_waypoints_from_patrols(patrols)

    node_rows = []
    usv_id_per_node = []

    for item in wp_list:
        usv_idx = item["usv_idx"]
        local_wp_idx = item["local_wp_idx"]
        xy = item["xy"].copy()

        node_path = sim_result["usv_node_paths"][usv_idx]
        edge_path = sim_result["usv_edge_paths"][usv_idx]

        visit_schedule = build_usv_node_visit_schedule(
            node_path=node_path,
            edge_path=edge_path,
            cost_e=cost_e,
            num_laps=num_laps_for_state_schedule,
        )

        vid = patrols[usv_idx]["waypoint_vids"][local_wp_idx]
        times = visit_schedule.get(int(vid), [])

        future_arr = select_future_visit_times(
            times=times,
            current_time=current_time,
            num_future_times=num_future_times,
            as_residual=as_residual,
        )

        if normalize_xy is not None:
            xy[0] /= float(normalize_xy[0])
            xy[1] /= float(normalize_xy[1])

        if normalize_t is not None:
            finite_mask = np.isfinite(future_arr)
            future_arr = future_arr.copy()
            future_arr[finite_mask] /= float(normalize_t)

        row = np. concatenate([xy, future_arr])
        node_rows.append(row)
        usv_id_per_node.append(usv_idx)

    node_features = np.asarray(node_rows, dtype=float)
    usv_id_per_node = np.asarray(usv_id_per_node, dtype=np.int64)

    return wp_list, node_features, usv_id_per_node
def build_rf_pair_feature_table(
    patrols: List[Dict],
    points: np.ndarray,
    blocked_mask: np.ndarray,
    total_ux: np.ndarray,
    total_uy: np.ndarray,
    rf_feature_names: List[str],
    *,
    inner_width: float = 90.0,
    outer_width: float = 180.0,
    exclude_same_usv: bool = True,
) -> Tuple[List[Dict], List[Tuple[int, int]], np.ndarray]:
    wp_list = flatten_waypoints_from_patrols(patrols)

    pair_index = []
    rows = []

    for i, src in enumerate(wp_list):
        for j, dst in enumerate(wp_list):
            if i == j:
                continue
            if exclude_same_usv and src["usv_idx"] == dst["usv_idx"]:
                continue

            feat_dict = extract_fixed_features(
                points=points,
                blocked_mask=blocked_mask,
                total_ux=total_ux,
                total_uy=total_uy,
                start_idx=src["vertex_idx"],
                goal_idx=dst["vertex_idx"],
                inner_width=inner_width,
                outer_width=outer_width,
            )

            row = []
            for name in rf_feature_names:
                if name not in feat_dict:
                    raise KeyError(f"RF feature '{name}' not found in extract_fixed_features output.")
                row.append(feat_dict[name])

            pair_index.append((i, j))
            rows.append(row)

    X_rf = np.asarray(rows, dtype=float)
    return wp_list, pair_index, X_rf


# =========================================================
# 7. RF prediction -> predicted time matrix
# =========================================================
def build_prediction_time_matrix_from_rf(
    patrols: List[Dict],
    points: np.ndarray,
    blocked_mask: np.ndarray,
    total_ux: np.ndarray,
    total_uy: np.ndarray,
    rf_model,
    rf_feature_names: List[str],
    *,
    usv_speed: float,
    residual_mode: str = "ratio",   # "ratio" or "percent"
    inner_width: float = 90.0,
    outer_width: float = 180.0,
    exclude_same_usv: bool = True,
    clip_min_time: float = 1e-6,
) -> Tuple[List[Dict], np.ndarray]:
    """
    RF가 residual ratio(또는 %)를 예측한다고 가정하고
    baseline = distance / usv_speed 로부터 pred_time 복원

    Returns:
      wp_list
      pred_time_mat: (N, N), same-USV or self pair는 np.nan
    """
    wp_list = flatten_waypoints_from_patrols(patrols)

    pair_index = []
    X_rows = []
    base_time_rows = []

    for i, src in enumerate(wp_list):
        for j, dst in enumerate(wp_list):
            if i == j:
                continue
            if exclude_same_usv and src["usv_idx"] == dst["usv_idx"]:
                continue

            feat_dict = extract_fixed_features(
                points=points,
                blocked_mask=blocked_mask,
                total_ux=total_ux,
                total_uy=total_uy,
                start_idx=src["vertex_idx"],
                goal_idx=dst["vertex_idx"],
                inner_width=inner_width,
                outer_width=outer_width,
                heading_diff_deg=0.0,
            )

            if "distance" not in feat_dict:
                raise KeyError("'distance' must be included in extract_fixed_features output.")

            base_time = float(feat_dict["distance"]) / float(usv_speed)

            row = []
            for name in rf_feature_names:
                if name not in feat_dict:
                    raise KeyError(f"RF feature '{name}' not found in extract_fixed_features output.")
                row.append(feat_dict[name])

            pair_index.append((i, j))
            X_rows.append(row)
            base_time_rows.append(base_time)

    X_rf = np.asarray(X_rows, dtype=float)
    base_time_arr = np.asarray(base_time_rows, dtype=float)

    pred_residual = np.asarray(rf_model.predict(X_rf), dtype=float)

    if residual_mode == "ratio":
        pred_time = base_time_arr * (1.0 + pred_residual)
    elif residual_mode == "percent":
        pred_time = base_time_arr * (1.0 + pred_residual / 100.0)
    else:
        raise ValueError("residual_mode must be 'ratio' or 'percent'.")

    pred_time = np.maximum(pred_time, clip_min_time)

    N = len(wp_list)
    pred_time_mat = np.full((N, N), np.nan, dtype=float)

    for (i, j), val in zip(pair_index, pred_time):
        pred_time_mat[i, j] = float(val)

    return wp_list, pred_time_mat

# =========================================================
# 8. same-USV attention mask
# =========================================================
def build_same_usv_mask_numpy(
    usv_id_per_node: np.ndarray,
    *,
    allow_self: bool = True,
    mask_value: float = -1e9,
) -> np.ndarray:
    N = len(usv_id_per_node)
    same = usv_id_per_node[:, None] == usv_id_per_node[None, :]
    mask = np.zeros((N, N), dtype=float)

    if allow_self:
        same = same & (~np.eye(N, dtype=bool))

    mask[same] = mask_value
    return mask


# =========================================================
# 9. simulation -> transformer inputs
# =========================================================

def reduce_state_dist_to_node_best(
    dist: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    best_k = np.argmin(dist, axis=1)
    best_t = dist[np.arange(dist.shape[0]), best_k]

    all_inf = ~np.isfinite(best_t)
    best_k = best_k.astype(np.int32)
    best_k[all_inf] = -1

    return best_t, best_k


def build_usv_path_timeline(
    node_path: List[int],
    edge_path: List[int],
    cost_e: np.ndarray,
    *,
    num_laps: int = 1,
) -> Tuple[List[int], np.ndarray]:
    if len(node_path) == 0:
        return [], np.array([], dtype=float)

    if len(edge_path) != len(node_path) - 1:
        raise ValueError("edge_path length must be len(node_path)-1")

    visit_nodes = [node_path[0]]
    visit_times = [0.0]

    t = 0.0
    for _ in range(num_laps):
        for i, e in enumerate(edge_path):
            t += float(cost_e[e])
            visit_nodes.append(node_path[i + 1])
            visit_times.append(t)

    return visit_nodes, np.array(visit_times, dtype=float)

def find_intercept_for_selected_usv_from_precomputed(
    wp_list,
    points: np.ndarray,
    sim_result: Dict,
    cost_e: np.ndarray,
    theta_e_deg: np.ndarray,
    *,
    usv_idx: int,
    command_current_vid: int,
    command_current_time: float,
    command_heading_deg: float,
    cmd_best_time_rel: np.ndarray,   # precomputed
    cmd_best_k: np.ndarray,          # precomputed
    prev_node_cmd: np.ndarray,       # precomputed
    prev_k_cmd: np.ndarray,          # precomputed
    cache,
    num_laps: int = 10,
):
    node_path = sim_result["usv_node_paths"][usv_idx]
    edge_path = sim_result["usv_edge_paths"][usv_idx]

    visit_nodes, visit_times = build_usv_path_timeline(
        node_path=node_path,
        edge_path=edge_path,
        cost_e=cost_e,
        num_laps=num_laps,
    )

    best = None
    best_meet_time = float("inf")
    waypoint_set={
        int(wp["vertex_idx"])
        for wp in wp_list
        if int(wp["usv_idx"])==int(usv_idx)
    }
    for v, t_usv_abs in zip(visit_nodes, visit_times):
        v=int(v)
        if v not in waypoint_set:
            continue

        if t_usv_abs < command_current_time:
            continue

        t_cmd_rel = float(cmd_best_time_rel[v])
        if not np.isfinite(t_cmd_rel):
            continue

        t_cmd_abs = command_current_time + t_cmd_rel

        if t_cmd_abs <= t_usv_abs:
            if t_usv_abs < best_meet_time:
                meet_k = int(cmd_best_k[v])

                arrival_heading_deg = get_arrival_heading_deg_at_node(
                    cache=cache,
                    theta_e_deg=theta_e_deg,
                    node_vid=int(v),
                    best_k_in=meet_k,
                    fallback_heading_deg=float(command_heading_deg),
                )

                best_meet_time = t_usv_abs
                best = {
                    "usv_idx": usv_idx,
                    "meet_node": int(v),
                    "meet_xy": points[int(v)].copy(),
                    "command_arrival_time_abs": float(t_cmd_abs),
                    "command_travel_time_rel": float(t_cmd_rel),
                    "usv_arrival_time_abs": float(t_usv_abs),
                    "waiting_time": float(t_usv_abs - t_cmd_abs),
                    "command_best_k": meet_k,
                    "arrival_heading_deg": float(arrival_heading_deg),
                }

    if best is None:
        return None

    meet_node = int(best["meet_node"])
    meet_k = int(best["command_best_k"])

    if meet_node == command_current_vid:
        cmd_node_path, cmd_edge_path, cmd_k_path = [command_current_vid], [], []
    else:
        cmd_node_path, cmd_edge_path, cmd_k_path = reconstruct_path_from_prev(
            cache=cache,
            prev_node=prev_node_cmd,
            prev_k=prev_k_cmd,
            start=command_current_vid,
            goal=meet_node,
            best_goal_k=meet_k,
        )

    best["command_node_path"] = cmd_node_path
    best["command_edge_path"] = cmd_edge_path
    best["command_k_path"] = cmd_k_path

    return best

def state_to_usv_tensor(state: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    state["node_features"] + state["wp_list"] -> (U, W, F)
    """
    wp_list = state["wp_list"]
    node_features = state["node_features"]
    alive_mask = state["alive_mask"]

    num_usv = len(alive_mask)
    num_waypoints = max(item["local_wp_idx"] for item in wp_list) + 1
    feat_dim = node_features.shape[1]

    usv_node_features = np.zeros((num_usv, num_waypoints, feat_dim), dtype=np.float32)

    for i, meta in enumerate(wp_list):
        u = meta["usv_idx"]
        w = meta["local_wp_idx"]
        usv_node_features[u, w] = node_features[i]

    return usv_node_features, alive_mask.astype(bool)

def compute_all_usv_intercept_candidates(env,precomputed: Optional[Dict]=None):
    if precomputed is None:
        precomputed = compute_command_all_shortest_once(env)
    candidates = []

    for usv_idx in range(env.num_usv):
        if not env.alive_mask[usv_idx]:
            candidates.append(None)
            continue

        cand = find_intercept_for_selected_usv_from_precomputed(
            wp_list=env.wp_list,
            points=env.points,
            sim_result=env.sim_result,
            cost_e=env.cost_e,
            usv_idx=usv_idx,
            command_current_vid=env.command_vid,
            command_current_time=env.command_time,
            cmd_best_time_rel=precomputed["cmd_best_time_rel"],
            cmd_best_k=precomputed["cmd_best_k"],
            prev_node_cmd=precomputed["prev_node_cmd"],
            prev_k_cmd=precomputed["prev_k_cmd"],
            cache=env.cache,
            num_laps=env.config.num_laps_for_intercept,
        )

        if cand is not None:
            meet_node = int(cand["meet_node"])

            # cand 안에 best incoming state가 이미 있다면 그걸 쓰고
            # 없다면 fallback 필요
            best_k_in = cand.get("command_best_k", None)

            if best_k_in is not None and np.isfinite(best_k_in):
                arrival_heading_deg = get_arrival_heading_deg_at_node(
                    cache=env.cache,
                    theta_e_deg=env.theta_e_deg,
                    node_vid=meet_node,
                    best_k_in=int(best_k_in),
                    fallback_heading_deg=env.command_heading_deg,
                )
            else:
                arrival_heading_deg = env.command_heading_deg

            cand["arrival_heading_deg"] = float(arrival_heading_deg)

        candidates.append(cand)

    return candidates

def get_representative_waypoint_indices_from_intercepts(
    patrols: List[Dict],
    intercept_candidates: List[Optional[Dict]],
    points: np.ndarray,
) -> np.ndarray:
    num_usv = len(patrols)
    rep_idx = np.zeros(num_usv, dtype=np.int64)

    for u in range(num_usv):
        cand = intercept_candidates[u]
        wp_vids = patrols[u]["waypoint_vids"]

        if cand is None:
            rep_idx[u] = 0
            continue

        meet_node = int(cand["meet_node"])
        meet_xy = points[meet_node]

        wp_xy = points[np.array(wp_vids)]
        d2 = np.sum((wp_xy - meet_xy[None, :]) ** 2, axis=1)
        rep_idx[u] = int(np.argmin(d2))

    return rep_idx

def build_actor_query_features_from_intercepts(
    patrols,
    intercept_candidates,
    command_xy,
    *,
    command_heading_deg: float,
    current_time: float,
    normalize_xy=None,
    normalize_t=None,
):
    """
    actor query feature (11-dim):
      [command_x, command_y,
       rep_x, rep_y,
       travel_time,
       sin(command_heading), cos(command_heading),
       sin(bearing),         cos(bearing),
       sin(arrival_heading), cos(arrival_heading)]
    """
    command_xy_raw = np.asarray(command_xy, dtype=np.float32).copy()
    command_xy_norm = command_xy_raw.copy()

    if normalize_xy is not None:
        command_xy_norm[0] /= float(normalize_xy[0])
        command_xy_norm[1] /= float(normalize_xy[1])

    rows = []
    for usv_idx in range(len(patrols)):
        cand = intercept_candidates[usv_idx]

        if cand is None:
            rep_xy_raw = command_xy_raw.copy()
            rep_xy_norm = command_xy_norm.copy()
            meet_elapsed_time = 1e6


            sin_ch, cos_ch = _heading_feature_pair_from_angle_deg(command_heading_deg)
            bearing_deg = float(command_heading_deg)
            arrival_heading_deg = float(command_heading_deg)

        else:
            rep_xy_raw = np.asarray(cand["meet_xy"], dtype=np.float32).copy()
            rep_xy_norm = rep_xy_raw.copy()

            if normalize_xy is not None:
                rep_xy_norm[0] /= float(normalize_xy[0])
                rep_xy_norm[1] /= float(normalize_xy[1])

            meet_elapsed_time = float(cand["usv_arrival_time_abs"] - current_time)

            sin_ch, cos_ch = _heading_feature_pair_from_angle_deg(command_heading_deg)

            bearing_deg = _bearing_deg_from_two_points(
                command_xy=command_xy_raw,
                goal_xy=rep_xy_raw,
                fallback_heading_deg=command_heading_deg,
            )

            if ("arrival_heading_deg" in cand) and np.isfinite(cand["arrival_heading_deg"]):
                arrival_heading_deg = float(cand["arrival_heading_deg"])
            else:
                arrival_heading_deg = float(bearing_deg)

        sin_gb, cos_gb = _heading_feature_pair_from_angle_deg(bearing_deg)
        sin_ah, cos_ah = _heading_feature_pair_from_angle_deg(arrival_heading_deg)

        if normalize_t is not None:
            meet_elapsed_time /= float(normalize_t)

        rows.append([
            command_xy_norm[0],
            command_xy_norm[1],
            meet_elapsed_time,
            sin_ch,
            cos_ch,
            sin_ah,
            cos_ah,
        ])

    return np.asarray(rows, dtype=np.float32)


def build_critic_query_features_from_surrogate_waypoints(
    env,
    state: Dict,
    *,
    pred_time_to_wp: np.ndarray,
    normalize_xy: Optional[Tuple[float, float]] = None,
    normalize_t: Optional[float] = None,
) -> np.ndarray:
    """
    critic query feature (11-dim), same layout as actor.
    In surrogate mode, arrival heading uses bearing as a proxy.
    """
    wp_list = state["wp_list"]
    meeting_node_features = state["surrogate_meeting_node_features"]
    command_xy_raw = np.asarray(state["command_xy"], dtype=np.float32).copy()
    command_xy_norm = command_xy_raw.copy()
    command_heading_deg = float(state["command_heading_deg"])

    t_scale = float(env.config.normalize_t) if env.config.normalize_t is not None else 1.0


    if normalize_xy is not None:
        command_xy_norm[0] /= float(normalize_xy[0])
        command_xy_norm[1] /= float(normalize_xy[1])

    rows = []
    for gidx, meta in enumerate(wp_list):
        wp_xy_raw = np.asarray(meta["xy"], dtype=np.float32).copy()
        wp_xy_norm = wp_xy_raw.copy()

        if normalize_xy is not None:
            wp_xy_norm[0] /= float(normalize_xy[0])
            wp_xy_norm[1] /= float(normalize_xy[1])
        pred_cmd_rel = float(pred_time_to_wp[gidx])

        future_residual_norm = meeting_node_features[gidx, 2:]
        future_residual = np.asarray(future_residual_norm, dtype=float) * t_scale
        future_residual = future_residual[np.isfinite(future_residual)]

        meet_elapsed_time = float("inf")
        if np.isfinite(pred_cmd_rel) and future_residual.size > 0:
            for t_rel in future_residual:
                if float(t_rel) >= pred_cmd_rel:
                    meet_elapsed_time = float(t_rel)
                    break

        sin_ch, cos_ch = _heading_feature_pair_from_angle_deg(command_heading_deg)

        bearing_deg = _bearing_deg_from_two_points(
            command_xy=command_xy_raw,
            goal_xy=wp_xy_raw,
            fallback_heading_deg=command_heading_deg,
        )
        sin_gb, cos_gb = _heading_feature_pair_from_angle_deg(bearing_deg)

        arrival_heading_deg = bearing_deg
        sin_ah, cos_ah = _heading_feature_pair_from_angle_deg(arrival_heading_deg)

        if normalize_t is not None:
            meet_elapsed_time /= float(normalize_t)

        rows.append([
            command_xy_norm[0],
            command_xy_norm[1],
            wp_xy_norm[0],
            wp_xy_norm[1],
            meet_elapsed_time,
            sin_ch,
            cos_ch,
            sin_gb,
            cos_gb,
            sin_ah,
            cos_ah,
        ])

    return np.asarray(rows, dtype=np.float32)
def build_surrogate_actor_critic_inputs_from_env(env, state: Optional[Dict] = None) -> Dict[str, Any]:
    """
    exact Dijkstra 없이 surrogate travel time만으로 actor/critic 입력 생성
    """
    if state is None:
        state = env.get_state()

    usv_node_features, alive_mask = state_to_usv_tensor(state)

    pred_out = predict_command_to_waypoints_surrogate(env, state)
    pred_time_to_wp = pred_out["pred_time_to_wp"]

    candidates = build_surrogate_intercept_candidates(
        env,
        state,
        pred_time_to_wp=pred_time_to_wp,
    )

    rep_node_indices = np.zeros(env.num_usv, dtype=np.int64)
    for u in range(env.num_usv):
        cand = candidates[u]
        if cand is None:
            rep_node_indices[u] = 0
        else:
            rep_node_indices[u] = int(cand["rep_local_wp_idx"])

    actor_query_features = build_actor_query_features_from_surrogate_candidates(
        env,
        state,
        candidates,
        normalize_xy=env.config.normalize_xy,
        normalize_t=env.config.normalize_t,
    )
    '''
    critic_query_features = build_critic_query_features_from_surrogate_waypoints(
        env,
        state,
        pred_time_to_wp=pred_time_to_wp,
        normalize_xy=env.config.normalize_xy,
        normalize_t=env.config.normalize_t,
    )
    '''
    critic_query_features=np.zeros((1,1))

    U, W, _ = usv_node_features.shape

    rf_bias = np.asarray(state["bias_base_mat"], dtype=np.float32)       # (N,N)
    same_usv_mask = np.asarray(state["same_usv_mask"], dtype=np.float32) # (N,N)
    dead_waypoint_mask = build_dead_waypoint_mask_from_alive(
        alive_mask=alive_mask,
        num_waypoints=W,
        mask_value=-1e9,
    )
    attn_soft_bias=rf_bias
    attn_hard_bias = same_usv_mask + dead_waypoint_mask

    return {
        "usv_node_features": usv_node_features,          # (U,W,F)
        "rep_node_indices": rep_node_indices,            # (U,)
        "actor_query_features": actor_query_features,    # (U,9)
        "critic_query_features": critic_query_features,  # (N_wp,9)
        "alive_mask": alive_mask,                        # (U,)
        "attn_hard_bias": attn_hard_bias,  # (N,N)
        "attn_soft_bias": attn_soft_bias,              # (N,N)
        "surrogate_candidates": candidates,
        "pred_time_to_wp": pred_time_to_wp,
        "state_before": state,
    }

def build_critic_query_features_from_state(
    env,
    state: Dict,
    *,
    precomputed: Dict,
    normalize_xy: Optional[Tuple[float, float]] = None,
    normalize_t: Optional[float] = None,
) -> np.ndarray:
    """
    critic query feature (11-dim), same layout as actor.
    time feature = earliest feasible meet elapsed time
    """
    command_xy_raw = np.asarray(state["command_xy"], dtype=np.float32).copy()
    command_xy_norm = command_xy_raw.copy()
    command_heading_deg = float(state["command_heading_deg"])
    current_time = float(state["command_time"])

    wp_list = state["wp_list"]
    cmd_best_time_rel = np.asarray(precomputed["cmd_best_time_rel"], dtype=float)

    if normalize_xy is not None:
        command_xy_norm[0] /= float(normalize_xy[0])
        command_xy_norm[1] /= float(normalize_xy[1])

    rows = []
    for meta in wp_list:
        usv_idx = int(meta["usv_idx"])
        local_wp_idx = int(meta["local_wp_idx"])
        vid = int(meta["vertex_idx"])

        wp_xy_raw = np.asarray(meta["xy"], dtype=np.float32).copy()
        wp_xy_norm = wp_xy_raw.copy()

        if normalize_xy is not None:
            wp_xy_norm[0] /= float(normalize_xy[0])
            wp_xy_norm[1] /= float(normalize_xy[1])

        cmd_arrival_rel = float(cmd_best_time_rel[vid])

        meet_elapsed_time = _earliest_feasible_meet_rel_from_schedule(
            env.absolute_visit_schedule,
            usv_idx=usv_idx,
            local_wp_idx=local_wp_idx,
            current_time=current_time,
            command_arrival_rel=cmd_arrival_rel,
        )
        meet_elapsed_time = _safe_normalize_time_value(
            meet_elapsed_time,
            normalize_t,
            invalid_fill=1e6,
        )

        sin_ch, cos_ch = _heading_feature_pair_from_angle_deg(command_heading_deg)

        bearing_deg = _bearing_deg_from_two_points(
            command_xy=command_xy_raw,
            goal_xy=wp_xy_raw,
            fallback_heading_deg=command_heading_deg,
        )
        sin_gb, cos_gb = _heading_feature_pair_from_angle_deg(bearing_deg)

        best_k_in = int(precomputed["cmd_best_k"][vid]) if "cmd_best_k" in precomputed else -1
        if best_k_in >= 0 and np.isfinite(cmd_best_time_rel[vid]):
            arrival_heading_deg = get_arrival_heading_deg_at_node(
                cache=env.cache,
                theta_e_deg=env.theta_e_deg,
                node_vid=vid,
                best_k_in=best_k_in,
                fallback_heading_deg=command_heading_deg,
            )
        else:
            arrival_heading_deg = bearing_deg

        sin_ah, cos_ah = _heading_feature_pair_from_angle_deg(arrival_heading_deg)

        rows.append([
            command_xy_norm[0],
            command_xy_norm[1],
            wp_xy_norm[0],
            wp_xy_norm[1],
            meet_elapsed_time,
            sin_ch,
            cos_ch,
            sin_gb,
            cos_gb,
            sin_ah,
            cos_ah,
        ])

    return np.asarray(rows, dtype=np.float32)

def build_actor_query_features_from_surrogate_candidates(
    env,
    state: Dict,
    candidates,
    *,
    normalize_xy: Optional[Tuple[float, float]] = None,
    normalize_t: Optional[float] = None,
) -> np.ndarray:
    """
    actor query feature (11-dim):
      [command_x, command_y,
       rep_x, rep_y,
       travel_time,
       sin(command_heading), cos(command_heading),
       sin(bearing),         cos(bearing),
       sin(arrival_heading), cos(arrival_heading)]
    """
    command_xy_raw = np.asarray(state["command_xy"], dtype=np.float32).copy()
    command_xy_norm = command_xy_raw.copy()
    command_heading_deg = float(state["command_heading_deg"])
    current_time = float(state["command_time"])
    surrogate_mode = getattr(env.config, "surrogate_mode", "rf_direct")

    if normalize_xy is not None:
        command_xy_norm[0] /= float(normalize_xy[0])
        command_xy_norm[1] /= float(normalize_xy[1])

    rows = []
    for u in range(env.num_usv):
        cand = candidates[u]

        if cand is None:
            rep_xy_raw = command_xy_raw.copy()
            rep_xy_norm = command_xy_norm.copy()
            meet_elapsed_time = 1e6

            sin_ch, cos_ch = _heading_feature_pair_from_angle_deg(command_heading_deg)
            bearing_deg = float(command_heading_deg)
            arrival_heading_deg = float(command_heading_deg)

        else:
            rep_xy_raw = np.asarray(cand["rep_xy"], dtype=np.float32).copy()
            rep_xy_norm = rep_xy_raw.copy()

            if normalize_xy is not None:
                rep_xy_norm[0] /= float(normalize_xy[0])
                rep_xy_norm[1] /= float(normalize_xy[1])

            meet_elapsed_time = float(cand["meet_time_abs"] - current_time)


            sin_ch, cos_ch = _heading_feature_pair_from_angle_deg(command_heading_deg)

            bearing_deg = _bearing_deg_from_two_points(
                command_xy=command_xy_raw,
                goal_xy=rep_xy_raw,
                fallback_heading_deg=command_heading_deg,
            )

            if surrogate_mode in ("rf_direct", "euclidean"):
                arrival_heading_deg = float(bearing_deg)
            else:
                if ("arrival_heading_deg" in cand) and np.isfinite(cand["arrival_heading_deg"]):
                    arrival_heading_deg = float(cand["arrival_heading_deg"])
                else:
                    arrival_heading_deg = float(bearing_deg)

        sin_gb, cos_gb = _heading_feature_pair_from_angle_deg(bearing_deg)
        sin_ah, cos_ah = _heading_feature_pair_from_angle_deg(arrival_heading_deg)

        if normalize_t is not None:
            meet_elapsed_time /= float(normalize_t)

        rows.append([
            command_xy_norm[0],
            command_xy_norm[1],
            meet_elapsed_time,
            sin_ch,
            cos_ch,
            sin_ah,
            cos_ah,
        ])

    return np.asarray(rows, dtype=np.float32)

def build_actor_critic_inputs_from_env(env, state: Optional[Dict] = None) -> Dict[str, Any]:
    if state is None:
        state = env.get_state()

    usv_node_features, alive_mask = state_to_usv_tensor(state)
    precomputed = state["precomputed_shortest"]

    intercept_candidates = compute_all_usv_intercept_candidates(
        env,
        precomputed=precomputed,
    )

    rep_node_indices = get_representative_waypoint_indices_from_intercepts(
        patrols=env.patrols,
        intercept_candidates=intercept_candidates,
        points=env.points,
    )

    actor_query_features = build_actor_query_features_from_intercepts(
        patrols=env.patrols,
        intercept_candidates=intercept_candidates,
        command_xy=state["command_xy"],
        command_heading_deg=state["command_heading_deg"],
        current_time=state["command_time"],
        normalize_xy=env.config.normalize_xy,
        normalize_t=env.config.normalize_t,
    )

    critic_query_features = build_critic_query_features_from_state(
        env,
        state,
        precomputed=precomputed,
        normalize_xy=env.config.normalize_xy,
        normalize_t=env.config.normalize_t,
    )

    U, W, _ = usv_node_features.shape

    rf_bias = np.asarray(state["bias_base_mat"], dtype=np.float32)       # (N,N)
    same_usv_mask = np.asarray(state["same_usv_mask"], dtype=np.float32) # (N,N)
    dead_waypoint_mask = build_dead_waypoint_mask_from_alive(
        alive_mask=alive_mask,
        num_waypoints=W,
        mask_value=-1e9,
    )

    attn_hard_bias = same_usv_mask + dead_waypoint_mask
    attn_soft_bias=rf_bias
    return {
        "usv_node_features": usv_node_features,
        "rep_node_indices": rep_node_indices,
        "actor_query_features": actor_query_features,
        "critic_query_features": critic_query_features,
        "alive_mask": alive_mask,
        "attn_hard_bias": attn_hard_bias,
        "attn_soft_bias": attn_soft_bias,
        "intercept_candidates": intercept_candidates,
        "state_before": state,
        "precomputed_shortest": precomputed,
    }
def actor_critic_inputs_to_torch(inputs: Dict[str, Any], device: str = "cpu") -> Dict[str, torch.Tensor]:
    return {
        "usv_node_features": torch.tensor(inputs["usv_node_features"], dtype=torch.float32, device=device),
        "rep_node_indices": torch.tensor(inputs["rep_node_indices"], dtype=torch.long, device=device),
        "actor_query_features": torch.tensor(inputs["actor_query_features"], dtype=torch.float32, device=device),
        "critic_query_features": torch.tensor(inputs["critic_query_features"], dtype=torch.float32, device=device),
        "alive_mask": torch.tensor(inputs["alive_mask"], dtype=torch.bool, device=device),
        "attn_hard_bias": torch.tensor(inputs["attn_hard_bias"], dtype=torch.float32, device=device),
        "attn_soft_bias": torch.tensor(inputs["attn_soft_bias"], dtype=torch.float32, device=device),

    }


def build_waypoint_absolute_visit_schedule(
    patrols: List[Dict],
    sim_result: Dict,
    cost_e: np.ndarray,
    *,
    num_laps_for_state_schedule: int = 20,
) -> Dict[int, Dict[int, List[float]]]:
    """
    문제 생성 시 한 번만 계산:
    각 USV의 각 waypoint(local index)에 대해 absolute visit times 저장

    Returns:
      schedule[usv_idx][local_wp_idx] = [t_abs0, t_abs1, ...]
    """
    out: Dict[int, Dict[int, List[float]]] = {}

    for usv_idx, patrol in enumerate(patrols):
        node_path = sim_result["usv_node_paths"][usv_idx]
        edge_path = sim_result["usv_edge_paths"][usv_idx]
        wp_vids = patrol["waypoint_vids"]

        visit_schedule = build_usv_node_visit_schedule(
            node_path=node_path,
            edge_path=edge_path,
            cost_e=cost_e,
            num_laps=num_laps_for_state_schedule,
        )

        out[usv_idx] = {}
        for local_wp_idx, vid in enumerate(wp_vids):
            out[usv_idx][local_wp_idx] = visit_schedule.get(int(vid), []).copy()

    return out

def build_problem_pairwise_rf_cache(
    patrols: List[Dict],
    points: np.ndarray,
    blocked_mask: np.ndarray,
    total_ux: np.ndarray,
    total_uy: np.ndarray,
    rf_model,
    rf_feature_names: List[str],
    *,
    usv_speed: float,
    residual_mode: str = "percent",
    inner_width: float = 90.0,
    outer_width: float = 180.0,
) -> Dict[str, Any]:
    """
    문제 생성 시 한 번만 계산:
      - waypoint list
      - pairwise predicted time matrix
      - pairwise bias base matrix
      - same usv mask
    """
    wp_list, pred_time_mat = build_prediction_time_matrix_from_rf(
        patrols=patrols,
        points=points,
        blocked_mask=blocked_mask,
        total_ux=total_ux,
        total_uy=total_uy,
        rf_model=rf_model,
        rf_feature_names=rf_feature_names,
        usv_speed=usv_speed,
        residual_mode=residual_mode,
        inner_width=inner_width,
        outer_width=outer_width,
        exclude_same_usv=True,
    )

    usv_id_per_node = np.asarray([meta["usv_idx"] for meta in wp_list], dtype=np.int64)
    pred_time_mat = np.asarray(pred_time_mat, dtype=np.float32).copy()
    pred_time_mat = np.nan_to_num(pred_time_mat, nan=0.0, posinf=1e6, neginf=0.0)

    bias_base_mat = -pred_time_mat / 2000.0 #normalize_t
    same_usv_mask = build_same_usv_mask_numpy(
        usv_id_per_node,
        allow_self=True,
        mask_value=-1e9,
    )

    return {
        "wp_list": wp_list,
        "usv_id_per_node": usv_id_per_node,
        "pred_time_mat": pred_time_mat,
        "bias_base_mat": bias_base_mat,
        "same_usv_mask": same_usv_mask,
    }
def build_prediction_time_matrix_from_distance(
    *,
    wp_list: List[Dict],
    usv_speed: float,
    exclude_same_usv: bool = True,
) -> np.ndarray:
    N = len(wp_list)
    xy = np.asarray([meta["xy"] for meta in wp_list], dtype=np.float32)   # (N,2)

    diff = xy[:, None, :] - xy[None, :, :]                                # (N,N,2)
    dist = np.linalg.norm(diff, axis=-1)                                  # (N,N)

    pred_time_mat = dist / float(usv_speed)

    if exclude_same_usv:
        usv_ids = np.asarray([meta["usv_idx"] for meta in wp_list], dtype=np.int64)
        same = (usv_ids[:, None] == usv_ids[None, :])
        pred_time_mat = pred_time_mat.copy()
        pred_time_mat[same] = 0.0   # same_usv_mask가 따로 막아줄 거라 값은 0 둬도 됨

    return pred_time_mat.astype(np.float32)
def build_waypoint_node_features_from_precomputed_schedule(
    *,
    wp_list: List[Dict],
    absolute_visit_schedule: Dict[int, Dict[int, List[float]]],
    current_time: float,
    num_future_times: int = 5,
    as_residual: bool = True,
    normalize_xy: Optional[Tuple[float, float]] = None,
    normalize_t: Optional[float] = None,
    due_dates_per_usv: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    문제 생성 시 미리 계산된 absolute visit schedule을 사용해
    step마다 빠르게 node feature 생성

    Returns:
      node_features: (N_wp, 2 + num_future_times)
      usv_id_per_node: (N_wp,)
    """
    node_rows = []
    usv_id_per_node = []

    for meta in wp_list:
        usv_idx = meta["usv_idx"]
        local_wp_idx = meta["local_wp_idx"]
        xy = np.asarray(meta["xy"], dtype=float).copy()

        times = absolute_visit_schedule[usv_idx][local_wp_idx]

        future_arr = select_future_visit_times(
            times=times,
            current_time=current_time,
            num_future_times=num_future_times,
            as_residual=as_residual,
        )

        if normalize_xy is not None:
            xy[0] /= float(normalize_xy[0])
            xy[1] /= float(normalize_xy[1])

        if normalize_t is not None:
            finite_mask = np.isfinite(future_arr)
            future_arr = future_arr.copy()
            future_arr[finite_mask] /= float(normalize_t)

        row = np.concatenate([xy, future_arr])
        if due_dates_per_usv is not None:
            due_date = float(due_dates_per_usv[usv_idx])
            due_remaining = due_date - float(current_time)
            due_remaining_norm = due_remaining / float(normalize_t)
            row = np.concatenate([row, np.asarray([due_remaining_norm], dtype=float)])

        node_rows.append(row)
        usv_id_per_node.append(usv_idx)

    return np.asarray(node_rows, dtype=float), np.asarray(usv_id_per_node, dtype=np.int64)

def select_action_greedy_earliest(env, state: Dict) -> int:
    """
    가장 먼저 만날 수 있는 USV를 선택.
    기준: surrogate/exact 입력 생성 함수가 만든 intercept candidate의 meet_time_abs 최소
    """
    inputs = build_actor_critic_inputs_from_env(env, state)
    candidates = inputs["intercept_candidates"]

    best_action = None
    best_meet_time = float("inf")

    for u, cand in enumerate(candidates):
        if not state["alive_mask"][u]:
            continue
        if cand is None:
            continue

        # exact candidate
        if "usv_arrival_time_abs" in cand:
            meet_time = float(cand["usv_arrival_time_abs"])
        elif "meet_time_abs" in cand:
            meet_time = float(cand["meet_time_abs"])
        else:
            continue

        if meet_time < best_meet_time:
            best_meet_time = meet_time
            best_action = u

    if best_action is None:
        # fallback: 첫 alive usv
        alive = np.where(state["alive_mask"])[0]
        return int(alive[0])

    return int(best_action)
def select_greedy_action_for_eval(
    env,
    state,
    *,
    greedy_mode: str = "meet_time",
    use_surrogate: bool = False,
):
    """
    현재 state에서 greedy action과 state_cache를 반환한다.
    run_one_eval_episode 안의 use_greedy 분기를 함수로 뺀 버전.
    """

    if greedy_mode == "euclidean":
        action = select_euclidean_greedy_action(env, state)

        if use_surrogate:
            state_cache = build_surrogate_actor_critic_inputs_from_env(env, state)
        else:
            state_cache = build_actor_critic_inputs_from_env(env, state)

        return int(action), state_cache


    elif greedy_mode == "meet_time_roll2":
        action = select_rolling_greedy_action_len2(
            env,
            state,
            greedy_mode="meet_time",
            use_surrogate=use_surrogate,
        )

        if use_surrogate:
            state_cache = build_surrogate_actor_critic_inputs_from_env(env, state)
        else:
            state_cache = build_actor_critic_inputs_from_env(env, state)

        return int(action), state_cache

    elif greedy_mode == "euclidean_roll2":
        action = select_rolling_greedy_action_len2(
            env,
            state,
            greedy_mode="euclidean",
            use_surrogate=use_surrogate,
        )

        if use_surrogate:
            state_cache = build_surrogate_actor_critic_inputs_from_env(env, state)
        else:
            state_cache = build_actor_critic_inputs_from_env(env, state)

        return int(action), state_cache

    elif greedy_mode == "meet_time":
        if use_surrogate:
            inputs = build_surrogate_actor_critic_inputs_from_env(env, state)
            candidates = inputs["surrogate_candidates"]
        else:
            inputs = build_actor_critic_inputs_from_env(env, state)
            candidates = inputs["intercept_candidates"]

        best_action = None
        best_meet_time = float("inf")

        for u, cand in enumerate(candidates):
            if cand is None:
                continue
            if not bool(state["alive_mask"][u]):
                continue

            if "meet_time_abs" in cand:
                meet_time = float(cand["meet_time_abs"])
            elif "usv_arrival_time_abs" in cand:
                meet_time = float(cand["usv_arrival_time_abs"])
            else:
                continue

            if meet_time < best_meet_time:
                best_meet_time = meet_time
                best_action = int(u)

        if best_action is None:
            alive = np.where(state["alive_mask"])[0]
            action = int(alive[0])
        else:
            action = int(best_action)

        return int(action), inputs

    else:
        raise ValueError(f"Unknown greedy_mode: {greedy_mode}")
def rollout_greedy_return_from_current_state(
    env,
    *,
    greedy_mode: str = "meet_time",
    use_surrogate: bool = False,
    gamma: float = 1.0,
    max_steps: int = 10_000,
):
    """
    현재 env 상태에서 greedy로 끝까지 갔을 때의 discounted return 계산.
    이 함수 자체는 env를 진행시키므로, 바깥에서 snapshot/restore로 감싸야 한다.
    """

    state = env.get_state()
    done = False
    total_return = 0.0
    discount = 1.0
    step = 0

    while not done:
        if step >= max_steps:
            raise RuntimeError("greedy rollout exceeded max_steps. Possible infinite loop.")

        action, state_cache = select_greedy_action_for_eval(
            env,
            state,
            greedy_mode=greedy_mode,
            use_surrogate=use_surrogate,
        )

        next_state, reward, done, info = env.step(action, state_cache=state_cache)

        total_return += discount * float(reward)
        discount *= gamma

        state = next_state
        step += 1

    return float(total_return), int(step)

def run_one_eval_episode(
    env,
    model,
    device: str = "cpu",
    deterministic: bool = False,
    use_greedy: bool = False,
    use_surrogate: bool = False,
    return_trace: bool = False,
    greedy_mode: str = "meet_time", #euclidean
):
    state = env.get_state()
    done = False
    ep_reward = 0.0
    tardiness_penalty_sum = 0.0
    step_count = 0

    command_trace_nodes = [int(state["command_vid"])]
    action_trace = []

    while not done:
        if use_greedy:
            if greedy_mode == "euclidean":
                action = select_euclidean_greedy_action(env, state)

                if use_surrogate:
                    state_cache = build_surrogate_actor_critic_inputs_from_env(env, state)
                else:
                    state_cache = build_actor_critic_inputs_from_env(env, state)

            elif greedy_mode == "roll2_greedy":
                action = select_rolling_greedy_action_len2(
                    env,
                    state,
                    greedy_mode="meet_time_tardiness",
                    use_surrogate=use_surrogate,
                )

                if use_surrogate:
                    state_cache = build_surrogate_actor_critic_inputs_from_env(env, state)
                else:
                    state_cache = build_actor_critic_inputs_from_env(env, state)

            elif greedy_mode == "roll2_tardy_first_greedy":
                action = select_rolling_greedy_action_len2(
                    env,
                    state,
                    greedy_mode="tardy_first_greedy",
                    use_surrogate=use_surrogate,
                )

                if use_surrogate:
                    state_cache = build_surrogate_actor_critic_inputs_from_env(env, state)
                else:
                    state_cache = build_actor_critic_inputs_from_env(env, state)

            elif greedy_mode == "meet_time_tardiness":
                action, state_cache = select_meet_time_with_tardiness_greedy_action(
                    env,
                    state,
                    use_surrogate=use_surrogate,
                    normalize_score=True,
                )

            elif greedy_mode == "tardy_first_greedy":
                action, state_cache = select_action_greedy_tardy_first_earliest(
                    env,
                    state,
                )

            elif greedy_mode == "meet_time":
                if use_surrogate:
                    inputs = build_surrogate_actor_critic_inputs_from_env(env, state)
                    candidates = inputs["surrogate_candidates"]
                else:
                    inputs = build_actor_critic_inputs_from_env(env, state)
                    candidates = inputs["intercept_candidates"]

                best_action = None
                best_meet_time = float("inf")

                for u, cand in enumerate(candidates):
                    if cand is None:
                        continue
                    if not bool(state["alive_mask"][u]):
                        continue

                    if "meet_time_abs" in cand:
                        meet_time = float(cand["meet_time_abs"])
                    elif "usv_arrival_time_abs" in cand:
                        meet_time = float(cand["usv_arrival_time_abs"])
                    else:
                        continue

                    if meet_time < best_meet_time:
                        best_meet_time = meet_time
                        best_action = int(u)

                if best_action is None:
                    alive = np.where(state["alive_mask"])[0]
                    action = int(alive[0])
                else:
                    action = int(best_action)

                state_cache = inputs

            else:
                raise ValueError(f"Unknown greedy_mode: {greedy_mode}")

        else:
            if use_surrogate:
                inputs_np = build_surrogate_actor_critic_inputs_from_env(env, state)
            else:
                inputs_np = build_actor_critic_inputs_from_env(env, state)

            inputs_t = actor_critic_inputs_to_torch(inputs_np, device=device)

            with torch.no_grad():
                out = model(
                    usv_node_features=inputs_t["usv_node_features"].unsqueeze(0),
                    rep_node_indices=inputs_t["rep_node_indices"].unsqueeze(0),
                    actor_query_features=inputs_t["actor_query_features"].unsqueeze(0),
                    critic_query_features=inputs_t["critic_query_features"].unsqueeze(0),
                    alive_mask=inputs_t["alive_mask"].unsqueeze(0),
                    attn_hard_bias=inputs_t["attn_hard_bias"].unsqueeze(0),
                    attn_soft_bias=inputs_t["attn_soft_bias"].unsqueeze(0),
                )

            logits = out["logits"]

            if deterministic:
                action = torch.argmax(logits, dim=-1).item()
            else:
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample().item()

            state_cache = inputs_np

        action_trace.append(int(action))

        state_before = state

        if use_surrogate:
            cand = None
            if state_cache is not None and "surrogate_candidates" in state_cache:
                cand = state_cache["surrogate_candidates"][int(action)]

            if cand is not None and env.config.surrogate_mode in ("astar_euclidean", "astar_rf"):
                segment_nodes = reconstruct_surrogate_astar_command_segment(
                    env=env,
                    state=state_before,
                    cand=cand,
                    heuristic_mode=None,
                    fallback_to_direct=True,
                )
            elif cand is not None:
                segment_nodes = [
                    int(state_before["command_vid"]),
                    int(cand["rep_vertex_idx"]),
                ]
            else:
                segment_nodes = [int(state_before["command_vid"])]

            next_state, reward, done, info = env.step(action, state_cache=state_cache)

        else:
            next_state, reward, done, info = env.step(action, state_cache=state_cache)

            if "intercept" in info:
                meet_node = int(info["intercept"]["meet_node"])
            else:
                meet_node = int(env.command_vid)

            segment_nodes = _reconstruct_command_segment_from_state(
                env=env,
                state=state_before,
                meet_node=meet_node,
            )

        _append_segment_to_trace(command_trace_nodes, segment_nodes)

        state = next_state
        ep_reward += float(reward)
        tardiness_penalty_sum += float(info.get("tardiness_penalty", 0.0))
        step_count += 1

    success = bool(np.all(~env.alive_mask))

    out = {
        "reward": float(ep_reward),
        "tardiness_penalty": float(tardiness_penalty_sum),
        "length": int(step_count),
        "success": success,
    }

    if return_trace:
        out["command_trace_nodes"] = command_trace_nodes
        out["actions"] = action_trace

    return out
def compute_candidate_incremental_tardiness(
    *,
    due_dates_per_usv: np.ndarray,
    alive_mask: np.ndarray,
    old_time: float,
    new_time: float,
) -> float:
    """
    candidate action으로 old_time -> new_time까지 시간이 진행될 때,
    alive 상태였던 모든 USV에서 새로 발생하는 tardiness 총량.

    incremental tardiness =
      sum_i alive_i * [
          max(0, new_time - due_i) - max(0, old_time - due_i)
      ]
    """
    due_dates = np.asarray(due_dates_per_usv, dtype=float)
    alive = np.asarray(alive_mask, dtype=bool)

    old_time = float(old_time)
    new_time = float(new_time)

    if new_time < old_time:
        return float("inf")

    old_tardiness = np.maximum(0.0, old_time - due_dates)
    new_tardiness = np.maximum(0.0, new_time - due_dates)

    incremental = new_tardiness - old_tardiness
    return float(np.sum(incremental[alive]))


def select_meet_time_with_tardiness_greedy_action(
    env,
    state: Dict,
    *,
    use_surrogate: bool = False,
    normalize_score: bool = False,
) -> Tuple[int, Dict]:
    """
    meet-time greedy를 확장한 버전.

    기존:
      argmin meet_time_abs

    변경:
      argmin [
          elapsed_meet_time
          + incremental_tardiness_over_alive_usvs
      ]

    여기서 incremental_tardiness는 step 사이에서 새로 발생한 tardiness이다.
    reward와 일관되게 보려면 normalize_t로 나눌 수 있지만,
    모든 candidate에 같은 양수 normalize_t를 나누므로 argmin 결과는 동일하다.
    """

    if use_surrogate:
        inputs = build_surrogate_actor_critic_inputs_from_env(env, state)
        candidates = inputs["surrogate_candidates"]
    else:
        inputs = build_actor_critic_inputs_from_env(env, state)
        candidates = inputs["intercept_candidates"]

    alive_mask = np.asarray(state["alive_mask"], dtype=bool)
    current_time = float(state["command_time"])

    # state에 있으면 state 기준, 없으면 env 기준으로 fallback
    due_dates = state.get("due_dates_per_usv", None)
    if due_dates is None:
        due_dates = getattr(env, "due_dates_per_usv", None)

    best_action = None
    best_score = float("inf")
    best_debug = None

    for u, cand in enumerate(candidates):
        if cand is None:
            continue
        if not bool(alive_mask[u]):
            continue

        # surrogate / exact 둘 다 대응
        if "meet_time_abs" in cand:
            meet_time = float(cand["meet_time_abs"])
        elif "usv_arrival_time_abs" in cand:
            meet_time = float(cand["usv_arrival_time_abs"])
        else:
            continue

        elapsed_meet_time = meet_time - current_time
        if elapsed_meet_time < 0:
            continue

        if due_dates is None:
            incremental_tardiness = 0.0
        else:
            incremental_tardiness = compute_candidate_incremental_tardiness(
                due_dates_per_usv=due_dates,
                alive_mask=alive_mask,
                old_time=current_time,
                new_time=meet_time,
            )

        score = elapsed_meet_time + incremental_tardiness

        if normalize_score:
            score = score / float(env.config.normalize_t)

        if score < best_score:
            best_score = float(score)
            best_action = int(u)
            best_debug = {
                "greedy_score": float(score),
                "elapsed_meet_time": float(elapsed_meet_time),
                "incremental_tardiness": float(incremental_tardiness),
                "meet_time_abs": float(meet_time),
                "due_date": float(due_dates[u]) if due_dates is not None else None,
            }

    if best_action is None:
        alive = np.where(alive_mask)[0]
        best_action = int(alive[0])
        best_debug = {
            "greedy_score": float("inf"),
            "fallback": True,
        }

    inputs["greedy_debug"] = best_debug
    return int(best_action), inputs
def select_action_greedy_earliest(env, state: Dict) -> int:
    """
    가장 먼저 만날 수 있는 USV를 선택.
    기준: surrogate/exact 입력 생성 함수가 만든 intercept candidate의 meet_time_abs 최소
    """
    inputs = build_actor_critic_inputs_from_env(env, state)
    candidates = inputs["intercept_candidates"]

    best_action = None
    best_meet_time = float("inf")

    for u, cand in enumerate(candidates):
        if not state["alive_mask"][u]:
            continue
        if cand is None:
            continue

        # exact candidate
        if "usv_arrival_time_abs" in cand:
            meet_time = float(cand["usv_arrival_time_abs"])
        elif "meet_time_abs" in cand:
            meet_time = float(cand["meet_time_abs"])
        else:
            continue

        if meet_time < best_meet_time:
            best_meet_time = meet_time
            best_action = u

    if best_action is None:
        # fallback: 첫 alive usv
        alive = np.where(state["alive_mask"])[0]
        return int(alive[0])

    return int(best_action)
def select_action_greedy_tardy_first_earliest(
    env,
    state: Dict,
    *,
    use_surrogate: bool = True,
) -> Tuple[int, Dict]:
    """
    Due-date-aware earliest greedy.

    기준:
      1. alive USV 중 due date가 가장 빠른 그룹을 먼저 선택
      2. 그 그룹 안에서 meet_time_abs가 가장 빠른 USV 선택

    주의:
      - use_surrogate=True이면 build_surrogate_actor_critic_inputs_from_env 사용
      - use_surrogate=False이면 build_actor_critic_inputs_from_env 사용
      - run_one_eval_episode에서 state_cache가 필요하므로 (action, inputs)를 반환
    """

    if use_surrogate:
        inputs = build_surrogate_actor_critic_inputs_from_env(env, state)
        candidates = inputs["surrogate_candidates"]
    else:
        inputs = build_actor_critic_inputs_from_env(env, state)
        candidates = inputs["intercept_candidates"]

    alive_mask = np.asarray(state["alive_mask"], dtype=bool)

    # state에 있으면 state 기준, 없으면 env 기준
    due_dates = state.get("due_dates_per_usv", None)
    if due_dates is None:
        due_dates = getattr(env, "due_dates_per_usv", None)

    def _get_meet_time(cand):
        if cand is None:
            return None

        if "meet_time_abs" in cand:
            return float(cand["meet_time_abs"])

        if "usv_arrival_time_abs" in cand:
            return float(cand["usv_arrival_time_abs"])

        return None

    # --------------------------------------------------
    # due date 정보가 없으면 그냥 meet-time greedy
    # --------------------------------------------------
    if due_dates is None:
        best_action = None
        best_meet_time = float("inf")

        for u, cand in enumerate(candidates):
            if not bool(alive_mask[u]):
                continue

            meet_time = _get_meet_time(cand)
            if meet_time is None:
                continue

            if meet_time < best_meet_time:
                best_meet_time = float(meet_time)
                best_action = int(u)

        if best_action is None:
            alive = np.where(alive_mask)[0]
            best_action = int(alive[0])
            inputs["greedy_debug"] = {
                "mode": "tardy_first_meet_time_greedy",
                "fallback": True,
                "reason": "no_due_dates_and_no_valid_candidate",
            }
            return best_action, inputs

        inputs["greedy_debug"] = {
            "mode": "tardy_first_meet_time_greedy",
            "fallback": False,
            "due_dates_available": False,
            "meet_time_abs": float(best_meet_time),
        }
        return int(best_action), inputs

    due_dates = np.asarray(due_dates, dtype=float)

    # --------------------------------------------------
    # 1. alive USV 중 가장 빠른 due date 찾기
    # --------------------------------------------------
    alive_due_dates = due_dates[alive_mask]
    min_due_date = float(np.min(alive_due_dates))

    # 가장 빠른 due date를 가진 alive USV만 후보로 제한
    priority_mask = alive_mask & np.isclose(due_dates, min_due_date)

    # --------------------------------------------------
    # 2. priority group 안에서 가장 빨리 만나는 USV 선택
    # --------------------------------------------------
    best_action = None
    best_meet_time = float("inf")

    for u, cand in enumerate(candidates):
        if not bool(priority_mask[u]):
            continue

        meet_time = _get_meet_time(cand)
        if meet_time is None:
            continue

        if meet_time < best_meet_time:
            best_meet_time = float(meet_time)
            best_action = int(u)

    # --------------------------------------------------
    # 3. priority group 안에 valid candidate가 없으면 전체 alive 중 meet-time fallback
    # --------------------------------------------------
    fallback = False

    if best_action is None:
        fallback = True
        best_meet_time = float("inf")

        for u, cand in enumerate(candidates):
            if not bool(alive_mask[u]):
                continue

            meet_time = _get_meet_time(cand)
            if meet_time is None:
                continue

            if meet_time < best_meet_time:
                best_meet_time = float(meet_time)
                best_action = int(u)

    # --------------------------------------------------
    # 4. 그래도 없으면 첫 alive fallback
    # --------------------------------------------------
    if best_action is None:
        alive = np.where(alive_mask)[0]
        best_action = int(alive[0])
        inputs["greedy_debug"] = {
            "mode": "tardy_first_meet_time_greedy",
            "fallback": True,
            "reason": "no_valid_candidate",
            "min_due_date": float(min_due_date),
        }
        return int(best_action), inputs

    inputs["greedy_debug"] = {
        "mode": "tardy_first_meet_time_greedy",
        "fallback": bool(fallback),
        "selected_usv": int(best_action),
        "selected_due_date": float(due_dates[best_action]),
        "min_due_date": float(min_due_date),
        "meet_time_abs": float(best_meet_time),
        "use_surrogate": bool(use_surrogate),
    }

    return int(best_action), inputs
def _as_xy_path(points, node_path):
    node_path = [int(v) for v in node_path if v is not None and int(v) >= 0]
    if len(node_path) == 0:
        return np.empty((0, 2), dtype=float)
    return np.asarray(points[node_path], dtype=float)


def _append_segment_to_trace(command_trace_nodes, segment_nodes):
    """
    command_trace_nodes에 segment를 이어 붙인다.
    중복되는 시작 노드는 한 번만 남긴다.
    """
    if segment_nodes is None or len(segment_nodes) == 0:
        return

    segment_nodes = [int(v) for v in segment_nodes]

    if len(command_trace_nodes) == 0:
        command_trace_nodes.extend(segment_nodes)
    else:
        if int(command_trace_nodes[-1]) == int(segment_nodes[0]):
            command_trace_nodes.extend(segment_nodes[1:])
        else:
            command_trace_nodes.extend(segment_nodes)


def _reconstruct_command_segment_from_state(env, state, meet_node):
    """
    exact mode에서 현재 command_vid -> meet_node까지의 실제 Dijkstra node path 복원.
    실패하면 시작점과 도착점만 반환한다.
    """
    start_vid = int(state["command_vid"])
    meet_node = int(meet_node)

    if start_vid == meet_node:
        return [start_vid]

    try:
        precomputed = state["precomputed_shortest"]

        best_k = int(precomputed["cmd_best_k"][meet_node])
        if best_k < 0:
            return [start_vid, meet_node]

        node_path, edge_path = reconstruct_path_from_prev(
            cache=env.cache,
            prev_node=precomputed["prev_node_cmd"],
            prev_k=precomputed["prev_k_cmd"],
            start=start_vid,
            goal=meet_node,
            best_goal_k=best_k,
        )

        if node_path is None or len(node_path) == 0:
            return [start_vid, meet_node]

        return [int(v) for v in node_path]

    except Exception:
        return [start_vid, meet_node]


from matplotlib.lines import Line2D
def plot_eval_episode_overview(
    env,
    command_trace_nodes,
    *,
    seed=None,
    rep=None,
    actions=None,
    save_path=None,
    current_stride=4,
    figsize=(12, 8),
    show=True,
):
    """
    하나의 episode에 대해 다음 항목을 하나의 그림에 표시한다.

    1) 해류 vector field
    2) 장애물
    3) 정찰 USV patrol route
    4) due date 기반 USV 우선순위
    5) 지휘함 전체 이동 경로

    우선순위는 env.due_dates_per_usv를 기준으로 자동 설정한다.
    작은 due date일수록 높은 우선순위이며 Priority 1로 표시된다.
    """

    # --------------------------------------------------
    # 기본 데이터
    # --------------------------------------------------
    points = np.asarray(env.points, dtype=float)

    total_ux = np.asarray(env.total_ux, dtype=float)
    total_uy = np.asarray(env.total_uy, dtype=float)

    sim_result = env.sim_result
    usv_node_paths = sim_result["usv_node_paths"]
    num_usv = len(usv_node_paths)

    # --------------------------------------------------
    # 0) due date 기반 priority 구성
    # --------------------------------------------------
    due_dates = np.asarray(
        env.due_dates_per_usv,
        dtype=float,
    )

    if len(due_dates) != num_usv:
        raise ValueError(
            f"due_dates_per_usv 길이({len(due_dates)})와 "
            f"USV 수({num_usv})가 일치하지 않습니다."
        )

    # due date가 작을수록 높은 우선순위
    unique_due_dates = np.sort(np.unique(due_dates))

    due_to_priority = {
        due_date: priority_idx + 1
        for priority_idx, due_date in enumerate(unique_due_dates)
    }

    priorities = np.asarray(
        [due_to_priority[due_date] for due_date in due_dates],
        dtype=int,
    )

    unique_priorities = np.sort(np.unique(priorities))

    # 우선순위별 시각적 스타일
    priority_cmap = plt.get_cmap("tab10")

    priority_colors = {
        priority: priority_cmap(i % 10)
        for i, priority in enumerate(unique_priorities)
    }

    # 높은 우선순위일수록 굵게 표시
    priority_linewidths = {
        priority: max(1.3, 3.2 - 0.55 * i)
        for i, priority in enumerate(unique_priorities)
    }

    priority_markers = [
        "o",
        "s",
        "^",
        "D",
        "v",
        "P",
        "*",
        "h",
    ]

    priority_marker_map = {
        priority: priority_markers[i % len(priority_markers)]
        for i, priority in enumerate(unique_priorities)
    }

    fig, ax = plt.subplots(figsize=figsize)

    # --------------------------------------------------
    # 1) current field
    # --------------------------------------------------
    free_mask = np.ones(len(points), dtype=bool)

    if getattr(env, "blocked_mask", None) is not None:
        free_mask = ~np.asarray(
            env.blocked_mask,
            dtype=bool,
        )

    idx = np.where(free_mask)[0][::current_stride]

    ax.quiver(
        points[idx, 0],
        points[idx, 1],
        total_ux[idx],
        total_uy[idx],
        angles="xy",
        scale_units="xy",
        scale=0.006,
        width=0.002,
        alpha=0.35,
        color="deepskyblue",
        label="Current direction",
        zorder=0,
    )

    # --------------------------------------------------
    # 2) obstacles
    # --------------------------------------------------
    if getattr(env, "obstacles", None) is not None:
        for obs in env.obstacles:

            # dict와 dataclass/object 모두 대응
            if isinstance(obs, dict):
                x0 = obs.get("x0", obs.get("x", None))
                y0 = obs.get("y0", obs.get("y", None))
                radius = obs.get(
                    "radius",
                    obs.get("r", None),
                )
            else:
                x0 = getattr(
                    obs,
                    "x0",
                    getattr(obs, "x", None),
                )
                y0 = getattr(
                    obs,
                    "y0",
                    getattr(obs, "y", None),
                )
                radius = getattr(
                    obs,
                    "radius",
                    getattr(obs, "r", None),
                )

            if (
                x0 is not None
                and y0 is not None
                and radius is not None
            ):
                circle = Circle(
                    (float(x0), float(y0)),
                    float(radius),
                    facecolor="lightgray",
                    edgecolor="black",
                    linewidth=1.0,
                    alpha=0.40,
                    zorder=1,
                )
                ax.add_patch(circle)

    # --------------------------------------------------
    # 3) priority-aware patrol routes
    # --------------------------------------------------
    for usv_idx, node_path in enumerate(usv_node_paths):
        patrol_xy = _as_xy_path(points, node_path)

        if len(patrol_xy) == 0:
            continue

        priority = priorities[usv_idx]
        due_date = due_dates[usv_idx]

        route_color = priority_colors[priority]
        route_width = priority_linewidths[priority]
        route_marker = priority_marker_map[priority]

        # 전체 patrol route
        ax.plot(
            patrol_xy[:, 0],
            patrol_xy[:, 1],
            color=route_color,
            linewidth=route_width,
            alpha=0.80,
            zorder=3,
        )

        # patrol waypoint
        if getattr(env, "patrols", None) is not None:
            patrol_info = env.patrols[usv_idx]

            if isinstance(patrol_info, dict):
                wp_vids = patrol_info.get(
                    "waypoint_vids",
                    [],
                )
            else:
                wp_vids = getattr(
                    patrol_info,
                    "waypoint_vids",
                    [],
                )

            wp_xy = _as_xy_path(points, wp_vids)

            if len(wp_xy) > 0:
                ax.scatter(
                    wp_xy[:, 0],
                    wp_xy[:, 1],
                    s=38,
                    marker=route_marker,
                    facecolor=route_color,
                    edgecolor="white",
                    linewidth=0.7,
                    alpha=0.95,
                    zorder=5,
                )

        # patrol 시작점 강조
        ax.scatter(
            patrol_xy[0, 0],
            patrol_xy[0, 1],
            s=85,
            marker=route_marker,
            facecolor=route_color,
            edgecolor="black",
            linewidth=0.9,
            zorder=6,
        )

        # USV 번호와 priority 표시
        ax.annotate(
            f"USV {usv_idx}",
            xy=(
                patrol_xy[0, 0],
                patrol_xy[0, 1],
            ),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
            fontweight=(
                "bold"
                if priority == unique_priorities[0]
                else "normal"
            ),
            color=route_color,
            bbox=dict(
                boxstyle="round,pad=0.20",
                facecolor="white",
                edgecolor=route_color,
                alpha=0.80,
            ),
            zorder=7,
        )

    # --------------------------------------------------
    # 4) command ship trajectory
    # --------------------------------------------------
    command_xy = _as_xy_path(
        points,
        command_trace_nodes,
    )

    if len(command_xy) > 0:
        ax.plot(
            command_xy[:, 0],
            command_xy[:, 1],
            color="black",
            linewidth=3.0,
            markeredgecolor="black",
            linestyle="-",
            label="Command ship trajectory",
            zorder=8,
        )

        # 시작점
        ax.scatter(
            command_xy[0, 0],
            command_xy[0, 1],
            s=150,
            marker="p",
            facecolor="limegreen",
            edgecolor="black",
            linewidth=1.0,
            label="Command start",
            zorder=10,
        )

        # 종료점
        ax.scatter(
            command_xy[-1, 0],
            command_xy[-1, 1],
            s=150,
            marker="X",
            facecolor="red",
            edgecolor="black",
            linewidth=1.0,
            label="Command end",
            zorder=10,
        )

    # --------------------------------------------------
    # 5) priority legend
    # --------------------------------------------------
    priority_handles = []

    for priority in unique_priorities:
        corresponding_due_date = unique_due_dates[priority - 1]

        priority_handles.append(
            Line2D(
                [0],
                [0],
                color=priority_colors[priority],
                linewidth=priority_linewidths[priority],
                marker=priority_marker_map[priority],
                markerfacecolor=priority_colors[priority],
                markeredgecolor="black",
                markersize=7,
                label=(
                    f"Priority {priority} "
                ),
            )
        )

    trajectory_handles, trajectory_labels = (
        ax.get_legend_handles_labels()
    )

    all_handles = priority_handles + trajectory_handles
    all_labels = [
        handle.get_label()
        for handle in priority_handles
    ] + trajectory_labels

    ax.legend(
        all_handles,
        all_labels,
        loc="upper right",
        fontsize=8,
        framealpha=0.90,
    )

    # --------------------------------------------------
    # 6) title / style
    # --------------------------------------------------

    ax.set_xlabel("x position")
    ax.set_ylabel("y position")
    ax.set_aspect(
        "equal",
        adjustable="box",
    )
    ax.grid(
        True,
        alpha=0.25,
    )

    plt.tight_layout()

    # --------------------------------------------------
    # 7) save / show
    # --------------------------------------------------
    if save_path is not None:
        save_dir = os.path.dirname(save_path)

        if save_dir:
            os.makedirs(
                save_dir,
                exist_ok=True,
            )

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

        print(f"[plot saved] {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

def evaluate_policy_on_fixed_testset_multi(
    env,
    model,
    test_problem_seeds,
    *,
    test_problem_bank: Dict[int, Dict],
    num_repeats_per_problem: int = 5,
    device: str = "cpu",
    deterministic: bool = False,
    use_greedy: bool = False,
    use_surrogate: bool = False,

    # 추가 옵션
    plot_best_case: bool = False,
    plot_seed_index: int = 1,
    plot_save_dir: str = "eval_episode_plots",
    plot_show: bool = False,
    greedy_mode: str = "meet_time",
    print_gate: Optional[bool] = None,
):
    problem_level_best_rewards = []
    problem_level_mean_rewards = []
    problem_level_best_lengths = []
    problem_level_mean_lengths = []
    problem_level_best_success = []
    problem_level_mean_success = []
    problem_level_mean_times = []
    problem_level_tardy_penalty=[]
    detailed_results = []

    for seed_idx, seed in enumerate(test_problem_seeds):
        if ( seed_idx!=plot_seed_index) and plot_best_case:
            continue
        run_rewards = []
        run_lengths = []
        run_success = []
        run_traces = []  # 추가
        run_actions = []  # 추가
        run_times=[]
        run_tardiness=[]

        # 문제 생성 대신 bank에서 로드
        env.load_problem_bundle(test_problem_bank[int(seed)])

        for rep in range(num_repeats_per_problem):
            # 같은 문제 위에서 episode state만 reset
            env.reset(resample_problem=False)
            st=time.time()
            ep_result = run_one_eval_episode(
                env=env,
                model=model,
                device=device,
                deterministic=deterministic,
                use_greedy=use_greedy,
                use_surrogate=use_surrogate,
                return_trace=True,  # 항상 trace 저장
                greedy_mode=greedy_mode,
            )
            cal_time=time.time()-st
            run_rewards.append(ep_result["reward"])
            run_lengths.append(ep_result["length"])
            run_success.append(float(ep_result["success"]))
            run_traces.append(ep_result["command_trace_nodes"])  # 추가
            run_actions.append(ep_result["actions"])  # 추가
            run_tardiness.append(ep_result["tardiness_penalty"])
            run_times.append(cal_time)
        run_rewards = np.asarray(run_rewards, dtype=float)
        run_lengths = np.asarray(run_lengths, dtype=float)
        run_success = np.asarray(run_success, dtype=float)
        run_tardiness=np.asarray(run_tardiness,dtype=float)
        run_times=np.asarray(run_times,dtype=float)
        best_idx = int(np.argmax(run_rewards))
        mean_reward = np.mean(run_rewards)
        mean_idx = int(np.argmin(np.abs(run_rewards - mean_reward)))

        # --------------------------------------------------
        # 하나의 문제에 대해 rep 중 best case만 plot
        # --------------------------------------------------
        should_plot_this_problem = (
                plot_best_case
                and int(seed_idx) == int(plot_seed_index)
        )

        if should_plot_this_problem:
            save_path = os.path.join(
                plot_save_dir,
                f"best_rep_seed_{int(seed)}_rep_{int(mean_idx)}_reward_{float(run_rewards[mean_idx]):.4f}.png"
            )

            plot_eval_episode_overview(
                env,
                run_traces[mean_idx],
                seed=int(seed),
                rep=int(mean_idx),
                actions=run_actions[mean_idx],
                save_path=save_path,
                current_stride=2,
                show=plot_show,
            )

            print(
                f"[best rep plotted] "
                f"seed={int(seed)}, "
                f"best_rep={int(mean_idx)}, "
                f"best_reward={float(run_rewards[mean_idx]):.6f}, "
                f"success={float(run_success[mean_idx]):.1f}"
            )

        problem_level_best_rewards.append(float(run_rewards[best_idx]))
        problem_level_mean_rewards.append(float(np.mean(run_rewards)))
        problem_level_best_lengths.append(int(run_lengths[best_idx]))
        problem_level_mean_lengths.append(float(np.mean(run_lengths)))
        problem_level_best_success.append(float(run_success[best_idx]))
        problem_level_mean_success.append(float(np.mean(run_success)))
        problem_level_mean_times.append(float(np.mean(run_times)))
        problem_level_tardy_penalty.append(float(np.mean(run_tardiness)))
        detailed_results.append({
            "seed": int(seed),
            "run_rewards": run_rewards.tolist(),
            "run_lengths": run_lengths.tolist(),
            "run_success": run_success.tolist(),
            "best_idx": int(best_idx),
            "best_reward": float(run_rewards[best_idx]),
            "best_length": int(run_lengths[best_idx]),
            "best_success": float(run_success[best_idx]),
            "average_time": float(np.mean(run_times)),
            "tardiness_reward":float(np.mean(run_tardiness)),
        })
    return {
        "best_mean_reward": float(np.mean(problem_level_best_rewards)),
        "best_std_reward": float(np.std(problem_level_best_rewards)),
        "mean_mean_reward": float(np.mean(problem_level_mean_rewards)),
        "mean_std_reward": float(np.std(problem_level_mean_rewards)),
        "best_mean_length": float(np.mean(problem_level_best_lengths)),
        "mean_mean_length": float(np.mean(problem_level_mean_lengths)),
        "best_success_rate": float(np.mean(problem_level_best_success)),
        "mean_success_rate": float(np.mean(problem_level_mean_success)),
        "tardiness_reward":float(np.mean(problem_level_tardy_penalty)),
        "details": detailed_results,
    }

def evaluate_policy_test_version(
    env,
    model,
    test_problem_seeds,
    *,
    test_problem_bank: Dict[int, Dict],
    num_repeats_per_problem: int = 5,
    device: str = "cpu",
    deterministic: bool = False,
    use_greedy: bool = False,
    use_surrogate: bool = False,

    # plot 옵션
    plot_best_case: bool = False,
    plot_seed_index: int = 1,
    plot_save_dir: str = "eval_episode_plots",
    plot_show: bool = False,
    greedy_mode: str = "meet_time",
    print_gate: Optional[bool] = None,
):
    """
    Returns
    -------
    {
        "seeds": List[int],
        "rewards": np.ndarray,
        "tardiness_rewards": np.ndarray,
        "calculation_times": np.ndarray,
    }

    각 배열의 shape:
        (평가한 문제 수, num_repeats_per_problem)
    """

    evaluated_seeds = []
    all_rewards = []
    all_tardiness_rewards = []
    all_calculation_times = []

    for seed_idx, seed in enumerate(test_problem_seeds):
        # 특정 문제의 best case만 plot하려는 경우
        if plot_best_case and seed_idx != plot_seed_index:
            continue

        run_rewards = []
        run_tardiness_rewards = []
        run_calculation_times = []

        # plot에 필요한 정보만 임시 저장
        run_traces = []
        run_actions = []

        # bank에서 고정 문제 로드
        env.load_problem_bundle(
            test_problem_bank[int(seed)]
        )

        for rep in range(num_repeats_per_problem):
            # 같은 문제를 유지한 채 episode 상태만 초기화
            env.reset(resample_problem=False)

            start_time = time.time()

            ep_result = run_one_eval_episode(
                env=env,
                model=model,
                device=device,
                deterministic=deterministic,
                use_greedy=use_greedy,
                use_surrogate=use_surrogate,
                return_trace=plot_best_case,
                greedy_mode=greedy_mode,
            )

            calculation_time = time.time() - start_time

            run_rewards.append(
                float(ep_result["reward"])
            )
            run_tardiness_rewards.append(
                float(ep_result["tardiness_penalty"])
            )
            run_calculation_times.append(
                float(calculation_time)
            )

            if plot_best_case:
                run_traces.append(
                    ep_result["command_trace_nodes"]
                )
                run_actions.append(
                    ep_result["actions"]
                )

        # --------------------------------------------------
        # 선택한 문제에서 reward가 가장 높은 반복만 plot
        # --------------------------------------------------
        if plot_best_case and seed_idx == plot_seed_index:
            best_idx = int(np.argmax(run_rewards))

            os.makedirs(plot_save_dir, exist_ok=True)

            save_path = os.path.join(
                plot_save_dir,
                (
                    f"best_rep_seed_{int(seed)}"
                    f"_rep_{best_idx}"
                    f"_reward_{run_rewards[best_idx]:.4f}.png"
                ),
            )

            plot_eval_episode_overview(
                env,
                run_traces[best_idx],
                seed=int(seed),
                rep=best_idx,
                actions=run_actions[best_idx],
                save_path=save_path,
                current_stride=2,
                show=plot_show,
            )

            print(
                f"[best rep plotted] "
                f"seed={int(seed)}, "
                f"best_rep={best_idx}, "
                f"best_reward={run_rewards[best_idx]:.6f}"
            )

        evaluated_seeds.append(int(seed))
        all_rewards.append(run_rewards)
        all_tardiness_rewards.append(run_tardiness_rewards)
        all_calculation_times.append(run_calculation_times)

    return {
        "seeds": evaluated_seeds,
        "rewards": np.asarray(all_rewards, dtype=float),
        "tardiness_rewards": np.asarray(
            all_tardiness_rewards,
            dtype=float,
        ),
        "calculation_times": np.asarray(
            all_calculation_times,
            dtype=float,
        ),
    }

def rollout_policy_collect_actions(
    env,
    model,
    *,
    device: str = "cpu",
    deterministic: bool = False,
    use_surrogate: bool = True,
):
    """
    현재 env에 이미 문제 bundle이 로드되어 있다고 가정.
    같은 문제에서 episode state만 초기화 후 action sequence 생성.
    """
    state = env.reset(resample_problem=False)
    done = False
    actions = []
    total_reward = 0.0

    while not done:
        if use_surrogate:
            inputs_np = build_surrogate_actor_critic_inputs_from_env(env, state)
        else:
            inputs_np = build_actor_critic_inputs_from_env(env, state)

        inputs_t = actor_critic_inputs_to_torch(inputs_np, device=device)

        with torch.no_grad():
            out = model(
                usv_node_features=inputs_t["usv_node_features"].unsqueeze(0),
                rep_node_indices=inputs_t["rep_node_indices"].unsqueeze(0),
                actor_query_features=inputs_t["actor_query_features"].unsqueeze(0),
                critic_query_features=inputs_t["critic_query_features"].unsqueeze(0),
                alive_mask=inputs_t["alive_mask"].unsqueeze(0),
            )

        logits = out["logits"]

        if deterministic:
            action = torch.argmax(logits, dim=-1).item()
        else:
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample().item()

        actions.append(int(action))

        if use_surrogate:
            state, reward, done, info = env.step(action, state_cache=inputs_np)
        else:
            state, reward, done, info = env.step(action)

        total_reward += reward

    return {
        "actions": actions,
        "rollout_reward": float(total_reward),
        "success": bool(np.all(~env.alive_mask)),
    }
def replay_action_sequence_on_exact_env(
    env,
    action_sequence,
):
    """
    현재 env에 이미 exact problem bundle이 로드되어 있다고 가정.
    같은 문제에서 episode state만 초기화 후 action sequence replay.
    """
    state = env.reset(resample_problem=False)
    total_reward = 0.0
    done = False
    step_count = 0

    for action in action_sequence:
        if done:
            break

        state, reward, done, info = env.step(int(action))
        total_reward += reward
        step_count += 1

    return {
        "exact_reward": float(total_reward),
        "done": bool(done),
        "success": bool(np.all(~env.alive_mask)),
        "num_actions_used": int(step_count),
    }
def evaluate_surrogate_best_then_exact_score(
    surrogate_env,
    exact_env,
    model,
    test_problem_seeds,
    test_problem_bank_surrogate,
    test_problem_bank_exact,
    *,
    num_surrogate_samples: int = 20,
    device: str = "cpu",
    deterministic_surrogate: bool = False,
):
    """
    각 테스트 문제에 대해:
      - surrogate env에서 num_surrogate_samples번 action sequence 샘플링
      - surrogate reward가 가장 좋은 sequence 1개 선택
      - exact env에서 그 sequence만 replay하여 exact reward 채점
    """
    problem_exact_rewards = []
    problem_success = []
    details = []

    for seed in test_problem_seeds:
        #print(seed)
        surrogate_bundle = test_problem_bank_surrogate[int(seed)]
        exact_bundle = test_problem_bank_exact[int(seed)]

        sampled_results = []

        # 1) surrogate에서만 n개 샘플링
        for s in range(num_surrogate_samples):
            #start=time.time()
            surrogate_env.load_problem_bundle(surrogate_bundle)
            roll = rollout_policy_collect_actions(
                surrogate_env,
                model,
                device=device,
                deterministic=deterministic_surrogate,
                use_surrogate=True,
            )

            sampled_results.append({
                "actions": roll["actions"],
                "surrogate_reward": roll["rollout_reward"],
                "surrogate_success": roll["success"],
            })
            #print(time.time()-start)

        # 2) surrogate reward 기준으로 best 선택
        surrogate_rewards = np.array(
            [x["surrogate_reward"] for x in sampled_results], dtype=float
        )
        best_idx = int(np.argmax(surrogate_rewards))
        best_sample = sampled_results[best_idx]

        # 3) 선택된 1개만 exact에서 채점
        exact_env.load_problem_bundle(exact_bundle)
        exact_eval = replay_action_sequence_on_exact_env(
            exact_env,
            best_sample["actions"],
        )

        exact_reward = float(exact_eval["exact_reward"])
        exact_success = float(exact_eval["success"])

        problem_exact_rewards.append(exact_reward)
        problem_success.append(exact_success)

        details.append({
            "seed": int(seed),
            "samples": sampled_results,
            "selected_idx_by_surrogate": best_idx,
            "selected_surrogate_reward": float(best_sample["surrogate_reward"]),
            "selected_surrogate_success": float(best_sample["surrogate_success"]),
            "exact_reward": exact_reward,
            "exact_success": exact_success,
            "num_actions_used": exact_eval["num_actions_used"],
        })

    return {
        "mean_exact_reward": float(np.mean(problem_exact_rewards)),
        "std_exact_reward": float(np.std(problem_exact_rewards)),
        "success_rate": float(np.mean(problem_success)),
        "problem_exact_rewards": problem_exact_rewards,
        "details": details,
    }

def get_arrival_heading_deg_at_node(
    cache,
    theta_e_deg: np.ndarray,
    node_vid: int,
    best_k_in: int,
    fallback_heading_deg: float,
) -> float:
    """
    node_vid에 best_k_in state로 도착했을 때의 heading_deg 반환.
    """
    if best_k_in is None or best_k_in < 0:
        return float(fallback_heading_deg)

    e_in = int(cache.in_edge_id[node_vid, best_k_in])
    if e_in < 0:
        return float(fallback_heading_deg)

    heading_deg = float(theta_e_deg[e_in])
    if not np.isfinite(heading_deg):
        return float(fallback_heading_deg)

    return heading_deg

def build_intra_usv_bias_blocks_from_global_wp_matrix(
    patrols,
    wp_list,
    global_bias_mat: np.ndarray,
    *,
    fill_value: float = -1e9,
) -> np.ndarray:
    """
    global_bias_mat: (N_wp, N_wp)
    returns: (U, W, W)
    """
    U = len(patrols)
    W = len(patrols[0]["waypoint_vids"])

    # wp_list에서 각 usv/local_wp -> global_wp_idx 매핑
    global_idx_of = {}
    for gidx, meta in enumerate(wp_list):
        global_idx_of[(int(meta["usv_idx"]), int(meta["local_wp_idx"]))] = int(gidx)

    out = np.full((U, W, W), fill_value, dtype=np.float32)

    for u in range(U):
        for i in range(W):
            gi = global_idx_of[(u, i)]
            for j in range(W):
                gj = global_idx_of[(u, j)]
                out[u, i, j] = float(global_bias_mat[gi, gj])

    return out

def predict_pair_time_rf_heading0(
    *,
    points: np.ndarray,
    blocked_mask: np.ndarray,
    total_ux: np.ndarray,
    total_uy: np.ndarray,
    start_idx: int,
    goal_idx: int,
    rf_model,
    rf_feature_names,
    usv_speed: float,
    residual_mode: str = "percent",
    inner_width: float = 90.0,
    outer_width: float = 180.0,
) -> float:
    feat = extract_fixed_features(
        points=points,
        blocked_mask=blocked_mask,
        total_ux=total_ux,
        total_uy=total_uy,
        start_idx=start_idx,
        goal_idx=goal_idx,
        inner_width=inner_width,
        outer_width=outer_width,
        heading_diff_deg=0.0,   # 핵심
    )

    distance = float(feat["distance"])
    base_time = distance / float(usv_speed) if usv_speed > 1e-12 else 1e6

    x = np.array([[float(feat[name]) for name in rf_feature_names]], dtype=np.float32)
    pred = float(rf_model.predict(x)[0])

    if residual_mode == "percent":
        time_pred = base_time * (1.0 + pred)
    elif residual_mode == "additive":
        time_pred = base_time + pred
    else:
        raise ValueError(f"Unknown residual_mode: {residual_mode}")

    if not np.isfinite(time_pred) or time_pred <= 0:
        time_pred = max(base_time, 1e-6)

    return float(time_pred)

def build_intra_usv_time_bias_blocks_rf_heading0(
    patrols,
    points: np.ndarray,
    blocked_mask: np.ndarray,
    total_ux: np.ndarray,
    total_uy: np.ndarray,
    rf_model,
    rf_feature_names,
    *,
    usv_speed: float,
    residual_mode: str = "percent",
    inner_width: float = 90.0,
    outer_width: float = 180.0,
    bias_scale: float = 1000.0,
    diag_bias: float = 0.0,
) -> Dict[str, np.ndarray]:
    """
    returns:
      time_blocks: (U,W,W)
      bias_blocks: (U,W,W) = -time / bias_scale
    """
    U = len(patrols)
    W = len(patrols[0]["waypoint_vids"])

    time_blocks = np.zeros((U, W, W), dtype=np.float32)

    for u, patrol in enumerate(patrols):
        wp_vids = patrol["waypoint_vids"]
        for i in range(W):
            s_vid = int(wp_vids[i])
            for j in range(W):
                g_vid = int(wp_vids[j])

                if i == j:
                    time_blocks[u, i, j] = 0.0
                else:
                    time_blocks[u, i, j] = predict_pair_time_rf_heading0(
                        points=points,
                        blocked_mask=blocked_mask,
                        total_ux=total_ux,
                        total_uy=total_uy,
                        start_idx=s_vid,
                        goal_idx=g_vid,
                        rf_model=rf_model,
                        rf_feature_names=rf_feature_names,
                        usv_speed=usv_speed,
                        residual_mode=residual_mode,
                        inner_width=inner_width,
                        outer_width=outer_width,
                    )

    bias_blocks = -time_blocks / float(bias_scale)
    for u in range(U):
        np.fill_diagonal(bias_blocks[u], float(diag_bias))

    return {
        "time_blocks": time_blocks,
        "bias_blocks": bias_blocks.astype(np.float32),
    }

def build_intra_usv_time_bias_blocks_distance(
    patrols,
    points: np.ndarray,
    *,
    usv_speed: float,
    bias_scale: float = 1000.0,
    diag_bias: float = 0.0,
) -> Dict[str, np.ndarray]:
    U = len(patrols)
    W = len(patrols[0]["waypoint_vids"])

    time_blocks = np.zeros((U, W, W), dtype=np.float32)

    for u, patrol in enumerate(patrols):
        wp_vids = patrol["waypoint_vids"]
        for i in range(W):
            pi = np.asarray(points[int(wp_vids[i])], dtype=np.float32)
            for j in range(W):
                pj = np.asarray(points[int(wp_vids[j])], dtype=np.float32)

                if i == j:
                    time_blocks[u, i, j] = 0.0
                else:
                    dist = float(np.linalg.norm(pj - pi))
                    time_blocks[u, i, j] = dist / float(usv_speed)

    bias_blocks = -time_blocks / float(bias_scale)
    for u in range(U):
        np.fill_diagonal(bias_blocks[u], float(diag_bias))

    return {
        "time_blocks": time_blocks,
        "bias_blocks": bias_blocks.astype(np.float32),
    }
def mask_intra_attn_bias_blocks_by_alive(
    intra_attn_bias_blocks: np.ndarray,   # (U, W, W)
    alive_mask: np.ndarray,
    *,
    mask_value: float = -1e9,
) -> np.ndarray:
    """
    dead USV의 intra-USV attention block을 사실상 비활성화.
    diagonal만 0으로 두어 softmax NaN을 방지한다.
    """
    bias = np.asarray(intra_attn_bias_blocks, dtype=np.float32).copy()
    alive_mask = np.asarray(alive_mask, dtype=bool)

    U, W, W2 = bias.shape
    if W != W2:
        raise ValueError("intra_attn_bias_blocks must have shape (U, W, W).")
    if alive_mask.shape[0] != U:
        raise ValueError("alive_mask length must match number of USVs.")

    eye = np.eye(W, dtype=np.float32)

    for u in range(U):
        if not alive_mask[u]:
            bias[u, :, :] = mask_value
            bias[u, :, :] += eye * (-mask_value)   # diagonal -> 0

    return bias

def _heading_feature_pair_from_angle_deg(angle_deg: float):
    th = math.radians(float(angle_deg))
    return (
        np.float32(math.sin(th)),
        np.float32(math.cos(th)),
    )
def _angle_to_sin_cos(angle_deg: float) -> Tuple[np.float32, np.float32]:
    th = math.radians(float(angle_deg))
    return np.float32(math.sin(th)), np.float32(math.cos(th))

def _bearing_deg_from_two_points(command_xy, goal_xy, fallback_heading_deg: float):
    command_xy = np.asarray(command_xy, dtype=np.float32)
    goal_xy = np.asarray(goal_xy, dtype=np.float32)

    move_vec = goal_xy - command_xy
    if np.linalg.norm(move_vec) > 1e-12:
        return float(angle_from_vec_y_clockwise_deg(move_vec))
    return float(fallback_heading_deg)

def build_dead_usv_waypoint_mask(
    alive_mask: np.ndarray,   # (U,)
    num_waypoints: int,
    *,
    mask_value: float = -1e9,
) -> np.ndarray:
    """
    dead USV에 속한 waypoint는 attention에서 제외하기 위한 (N,N) additive mask 생성.
    N = U * W
    dead waypoint가 key로 들어가는 열(column)을 막는다.
    dead query row도 같이 막되, softmax NaN 방지를 위해 diagonal은 0으로 복구한다.
    """
    alive_mask = np.asarray(alive_mask, dtype=bool)
    U = alive_mask.shape[0]
    W = int(num_waypoints)
    N = U * W

    dead_wp = np.repeat(~alive_mask, W)   # (N,)
    mask = np.zeros((N, N), dtype=np.float32)

    # dead key/value column 차단
    mask[:, dead_wp] = mask_value

    # dead query row도 차단
    mask[dead_wp, :] = mask_value

    # diagonal 복구
    np.fill_diagonal(mask, 0.0)
    return mask

def build_dead_waypoint_mask_from_alive(
    alive_mask: np.ndarray,   # (U,)
    num_waypoints: int,
    *,
    mask_value: float = -1e9,
) -> np.ndarray:
    """
    alive_mask를 waypoint level mask로 확장해서 (N,N) additive mask 생성
    N = U * W

    역할:
      - dead USV에 속한 waypoint는 attention에서 제외
      - dead row/query, dead col/key 모두 막음
      - diagonal은 0으로 복구해서 softmax NaN 방지
    """
    alive_mask = np.asarray(alive_mask, dtype=bool)
    U = int(alive_mask.shape[0])
    W = int(num_waypoints)
    N = U * W

    dead_wp = np.repeat(~alive_mask, W)   # (N,)
    mask = np.zeros((N, N), dtype=np.float32)

    # dead waypoint가 key/value로 선택되지 않도록 열 차단
    mask[:, dead_wp] = mask_value

    # dead waypoint가 query로도 사용되지 않도록 행 차단
    mask[dead_wp, :] = mask_value

    # diagonal 복구
    return mask

def _safe_normalize_time_value(
    time_value: float,
    normalize_t: Optional[float],
    *,
    invalid_fill: float = 1e6,
) -> float:
    if not np.isfinite(time_value):
        time_value = float(invalid_fill)
    if normalize_t is not None:
        time_value /= float(normalize_t)
    return float(time_value)


def _earliest_feasible_meet_rel_from_schedule(
    absolute_visit_schedule: Dict[int, Dict[int, List[float]]],
    *,
    usv_idx: int,
    local_wp_idx: int,
    current_time: float,
    command_arrival_rel: float,
) -> float:
    """
    해당 waypoint(local_wp_idx)에 command가 도착 가능한 시점 이후의
    가장 이른 meet residual time (= meet_time_abs - current_time)을 반환.
    없으면 inf.
    """
    if not np.isfinite(command_arrival_rel):
        return float("inf")

    times = absolute_visit_schedule[int(usv_idx)][int(local_wp_idx)]
    earliest_cmd_abs = float(current_time) + float(command_arrival_rel)

    for t_abs in times:
        if float(t_abs) >= earliest_cmd_abs:
            return float(t_abs - current_time)

    return float("inf")

def get_usv_current_node_at_time(
    env,
    usv_idx: int,
    current_time: float,
    *,
    num_laps: int = None,
):
    """
    current_time에서 해당 USV가 가장 최근에 도달한 node를 반환한다.
    즉, 연속적인 위치 보간이 아니라 현재까지 방문한 마지막 graph node 기준이다.
    """

    if num_laps is None:
        num_laps = env.config.num_laps_for_intercept

    node_path = env.sim_result["usv_node_paths"][usv_idx]
    edge_path = env.sim_result["usv_edge_paths"][usv_idx]

    visit_nodes, visit_times = build_usv_path_timeline(
        node_path=node_path,
        edge_path=edge_path,
        cost_e=env.cost_e,
        num_laps=num_laps,
    )

    if len(visit_nodes) == 0:
        return None

    visit_times = np.asarray(visit_times, dtype=float)

    # current_time 이하에서 가장 마지막 방문 index
    idx = np.searchsorted(visit_times, current_time, side="right") - 1

    if idx < 0:
        idx = 0

    return int(visit_nodes[idx])

def select_euclidean_greedy_action(env, state):
    """
    현재 지휘함 위치와 각 alive USV의 현재 위치 사이의 Euclidean distance가
    가장 작은 USV를 선택한다.
    """

    command_xy = np.asarray(state["command_xy"], dtype=float)
    current_time = float(state["command_time"])
    alive_mask = np.asarray(state["alive_mask"], dtype=bool)

    best_action = None
    best_dist = float("inf")

    for u in range(env.num_usv):
        if not alive_mask[u]:
            continue

        usv_node = get_usv_current_node_at_time(
            env=env,
            usv_idx=u,
            current_time=current_time,
            num_laps=env.config.num_laps_for_intercept,
        )

        if usv_node is None:
            continue

        usv_xy = np.asarray(env.points[usv_node], dtype=float)
        dist = float(np.linalg.norm(usv_xy - command_xy))

        if dist < best_dist:
            best_dist = dist
            best_action = int(u)

    if best_action is None:
        alive = np.where(alive_mask)[0]
        return int(alive[0])

    return int(best_action)

def snapshot_episode_state(env):
    return {
        "command_vid": int(env.command_vid),
        "command_time": float(env.command_time),
        "command_heading_deg": float(env.command_heading_deg),
        "alive_mask": env.alive_mask.copy(),
        "done": bool(env.done),
    }


def restore_episode_state(env, snap):
    env.command_vid = int(snap["command_vid"])
    env.command_time = float(snap["command_time"])
    env.command_heading_deg = float(snap["command_heading_deg"])
    env.alive_mask = snap["alive_mask"].copy()
    env.done = bool(snap["done"])

def select_rolling_greedy_action_len2(
    env,
    state,
    *,
    greedy_mode: str = "meet_time",
    use_surrogate: bool = True,
    objective: str = "total_reward",   # "total_reward" or "final_time"
):
    """
    길이 2 rolling greedy:
      현재 상태에서 a1 후보를 고르고,
      a1을 가상 수행한 뒤,
      다음 상태에서 greedy 기준으로 a2를 고른다.

      objective="total_reward"이면 길이 2 동안의 reward 합이 가장 작은 a1을 반환한다.
      objective="final_time"이면 기존처럼 길이 2 이후 command_time이 가장 작은 a1을 반환한다.
    """

    alive_actions = [int(u) for u in np.where(state["alive_mask"])[0]]

    if len(alive_actions) == 1:
        return alive_actions[0]

    root_snap = snapshot_episode_state(env)

    best_a1 = None
    best_score = -float("inf")

    for a1 in alive_actions:
        restore_episode_state(env, root_snap)

        if use_surrogate:
            inputs1 = build_surrogate_actor_critic_inputs_from_env(env, state)
        else:
            inputs1 = build_actor_critic_inputs_from_env(env, state)

        next_state1, reward1, done1, info1 = env.step(a1, state_cache=inputs1)

        if info1.get("invalid_action", False) or info1.get("intercept_failed", False):
            continue

        if done1:
            if objective == "total_reward":
                score = float(reward1)
            elif objective == "final_time":
                score = float(env.command_time)
            else:
                raise ValueError(f"Unknown objective: {objective}")

        else:
            if greedy_mode == "meet_time":
                a2 = select_meet_time_greedy_action(
                    env,
                    next_state1,
                    use_surrogate=use_surrogate,
                )

            elif greedy_mode == "meet_time_tardiness":
                a2, _ = select_meet_time_with_tardiness_greedy_action(
                    env,
                    next_state1,
                    use_surrogate=use_surrogate,
                    normalize_score=True,
                )

            elif greedy_mode == "tardy_first_greedy":
                a2, _ = select_action_greedy_tardy_first_earliest(
                    env,
                    next_state1,
                )

            elif greedy_mode == "euclidean":
                a2 = select_euclidean_greedy_action(env, next_state1)

            else:
                raise ValueError(f"Unknown greedy_mode: {greedy_mode}")

            if use_surrogate:
                inputs2 = build_surrogate_actor_critic_inputs_from_env(env, next_state1)
            else:
                inputs2 = build_actor_critic_inputs_from_env(env, next_state1)

            next_state2, reward2, done2, info2 = env.step(a2, state_cache=inputs2)

            if info2.get("invalid_action", False) or info2.get("intercept_failed", False):
                continue

            if objective == "total_reward":
                score = float(reward1) + float(reward2)
            elif objective == "final_time":
                score = float(env.command_time)
            else:
                raise ValueError(f"Unknown objective: {objective}")

        if score > best_score:
            best_score = score
            best_a1 = int(a1)

    restore_episode_state(env, root_snap)

    if best_a1 is None:
        return alive_actions[0]

    return best_a1

def select_meet_time_greedy_action(env, state, *, use_surrogate: bool = False):
    if use_surrogate:
        inputs = build_surrogate_actor_critic_inputs_from_env(env, state)
        candidates = inputs["surrogate_candidates"]
    else:
        inputs = build_actor_critic_inputs_from_env(env, state)
        candidates = inputs["intercept_candidates"]

    best_action = None
    best_meet_time = float("inf")

    for u, cand in enumerate(candidates):
        if cand is None:
            continue
        if not bool(state["alive_mask"][u]):
            continue

        if "meet_time_abs" in cand:
            meet_time = float(cand["meet_time_abs"])
        elif "usv_arrival_time_abs" in cand:
            meet_time = float(cand["usv_arrival_time_abs"])
        else:
            continue

        if meet_time < best_meet_time:
            best_meet_time = meet_time
            best_action = int(u)

    if best_action is None:
        alive = np.where(state["alive_mask"])[0]
        return int(alive[0])

    return int(best_action)

def estimate_return_to_base_surrogate(env, *, start_vid: int, start_heading_deg: float):
    """
    현재 command 위치(start_vid)에서 base_xy까지 돌아가는 시간을 surrogate_mode에 맞춰 계산.

    Returns:
      {
        "return_time": float,
        "base_vid": int,
        "arrival_heading_deg": float,
        "return_mode": str,
        "return_failed": bool,
      }
    """
    mode = getattr(env.config, "surrogate_mode", "rf_direct")

    base_xy = np.asarray(env.config.base_xy, dtype=float)

    if env.blocked_mask is not None:
        base_vid = nearest_free_vertex_index(env.points, base_xy, env.blocked_mask)
    else:
        base_vid = nearest_vertex_index(env.points, base_xy)

    start_vid = int(start_vid)

    if start_vid == base_vid:
        return {
            "return_time": 0.0,
            "base_vid": int(base_vid),
            "arrival_heading_deg": float(start_heading_deg),
            "return_mode": mode,
            "return_failed": False,
        }

    # --------------------------------------------------
    # 1) A* surrogate mode: graph 기반 실제 이동시간
    # --------------------------------------------------
    if mode in ("astar_euclidean", "astar_rf"):
        heuristic_mode = "rf" if mode == "astar_rf" else "euclidean"

        _, _, _, best_goal, best_goal_k = astar_turn_state_core(
            cache=env.cache,
            base_cost_e=env.cost_e,
            theta_e_deg=env.theta_e_deg,
            points=env.points,
            start=start_vid,
            goal=int(base_vid),

            heuristic_mode=heuristic_mode,
            rf_model=env.rf_model if heuristic_mode == "rf" else None,
            rf_feature_names=env.rf_feature_names if heuristic_mode == "rf" else None,
            blocked_mask=env.blocked_mask,
            total_ux=env.total_ux,
            total_uy=env.total_uy,
            current_heading_deg=float(start_heading_deg),
            inner_width=float(env.config.inner_width),
            outer_width=float(env.config.outer_width),
            usv_speed=float(env.config.usv_speed),
            residual_mode=env.config.residual_mode,

            th1=float(env.config.th1),
            lam1=float(env.config.lam1),
            th2=float(env.config.th2),
            lam2=float(env.config.lam2),
            use_start_heading=bool(env.config.use_start_heading),
            start_heading_deg=float(start_heading_deg),
            termination="goal_best",
        )

        if np.isfinite(best_goal):
            arrival_heading_deg = get_arrival_heading_deg_at_node(
                cache=env.cache,
                theta_e_deg=env.theta_e_deg,
                node_vid=int(base_vid),
                best_k_in=int(best_goal_k),
                fallback_heading_deg=float(start_heading_deg),
            )

            return {
                "return_time": float(best_goal),
                "base_vid": int(base_vid),
                "arrival_heading_deg": float(arrival_heading_deg),
                "return_mode": mode,
                "return_failed": False,
            }



    # --------------------------------------------------
    # 2) rf_direct: RF surrogate로 base까지 시간 예측
    # --------------------------------------------------
    if mode == "rf_direct":
        return_time = rf_time_heuristic(
            env,
            start_vid=int(start_vid),
            goal_vid=int(base_vid),
            current_heading_deg=float(start_heading_deg),
        )

        p0 = np.asarray(env.points[start_vid], dtype=float)
        p1 = np.asarray(env.points[base_vid], dtype=float)
        move_vec = p1 - p0

        if np.linalg.norm(move_vec) > 1e-12:
            arrival_heading_deg = angle_from_vec_y_clockwise_deg(move_vec)
        else:
            arrival_heading_deg = float(start_heading_deg)

        return {
            "return_time": float(return_time),
            "base_vid": int(base_vid),
            "arrival_heading_deg": float(arrival_heading_deg),
            "return_mode": mode,
            "return_failed": False,
        }

    # --------------------------------------------------
    # 3) euclidean: 거리 / 속도
    # --------------------------------------------------
    if mode == "euclidean":
        p0 = np.asarray(env.points[start_vid], dtype=float)
        p1 = np.asarray(env.points[base_vid], dtype=float)
        dist = float(np.linalg.norm(p1 - p0))
        return_time = max(dist / float(env.config.usv_speed), 1e-6)

        move_vec = p1 - p0
        if np.linalg.norm(move_vec) > 1e-12:
            arrival_heading_deg = angle_from_vec_y_clockwise_deg(move_vec)
        else:
            arrival_heading_deg = float(start_heading_deg)

        return {
            "return_time": float(return_time),
            "base_vid": int(base_vid),
            "arrival_heading_deg": float(arrival_heading_deg),
            "return_mode": mode,
            "return_failed": False,
        }

    raise ValueError(f"Unknown surrogate_mode: {mode}")