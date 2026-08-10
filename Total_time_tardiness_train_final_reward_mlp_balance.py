import warnings
warnings.simplefilter("ignore")
from Simulation_kgreedy_V3 import *
from Network_no_graph import *
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import copy
import pickle
import joblib
import glob
import gc
os.makedirs("checkpoints_direct_tardy_mlp_triple_reward_balance_v2", exist_ok=True)
dir="checkpoints_direct_tardy_mlp_triple_reward_balance_v2"
def make_fixed_test_problem_seeds(num_test_problems: int = 10, base_seed: int = 2025):
    rng = np.random.default_rng(base_seed)
    return [int(rng.integers(0, 10**9)) for _ in range(num_test_problems)]

@dataclass
class CommandEnvConfig:
    command_start_xy: Tuple[float, float] = (1800.0, 900.0)
    command_start_heading_deg: float = 0.0

    num_waypoints: int = 4

    # state node feature
    num_future_times: int = 3
    num_laps_for_state_schedule: int = 20
    # surrogate meeting 전용
    surrogate_num_future_times: int = 20

    # intercept planning
    num_laps_for_intercept: int = 20

    # planner params
    th1: float = 20.0
    lam1: float = 1.0
    th2: float = 30.0
    lam2: float = 3.0
    use_start_heading: bool = False

    # normalization
    normalize_xy: Tuple[float, float] = (3600.0, 1800.0)
    normalize_t: float = 1000.0
    use_due_date: bool = True
    high_priority_due_date: float = 3000.0
    low_priority_due_date: float = 6000.0
    # RF feature extraction params
    inner_width: float = 90.0
    outer_width: float = 180.0
    # RF residual restore
    usv_speed: float = 2.5
    residual_mode: str = "percent"
    pair_time_mode: str = "rf_zero_heading"  # "rf_zero_heading" or "distance"
    pair_bias_scale: float = 2000.0

    surrogate_mode: str = "rf_direct" #rf_direct, euclidean, astar_euclidean, astar_rf
    surrogate_topk_per_usv: int = 24
    return_to_base: bool = False
    base_xy: Tuple[float, float] = (1800.0, 1800.0)

@dataclass
class ProblemGenConfig:
    # domain / mesh
    x_s: float = 0.0
    x_f: float = 3600.0
    y_s: float = 0.0
    y_f: float = 1800.0
    x_c: float = 1800.0
    y_c: float = 900.0
    r: float = 60.0
    mod: str = "Hexa"
    neigh_mode: str = "Extra_extended_edges"

    # obstacles
    num_obstacles: int = 6
    radius_range: Tuple[float, float] = (120.0, 200.0)
    border_margin: float = 50.0
    allow_overlap: bool = False
    obstacle_clearance: float = 200.0

    # current
    usv_speed: float = 2.5

    # patrol
    num_waypoints: int = 8
    patrol_margin: float = 240.0
    patrol_close_loop: bool = True

    # path planner params for patrol generation
    th1: float = 40.0
    lam1: float = 0.0
    th2: float = 50.0
    lam2: float = np.inf

    # sim_result
    sim_num_laps: int = 1
    assign_only_inside_region: bool = False
class ProblemGenerator:
    def __init__(self, config: ProblemGenConfig):
        self.config = config

    def sample_problem(self, seed: Optional[int] = None,region_type=None) -> Dict:
        cfg = self.config
        rng = np.random.default_rng(seed)

        # seed를 통합해서 각 구성요소에 분배
        seed_vortex = int(rng.integers(0, 10**9))
        seed_obstacle = int(rng.integers(0, 10**9))
        seed_patrol = int(rng.integers(0, 10**9))

        # 1) mesh
        points, idx_map = mesh(
            cfg.x_s, cfg.x_f,
            cfg.y_s, cfg.y_f,
            cfg.x_c, cfg.y_c,
            cfg.r, mod=cfg.mod
        )

        adj = build_vertex_adjacency(
            idx_map.copy(),
            len(points),
            mod=cfg.mod,
            mode=cfg.neigh_mode
        )

        # 2) obstacles
        obstacles = make_random_circle_obstacles(
            x_s=cfg.x_s, x_f=cfg.x_f,
            y_s=cfg.y_s, y_f=cfg.y_f,
            num_obstacles=cfg.num_obstacles,
            radius_range=cfg.radius_range,
            border_margin=cfg.border_margin,
            allow_overlap=cfg.allow_overlap,
            obstacle_clearance=cfg.obstacle_clearance,
            seed=seed_obstacle,
        )

        blocked_mask, adj_pruned = apply_circle_obstacles_to_graph(
            points=points,
            adj=adj,
            obstacles=obstacles,
            missing=-1,
            inclusive=True,
        )

        cache = build_edge_cache(points.copy(), adj_pruned.copy())

        # 3) currents
        vortices, uniform = make_random_vortices(Lx=cfg.x_f,Ly=cfg.y_f,seed=seed_vortex)
        res = composite_current(points.copy(), vortices.copy(), uniform=uniform, return_per_vortex=False)

        current_speed = res["speed"]
        current_angle_deg = res["angle_deg"]
        #print(current_speed.mean()) #1.05
        #print("current speed mean =", float(np.mean(current_speed)))
        #print("current speed max  =", float(np.max(current_speed)))
        theta_deg_e, denom_e, cost_e, feasible_e, valid_e = compute_theta_cost_from_cache(
            cache,
            theta_f_deg=current_angle_deg,
            v_f=current_speed,
            V=cfg.usv_speed,
        )

        # 4) patrols
        patrols,region_mode = build_all_patrols(
            points=points,
            cache=cache,
            cost_e=cost_e,
            theta_e_deg=theta_deg_e,
            blocked_mask=blocked_mask,
            x_s=cfg.x_s, x_f=cfg.x_f,
            y_s=cfg.y_s, y_f=cfg.y_f,
            margin=cfg.patrol_margin,
            num_waypoints=cfg.num_waypoints,
            close_loop=cfg.patrol_close_loop,
            th1=cfg.th1,
            lam1=cfg.lam1,
            th2=cfg.th2,
            lam2=cfg.lam2,
            region_mode=region_type
        )

        # 5) random initial patrol phase/start waypoint
        sim_result = simulate_multi_usv_patrol_arrival(
            points=points,
            patrols=patrols,
            cost_e=cost_e,
            seed=seed_patrol,
            num_laps=cfg.sim_num_laps,
            assign_only_inside_region=cfg.assign_only_inside_region,
            region_mode=None #16용
        )

        return {
            "points": points,
            "idx_map": idx_map,
            "adj": adj,
            "adj_pruned": adj_pruned,
            "blocked_mask": blocked_mask,
            "obstacles": obstacles,
            "cache": cache,
            "vortices": vortices,
            "uniform": uniform,
            "res": res,
            "theta_deg_e": theta_deg_e,
            "cost_e": cost_e,
            "patrols": patrols,
            "sim_result": sim_result,
            "seed_vortex": seed_vortex,
            "seed_obstacle": seed_obstacle,
            "seed_patrol": seed_patrol,
            "region_type": region_type,
        }

def make_fixed_half_due_dates(
    num_usv: int,
    *,
    high_due_date: float = 3000.0,
    low_due_date: float = 6000.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    num_usv = int(num_usv)
    rng = np.random.default_rng(seed)

    due_dates = np.full(num_usv, float(low_due_date), dtype=np.float32)

    n_high = num_usv // 2
    high_indices = rng.choice(num_usv, size=n_high, replace=False)

    due_dates[high_indices] = float(high_due_date)

    return due_dates

def make_fixed_triple_due_dates(
    num_usv: int,
    *,
    high_due_date: float = 2000.0,
    medium_due_date: float = 4000.0,
    low_due_date: float = 6000.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    USV들을 high / medium / low due date 그룹으로 나눈다.

    기본 배정:
      - high:   num_usv // 3 개
      - medium: num_usv // 3 개
      - low:    나머지

    due date 값이 작을수록 더 급한 USV로 해석.
    """
    num_usv = int(num_usv)
    rng = np.random.default_rng(seed)

    due_dates = np.full(num_usv, float(low_due_date), dtype=np.float32)

    n_high = num_usv // 3
    n_medium = num_usv // 3

    all_indices = np.arange(num_usv)
    rng.shuffle(all_indices)

    high_indices = all_indices[:n_high]
    medium_indices = all_indices[n_high:n_high + n_medium]

    due_dates[high_indices] = float(high_due_date)
    due_dates[medium_indices] = float(medium_due_date)

    return due_dates


def compute_incremental_tardiness_penalty(
    *,
    due_dates_per_usv: np.ndarray,
    alive_mask_before_step: np.ndarray,
    old_time: float,
    new_time: float,
    normalize_t: float,
) -> float:
    """
    step 사이 [old_time, new_time]에서 새로 발생한 tardiness 총량을 계산한다.

    unfinished 상태였던 모든 USV에 대해:
      max(0, new_time - due_i) - max(0, old_time - due_i)

    를 더한다.

    반환값은 이미 normalize_t로 나눈 값이며,
    reward에는 음수로 더하면 된다.
    """
    due_dates = np.asarray(due_dates_per_usv, dtype=float)
    alive = np.asarray(alive_mask_before_step, dtype=bool)

    old_time = float(old_time)
    new_time = float(new_time)
    normalize_t = float(normalize_t)

    if new_time < old_time:
        raise ValueError(f"new_time({new_time}) must be >= old_time({old_time}).")

    if normalize_t <= 0:
        raise ValueError("normalize_t must be positive.")

    old_tardiness = np.maximum(0.0, old_time - due_dates)
    new_tardiness = np.maximum(0.0, new_time - due_dates)

    incremental = new_tardiness - old_tardiness
    incremental_total = float(np.sum(incremental[alive]))

    return incremental_total / normalize_t
def compute_selected_tardiness_penalty(
    *,
    due_date: float,
    meet_time: float,
    normalize_t: float,
) -> float:
    """
    선택한 USV를 만난 시점의 tardiness를 계산한다.

    tardiness = max(0, meet_time - due_date)

    반환값은 normalize_t로 나눈 값.
    reward에는 음수로 더하면 된다.
    """
    due_date = float(due_date)
    meet_time = float(meet_time)
    normalize_t = float(normalize_t)

    if normalize_t <= 0:
        raise ValueError("normalize_t must be positive.")

    tardiness = max(0.0, meet_time - due_date)

    return tardiness / normalize_t

class CommandInterceptEnv:
    def __init__(
        self,
        *,
        problem_generator: ProblemGenerator,
        rf_model,
        rf_feature_names: List[str],
        config: Optional[CommandEnvConfig] = None,
    ):
        self.problem_generator = problem_generator
        self.rf_model = rf_model
        self.rf_feature_names = rf_feature_names
        self.config = config if config is not None else CommandEnvConfig()

        # problem-specific fields
        self.points = None
        self.cache = None
        self.cost_e = None
        self.theta_e_deg = None
        self.total_ux = None
        self.total_uy = None
        self.patrols = None
        self.sim_result = None
        self.blocked_mask = None
        self.obstacles = None
        self.problem_info = None

        # episode state
        self.num_usv = 12
        self.command_vid: Optional[int] = None
        self.command_time: float = 0.0
        self.command_heading_deg: float = self.config.command_start_heading_deg
        self.alive_mask: Optional[np.ndarray] = None
        self.done: bool = False
        self.wp_list = None
        self.usv_id_per_node = None
        self.pred_time_mat = None
        self.bias_base_mat = None
        self.same_usv_mask = None
        self.absolute_visit_schedule = None
        self.due_dates_per_usv = None


    def sample_new_problem(self, seed: Optional[int] = None, region_type=None):
        prob = self.problem_generator.sample_problem(seed=seed,region_type=region_type)

        self.points = prob["points"]
        self.cache = prob["cache"]
        self.cost_e = prob["cost_e"]
        self.theta_e_deg = prob["theta_deg_e"]
        self.total_ux = prob["res"]["total_ux"]
        self.total_uy = prob["res"]["total_uy"]
        self.patrols = prob["patrols"]
        self.sim_result = prob["sim_result"]
        self.blocked_mask = prob["blocked_mask"]
        self.obstacles = prob["obstacles"]
        self.problem_info = None

        self.num_usv = len(self.sim_result["usv_node_paths"])

        # --------------------------------------------------
        # 1) 문제 생성 시 한 번만: waypoint absolute visit schedule
        # --------------------------------------------------
        self.absolute_visit_schedule = build_waypoint_absolute_visit_schedule(
            patrols=self.patrols,
            sim_result=self.sim_result,
            cost_e=self.cost_e,
            num_laps_for_state_schedule=self.config.num_laps_for_state_schedule,
        )

        # --------------------------------------------------
        # 2) 문제 생성 시 한 번만: pairwise RF cache
        # --------------------------------------------------
        pair_cache = build_problem_pairwise_rf_cache(
            patrols=self.patrols,
            points=self.points,
            blocked_mask=self.blocked_mask,
            total_ux=self.total_ux,
            total_uy=self.total_uy,
            rf_model=self.rf_model,
            rf_feature_names=self.rf_feature_names,
            usv_speed=self.config.usv_speed,
            residual_mode=self.config.residual_mode,
            inner_width=self.config.inner_width,
            outer_width=self.config.outer_width,
        )

        self.wp_list = pair_cache["wp_list"]
        self.usv_id_per_node = pair_cache["usv_id_per_node"]
        self.pred_time_mat = pair_cache["pred_time_mat"]
        self.bias_base_mat = pair_cache["bias_base_mat"]
        self.same_usv_mask = pair_cache["same_usv_mask"]
        if region_type=="type1":
            self.due_dates_per_usv = make_fixed_triple_due_dates(
                self.num_usv,
                high_due_date=6750/3,
                medium_due_date=2*6750/3,
                low_due_date=6750
            )
        if region_type=="type2":
            self.due_dates_per_usv = make_fixed_triple_due_dates(
                self.num_usv,
                high_due_date=7650/3,
                medium_due_date=2*7650/3,
                low_due_date=7650
            )

    def reset(self, *, resample_problem: bool = True, seed: Optional[int] = None,region_type: Optional[str]=None) -> Dict:
        if resample_problem or (self.points is None):
            self.sample_new_problem(seed=seed,region_type=region_type)

        start_xy = np.array(self.config.command_start_xy, dtype=float)

        if self.blocked_mask is not None:
            self.command_vid = nearest_free_vertex_index(
                self.points, start_xy, self.blocked_mask
            )
        else:
            self.command_vid = nearest_vertex_index(self.points, start_xy)

        self.command_time = 0.0
        self.command_heading_deg = self.config.command_start_heading_deg
        self.alive_mask = np.ones(self.num_usv, dtype=bool)
        self.done = False

        return self.get_state()

    def get_state(self) -> Dict:
        # step마다 바뀌는 건 current_time 기준 node feature뿐
        node_features, usv_id_per_node = build_waypoint_node_features_from_precomputed_schedule(
            wp_list=self.wp_list,
            absolute_visit_schedule=self.absolute_visit_schedule,
            current_time=self.command_time,
            num_future_times=self.config.num_future_times,
            as_residual=True,
            normalize_xy=self.config.normalize_xy,
            normalize_t=self.config.normalize_t,
            due_dates_per_usv=self.due_dates_per_usv if self.config.use_due_date else None,
        )

        precomputed_shortest = compute_command_all_shortest_once(self)

        state = {
            "command_xy": self.points[self.command_vid].copy(),
            "command_vid": int(self.command_vid),
            "command_time": float(self.command_time),
            "command_heading_deg": float(self.command_heading_deg),
            "alive_mask": self.alive_mask.copy(),

            # precomputed static caches
            "wp_list": self.wp_list,
            "usv_id_per_node": self.usv_id_per_node,
            "pred_time_mat": self.pred_time_mat,
            "bias_base_mat": self.bias_base_mat,
            "same_usv_mask": self.same_usv_mask,

            # only this changes with current_time
            "node_features": node_features,

            # exact mode only
            "precomputed_shortest": precomputed_shortest,
        }
        return state

    def step(self, action: int, state_cache: Optional[Dict] = None):
        if self.done:
            raise RuntimeError("Environment is done. Call reset().")

        action = int(action)

        if action < 0 or action >= self.num_usv:
            raise ValueError(f"Invalid action {action}")

        if not self.alive_mask[action]:
            reward = -200.0
            info = {"invalid_action": True}
            return self.get_state(), reward, self.done, info

        if state_cache is not None and "precomputed_shortest" in state_cache:
            precomputed = state_cache["precomputed_shortest"]
        else:
            precomputed = compute_command_all_shortest_once(self)

        intercept = find_intercept_for_selected_usv_from_precomputed(
            wp_list=self.wp_list,
            points=self.points,
            sim_result=self.sim_result,
            cost_e=self.cost_e,
            usv_idx=action,
            theta_e_deg=self.theta_e_deg,
            command_heading_deg=self.command_heading_deg,
            command_current_vid=self.command_vid,
            command_current_time=self.command_time,
            cmd_best_time_rel=precomputed["cmd_best_time_rel"],
            cmd_best_k=precomputed["cmd_best_k"],
            prev_node_cmd=precomputed["prev_node_cmd"],
            prev_k_cmd=precomputed["prev_k_cmd"],
            cache=self.cache,
            num_laps=self.config.num_laps_for_intercept,
        )

        if intercept is None:
            reward = -10.0
            info = {"intercept_failed": True, "action_usv": action}
            return self.get_state(), reward, self.done, info

        old_time = float(self.command_time)
        meet_time_abs = float(intercept["usv_arrival_time_abs"])
        elapsed_meet_time = float(meet_time_abs - old_time)

        alive_before_step = self.alive_mask.copy()

        base_reward = -elapsed_meet_time / float(self.config.normalize_t)
        tardiness_penalty = 0.0
        if self.config.use_due_date and self.due_dates_per_usv is not None:
            tardiness_penalty = compute_selected_tardiness_penalty(
                due_date=self.due_dates_per_usv[action],
                meet_time=meet_time_abs,
                normalize_t=self.config.normalize_t,
            )
        reward=base_reward-tardiness_penalty
        # --- heading update before state overwrite ---
        meet_node = int(intercept["meet_node"])

        # precomputed shortest 결과에서 meet_node에 도달하는 best incoming state
        meet_best_k = int(precomputed["cmd_best_k"][meet_node])

        new_heading_deg = get_arrival_heading_deg_at_node(
            cache=self.cache,
            theta_e_deg=self.theta_e_deg,
            node_vid=meet_node,
            best_k_in=meet_best_k,
            fallback_heading_deg=self.command_heading_deg,
        )

        self.command_vid = meet_node
        self.command_time = float(intercept["usv_arrival_time_abs"])
        self.command_heading_deg = float(new_heading_deg)
        self.alive_mask[action] = False

        if not np.any(self.alive_mask):
            self.done = True

        next_state = self.get_state()
        info = {
            "intercept": intercept,
            "action_usv": action,
            "elapsed_meet_time": elapsed_meet_time,
            "base_reward": base_reward,
            "tardiness_penalty": tardiness_penalty,
            "scaled_reward": reward,
            "due_date": float(self.due_dates_per_usv[action]) if self.due_dates_per_usv is not None else None,

        }
        return next_state, reward, self.done, info

    def export_problem_bundle(self) -> Dict:
        return {
            "points": copy.deepcopy(self.points),
            "cache": copy.deepcopy(self.cache),
            "cost_e": copy.deepcopy(self.cost_e),
            "theta_e_deg": copy.deepcopy(self.theta_e_deg),
            "total_ux": copy.deepcopy(self.total_ux),
            "total_uy": copy.deepcopy(self.total_uy),
            "patrols": copy.deepcopy(self.patrols),
            "sim_result": copy.deepcopy(self.sim_result),
            "blocked_mask": copy.deepcopy(self.blocked_mask),
            "obstacles": copy.deepcopy(self.obstacles),
            #"problem_info": copy.deepcopy(self.problem_info),
            "num_usv": copy.deepcopy(self.num_usv),
            "absolute_visit_schedule": copy.deepcopy(self.absolute_visit_schedule),
            "wp_list": copy.deepcopy(self.wp_list),
            "usv_id_per_node": copy.deepcopy(self.usv_id_per_node),
            "pred_time_mat": copy.deepcopy(self.pred_time_mat),
            "bias_base_mat": copy.deepcopy(self.bias_base_mat),
            "same_usv_mask": copy.deepcopy(self.same_usv_mask),
            "due_dates_per_usv": copy.deepcopy(self.due_dates_per_usv),

        }

    def load_problem_bundle(self, bundle: Dict):
        self.points = copy.deepcopy(bundle["points"])
        self.cache = copy.deepcopy(bundle["cache"])
        self.cost_e = copy.deepcopy(bundle["cost_e"])
        self.theta_e_deg = copy.deepcopy(bundle["theta_e_deg"])
        self.total_ux = copy.deepcopy(bundle["total_ux"])
        self.total_uy = copy.deepcopy(bundle["total_uy"])
        self.patrols = copy.deepcopy(bundle["patrols"])
        self.sim_result = copy.deepcopy(bundle["sim_result"])
        self.blocked_mask = copy.deepcopy(bundle["blocked_mask"])
        self.obstacles = copy.deepcopy(bundle["obstacles"])
        #self.problem_info = copy.deepcopy(bundle["problem_info"])
        self.num_usv = copy.deepcopy(bundle["num_usv"])
        self.absolute_visit_schedule = copy.deepcopy(bundle["absolute_visit_schedule"])
        self.wp_list = copy.deepcopy(bundle["wp_list"])
        self.usv_id_per_node = copy.deepcopy(bundle["usv_id_per_node"])
        self.pred_time_mat = copy.deepcopy(bundle["pred_time_mat"])
        self.bias_base_mat = copy.deepcopy(bundle["bias_base_mat"])
        self.same_usv_mask = copy.deepcopy(bundle["same_usv_mask"])
        self.due_dates_per_usv = copy.deepcopy(bundle["due_dates_per_usv"])

def build_fixed_test_problem_bank(env, test_problem_seeds, pkl_path="data_eval.pkl"):
    bank = {}

    for seed in test_problem_seeds:
        #st=time.time()
        env.reset(resample_problem=True, seed=seed)
        bank[int(seed)] = env.export_problem_bundle()
        #print(time.time() - st)
    save_data = {
        "test_problem_seeds": list(map(int, test_problem_seeds)),
        "bank": bank,
    }

    with open(pkl_path, "wb") as f:
        pickle.dump(save_data, f)

    return bank
def load_fixed_test_problem_bank(pkl_path="data_eval.pkl"):
    with open(pkl_path, "rb") as f:
        save_data = pickle.load(f)

    test_problem_seeds = save_data["test_problem_seeds"]
    bank = save_data["bank"]

    return test_problem_seeds, bank
class SurrogatePretrainEnv(CommandInterceptEnv):
    """
    1차 pretrain 전용:
      - command shortest path exact Dijkstra 사용 안 함
      - surrogate waypoint candidate만 사용
      - meet node = representative waypoint
    """

    def get_state(self) -> Dict:
        policy_node_features, usv_id_per_node = build_waypoint_node_features_from_precomputed_schedule(
            wp_list=self.wp_list,
            absolute_visit_schedule=self.absolute_visit_schedule,
            current_time=self.command_time,
            num_future_times=self.config.num_future_times,  # policy용
            as_residual=True,
            normalize_xy=self.config.normalize_xy,
            normalize_t=self.config.normalize_t,
            due_dates_per_usv=self.due_dates_per_usv if self.config.use_due_date else None,
        )

        surrogate_meeting_node_features, _ = build_waypoint_node_features_from_precomputed_schedule(
            wp_list=self.wp_list,
            absolute_visit_schedule=self.absolute_visit_schedule,
            current_time=self.command_time,
            num_future_times=self.config.surrogate_num_future_times,  # surrogate meeting용
            as_residual=True,
            normalize_xy=self.config.normalize_xy,
            normalize_t=self.config.normalize_t,
            due_dates_per_usv=None
        )
        state = {
            "command_xy": self.points[self.command_vid].copy(),
            "command_vid": int(self.command_vid),
            "command_time": float(self.command_time),
            "command_heading_deg": float(self.command_heading_deg),
            "alive_mask": self.alive_mask.copy(),
            "due_dates_per_usv": None if self.due_dates_per_usv is None else self.due_dates_per_usv.copy(),

            "wp_list": self.wp_list,
            "usv_id_per_node": self.usv_id_per_node,
            "pred_time_mat": self.pred_time_mat,
            "bias_base_mat": self.bias_base_mat,
            "same_usv_mask": self.same_usv_mask,

            "node_features": policy_node_features,
            "surrogate_meeting_node_features": surrogate_meeting_node_features,
        }
        return state
    def step(self, action: int, state_cache: Optional[Dict] = None):
        if self.done:
            raise RuntimeError("Environment is done. Call reset().")

        action = int(action)

        if action < 0 or action >= self.num_usv:
            raise ValueError(f"Invalid action {action}")

        if not self.alive_mask[action]:
            reward = -20.0
            info = {"invalid_action": True}
            return self.get_state(), reward, self.done, info

        if state_cache is None or "surrogate_candidates" not in state_cache:
            state_cache = build_surrogate_actor_critic_inputs_from_env(self, self.get_state())

        candidates = state_cache["surrogate_candidates"]
        cand = candidates[action]

        if cand is None:
            reward = -10.0
            info = {"intercept_failed": True, "action_usv": action}
            return self.get_state(), reward, self.done, info

        old_time = float(self.command_time)
        meet_time_abs = float(cand["meet_time_abs"])
        elapsed_meet_time = float(meet_time_abs - old_time)
        base_reward = -elapsed_meet_time / float(self.config.normalize_t)
        tardiness_penalty = 0.0
        alive_before_step = self.alive_mask.copy()
        if self.config.use_due_date and self.due_dates_per_usv is not None:
            tardiness_penalty = compute_selected_tardiness_penalty(
                due_date=self.due_dates_per_usv[action],
                meet_time=meet_time_abs,
                normalize_t=self.config.normalize_t,
            )

        reward = base_reward - tardiness_penalty

        # --- approximate heading update: current -> representative waypoint ---
        # --- heading update ---
        old_vid = int(self.command_vid)
        new_vid = int(cand["rep_vertex_idx"])

        if (
                self.config.surrogate_mode in ("astar_euclidean", "astar_rf")
                and ("arrival_heading_deg" in cand)
                and np.isfinite(cand["arrival_heading_deg"])
        ):
            new_heading_deg = float(cand["arrival_heading_deg"])
        else:
            p0 = np.asarray(self.points[old_vid], dtype=float)
            p1 = np.asarray(self.points[new_vid], dtype=float)
            move_vec = p1 - p0

            if np.linalg.norm(move_vec) > 1e-12:
                new_heading_deg = angle_from_vec_y_clockwise_deg(move_vec)
            else:
                new_heading_deg = float(self.command_heading_deg)

        # representative waypoint로 command ship 이동
        # representative waypoint로 command ship 이동
        self.command_vid = new_vid
        self.command_time = float(cand["meet_time_abs"])
        self.command_heading_deg = float(new_heading_deg)
        self.alive_mask[action] = False

        # --------------------------------------------------
        # 마지막 USV를 만난 뒤 base로 복귀
        # --------------------------------------------------
        return_info = None
        return_time = 0.0
        return_reward = 0.0

        if not np.any(self.alive_mask):
            if bool(getattr(self.config, "return_to_base", False)):
                return_info = estimate_return_to_base_surrogate(
                    self,
                    start_vid=int(self.command_vid),
                    start_heading_deg=float(self.command_heading_deg),
                )

                return_time = float(return_info["return_time"])
                return_reward = -return_time / float(self.config.normalize_t)

                reward += return_reward

                # command ship을 실제로 base로 이동시켜 episode 종료 상태를 일관되게 만듦
                self.command_vid = int(return_info["base_vid"])
                self.command_time += return_time
                self.command_heading_deg = float(return_info["arrival_heading_deg"])

            self.done = True

        next_state = self.get_state()

        info = {
            "surrogate_intercept": cand,
            "action_usv": action,

            # meeting part
            "elapsed_meet_time": elapsed_meet_time,
            "travel_time": float(cand["command_travel_time_rel"]),
            "waiting_time": float(cand["waiting_time"]),

            # return-to-base part
            "return_to_base": bool(return_info is not None),
            "return_time": float(return_time),
            "return_reward": float(return_reward),
            "return_info": return_info,

            # total
            "scaled_reward": float(reward),
            "base_reward": base_reward,
            "tardiness_penalty": tardiness_penalty,
            "due_date": float(self.due_dates_per_usv[action]) if self.due_dates_per_usv is not None else None,
        }
        return next_state, reward, self.done, info
def run_actor_critic_one_step_surrogate(
    env: SurrogatePretrainEnv,
    model: nn.Module,
    device: str = "cuda",
    deterministic: bool = False,
    greedy_mode: str = "meet_time",
    gamma: float = 1.0,
    compute_greedy_diag: bool = True,
):
    state = env.get_state()
    inputs_np = build_surrogate_actor_critic_inputs_from_env(env, state)
    inputs_t = actor_critic_inputs_to_torch(inputs_np, device=device)

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
    probs = out["probs"]
    value = out["value"]

    if deterministic:
        action = torch.argmax(logits, dim=-1).item()
        log_prob = F.log_softmax(logits, dim=-1)[0, action]
    else:
        dist = torch.distributions.Categorical(probs=probs)
        action_tensor = dist.sample()
        action = action_tensor.item()
        log_prob = dist.log_prob(action_tensor)[0]

    next_state, reward, done, info = env.step(action, state_cache=inputs_np)

    extra = {
        "state_before": state,
        "inputs_np": inputs_np,
        "inputs_t": inputs_t,
        "model_output": out,
        "action": action,
        "value": value[0],
        "log_prob": log_prob,
        "env_info": info,
        "attn_hard_bias": inputs_t["attn_hard_bias"],
        "attn_soft_bias": inputs_t["attn_soft_bias"],
    }

    return next_state, reward, done, extra


def collect_one_episode_rollout_surrogate(
    env: SurrogatePretrainEnv,
    model: nn.Module,
    *,
    device: str = "cuda",
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
    resample_problem: bool = True,
    problem_seed: Optional[int] = None,
    use_greedy_advantage: bool = True,
    greedy_mode: str = "meet_time",
    region_type: str = "type1",
):
    buffer = PPORolloutBuffer()
    state = env.reset(resample_problem=resample_problem, seed=problem_seed,region_type=region_type)
    done = False
    episode_reward = 0.0

    while not done:
        next_state, reward, done, extra = run_actor_critic_one_step_surrogate(
            env=env,
            model=model,
            device=device,
            deterministic=False,
            greedy_mode=greedy_mode,
            gamma=gamma,
            compute_greedy_diag=use_greedy_advantage,
        )

        inp = extra["inputs_t"]

        buffer.add(
            usv_node_features=inp["usv_node_features"],
            rep_node_indices=inp["rep_node_indices"],
            actor_query_features=inp["actor_query_features"],
            critic_query_features=inp["critic_query_features"],
            alive_mask=inp["alive_mask"],
            action=torch.tensor(extra["action"], dtype=torch.long),
            log_prob=extra["log_prob"],
            reward=reward,
            done=done,
            attn_hard_bias=extra["attn_hard_bias"],
            attn_soft_bias=extra["attn_soft_bias"],
            value=extra["value"],
        )

        episode_reward += reward
        state = next_state

    last_value = 0.0
    buffer.compute_returns_and_advantages(
        last_value=last_value,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    return buffer, episode_reward





def run_actor_critic_one_step(env, model: nn.Module, device: str = "cuda", deterministic: bool = False):
    state = env.get_state()
    inputs_np = build_actor_critic_inputs_from_env(env, state)
    inputs_t = actor_critic_inputs_to_torch(inputs_np, device=device)

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
    probs = out["probs"]
    value = out["value"]

    if deterministic:
        action = torch.argmax(logits, dim=-1).item()
        log_prob = F.log_softmax(logits, dim=-1)[0, action]
    else:
        dist = torch.distributions.Categorical(probs=probs)
        action_tensor = dist.sample()
        action = action_tensor.item()
        log_prob = dist.log_prob(action_tensor)[0]

    # 핵심: state cache 재사용
    next_state, reward, done, info = env.step(action, state_cache=state)

    extra = {
        "state_before": state,
        "inputs_np": inputs_np,
        "inputs_t": inputs_t,
        "model_output": out,
        "action": action,
        "log_prob": log_prob,
        "env_info": info,
        "value": value[0],
        "attn_hard_bias": inputs_t["attn_hard_bias"],
        "attn_soft_bias": inputs_t["attn_soft_bias"],
    }

    return next_state, reward, done, extra

def collect_ppo_rollout(
    env,
    model: nn.Module,
    rollout_steps: int,
    *,
    device: str = "cuda",
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
):
    buffer = PPORolloutBuffer()

    state = env.reset()
    done = False
    episode_reward = 0.0

    for _ in range(rollout_steps):
        next_state, reward, done, extra = run_actor_critic_one_step(
            env=env,
            model=model,
            device=device,
            deterministic=False,
        )

        inp = extra["inputs_t"]

        buffer.add(
            usv_node_features=inp["usv_node_features"],
            rep_node_indices=inp["rep_node_indices"],
            actor_query_features=inp["actor_query_features"],
            critic_query_features=inp["critic_query_features"],
            alive_mask=inp["alive_mask"],
            action=torch.tensor(extra["action"], dtype=torch.long),
            log_prob=extra["log_prob"],
            reward=reward,
            done=done,
            value=extra["value"],
            attn_hard_bias=extra["attn_hard_bias"],
            attn_soft_bias=extra["attn_soft_bias"],
        )

        episode_reward += reward
        state = next_state

        if done:
            break

    # bootstrap
    if done:
        last_value = 0.0
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
        last_value = out["value"][0].item()

    buffer.compute_returns_and_advantages(last_value=last_value, gamma=gamma, gae_lambda=gae_lambda)

    return buffer, episode_reward, done

def collect_one_episode_rollout(
    env,
    model: nn.Module,
    *,
    device: str = "cuda",
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    resample_problem: bool = True,
    problem_seed: Optional[int] = None,
):
    buffer = PPORolloutBuffer()

    state = env.reset(resample_problem=resample_problem, seed=problem_seed)
    done = False
    episode_reward = 0.0

    while not done:
        next_state, reward, done, extra = run_actor_critic_one_step(
            env=env,
            model=model,
            device=device,
            deterministic=False,
        )

        inp = extra["inputs_t"]

        buffer.add(
            usv_node_features=inp["usv_node_features"],
            rep_node_indices=inp["rep_node_indices"],
            actor_query_features=inp["actor_query_features"],
            critic_query_features=inp["critic_query_features"],
            alive_mask=inp["alive_mask"],
            action=torch.tensor(extra["action"], dtype=torch.long),
            log_prob=extra["log_prob"],
            reward=reward,
            done=done,
            attn_hard_bias=extra["attn_hard_bias"],
            attn_soft_bias=extra["attn_soft_bias"],
            value=extra["value"],
        )

        episode_reward += reward
        state = next_state

    last_value = 0.0
    buffer.compute_returns_and_advantages(
        last_value=last_value,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )

    return buffer, episode_reward

def load_one_chunk(chunk_path,duedate):
    with open(chunk_path, "rb") as f:
        save_data = pickle.load(f)

    test_problem_seeds = list(map(int, save_data["test_problem_seeds"]))
    test_problem_bank = {int(k): v for k, v in save_data["bank"].items()}

    # problem_info 없는 버전도 있고, 혹시 있으면 제거
    for seed, bundle in test_problem_bank.items():
        bundle.pop("problem_info", None)

        num_usv = int(bundle["num_usv"])
        if duedate=='type1':
            bundle["due_dates_per_usv"] = make_fixed_triple_due_dates(
                num_usv,
                high_due_date=6750/3,
                medium_due_date=2*6750.0/3,
                low_due_date=6750.0,
                seed=int(seed),
            )
        if duedate=='type2':
            bundle["due_dates_per_usv"] = make_fixed_triple_due_dates(
                num_usv,
                high_due_date=7650.0/3,
                medium_due_date=7650.0*2/3,
                low_due_date=7650.0,
                seed=int(seed),
            )

    return test_problem_seeds, test_problem_bank


def evaluate_fixed_testset_over_chunks(
    *,
    env,
    model,
    chunk_paths,
    device,
    num_repeats_per_problem=20,
    deterministic=False,
    use_greedy=False,
    use_surrogate=True,
    greedy_mode="meet_time",
):
    """
    저장된 chunk pkl들을 하나씩 로드하면서 전체 eval set을 평가한다.
    한 번에 전체 test_problem_bank를 메모리에 올리지 않는다.
    """
    all_details = []
    chunk_results = []

    for chunk_idx, chunk_path in enumerate(chunk_paths):
        if chunk_idx >= len(chunk_paths)//2:
            test_problem_seeds, test_problem_bank = load_one_chunk(chunk_path,duedate='type1')
        else:
            test_problem_seeds, test_problem_bank = load_one_chunk(chunk_path,duedate='type2')
        result = evaluate_policy_on_fixed_testset_multi(
            env=env,
            model=model,
            test_problem_seeds=test_problem_seeds,
            test_problem_bank=test_problem_bank,
            num_repeats_per_problem=num_repeats_per_problem,
            device=device,
            deterministic=deterministic,
            use_greedy=use_greedy,
            use_surrogate=use_surrogate,
            greedy_mode=greedy_mode,
            print_gate= True if chunk_idx==0 else False
        )

        all_details.extend(result["details"])

        chunk_results.append({
            "chunk_idx": chunk_idx,
            "chunk_path": chunk_path,
            "num_problems": len(test_problem_seeds),
            "best_mean_reward": result["best_mean_reward"],
            "mean_mean_reward": result["mean_mean_reward"],
            "best_success_rate": result["best_success_rate"],
            "mean_success_rate": result["mean_success_rate"],
        })

        del test_problem_seeds
        del test_problem_bank
        del result
        gc.collect()

    weights = np.array([r["num_problems"] for r in chunk_results], dtype=float)

    summary = {
        "best_mean_reward": float(np.average(
            [r["best_mean_reward"] for r in chunk_results],
            weights=weights
        )),
        "mean_mean_reward": float(np.average(
            [r["mean_mean_reward"] for r in chunk_results],
            weights=weights
        )),
        "best_success_rate": float(np.average(
            [r["best_success_rate"] for r in chunk_results],
            weights=weights
        )),
        "mean_success_rate": float(np.average(
            [r["mean_success_rate"] for r in chunk_results],
            weights=weights
        )),
        "details": all_details,
        "chunk_results": chunk_results,
    }

    return summary
def train_ppo_problem_batch_surrogate(
    env: SurrogatePretrainEnv,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    device: str = "cuda",
    n1_num_problems: int = 4,
    n2_rollouts_per_problem: int = 3,
    n3_iterations: int = 100,
    ppo_epochs: int = 4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    base_seed: int = 0,
    eval_every: int = 10,

    # 기존 test_problem_seeds/test_problem_bank 대신 chunk path 사용
    eval_chunk_paths=None,
    eval_num_repeats_policy: int = 20,
    eval_num_repeats_greedy: int = 1,
    eval_greedy_mode: str = "meet_time",
):
    history = []
    rng = np.random.default_rng(base_seed)

    if eval_chunk_paths is not None:
        eval_chunk_paths = list(eval_chunk_paths)
        if len(eval_chunk_paths) == 0:
            eval_chunk_paths = None

    for it in range(n3_iterations):

        merged_buffer = MultiPPORolloutBuffer()

        episode_rewards = []
        problem_seeds = []

        for p in range(n1_num_problems):
            problem_seed = int(rng.integers(0, 10**9))
            problem_seeds.append(problem_seed)

            for k in range(n2_rollouts_per_problem):
                #st=time.time()
                if p%2==0:
                    region_type="type1"
                else:
                    region_type="type2"
                rollout_buffer, episode_reward = collect_one_episode_rollout_surrogate(
                    env=env,
                    model=model,
                    device=device,
                    gamma=gamma,
                    gae_lambda=gae_lambda,
                    resample_problem=(k == 0),
                    problem_seed=problem_seed if k == 0 else None,
                    region_type=region_type
                )
                #print(time.time() - st)
                merged_buffer.extend_from_buffer(rollout_buffer)
                episode_rewards.append(episode_reward)

        batch = merged_buffer.to_batch(
            device=device
        )

        update_infos = []
        for _ in range(ppo_epochs):
            info = ppo_update_minibatch(
                model=model,
                optimizer=optimizer,
                batch=batch,
                device=device,
                mini_batch_size=12,
                clip_eps=clip_eps,
                value_coef=value_coef,
                entropy_coef=entropy_coef
            )
            update_infos.append(info)

        mean_info = {
            k: float(np.mean([x[k] for x in update_infos]))
            for k in update_infos[0].keys()
        }

        mean_info["iteration"] = it
        mean_info["mean_episode_reward"] = float(np.mean(episode_rewards))
        mean_info["std_episode_reward"] = float(np.std(episode_rewards))
        mean_info["num_episodes"] = len(episode_rewards)
        mean_info["num_transitions"] = len(merged_buffer)
        mean_info["problem_seeds"] = problem_seeds


        history.append(mean_info)

        if it % eval_every == 0:
            torch.save({
                "iteration": it,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
            }, os.path.join(dir, f"checkpoint_{it:04d}.pt"))

            print(
                f"[surrogate {it:04d}] "
                f"episodes={mean_info['num_episodes']} "
                f"transitions={mean_info['num_transitions']} "
                f"reward={mean_info['mean_episode_reward']:.3f}±{mean_info['std_episode_reward']:.3f} "
                f"loss={mean_info['loss']:.4f} "
                f"policy={mean_info['policy_loss']:.4f} "
                f"value={mean_info['value_loss']:.4f} "
                f"entropy={mean_info['entropy']:.4f}"
            )

        # --------------------------------------------------
        # chunk 기반 eval
        # --------------------------------------------------
        if (eval_chunk_paths is not None) and (it % eval_every == 0):
            start = time.time()

            print(f"\n  [eval start at iteration {it}] policy over chunks")

            eval_result = evaluate_fixed_testset_over_chunks(
                env=env,
                model=model,
                chunk_paths=eval_chunk_paths,
                device=device,
                num_repeats_per_problem=eval_num_repeats_policy,
                deterministic=False,
                use_greedy=False,
                use_surrogate=True,
                greedy_mode="euclidean",
            )
            print(time.time() - start)
            start = time.time()
            print(f"  [eval start at iteration {it}] greedy over chunks")

            greedy_result = evaluate_fixed_testset_over_chunks(
                env=env,
                model=model,
                chunk_paths=eval_chunk_paths,
                device=device,
                num_repeats_per_problem=eval_num_repeats_greedy,
                deterministic=True,
                use_greedy=True,
                use_surrogate=True,
                greedy_mode=eval_greedy_mode,
            )
            print(time.time() - start)
            mean_info["eval_best_mean_reward"] = eval_result["best_mean_reward"]
            mean_info["eval_mean_mean_reward"] = eval_result["mean_mean_reward"]
            mean_info["eval_best_success_rate"] = eval_result["best_success_rate"]
            mean_info["eval_mean_success_rate"] = eval_result["mean_success_rate"]

            mean_info["greedy_best_mean_reward"] = greedy_result["best_mean_reward"]
            mean_info["greedy_mean_mean_reward"] = greedy_result["mean_mean_reward"]
            mean_info["greedy_best_success_rate"] = greedy_result["best_success_rate"]
            mean_info["greedy_mean_success_rate"] = greedy_result["mean_success_rate"]

            print(
                f"  [eval-policy] best={eval_result['best_mean_reward']:.3f}, "
                f"mean={eval_result['mean_mean_reward']:.3f}, "
                f"best_succ={eval_result['best_success_rate']:.3f}, "
                f"mean_succ={eval_result['mean_success_rate']:.3f}"
            )

            print(
                f"  [eval-greedy/{eval_greedy_mode}] best={greedy_result['best_mean_reward']:.3f}, "
                f"mean={greedy_result['mean_mean_reward']:.3f}, "
                f"best_succ={greedy_result['best_success_rate']:.3f}, "
                f"mean_succ={greedy_result['mean_success_rate']:.3f}"
            )

            # eval details는 너무 클 수 있으니 매번 저장하고 메모리에서 제거
            np.save(
                os.path.join(dir, f"eval_policy_details_{it:04d}.npy"),
                np.array(eval_result["details"], dtype=object)
            )
            np.save(
                os.path.join(dir, f"eval_greedy_details_{it:04d}.npy"),
                np.array(greedy_result["details"], dtype=object)
            )

            del eval_result
            del greedy_result
            gc.collect()

    return history



def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    # --------------------------------------------------
    # 0) seed / device
    # --------------------------------------------------
    base_seed = 101
    set_global_seed(base_seed)
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    device="cuda"
    print("device:", device)

    # --------------------------------------------------
    # 1) RF model load
    # --------------------------------------------------
    rf_payload = joblib.load("best_residual_xgb_final.joblib")
    rf_model = rf_payload["model"]
    rf_model.n_jobs=1
    rf_feature_names = rf_payload["feature_names"]
    print("rf_feature_names:", rf_feature_names)

    # --------------------------------------------------
    # 2) problem generator
    # --------------------------------------------------
    num_waypoints = 18

    problem_gen = ProblemGenerator(
        ProblemGenConfig(
            x_s=0.0, x_f=3600.0,
            y_s=0.0, y_f=3600.0,
            x_c=1800.0, y_c=1800.0,
            r=60.0,
            mod="Hexa",
            neigh_mode="Extra_extended_edges",

            num_obstacles=0,
            radius_range=(120.0, 200.0),
            border_margin=50.0,
            allow_overlap=False,
            obstacle_clearance=200.0,

            usv_speed=2.5,

            num_waypoints=num_waypoints,
            patrol_margin=300.0,
            patrol_close_loop=True,

            th1=40.0,
            lam1=0.0,
            th2=60.0,
            lam2=np.inf,

            sim_num_laps=1,
            assign_only_inside_region=False,
        )
    )

    # --------------------------------------------------
    # 3) environment
    # --------------------------------------------------
    # 1차 pretrain env
    pretrain_env = SurrogatePretrainEnv(
        problem_generator=problem_gen,
        rf_model=rf_model,
        rf_feature_names=rf_feature_names,
        config=CommandEnvConfig(
            command_start_xy=(1800.0, 1800.0),
            command_start_heading_deg=0.0,
            num_waypoints=num_waypoints,
            num_future_times=3,
            num_laps_for_state_schedule=60,
            num_laps_for_intercept=10,  # surrogate에서는 직접 안 쓰지만 유지 가능
            th1=40.0,
            lam1=0.0,
            th2=60.0,
            lam2=np.inf,
            use_start_heading=True,
            normalize_xy=(3600.0, 3600.0),
            normalize_t=2000.0,
            inner_width=90.0,
            outer_width=180.0,
            usv_speed=2.5,
            residual_mode="percent",
            surrogate_mode="rf_direct" #"rf_direct", "euclidean", "astar_euclidean", "astar_rf"
        )
    )

    #device = "cuda" if torch.cuda.is_available() else "cpu"
    device="cuda"
    model = GraphREINFORCE(
        node_input_dim=6,
        actor_query_dim=7,
        d_model=128,
        num_actor_layers=3,
        num_critic_layers=3,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.00025)



    chunk_paths = sorted(glob.glob(
        os.path.join(
            "evaluation_chunks",
            "eval_*.pkl"
        )
    ))

    print("num eval chunks:", len(chunk_paths))
    for p in chunk_paths:
        print("  ", p)

    if len(chunk_paths) == 0:
        raise FileNotFoundError("eval chunk 파일을 찾지 못했습니다.")

    print('start')
    pretrain_history = train_ppo_problem_batch_surrogate(
        env=pretrain_env,
        model=model,
        optimizer=optimizer,
        device=device,
        n1_num_problems=8,
        n2_rollouts_per_problem=6,
        n3_iterations=501,
        ppo_epochs=2,
        gamma=1,
        gae_lambda=0.95,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.00,
        base_seed=30,
        eval_every=5,

        # 여기만 바뀜
        eval_chunk_paths=chunk_paths,
        eval_num_repeats_policy=5,
        eval_num_repeats_greedy=1,
        eval_greedy_mode="meet_time_tardiness",
    )