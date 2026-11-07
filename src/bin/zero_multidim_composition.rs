//! ZERO Multidim Composition Benchmark
//! Evaluates the composition: ZERO Multidim + Runtime Circuit Breaker + OOD Guard.
//! Tested against:
//!  1. ZERO Multidim (frozen baseline)
//!  2. ZERO + Circuit Breaker
//!  3. ZERO + OOD Guard
//!  4. ZERO + Full Composition (both mechanisms)
//!  5. Universal
//!  6. Fixed 2D Grid
//!  7. Single Stump
//!  8. Oracle
//!
//! Strict isolation: Development tuning seeds are completely separate from test seeds.
//! No ground-truth environment or phase information is leaked to the router.

use std::fs::{self, File};
use std::hint::black_box;
use std::io::{self, Write};
use std::time::Instant;

pub const TOOLS: usize = 5;
pub const FEATURES: usize = 7;
pub const TRAIN_TASKS: usize = 2400;
pub const TEST_TASKS: usize = 3000;
pub const MIN_LEAF: usize = 25;

pub const TOOL_NAMES: [&str; TOOLS] = [
    "X0_universal",
    "X1_conjunction",
    "X2_diagonal",
    "X3_disconnected",
    "X4_flaky_sigmoid",
];

pub const SCENARIO_KINDS: [&str; 5] = [
    "conjunction_box",
    "diagonal_asymmetric",
    "disconnected_islands",
    "flaky_transition",
    "extreme_universal",
];

// Circuit Breaker constants (identical to proven ZERO 0.6 parameters)
pub const CB_STREAK_TO_BENCH: u32 = 2;
pub const CB_BENCH_BASE: u32 = 4;
pub const CB_BENCH_MAX: u32 = 32;

// OOD Guard spatial coverage grid (resolution 32x32 over coordinate space [0, 380])
pub const OOD_GRID_SIZE: usize = 32;
pub const OOD_MAX_COORD: f64 = 380.0;

#[derive(Clone, Copy, Debug)]
pub struct Task {
    pub a: u32,
    pub b: u32,
    pub kind_tag: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct Feedback {
    pub tool_cost: f64,
    pub first_ok: bool,
    pub total_cost: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct EnvState {
    pub phase: usize,
    pub x2_cost: f64,
    pub x3_active: bool,
    pub x4_midpoint: f64,
}

impl Default for EnvState {
    fn default() -> Self {
        Self {
            phase: 1,
            x2_cost: 1.6,
            x3_active: true,
            x4_midpoint: 180.0,
        }
    }
}

impl EnvState {
    pub fn for_task(task_idx: usize) -> Self {
        match task_idx {
            0..=999 => Self {
                phase: 1,
                x2_cost: 1.6,
                x3_active: true,
                x4_midpoint: 180.0,
            },
            1000..=1499 => Self {
                phase: 2, // Flaky failure rate surge
                x2_cost: 1.6,
                x3_active: true,
                x4_midpoint: 130.0,
            },
            1500..=1999 => Self {
                phase: 3, // Cost surge on X2
                x2_cost: 5.0,
                x3_active: true,
                x4_midpoint: 180.0,
            },
            2000..=2499 => Self {
                phase: 4, // Full outage of specialist X3
                x2_cost: 1.6,
                x3_active: false,
                x4_midpoint: 180.0,
            },
            _ => Self {
                phase: 5, // Recovery
                x2_cost: 1.6,
                x3_active: true,
                x4_midpoint: 180.0,
            },
        }
    }
}

pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    pub fn range(&mut self, lo: u32, hi: u32) -> u32 {
        lo + (self.next() % (hi - lo + 1) as u64) as u32
    }
    pub fn next_f64(&mut self) -> f64 {
        (self.next() & 0x001F_FFFF_FFFF_FFFF) as f64 / (0x0020_0000_0000_0000u64 as f64)
    }
}

pub fn generate_task(r: &mut Rng, test_shift: bool) -> Task {
    let mode = r.range(0, 4) as usize;
    let (a, b) = match mode {
        0 => {
            let a = r.range(40, 180);
            let b = (a as i64 + r.range(0, 60) as i64 - 30).clamp(20, 320) as u32;
            (a, b)
        }
        1 => {
            let b = r.range(20, 160);
            let a = (b as f64 * (1.3 + r.next_f64() * 0.8)).clamp(30.0, 340.0) as u32;
            (a, b)
        }
        2 => {
            if r.next() % 2 == 0 {
                (r.range(110, 280), r.range(110, 280))
            } else {
                (r.range(20, 65), r.range(200, 330))
            }
        }
        3 => {
            let sum = r.range(130, 230);
            let a = r.range(20, sum.saturating_sub(20));
            (a, sum - a)
        }
        _ => {
            let (lo, hi) = if test_shift { (20, 360) } else { (15, 320) };
            (r.range(lo, hi), r.range(lo, hi))
        }
    };

    let kind_tag = classify_scenario(a, b);
    Task { a, b, kind_tag }
}

fn classify_scenario(a: u32, b: u32) -> usize {
    if a + b <= 160 && a.abs_diff(b) <= 40 {
        0
    } else if a as f64 >= 1.4 * b as f64 && a + b <= 350 {
        1
    } else if (a + b >= 240 && a >= 100 && b >= 100) || (a <= 60 && b >= 200) {
        2
    } else if (130..=230).contains(&(a + b)) {
        3
    } else {
        4
    }
}

pub fn extract_features(t: Task) -> [u32; FEATURES] {
    [
        t.a,
        t.b,
        t.a.min(t.b),
        t.a.max(t.b),
        t.a + t.b,
        t.a.abs_diff(t.b),
        ((t.a as u64 * t.b as u64) / 100).min(u32::MAX as u64) as u32,
    ]
}

pub fn call_cost(tool: usize, env: &EnvState) -> f64 {
    match tool {
        0 => 9.5,
        1 => 1.0,
        2 => env.x2_cost,
        3 => 2.0,
        4 => 1.2,
        _ => 10.0,
    }
}

pub fn reference_answer(t: Task) -> i64 {
    t.a as i64 * 41 + t.b as i64 * 59
}

pub fn execute_tool(t: Task, tool: usize, env: &EnvState, rng: &mut Rng) -> Option<i64> {
    let ans = reference_answer(t);
    match tool {
        0 => Some(ans),
        1 => {
            if (t.a + t.b <= 160) && (t.a.abs_diff(t.b) <= 40) {
                Some(ans)
            } else {
                None
            }
        }
        2 => {
            if (t.a as f64 >= 1.4 * t.b as f64) && (t.a + t.b <= 350) {
                Some(ans)
            } else {
                None
            }
        }
        3 => {
            if !env.x3_active {
                None
            } else if (t.a + t.b >= 240 && t.a >= 100 && t.b >= 100) || (t.a <= 60 && t.b >= 200) {
                Some(ans)
            } else {
                None
            }
        }
        4 => {
            let sum = (t.a + t.b) as f64;
            let logit = (sum - env.x4_midpoint) / 25.0;
            let p_fail = 1.0 / (1.0 + (-logit).exp());
            if rng.next_f64() >= p_fail {
                Some(ans)
            } else {
                None
            }
        }
        _ => None,
    }
}

pub fn run_feedback(t: Task, tool: usize, env: &EnvState, rng: &mut Rng) -> Feedback {
    let cost = call_cost(tool, env);
    let ok = execute_tool(t, tool, env, rng) == Some(reference_answer(t));
    let fallback = if ok { 0.0 } else { call_cost(0, env) };
    Feedback {
        tool_cost: cost,
        first_ok: ok,
        total_cost: cost + fallback,
    }
}

pub fn true_expected_cost(t: Task, tool: usize, env: &EnvState) -> f64 {
    let base = call_cost(tool, env);
    let p_fail = match tool {
        0 => 0.0,
        1 => {
            if (t.a + t.b <= 160) && (t.a.abs_diff(t.b) <= 40) {
                0.0
            } else {
                1.0
            }
        }
        2 => {
            if (t.a as f64 >= 1.4 * t.b as f64) && (t.a + t.b <= 350) {
                0.0
            } else {
                1.0
            }
        }
        3 => {
            if !env.x3_active {
                1.0
            } else if (t.a + t.b >= 240 && t.a >= 100 && t.b >= 100) || (t.a <= 60 && t.b >= 200) {
                0.0
            } else {
                1.0
            }
        }
        4 => {
            let sum = (t.a + t.b) as f64;
            let logit = (sum - env.x4_midpoint) / 25.0;
            1.0 / (1.0 + (-logit).exp())
        }
        _ => 1.0,
    };
    base + p_fail * call_cost(0, env)
}

pub fn oracle_choice(t: Task, env: &EnvState) -> usize {
    let mut best_i = 0;
    let mut best_c = f64::INFINITY;
    for i in 0..TOOLS {
        let c = true_expected_cost(t, i, env);
        if c < best_c {
            best_c = c;
            best_i = i;
        }
    }
    best_i
}

#[derive(Clone, Copy)]
pub struct Sample {
    pub task: Task,
    pub outcomes: [Feedback; TOOLS],
}

pub fn collect_training_samples(seed: u64, n: usize) -> (Vec<Sample>, f64) {
    let mut rng = Rng::new(seed);
    let default_env = EnvState::default();
    let mut total_probe_cost = 0.0;
    let samples = (0..n)
        .map(|_| {
            let t = generate_task(&mut rng, false);
            let outcomes = std::array::from_fn(|i| {
                let mut tool_rng = Rng::new(rng.next());
                let fb = run_feedback(t, i, &default_env, &mut tool_rng);
                total_probe_cost += fb.total_cost;
                fb
            });
            Sample { task: t, outcomes }
        })
        .collect();
    (samples, total_probe_cost)
}

// ---------------- Tree2D Implementation (Identical to zero_multidim_robustness) ----------------
#[derive(Clone, Copy)]
pub struct Tree2D {
    pub f1: usize,
    pub cut1: u32,
    pub left_f2: usize,
    pub left_cut2: u32,
    pub left_left_val: f64,
    pub left_right_val: f64,
    pub right_f2: usize,
    pub right_cut2: u32,
    pub right_left_val: f64,
    pub right_right_val: f64,
}
impl Tree2D {
    pub fn predict(&self, t: Task) -> f64 {
        let feats = extract_features(t);
        if feats[self.f1] <= self.cut1 {
            if feats[self.left_f2] <= self.left_cut2 {
                self.left_left_val
            } else {
                self.left_right_val
            }
        } else {
            if feats[self.right_f2] <= self.right_cut2 {
                self.right_left_val
            } else {
                self.right_right_val
            }
        }
    }
}

fn fit_stump_indices(
    indices: &[usize],
    feat_matrix: &[[u32; FEATURES]],
    targets: &[f64],
) -> (usize, u32, f64, f64, f64) {
    let n = indices.len();
    if n == 0 {
        return (0, 0, 0.0, 0.0, 0.0);
    }
    let sum_y: f64 = indices.iter().map(|&i| targets[i]).sum();
    let mean = sum_y / n as f64;
    let mut best_f = 0;
    let mut best_cut = u32::MAX;
    let mut best_l = mean;
    let mut best_r = mean;
    let mut best_sse: f64 = indices.iter().map(|&i| (targets[i] - mean).powi(2)).sum();

    if n < MIN_LEAF * 2 {
        return (best_f, best_cut, best_l, best_r, best_sse);
    }

    for f in 0..FEATURES {
        let mut pairs: Vec<(u32, f64)> = indices
            .iter()
            .map(|&i| (feat_matrix[i][f], targets[i]))
            .collect();
        pairs.sort_unstable_by_key(|p| p.0);

        let sum_total: f64 = pairs.iter().map(|p| p.1).sum();
        let sq_total: f64 = pairs.iter().map(|p| p.1 * p.1).sum();

        let mut l_sum = 0.0;
        let mut l_sq = 0.0;

        for k in 0..n - 1 {
            let y = pairs[k].1;
            l_sum += y;
            l_sq += y * y;
            let l_n = k + 1;
            let r_n = n - l_n;

            if l_n < MIN_LEAF || r_n < MIN_LEAF || pairs[k].0 == pairs[k + 1].0 {
                continue;
            }

            let r_sum = sum_total - l_sum;
            let sse = (l_sq - l_sum * l_sum / l_n as f64)
                + ((sq_total - l_sq) - r_sum * r_sum / r_n as f64);

            if sse + 1e-9 < best_sse {
                best_sse = sse;
                best_f = f;
                best_cut = pairs[k].0;
                best_l = l_sum / l_n as f64;
                best_r = r_sum / r_n as f64;
            }
        }
    }
    (best_f, best_cut, best_l, best_r, best_sse)
}

pub fn fit_tree2d(data: &[Sample], tool: usize) -> Tree2D {
    let n = data.len();
    let targets: Vec<f64> = data.iter().map(|s| s.outcomes[tool].total_cost).collect();
    let feat_matrix: Vec<[u32; FEATURES]> = data.iter().map(|s| extract_features(s.task)).collect();
    let all_idx: Vec<usize> = (0..n).collect();

    let (root_f, root_cut, _, _, _) = fit_stump_indices(&all_idx, &feat_matrix, &targets);

    let (left_idx, right_idx): (Vec<usize>, Vec<usize>) = all_idx
        .into_iter()
        .partition(|&i| feat_matrix[i][root_f] <= root_cut);

    let (left_f2, left_cut2, left_left, left_right, _) =
        fit_stump_indices(&left_idx, &feat_matrix, &targets);

    let (right_f2, right_cut2, right_left, right_right, _) =
        fit_stump_indices(&right_idx, &feat_matrix, &targets);

    Tree2D {
        f1: root_f,
        cut1: root_cut,
        left_f2,
        left_cut2,
        left_left_val: left_left,
        left_right_val: left_right,
        right_f2,
        right_cut2,
        right_left_val: right_left,
        right_right_val: right_right,
    }
}

// ---------------- Component 1: Out-of-Distribution (OOD) Guard ----------------
// Measures spatial coverage of training tasks where specialists had valid utility.
// If a task falls into an unobserved coordinate region (with safety margin), OOD Guard
// flags it and triggers safe Universal fallback (cost 9.5), preventing double fallback (cost 11.5).
#[derive(Clone)]
pub struct OodGuard {
    coverage_grid: [[bool; OOD_GRID_SIZE]; OOD_GRID_SIZE],
}

impl OodGuard {
    pub fn build(train_data: &[Sample]) -> Self {
        let mut raw_grid = [[false; OOD_GRID_SIZE]; OOD_GRID_SIZE];

        // Mark cells where at least one specialist succeeded during training
        for s in train_data {
            let specialist_succeeded = (1..TOOLS).any(|i| s.outcomes[i].first_ok);
            if specialist_succeeded {
                let cell_a = ((s.task.a as f64 / OOD_MAX_COORD) * OOD_GRID_SIZE as f64) as usize;
                let cell_b = ((s.task.b as f64 / OOD_MAX_COORD) * OOD_GRID_SIZE as f64) as usize;
                if cell_a < OOD_GRID_SIZE && cell_b < OOD_GRID_SIZE {
                    raw_grid[cell_a][cell_b] = true;
                }
            }
        }

        // Apply a 1-cell safety dilation margin (Chebyshev distance <= 1)
        // so that slightly shifted in-distribution tasks are NOT falsely rejected
        let mut dilated = [[false; OOD_GRID_SIZE]; OOD_GRID_SIZE];
        for a in 0..OOD_GRID_SIZE {
            for b in 0..OOD_GRID_SIZE {
                let mut covered = false;
                for da in [0, 1, 2] {
                    for db in [0, 1, 2] {
                        let na = (a + da).saturating_sub(1);
                        let nb = (b + db).saturating_sub(1);
                        if na < OOD_GRID_SIZE && nb < OOD_GRID_SIZE && raw_grid[na][nb] {
                            covered = true;
                            break;
                        }
                    }
                    if covered {
                        break;
                    }
                }
                dilated[a][b] = covered;
            }
        }

        Self {
            coverage_grid: dilated,
        }
    }

    /// Returns true if task is Out-Of-Distribution (not covered by specialist support)
    pub fn is_ood(&self, t: Task) -> bool {
        let cell_a = ((t.a as f64 / OOD_MAX_COORD) * OOD_GRID_SIZE as f64) as usize;
        let cell_b = ((t.b as f64 / OOD_MAX_COORD) * OOD_GRID_SIZE as f64) as usize;
        if cell_a >= OOD_GRID_SIZE || cell_b >= OOD_GRID_SIZE {
            true // Outside coordinate space
        } else {
            !self.coverage_grid[cell_a][cell_b]
        }
    }
}

// ---------------- Component 2: Runtime Circuit Breaker ----------------
// Independent tracker of execution health for each tool.
// Does NOT know about test phases or ground truth.
#[derive(Clone)]
pub struct CircuitBreaker {
    consecutive_failures: [u32; TOOLS],
    benched_cooldown: [u32; TOOLS],
    bench_timer: [u32; TOOLS],
    pub false_benches: usize,
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        Self {
            consecutive_failures: [0; TOOLS],
            benched_cooldown: [0; TOOLS],
            bench_timer: [0; TOOLS],
            false_benches: 0,
        }
    }
}

impl CircuitBreaker {
    pub fn is_benched(&self, tool: usize) -> bool {
        self.bench_timer[tool] > 0
    }

    /// Tick down timers on tasks
    pub fn tick(&mut self) {
        for t in 1..TOOLS {
            if self.bench_timer[t] > 0 {
                self.bench_timer[t] -= 1;
            }
        }
    }

    /// Update status after observing tool call feedback
    pub fn observe(&mut self, tool: usize, fb: Feedback, is_healthy_env: bool) {
        if tool == 0 {
            return; // Universal cannot be benched
        }

        if fb.first_ok {
            // Success: restore tool trust and reset failure streak
            self.consecutive_failures[tool] = 0;
            self.benched_cooldown[tool] = 0;
            self.bench_timer[tool] = 0;
        } else {
            // Failure
            self.consecutive_failures[tool] += 1;
            if self.consecutive_failures[tool] >= CB_STREAK_TO_BENCH {
                let current_cd = self.benched_cooldown[tool];
                let next_cd = if current_cd == 0 {
                    CB_BENCH_BASE
                } else {
                    (current_cd * 2).min(CB_BENCH_MAX)
                };
                self.benched_cooldown[tool] = next_cd;
                self.bench_timer[tool] = next_cd;
                self.consecutive_failures[tool] = 0;

                // Track false positive benching: was benched even though environment was healthy
                if is_healthy_env {
                    self.false_benches += 1;
                }
            }
        }
    }
}

// ---------------- ZERO Router Supporting Modularity ----------------
#[derive(Clone)]
pub struct ZeroRouter {
    pub trees: [Tree2D; TOOLS],
    pub ood_guard: Option<OodGuard>,
    pub circuit_breaker: Option<CircuitBreaker>,
    pub false_ood_rejections: usize,
}

impl ZeroRouter {
    pub fn new(
        train_data: &[Sample],
        enable_cb: bool,
        enable_ood: bool,
    ) -> Self {
        let trees = std::array::from_fn(|i| fit_tree2d(train_data, i));
        let ood_guard = if enable_ood {
            Some(OodGuard::build(train_data))
        } else {
            None
        };
        let circuit_breaker = if enable_cb {
            Some(CircuitBreaker::default())
        } else {
            None
        };
        Self {
            trees,
            ood_guard,
            circuit_breaker,
            false_ood_rejections: 0,
        }
    }

    /// Choose best tool with optional OOD and Circuit Breaker filtering
    pub fn choose(&mut self, t: Task, is_specialist_actually_valid: bool) -> usize {
        // Step 1: OOD Guard Check
        if let Some(ref guard) = self.ood_guard {
            if guard.is_ood(t) {
                if is_specialist_actually_valid {
                    self.false_ood_rejections += 1;
                }
                return 0; // Safe Universal fallback
            }
        }

        // Step 2: Model Prediction across tools
        let preds: [f64; TOOLS] = std::array::from_fn(|i| self.trees[i].predict(t));

        // Step 3: Candidate Selection respecting Circuit Breaker
        if let Some(ref cb) = self.circuit_breaker {
            let mut best_i = 0;
            let mut best_val = f64::INFINITY;
            for i in 0..TOOLS {
                if i != 0 && cb.is_benched(i) {
                    continue; // Skip benched tool
                }
                if preds[i] < best_val {
                    best_val = preds[i];
                    best_i = i;
                }
            }
            best_i
        } else {
            // Standard argmin without CB
            (0..TOOLS)
                .min_by(|&i, &j| preds[i].total_cmp(&preds[j]))
                .unwrap_or(0)
        }
    }

    pub fn observe_feedback(&mut self, tool: usize, fb: Feedback, is_healthy_env: bool) {
        if let Some(ref mut cb) = self.circuit_breaker {
            cb.observe(tool, fb, is_healthy_env);
            cb.tick();
        }
    }
}

// ---------------- Control Baselines ----------------
#[derive(Clone)]
pub struct Fixed2DGrid {
    grid: [[[f64; TOOLS]; 6]; 6],
}
impl Fixed2DGrid {
    pub fn train(data: &[Sample]) -> Self {
        let mut sums = [[[0.0; TOOLS]; 6]; 6];
        let mut counts = [[0usize; 6]; 6];
        let mut prior_sums = [0.0; TOOLS];
        for s in data {
            let ia = (s.task.a as usize / 55).min(5);
            let ib = (s.task.b as usize / 55).min(5);
            counts[ia][ib] += 1;
            for i in 0..TOOLS {
                sums[ia][ib][i] += s.outcomes[i].total_cost;
                prior_sums[i] += s.outcomes[i].total_cost;
            }
        }
        let prior: [f64; TOOLS] = std::array::from_fn(|i| prior_sums[i] / data.len() as f64);
        let mut grid = [[[0.0; TOOLS]; 6]; 6];
        const PSEUDO: f64 = 8.0;
        for ia in 0..6 {
            for ib in 0..6 {
                let n = counts[ia][ib] as f64;
                for i in 0..TOOLS {
                    grid[ia][ib][i] = (sums[ia][ib][i] + PSEUDO * prior[i]) / (n + PSEUDO);
                }
            }
        }
        Self { grid }
    }
    pub fn choose(&self, t: Task) -> usize {
        let ia = (t.a as usize / 55).min(5);
        let ib = (t.b as usize / 55).min(5);
        (0..TOOLS)
            .min_by(|&i, &j| self.grid[ia][ib][i].total_cmp(&self.grid[ia][ib][j]))
            .unwrap_or(0)
    }
}

#[derive(Clone, Copy)]
pub struct Stump {
    pub feature: usize,
    pub cut: u32,
    pub below: f64,
    pub above: f64,
}
impl Stump {
    pub fn predict(&self, t: Task) -> f64 {
        if extract_features(t)[self.feature] <= self.cut {
            self.below
        } else {
            self.above
        }
    }
}

pub fn fit_stump(data: &[Sample], tool: usize) -> Stump {
    let ys: Vec<f64> = data.iter().map(|s| s.outcomes[tool].total_cost).collect();
    let n = ys.len();
    let mean = ys.iter().sum::<f64>() / n as f64;
    let mut best_f = 0;
    let mut best_cut = u32::MAX;
    let mut best_l = mean;
    let mut best_r = mean;
    let mut best_sse = ys.iter().map(|y| (y - mean).powi(2)).sum::<f64>();

    for f in 0..FEATURES {
        let mut pairs: Vec<(u32, f64)> = data
            .iter()
            .zip(&ys)
            .map(|(s, y)| (extract_features(s.task)[f], *y))
            .collect();
        pairs.sort_unstable_by_key(|p| p.0);

        let sum_total: f64 = pairs.iter().map(|p| p.1).sum();
        let sq_total: f64 = pairs.iter().map(|p| p.1 * p.1).sum();
        let mut l_sum = 0.0;
        let mut l_sq = 0.0;

        for k in 0..n - 1 {
            let y = pairs[k].1;
            l_sum += y;
            l_sq += y * y;
            let l_n = k + 1;
            let r_n = n - l_n;
            if l_n < MIN_LEAF || r_n < MIN_LEAF || pairs[k].0 == pairs[k + 1].0 {
                continue;
            }
            let r_sum = sum_total - l_sum;
            let sse = (l_sq - l_sum * l_sum / l_n as f64)
                + ((sq_total - l_sq) - r_sum * r_sum / r_n as f64);
            if sse + 1e-9 < best_sse {
                best_sse = sse;
                best_f = f;
                best_cut = pairs[k].0;
                best_l = l_sum / l_n as f64;
                best_r = r_sum / r_n as f64;
            }
        }
    }
    Stump {
        feature: best_f,
        cut: best_cut,
        below: best_l,
        above: best_r,
    }
}

#[derive(Clone)]
pub struct SingleStumpPolicy {
    pub stumps: [Stump; TOOLS],
}
impl SingleStumpPolicy {
    pub fn train(data: &[Sample]) -> Self {
        Self {
            stumps: std::array::from_fn(|i| fit_stump(data, i)),
        }
    }
    pub fn choose(&self, t: Task) -> usize {
        let preds: [f64; TOOLS] = std::array::from_fn(|i| self.stumps[i].predict(t));
        (0..TOOLS)
            .min_by(|&i, &j| preds[i].total_cmp(&preds[j]))
            .unwrap_or(0)
    }
}

// ---------------- Aggregated Metrics ----------------
#[derive(Default, Clone)]
pub struct Metric {
    pub n: usize,
    pub first_ok: usize,
    pub total_cost: f64,
    pub oracle_matches: usize,
    pub elapsed_nanos: u128,
    pub costs: Vec<f64>,
}
impl Metric {
    pub fn add(&mut self, chosen: usize, fb: Feedback, oracle: usize, nanos: u128) {
        self.n += 1;
        if fb.first_ok {
            self.first_ok += 1;
        }
        self.total_cost += fb.total_cost;
        if chosen == oracle {
            self.oracle_matches += 1;
        }
        self.elapsed_nanos += nanos;
        self.costs.push(fb.total_cost);
    }
    pub fn mean_cost(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.total_cost / self.n as f64
        }
    }
    pub fn first_pct(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            100.0 * self.first_ok as f64 / self.n as f64
        }
    }
    pub fn oracle_pct(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            100.0 * self.oracle_matches as f64 / self.n as f64
        }
    }
    pub fn p50_p95(&mut self) -> (f64, f64) {
        if self.costs.is_empty() {
            return (0.0, 0.0);
        }
        self.costs.sort_unstable_by(|a, b| a.total_cmp(b));
        let idx50 = ((self.costs.len() as f64 * 0.50) as usize).min(self.costs.len() - 1);
        let idx95 = ((self.costs.len() as f64 * 0.95) as usize).min(self.costs.len() - 1);
        (self.costs[idx50], self.costs[idx95])
    }
}

pub fn run() -> io::Result<()> {
    let mut seeds = 30usize;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--seeds" => seeds = args.next().and_then(|x| x.parse().ok()).unwrap_or(30),
            _ => {}
        }
    }

    let out_dir = "reports/zero_multidim_composition";
    fs::create_dir_all(out_dir)?;

    let mut summary = File::create(format!("{out_dir}/summary.csv"))?;
    let mut phases_csv = File::create(format!("{out_dir}/phases.csv"))?;
    let mut scenario_csv = File::create(format!("{out_dir}/scenario_breakdown.csv"))?;
    let mut comp_csv = File::create(format!("{out_dir}/composition_metrics.csv"))?;

    writeln!(
        summary,
        "seed,policy,mean_cost,p50_cost,p95_cost,first_ok_pct,oracle_match_pct,mean_decision_ns"
    )?;
    writeln!(
        phases_csv,
        "seed,phase,phase_name,policy,tasks,mean_cost,first_ok_pct,oracle_match_pct"
    )?;
    writeln!(
        scenario_csv,
        "seed,scenario,policy,count,mean_cost,first_ok_pct,oracle_match_pct"
    )?;
    writeln!(
        comp_csv,
        "seed,policy,outage_cost,outage_calls_to_broken,recovery_delay_tasks,false_benches,false_ood_rejections,ood_extreme_cost"
    )?;

    println!("=========================================================================================");
    println!("ZERO MULTIDIM COMPOSITION BENCHMARK (Circuit Breaker + OOD Guard)");
    println!("Seeds: {seeds} (Fresh independent evaluation seeds, distinct from dev tuning)");
    println!("Tasks per seed: {TRAIN_TASKS} train / {TEST_TASKS} test (90,000 paired decisions per strategy)");
    println!("=========================================================================================");

    const NUM_POLICIES: usize = 8;
    let policy_names = [
        "zero_baseline",
        "zero_circuit_breaker",
        "zero_ood_guard",
        "zero_full_composition",
        "universal",
        "fixed_2d_grid",
        "single_stump",
        "oracle",
    ];

    // Using fresh evaluation seed range starting from 2027_0001
    for s in 0..seeds {
        let seed = 20270001u64 + (s as u64) * 7919;
        let (train_data, _probe_cost) = collect_training_samples(seed ^ 0xAAAA_1111, TRAIN_TASKS);

        // Train 4 ZERO variants
        let mut p_zero_base = ZeroRouter::new(&train_data, false, false);
        let mut p_zero_cb = ZeroRouter::new(&train_data, true, false);
        let mut p_zero_ood = ZeroRouter::new(&train_data, false, true);
        let mut p_zero_full = ZeroRouter::new(&train_data, true, true);

        // Train controls
        let p_fixed2d = Fixed2DGrid::train(&train_data);
        let p_stump = SingleStumpPolicy::train(&train_data);

        // Pre-generate paired test tasks
        let mut test_rng = Rng::new(seed ^ 0xCCCC_3333);
        let test_tasks: Vec<Task> = (0..TEST_TASKS)
            .map(|_| generate_task(&mut test_rng, true))
            .collect();

        let mut pol_metrics: Vec<Metric> = vec![Metric::default(); policy_names.len()];
        let mut phase_metrics: Vec<Vec<Metric>> = vec![vec![Metric::default(); 5]; policy_names.len()];
        let mut scen_metrics: Vec<Vec<Metric>> = vec![vec![Metric::default(); 5]; policy_names.len()];

        // Outage and recovery tracking variables
        let mut outage_broken_calls = [0usize; NUM_POLICIES];
        let mut first_x3_recovery_task = [None; NUM_POLICIES];

        for (task_idx, &task) in test_tasks.iter().enumerate() {
            let env = EnvState::for_task(task_idx);
            let opt = oracle_choice(task, &env);
            let phase_idx = env.phase - 1;
            let scen_idx = task.kind_tag;
            let is_spec_valid = opt != 0;

            for (p_idx, &pol) in policy_names.iter().enumerate() {
                let mut exec_rng = Rng::new(seed ^ 0x5555_4444 ^ ((task_idx as u64) << 16));

                let t0 = Instant::now();
                let chosen = black_box(match pol {
                    "zero_baseline" => p_zero_base.choose(black_box(task), is_spec_valid),
                    "zero_circuit_breaker" => p_zero_cb.choose(black_box(task), is_spec_valid),
                    "zero_ood_guard" => p_zero_ood.choose(black_box(task), is_spec_valid),
                    "zero_full_composition" => p_zero_full.choose(black_box(task), is_spec_valid),
                    "universal" => 0,
                    "fixed_2d_grid" => p_fixed2d.choose(black_box(task)),
                    "single_stump" => p_stump.choose(black_box(task)),
                    "oracle" => opt,
                    _ => 0,
                });
                let elapsed = t0.elapsed().as_nanos();

                let fb = run_feedback(task, chosen, &env, &mut exec_rng);

                // Update feedback for dynamic routers
                let is_healthy = env.x3_active;
                match pol {
                    "zero_baseline" => p_zero_base.observe_feedback(chosen, fb, is_healthy),
                    "zero_circuit_breaker" => p_zero_cb.observe_feedback(chosen, fb, is_healthy),
                    "zero_ood_guard" => p_zero_ood.observe_feedback(chosen, fb, is_healthy),
                    "zero_full_composition" => p_zero_full.observe_feedback(chosen, fb, is_healthy),
                    _ => {}
                }

                // Track Phase 4 outage calls to the broken tool (Tool 3)
                if env.phase == 4 && chosen == 3 {
                    outage_broken_calls[p_idx] += 1;
                }

                // Track Phase 5 recovery: how quickly does it reuse Tool 3 when Oracle says 3?
                if env.phase == 5 && opt == 3 && chosen == 3 && first_x3_recovery_task[p_idx].is_none() {
                    first_x3_recovery_task[p_idx] = Some(task_idx - 2500);
                }

                pol_metrics[p_idx].add(chosen, fb, opt, elapsed);
                phase_metrics[p_idx][phase_idx].add(chosen, fb, opt, elapsed);
                scen_metrics[p_idx][scen_idx].add(chosen, fb, opt, elapsed);
            }
        }

        // Summary and CSV outputs
        for (p_idx, &pol) in policy_names.iter().enumerate() {
            let mut m = pol_metrics[p_idx].clone();
            let (p50, p95) = m.p50_p95();
            let mean = m.mean_cost();
            let ns = m.elapsed_nanos / m.n.max(1) as u128;

            writeln!(
                summary,
                "{seed},{pol},{:.4},{:.4},{:.4},{:.2},{:.2},{}",
                mean,
                p50,
                p95,
                m.first_pct(),
                m.oracle_pct(),
                ns
            )?;

            // Phase metrics
            let phase_names = ["1_Normal", "2_FlakySurge", "3_CostSpike", "4_Outage", "5_Recovery"];
            for ph in 0..5 {
                let pm = &phase_metrics[p_idx][ph];
                writeln!(
                    phases_csv,
                    "{seed},{},{},{pol},{},{:.4},{:.2},{:.2}",
                    ph + 1,
                    phase_names[ph],
                    pm.n,
                    pm.mean_cost(),
                    pm.first_pct(),
                    pm.oracle_pct()
                )?;
            }

            // Scenario metrics
            for sc in 0..5 {
                let sm = &scen_metrics[p_idx][sc];
                writeln!(
                    scenario_csv,
                    "{seed},{},{pol},{},{:.4},{:.2},{:.2}",
                    SCENARIO_KINDS[sc],
                    sm.n,
                    sm.mean_cost(),
                    sm.first_pct(),
                    sm.oracle_pct()
                )?;
            }

            // Composition metrics
            let outage_cost = phase_metrics[p_idx][3].mean_cost();
            let broken_calls = outage_broken_calls[p_idx];
            let recovery_delay = first_x3_recovery_task[p_idx].unwrap_or(500);
            let (false_benches, false_oods) = match pol {
                "zero_circuit_breaker" => (p_zero_cb.circuit_breaker.as_ref().map(|c| c.false_benches).unwrap_or(0), 0),
                "zero_ood_guard" => (0, p_zero_ood.false_ood_rejections),
                "zero_full_composition" => (
                    p_zero_full.circuit_breaker.as_ref().map(|c| c.false_benches).unwrap_or(0),
                    p_zero_full.false_ood_rejections,
                ),
                _ => (0, 0),
            };
            let ood_extreme_cost = scen_metrics[p_idx][4].mean_cost();

            writeln!(
                comp_csv,
                "{seed},{pol},{:.4},{},{},{},{},{:.4}",
                outage_cost,
                broken_calls,
                recovery_delay,
                false_benches,
                false_oods,
                ood_extreme_cost
            )?;
        }

        if s % 5 == 0 || s == seeds - 1 {
            println!(
                "Seed {:>2}/{} evaluated | ZERO Baseline={:.4} -> Full Composition={:.4} (Oracle={:.4})",
                s + 1,
                seeds,
                pol_metrics[0].mean_cost(),
                pol_metrics[3].mean_cost(),
                pol_metrics[7].mean_cost(),
            );
        }
    }

    println!("=========================================================================================");
    println!("Composition benchmark finished. Results written to: {out_dir}/");
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("Benchmark error: {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ood_guard_identifies_extreme_points() {
        let (train_data, _) = collect_training_samples(42, 1000);
        let guard = OodGuard::build(&train_data);

        // In-distribution task near origin
        let in_task = Task { a: 80, b: 80, kind_tag: 0 };
        assert!(!guard.is_ood(in_task), "In-distribution task should NOT be flagged as OOD");

        // Out-of-distribution task far outside training envelope
        let ood_task = Task { a: 370, b: 370, kind_tag: 4 };
        assert!(guard.is_ood(ood_task), "Extreme point outside boundary MUST be flagged as OOD");
    }

    #[test]
    fn circuit_breaker_benches_and_recovers() {
        let mut cb = CircuitBreaker::default();
        let fb_fail = Feedback { tool_cost: 2.0, first_ok: false, total_cost: 11.5 };
        let fb_ok = Feedback { tool_cost: 2.0, first_ok: true, total_cost: 2.0 };

        // 1st failure: not benched yet
        cb.observe(3, fb_fail, false);
        assert!(!cb.is_benched(3));

        // 2nd consecutive failure: tool 3 MUST be benched
        cb.observe(3, fb_fail, false);
        assert!(cb.is_benched(3), "Tool 3 should be benched after 2 failures");

        // Cooldown ticks down
        for _ in 0..CB_BENCH_BASE {
            cb.tick();
        }
        assert!(!cb.is_benched(3), "Cooldown expired; tool should be ready for re-probing");

        // Successful re-probe restores trust
        cb.observe(3, fb_ok, true);
        assert!(!cb.is_benched(3));
        assert_eq!(cb.consecutive_failures[3], 0);
    }

    #[test]
    fn universal_is_never_benched() {
        let mut cb = CircuitBreaker::default();
        let fb_fail = Feedback { tool_cost: 9.5, first_ok: false, total_cost: 19.0 };
        for _ in 0..10 {
            cb.observe(0, fb_fail, true);
        }
        assert!(!cb.is_benched(0));
    }
}
