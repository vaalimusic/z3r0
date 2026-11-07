//! ZERO Multidimensional Robustness Benchmark 0.1
//! Independent stress test of ZERO Multidim vs Strong Baselines:
//! - Unknown non-linear 2D capability bounds (conjunctions, diagonal, disconnected, flaky)
//! - Unannounced environment shifts (failure rate shift, cost spike, temporary outage, recovery)
//! - Full economic lifecycle accounting: probe training cost vs breakeven task horizon
//! - Strong 2D fixed baseline (6x6 grid with empirical Bayesian shrinkage)
//! - 30+ independent seeds, paired evaluations, p50/p95 distributions, black_box timing

use std::fs::{self, File};
use std::hint::black_box;
use std::io::{self, Write};
use std::time::Instant;

pub const TOOLS: usize = 5;
pub const FEATURES: usize = 7;
pub const TRAIN_TASKS: usize = 2400;
pub const VALID_TASKS: usize = 600;
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
                phase: 2, // Flaky failure rate surge: tool degrades much earlier
                x2_cost: 1.6,
                x3_active: true,
                x4_midpoint: 130.0,
            },
            1500..=1999 => Self {
                phase: 3, // Cost surge: X2 becomes expensive (5.0 instead of 1.6)
                x2_cost: 5.0,
                x3_active: true,
                x4_midpoint: 180.0,
            },
            2000..=2499 => Self {
                phase: 4, // Outage: X3 is completely down (100% fail)
                x2_cost: 1.6,
                x3_active: false,
                x4_midpoint: 180.0,
            },
            _ => Self {
                phase: 5, // Recovery: everything back to normal
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
            // Conjunction focus: moderate values near 140
            let a = r.range(40, 180);
            let b = (a as i64 + r.range(0, 60) as i64 - 30).clamp(20, 320) as u32;
            (a, b)
        }
        1 => {
            // Diagonal / Asymmetric focus: a dominant
            let b = r.range(20, 160);
            let a = (b as f64 * (1.3 + r.next_f64() * 0.8)).clamp(30.0, 340.0) as u32;
            (a, b)
        }
        2 => {
            // Disconnected regions focus: high load or corner
            if r.next() % 2 == 0 {
                (r.range(110, 280), r.range(110, 280))
            } else {
                (r.range(20, 65), r.range(200, 330))
            }
        }
        3 => {
            // Flaky sigmoid transition zone
            let sum = r.range(130, 230);
            let a = r.range(20, sum.saturating_sub(20));
            (a, sum - a)
        }
        _ => {
            // Extreme unhandled values: only universal works
            let (lo, hi) = if test_shift { (20, 360) } else { (15, 320) };
            (r.range(lo, hi), r.range(lo, hi))
        }
    };

    let kind_tag = classify_scenario(a, b);
    Task { a, b, kind_tag }
}

fn classify_scenario(a: u32, b: u32) -> usize {
    if a + b <= 160 && a.abs_diff(b) <= 40 {
        0 // conjunction_box
    } else if a as f64 >= 1.4 * b as f64 && a + b <= 350 {
        1 // diagonal_asymmetric
    } else if (a + b >= 240 && a >= 100 && b >= 100) || (a <= 60 && b >= 200) {
        2 // disconnected_islands
    } else if (130..=230).contains(&(a + b)) {
        3 // flaky_transition
    } else {
        4 // extreme_universal
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
        0 => Some(ans), // Universal always works
        1 => {
            // 2D Conjunction: sum <= 160 AND abs_diff <= 40
            if (t.a + t.b <= 160) && (t.a.abs_diff(t.b) <= 40) {
                Some(ans)
            } else {
                None
            }
        }
        2 => {
            // Diagonal / Asymmetric: a >= 1.4 * b AND a + b <= 350
            if (t.a as f64 >= 1.4 * t.b as f64) && (t.a + t.b <= 350) {
                Some(ans)
            } else {
                None
            }
        }
        3 => {
            // Disconnected / Non-convex regions:
            // Island 1: both large (sum >= 240, a >= 100, b >= 100)
            // Island 2: asymmetric corner (a <= 60, b >= 200)
            if !env.x3_active {
                None // Outage in Phase 4
            } else if (t.a + t.b >= 240 && t.a >= 100 && t.b >= 100) || (t.a <= 60 && t.b >= 200) {
                Some(ans)
            } else {
                None
            }
        }
        4 => {
            // Flaky tool: smooth sigmoid failure based on sum load
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

// ---------------- Policy: KindOnly ----------------
#[derive(Clone)]
pub struct KindOnly {
    means: [f64; TOOLS],
}
impl KindOnly {
    pub fn train(data: &[Sample]) -> Self {
        let mut sums = [0.0; TOOLS];
        for s in data {
            for i in 0..TOOLS {
                sums[i] += s.outcomes[i].total_cost;
            }
        }
        let n = data.len() as f64;
        Self {
            means: std::array::from_fn(|i| sums[i] / n),
        }
    }
    pub fn choose(&self) -> usize {
        (0..TOOLS)
            .min_by(|&i, &j| self.means[i].total_cmp(&self.means[j]))
            .unwrap_or(0)
    }
}

// ---------------- Policy: Fixed 1D Bins ----------------
#[derive(Clone)]
pub struct Fixed1DBins {
    grid: [[f64; TOOLS]; 8],
}
impl Fixed1DBins {
    pub fn train(data: &[Sample]) -> Self {
        let mut sums = [[0.0; TOOLS]; 8];
        let mut cnt = [0usize; 8];
        let prior = KindOnly::train(data);

        for s in data {
            let b = ((s.task.a + s.task.b) as usize / 60).min(7);
            cnt[b] += 1;
            for i in 0..TOOLS {
                sums[b][i] += s.outcomes[i].total_cost;
            }
        }
        let mut grid = [[0.0; TOOLS]; 8];
        for b in 0..8 {
            for i in 0..TOOLS {
                grid[b][i] = if cnt[b] < 10 {
                    prior.means[i]
                } else {
                    sums[b][i] / cnt[b] as f64
                };
            }
        }
        Self { grid }
    }
    pub fn choose(&self, t: Task) -> usize {
        let b = ((t.a + t.b) as usize / 60).min(7);
        (0..TOOLS)
            .min_by(|&i, &j| self.grid[b][i].total_cmp(&self.grid[b][j]))
            .unwrap_or(0)
    }
}

// ---------------- Strong 2D Fixed Baseline: 6x6 Grid with Bayesian Shrinkage ----------------
#[derive(Clone)]
pub struct Fixed2DGrid {
    grid: [[[f64; TOOLS]; 6]; 6],
}
impl Fixed2DGrid {
    pub fn train(data: &[Sample]) -> Self {
        let prior = KindOnly::train(data);
        let mut sums = [[[0.0; TOOLS]; 6]; 6];
        let mut counts = [[0usize; 6]; 6];

        for s in data {
            let ia = (s.task.a as usize / 55).min(5);
            let ib = (s.task.b as usize / 55).min(5);
            counts[ia][ib] += 1;
            for i in 0..TOOLS {
                sums[ia][ib][i] += s.outcomes[i].total_cost;
            }
        }

        let mut grid = [[[0.0; TOOLS]; 6]; 6];
        const PSEUDO_COUNT: f64 = 8.0;
        for ia in 0..6 {
            for ib in 0..6 {
                let n = counts[ia][ib] as f64;
                for i in 0..TOOLS {
                    // Empirical Bayesian shrinkage toward global prior
                    grid[ia][ib][i] = (sums[ia][ib][i] + PSEUDO_COUNT * prior.means[i])
                        / (n + PSEUDO_COUNT);
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

// ---------------- Policy: Single Stump (1D ZERO) ----------------
#[derive(Clone, Copy)]
pub struct Stump {
    pub feature: usize,
    pub cut: u32,
    pub below: f64,
    pub above: f64,
    pub score: f64,
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

pub fn fit_single_stump(data: &[Sample], tool: usize) -> Stump {
    let ys: Vec<f64> = data.iter().map(|s| s.outcomes[tool].total_cost).collect();
    let n = ys.len();
    let mean = ys.iter().sum::<f64>() / n as f64;
    let mut best = Stump {
        feature: 0,
        cut: u32::MAX,
        below: mean,
        above: mean,
        score: ys.iter().map(|y| (y - mean).powi(2)).sum(),
    };

    for feature in 0..FEATURES {
        let mut pairs: Vec<(u32, f64)> = data
            .iter()
            .zip(&ys)
            .map(|(s, y)| (extract_features(s.task)[feature], *y))
            .collect();
        pairs.sort_unstable_by_key(|p| p.0);

        let sum_total: f64 = pairs.iter().map(|p| p.1).sum();
        let sq_total: f64 = pairs.iter().map(|p| p.1 * p.1).sum();

        let mut left_sum = 0.0;
        let mut left_sq = 0.0;

        for idx in 0..n - 1 {
            let y = pairs[idx].1;
            left_sum += y;
            left_sq += y * y;
            let left_n = idx + 1;
            let right_n = n - left_n;

            if left_n < MIN_LEAF || right_n < MIN_LEAF || pairs[idx].0 == pairs[idx + 1].0 {
                continue;
            }

            let right_sum = sum_total - left_sum;
            let mse = (left_sq - left_sum * left_sum / left_n as f64)
                + ((sq_total - left_sq) - right_sum * right_sum / right_n as f64);

            if mse + 1e-9 < best.score {
                best = Stump {
                    feature,
                    cut: pairs[idx].0,
                    below: left_sum / left_n as f64,
                    above: right_sum / right_n as f64,
                    score: mse,
                };
            }
        }
    }
    best
}

#[derive(Clone)]
pub struct SingleStumpPolicy {
    pub stumps: [Stump; TOOLS],
}
impl SingleStumpPolicy {
    pub fn train(data: &[Sample]) -> Self {
        Self {
            stumps: std::array::from_fn(|i| fit_single_stump(data, i)),
        }
    }
    pub fn choose(&self, t: Task) -> usize {
        let preds: [f64; TOOLS] = std::array::from_fn(|i| self.stumps[i].predict(t));
        (0..TOOLS)
            .min_by(|&i, &j| preds[i].total_cmp(&preds[j]))
            .unwrap_or(0)
    }
}

// ---------------- Primary Tested Strategy: ZERO Multidim (2D Tree) ----------------
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

#[derive(Clone)]
pub struct ZeroMultidimensional {
    pub trees: [Tree2D; TOOLS],
}
impl ZeroMultidimensional {
    pub fn train(data: &[Sample]) -> Self {
        Self {
            trees: std::array::from_fn(|i| fit_tree2d(data, i)),
        }
    }
    pub fn choose(&self, t: Task) -> usize {
        let preds: [f64; TOOLS] = std::array::from_fn(|i| self.trees[i].predict(t));
        (0..TOOLS)
            .min_by(|&i, &j| preds[i].total_cmp(&preds[j]))
            .unwrap_or(0)
    }
}

// ---------------- Policy: Randomized Exploration + Greedy ----------------
#[derive(Clone)]
pub struct RandomizedGreedy {
    pub base_2d: Fixed2DGrid,
}
impl RandomizedGreedy {
    pub fn train(data: &[Sample]) -> Self {
        Self {
            base_2d: Fixed2DGrid::train(data),
        }
    }
    pub fn choose(&self, t: Task, rng: &mut Rng) -> usize {
        // 5% random exploratory probe among specialists 1..4
        if rng.next_f64() < 0.05 {
            rng.range(1, 4) as usize
        } else {
            self.base_2d.choose(t)
        }
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

    let out_dir = "reports/zero_multidim_robustness";
    fs::create_dir_all(out_dir)?;

    let mut summary = File::create(format!("{out_dir}/summary.csv"))?;
    let mut phases_csv = File::create(format!("{out_dir}/phases.csv"))?;
    let mut decisions_csv = File::create(format!("{out_dir}/decisions_sample.csv"))?;
    let mut learning_csv = File::create(format!("{out_dir}/learning_cost.csv"))?;
    let mut scenario_csv = File::create(format!("{out_dir}/scenario_breakdown.csv"))?;

    writeln!(
        summary,
        "seed,policy,mean_cost,p50_cost,p95_cost,first_ok_pct,oracle_match_pct,mean_decision_ns"
    )?;
    writeln!(
        phases_csv,
        "seed,phase,phase_name,policy,tasks,mean_cost,first_ok_pct,oracle_match_pct"
    )?;
    writeln!(
        decisions_csv,
        "seed,task_idx,phase,a,b,scenario,policy,chosen_tool,first_ok,cost,oracle_tool"
    )?;
    writeln!(
        learning_csv,
        "seed,policy,train_probe_calls,train_probe_cost,test_tasks,test_total_cost,lifetime_cost,breakeven_vs_universal"
    )?;
    writeln!(
        scenario_csv,
        "seed,scenario,policy,count,mean_cost,first_ok_pct,oracle_match_pct"
    )?;

    println!("=========================================================================================");
    println!("ZERO MULTIDIMENSIONAL ROBUSTNESS BENCHMARK 0.1");
    println!("Seeds: {seeds} | Tasks per seed: {TRAIN_TASKS} train / {VALID_TASKS} valid / {TEST_TASKS} test");
    println!("Non-Stationary Phases: P1(Normal) -> P2(FlakySurge) -> P3(CostSpike) -> P4(Outage) -> P5(Recovery)");
    println!("=========================================================================================");

    let policy_names = [
        "universal",
        "kind_only",
        "fixed_1d",
        "fixed_2d_grid",
        "single_stump",
        "zero_multidim",
        "random_greedy",
        "oracle",
    ];

    for s in 0..seeds {
        let seed = 20261107u64 + (s as u64) * 7919;
        let (train_data, probe_cost) = collect_training_samples(seed ^ 0xAAAA_1111, TRAIN_TASKS);

        // Train all models
        let p_kind = KindOnly::train(&train_data);
        let p_fixed1d = Fixed1DBins::train(&train_data);
        let p_fixed2d = Fixed2DGrid::train(&train_data);
        let p_stump = SingleStumpPolicy::train(&train_data);
        let p_zero = ZeroMultidimensional::train(&train_data);
        let p_rand = RandomizedGreedy::train(&train_data);

        // Pre-generate paired test tasks
        let mut test_rng = Rng::new(seed ^ 0xCCCC_3333);
        let test_tasks: Vec<Task> = (0..TEST_TASKS)
            .map(|_| generate_task(&mut test_rng, true))
            .collect();

        // Metrics tracking per policy
        let mut pol_metrics: Vec<Metric> = vec![Metric::default(); policy_names.len()];
        let mut phase_metrics: Vec<Vec<Metric>> = vec![vec![Metric::default(); 5]; policy_names.len()];
        let mut scen_metrics: Vec<Vec<Metric>> = vec![vec![Metric::default(); 5]; policy_names.len()];

        for (task_idx, &task) in test_tasks.iter().enumerate() {
            let env = EnvState::for_task(task_idx);
            let opt = oracle_choice(task, &env);
            let phase_idx = env.phase - 1;
            let scen_idx = task.kind_tag;

            for (p_idx, &pol) in policy_names.iter().enumerate() {
                // Ensure paired random seed for execution outcomes across policies
                let mut exec_rng = Rng::new(seed ^ 0x5555_4444 ^ ((task_idx as u64) << 16));
                let mut choose_rng = Rng::new(seed ^ 0x7777_8888 ^ ((task_idx as u64) << 16));

                let t0 = Instant::now();
                let chosen = black_box(match pol {
                    "universal" => 0,
                    "kind_only" => p_kind.choose(),
                    "fixed_1d" => p_fixed1d.choose(black_box(task)),
                    "fixed_2d_grid" => p_fixed2d.choose(black_box(task)),
                    "single_stump" => p_stump.choose(black_box(task)),
                    "zero_multidim" => p_zero.choose(black_box(task)),
                    "random_greedy" => p_rand.choose(black_box(task), &mut choose_rng),
                    "oracle" => opt,
                    _ => 0,
                });
                let elapsed = t0.elapsed().as_nanos();

                let fb = run_feedback(task, chosen, &env, &mut exec_rng);

                pol_metrics[p_idx].add(chosen, fb, opt, elapsed);
                phase_metrics[p_idx][phase_idx].add(chosen, fb, opt, elapsed);
                scen_metrics[p_idx][scen_idx].add(chosen, fb, opt, elapsed);

                if s == 0 && task_idx < 60 {
                    writeln!(
                        decisions_csv,
                        "{seed},{task_idx},{},{},{},{},{pol},{},{},{:.4},{}",
                        env.phase,
                        task.a,
                        task.b,
                        SCENARIO_KINDS[scen_idx],
                        TOOL_NAMES[chosen],
                        fb.first_ok,
                        fb.total_cost,
                        TOOL_NAMES[opt]
                    )?;
                }
            }
        }

        // Write summary and learning accounting for this seed
        let univ_cost = pol_metrics[0].total_cost;
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

            // Learning & lifecycle accounting
            let calls = if pol == "universal" { 0 } else { TRAIN_TASKS * TOOLS };
            let p_cost = if pol == "universal" { 0.0 } else { probe_cost };
            let life_cost = p_cost + m.total_cost;
            let diff_per_task = (univ_cost - m.total_cost) / TEST_TASKS as f64;
            let breakeven = if diff_per_task > 0.0 {
                (p_cost / diff_per_task).ceil() as i64
            } else {
                -1 // Never breaks even
            };

            writeln!(
                learning_csv,
                "{seed},{pol},{},{:.2},{},{:.2},{:.2},{}",
                calls,
                p_cost,
                TEST_TASKS,
                m.total_cost,
                life_cost,
                breakeven
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

            // Scenario breakdown
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
        }

        if s % 5 == 0 || s == seeds - 1 {
            println!(
                "Seed {:>2}/{} done: ZERO mean={:.4} (Oracle match={:.1}%), Fixed2D={:.4}, Univ={:.4}",
                s + 1,
                seeds,
                pol_metrics[5].mean_cost(),
                pol_metrics[5].oracle_pct(),
                pol_metrics[3].mean_cost(),
                pol_metrics[0].mean_cost(),
            );
        }
    }

    println!("=========================================================================================");
    println!("Robustness benchmark finished. Results written to: {out_dir}/");
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("Benchmark failed: {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_matches_formula() {
        let t = Task { a: 10, b: 20, kind_tag: 0 };
        assert_eq!(reference_answer(t), 10 * 41 + 20 * 59);
    }

    #[test]
    fn true_expected_cost_matches_definition() {
        let env = EnvState::default();
        let t = Task { a: 50, b: 50, kind_tag: 0 }; // in conjunction box
        // Tool 1 succeeds with p=1.0 -> cost = 1.0 + 0 * 9.5 = 1.0
        assert_eq!(true_expected_cost(t, 1, &env), 1.0);

        // Tool 1 outside box: sum = 200 > 160 -> p_fail = 1.0 -> cost = 1.0 + 9.5 = 10.5
        let t_out = Task { a: 100, b: 100, kind_tag: 0 };
        assert_eq!(true_expected_cost(t_out, 1, &env), 10.5);
    }

    #[test]
    fn disconnected_region_boundaries() {
        let env = EnvState::default();
        let mut rng = Rng::new(101);
        // Island 1: a=120, b=130 (sum >= 240, both >= 100)
        let t1 = Task { a: 120, b: 130, kind_tag: 2 };
        assert!(execute_tool(t1, 3, &env, &mut rng).is_some());

        // Island 2: a=30, b=220 (a <= 60, b >= 200)
        let t2 = Task { a: 30, b: 220, kind_tag: 2 };
        assert!(execute_tool(t2, 3, &env, &mut rng).is_some());

        // Gap / Outside: a=80, b=150 (sum = 230 < 240, not a <= 60)
        let t_gap = Task { a: 80, b: 150, kind_tag: 2 };
        assert!(execute_tool(t_gap, 3, &env, &mut rng).is_none());
    }

    #[test]
    fn environment_phases_behave_correctly() {
        // Phase 1 (0..999): normal
        assert_eq!(EnvState::for_task(500).phase, 1);
        assert_eq!(EnvState::for_task(500).x2_cost, 1.6);
        assert!(EnvState::for_task(500).x3_active);

        // Phase 2 (1000..1499): flaky midpoint shifted
        assert_eq!(EnvState::for_task(1200).phase, 2);
        assert_eq!(EnvState::for_task(1200).x4_midpoint, 130.0);

        // Phase 3 (1500..1999): cost surge
        assert_eq!(EnvState::for_task(1700).phase, 3);
        assert_eq!(EnvState::for_task(1700).x2_cost, 5.0);

        // Phase 4 (2000..2499): outage
        assert_eq!(EnvState::for_task(2200).phase, 4);
        assert!(!EnvState::for_task(2200).x3_active);

        // Phase 5 (2500..2999): recovered
        assert_eq!(EnvState::for_task(2700).phase, 5);
        assert!(EnvState::for_task(2700).x3_active);
    }

    #[test]
    fn oracle_achieves_minimal_expected_cost() {
        let env = EnvState::default();
        let mut rng = Rng::new(555);
        for _ in 0..100 {
            let t = generate_task(&mut rng, true);
            let opt = oracle_choice(t, &env);
            let opt_cost = true_expected_cost(t, opt, &env);
            for i in 0..TOOLS {
                assert!(opt_cost <= true_expected_cost(t, i, &env) + 1e-9);
            }
        }
    }
}
