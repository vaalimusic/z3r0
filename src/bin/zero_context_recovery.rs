//! ZERO Context-Aware Recovery Benchmark 0.1
//! Evaluates Two-Level Tool Health: Global Health vs Context Health.
//!
//! Architectures compared:
//! 1. ZERO Baseline (frozen 2D tree)
//! 2. ZERO + Global Circuit Breaker
//! 3. ZERO + Context Circuit Breaker
//! 4. ZERO + Dual Health (Global + Context)
//! 5. Oracle (Optimal expected cost)
//!
//! Event Scenarios:
//! - Full Outage (100% failure on all inputs)
//! - Contextual Outage (failure restricted to specific input sub-region)
//! - Probabilistic Degradation (smooth failure rate shift)
//! - Recovery (unannounced restoration)
//! - Task-induced OOD stress (all specialists fail)
//!
//! 30 fresh independent evaluation seeds, paired evaluations, no secret simulator leakage.

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

// Discrete Context Grid: 4x4 = 16 contexts based on input magnitude
pub const CONTEXT_BINS: usize = 4;
pub const CONTEXTS: usize = CONTEXT_BINS * CONTEXT_BINS; // 16

// Circuit Breaker parameters
pub const CB_STREAK_TO_BENCH: u32 = 2;
pub const CB_BENCH_BASE: u32 = 4;
pub const CB_BENCH_MAX: u32 = 32;

// Dual Health threshold: if tool fails in >= 2 distinct contexts within window, escalate to Global Outage
pub const DUAL_DISTINCT_CONTEXTS_FOR_GLOBAL: usize = 2;

#[derive(Clone, Copy, Debug)]
pub struct Task {
    pub a: u32,
    pub b: u32,
    pub kind_tag: usize,
}

impl Task {
    #[inline(always)]
    pub fn context_id(&self) -> usize {
        let ca = (self.a as usize / 85).min(CONTEXT_BINS - 1);
        let cb = (self.b as usize / 85).min(CONTEXT_BINS - 1);
        ca * CONTEXT_BINS + cb
    }
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
    pub x2_contextual_outage: bool, // When true, X2 fails if a > 200
    pub x3_full_outage: bool,       // When true, X3 fails 100%
    pub x4_midpoint: f64,           // Sigmoid threshold for X4
}

impl Default for EnvState {
    fn default() -> Self {
        Self {
            phase: 1,
            x2_contextual_outage: false,
            x3_full_outage: false,
            x4_midpoint: 180.0,
        }
    }
}

impl EnvState {
    pub fn for_task(task_idx: usize) -> Self {
        match task_idx {
            0..=599 => Self {
                phase: 1, // Phase 1: Normal
                x2_contextual_outage: false,
                x3_full_outage: false,
                x4_midpoint: 180.0,
            },
            600..=1199 => Self {
                phase: 2, // Phase 2: Contextual failure: X2 fails ONLY when a > 200
                x2_contextual_outage: true,
                x3_full_outage: false,
                x4_midpoint: 180.0,
            },
            1200..=1799 => Self {
                phase: 3, // Phase 3: Probabilistic degradation: X4 midpoint shifts to 130
                x2_contextual_outage: false,
                x3_full_outage: false,
                x4_midpoint: 130.0,
            },
            1800..=2399 => Self {
                phase: 4, // Phase 4: Full Outage: X3 is 100% offline
                x2_contextual_outage: false,
                x3_full_outage: true,
                x4_midpoint: 180.0,
            },
            _ => Self {
                phase: 5, // Phase 5: Recovery: everything back to normal
                x2_contextual_outage: false,
                x3_full_outage: false,
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

pub fn call_cost(tool: usize) -> f64 {
    match tool {
        0 => 9.5,
        1 => 1.0,
        2 => 1.6,
        3 => 2.0,
        4 => 1.2,
        _ => 10.0,
    }
}

pub fn reference_answer(t: Task) -> i64 {
    t.a as i64 * 43 + t.b as i64 * 67
}

pub fn execute_tool(t: Task, tool: usize, env: &EnvState, rng: &mut Rng) -> Option<i64> {
    let ans = reference_answer(t);
    match tool {
        0 => Some(ans), // Universal always works
        1 => {
            // Conjunction
            if (t.a + t.b <= 160) && (t.a.abs_diff(t.b) <= 40) {
                Some(ans)
            } else {
                None
            }
        }
        2 => {
            // Diagonal specialist: if contextual outage active and a > 200, it fails
            if env.x2_contextual_outage && t.a > 200 {
                None
            } else if (t.a as f64 >= 1.4 * t.b as f64) && (t.a + t.b <= 350) {
                Some(ans)
            } else {
                None
            }
        }
        3 => {
            // Disconnected specialist: full outage in Phase 4
            if env.x3_full_outage {
                None
            } else if (t.a + t.b >= 240 && t.a >= 100 && t.b >= 100) || (t.a <= 60 && t.b >= 200) {
                Some(ans)
            } else {
                None
            }
        }
        4 => {
            // Flaky tool: sigmoid degradation
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
    let cost = call_cost(tool);
    let ok = execute_tool(t, tool, env, rng) == Some(reference_answer(t));
    let fallback = if ok { 0.0 } else { call_cost(0) };
    Feedback {
        tool_cost: cost,
        first_ok: ok,
        total_cost: cost + fallback,
    }
}

pub fn true_expected_cost(t: Task, tool: usize, env: &EnvState) -> f64 {
    let base = call_cost(tool);
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
            if env.x2_contextual_outage && t.a > 200 {
                1.0
            } else if (t.a as f64 >= 1.4 * t.b as f64) && (t.a + t.b <= 350) {
                0.0
            } else {
                1.0
            }
        }
        3 => {
            if env.x3_full_outage {
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
    base + p_fail * call_cost(0)
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

pub fn collect_training_samples(seed: u64, n: usize) -> Vec<Sample> {
    let mut rng = Rng::new(seed);
    let default_env = EnvState::default();
    (0..n)
        .map(|_| {
            let t = generate_task(&mut rng, false);
            let outcomes = std::array::from_fn(|i| {
                let mut tool_rng = Rng::new(rng.next());
                run_feedback(t, i, &default_env, &mut tool_rng)
            });
            Sample { task: t, outcomes }
        })
        .collect()
}

// ---------------- Tree2D (Unmodified core learner) ----------------
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

// ---------------- Health Tracker Implementations ----------------

// Mode 1: Global Circuit Breaker Only
#[derive(Clone, Default)]
pub struct GlobalCb {
    consecutive_failures: [u32; TOOLS],
    bench_timer: [u32; TOOLS],
    bench_cooldown: [u32; TOOLS],
    pub false_global_benches: usize,
    pub probe_calls: usize,
}
impl GlobalCb {
    pub fn is_benched(&self, tool: usize, _ctx: usize) -> bool {
        self.bench_timer[tool] > 0
    }
    pub fn tick(&mut self) {
        for t in 1..TOOLS {
            if self.bench_timer[t] > 0 {
                self.bench_timer[t] -= 1;
            }
        }
    }
    pub fn observe(&mut self, tool: usize, _ctx: usize, ok: bool, is_tool_globally_dead: bool) {
        if tool == 0 {
            return;
        }
        if ok {
            self.consecutive_failures[tool] = 0;
            self.bench_cooldown[tool] = 0;
            self.bench_timer[tool] = 0;
        } else {
            self.consecutive_failures[tool] += 1;
            if self.consecutive_failures[tool] >= CB_STREAK_TO_BENCH {
                let cur = self.bench_cooldown[tool];
                let next = if cur == 0 { CB_BENCH_BASE } else { (cur * 2).min(CB_BENCH_MAX) };
                self.bench_cooldown[tool] = next;
                self.bench_timer[tool] = next;
                self.consecutive_failures[tool] = 0;
                if !is_tool_globally_dead {
                    self.false_global_benches += 1;
                }
            }
        }
    }
}

// Mode 2: Context-Only Circuit Breaker
#[derive(Clone)]
pub struct ContextCb {
    ctx_failures: [[u32; CONTEXTS]; TOOLS],
    ctx_timer: [[u32; CONTEXTS]; TOOLS],
    ctx_cooldown: [[u32; CONTEXTS]; TOOLS],
    pub false_context_benches: usize,
    pub probe_calls: usize,
}
impl Default for ContextCb {
    fn default() -> Self {
        Self {
            ctx_failures: [[0; CONTEXTS]; TOOLS],
            ctx_timer: [[0; CONTEXTS]; TOOLS],
            ctx_cooldown: [[0; CONTEXTS]; TOOLS],
            false_context_benches: 0,
            probe_calls: 0,
        }
    }
}
impl ContextCb {
    pub fn is_benched(&self, tool: usize, ctx: usize) -> bool {
        self.ctx_timer[tool][ctx] > 0
    }
    pub fn tick(&mut self) {
        for t in 1..TOOLS {
            for c in 0..CONTEXTS {
                if self.ctx_timer[t][c] > 0 {
                    self.ctx_timer[t][c] -= 1;
                }
            }
        }
    }
    pub fn observe(&mut self, tool: usize, ctx: usize, ok: bool, is_context_actually_dead: bool) {
        if tool == 0 {
            return;
        }
        if ok {
            self.ctx_failures[tool][ctx] = 0;
            self.ctx_cooldown[tool][ctx] = 0;
            self.ctx_timer[tool][ctx] = 0;
        } else {
            self.ctx_failures[tool][ctx] += 1;
            if self.ctx_failures[tool][ctx] >= CB_STREAK_TO_BENCH {
                let cur = self.ctx_cooldown[tool][ctx];
                let next = if cur == 0 { CB_BENCH_BASE } else { (cur * 2).min(CB_BENCH_MAX) };
                self.ctx_cooldown[tool][ctx] = next;
                self.ctx_timer[tool][ctx] = next;
                self.ctx_failures[tool][ctx] = 0;
                if !is_context_actually_dead {
                    self.false_context_benches += 1;
                }
            }
        }
    }
}

// Mode 3: Dual Health (Global + Context Health)
#[derive(Clone, Default)]
pub struct DualHealth {
    // Context-level state
    ctx_failures: [[u32; CONTEXTS]; TOOLS],
    ctx_timer: [[u32; CONTEXTS]; TOOLS],
    ctx_cooldown: [[u32; CONTEXTS]; TOOLS],

    // Global-level state
    global_timer: [u32; TOOLS],
    global_cooldown: [u32; TOOLS],
    recent_failed_contexts: [u32; TOOLS], // Bitmask of distinct contexts where tool failed recently

    pub false_global_benches: usize,
    pub false_context_benches: usize,
    pub probe_calls: usize,
}
impl DualHealth {
    pub fn is_benched(&self, tool: usize, ctx: usize) -> bool {
        if self.global_timer[tool] > 0 {
            return true; // Globally benched across all contexts
        }
        self.ctx_timer[tool][ctx] > 0 // Contextually benched
    }

    pub fn tick(&mut self) {
        for t in 1..TOOLS {
            if self.global_timer[t] > 0 {
                self.global_timer[t] -= 1;
            }
            for c in 0..CONTEXTS {
                if self.ctx_timer[t][c] > 0 {
                    self.ctx_timer[t][c] -= 1;
                }
            }
        }
    }

    pub fn observe(
        &mut self,
        tool: usize,
        ctx: usize,
        ok: bool,
        is_tool_globally_dead: bool,
        is_context_actually_dead: bool,
    ) {
        if tool == 0 {
            return;
        }

        if ok {
            // Success restores context trust
            self.ctx_failures[tool][ctx] = 0;
            self.ctx_cooldown[tool][ctx] = 0;
            self.ctx_timer[tool][ctx] = 0;

            // If it succeeds, it also clears global outage suspicion
            self.recent_failed_contexts[tool] = 0;
            self.global_timer[tool] = 0;
            self.global_cooldown[tool] = 0;
        } else {
            // Failure in context `ctx`
            self.ctx_failures[tool][ctx] += 1;
            self.recent_failed_contexts[tool] |= 1 << ctx;

            // 1. Context-level escalation
            if self.ctx_failures[tool][ctx] >= CB_STREAK_TO_BENCH {
                let cur = self.ctx_cooldown[tool][ctx];
                let next = if cur == 0 { CB_BENCH_BASE } else { (cur * 2).min(CB_BENCH_MAX) };
                self.ctx_cooldown[tool][ctx] = next;
                self.ctx_timer[tool][ctx] = next;
                self.ctx_failures[tool][ctx] = 0;
                if !is_context_actually_dead {
                    self.false_context_benches += 1;
                }
            }

            // 2. Global-level escalation: failure confirmed across >= 2 distinct contexts
            let distinct_contexts = self.recent_failed_contexts[tool].count_ones() as usize;
            if distinct_contexts >= DUAL_DISTINCT_CONTEXTS_FOR_GLOBAL {
                let cur_g = self.global_cooldown[tool];
                let next_g = if cur_g == 0 { CB_BENCH_BASE } else { (cur_g * 2).min(CB_BENCH_MAX) };
                self.global_cooldown[tool] = next_g;
                self.global_timer[tool] = next_g;
                self.recent_failed_contexts[tool] = 0; // Reset after global bench

                if !is_tool_globally_dead {
                    self.false_global_benches += 1;
                }
            }
        }
    }
}

// ---------------- Health Strategy Wrapper ----------------
#[derive(Clone)]
pub enum HealthStrategy {
    None,
    Global(GlobalCb),
    Context(ContextCb),
    Dual(DualHealth),
}

#[derive(Clone)]
pub struct ZeroRouter {
    pub trees: [Tree2D; TOOLS],
    pub health: HealthStrategy,
}

impl ZeroRouter {
    pub fn new(train_data: &[Sample], strategy: HealthStrategy) -> Self {
        let trees = std::array::from_fn(|i| fit_tree2d(train_data, i));
        Self {
            trees,
            health: strategy,
        }
    }

    pub fn choose(&self, t: Task) -> usize {
        let preds: [f64; TOOLS] = std::array::from_fn(|i| self.trees[i].predict(t));
        let ctx = t.context_id();

        let mut best_i = 0;
        let mut best_val = f64::INFINITY;

        for i in 0..TOOLS {
            if i != 0 {
                let is_benched = match self.health {
                    HealthStrategy::None => false,
                    HealthStrategy::Global(ref g) => g.is_benched(i, ctx),
                    HealthStrategy::Context(ref c) => c.is_benched(i, ctx),
                    HealthStrategy::Dual(ref d) => d.is_benched(i, ctx),
                };
                if is_benched {
                    continue;
                }
            }
            if preds[i] < best_val {
                best_val = preds[i];
                best_i = i;
            }
        }
        best_i
    }

    pub fn observe_feedback(
        &mut self,
        tool: usize,
        task: Task,
        fb: Feedback,
        is_tool_globally_dead: bool,
        is_context_actually_dead: bool,
    ) {
        let ctx = task.context_id();
        match self.health {
            HealthStrategy::None => {}
            HealthStrategy::Global(ref mut g) => {
                g.observe(tool, ctx, fb.first_ok, is_tool_globally_dead);
                g.tick();
            }
            HealthStrategy::Context(ref mut c) => {
                c.observe(tool, ctx, fb.first_ok, is_context_actually_dead);
                c.tick();
            }
            HealthStrategy::Dual(ref mut d) => {
                d.observe(
                    tool,
                    ctx,
                    fb.first_ok,
                    is_tool_globally_dead,
                    is_context_actually_dead,
                );
                d.tick();
            }
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

    let out_dir = "reports/zero_context_recovery";
    fs::create_dir_all(out_dir)?;

    let mut summary = File::create(format!("{out_dir}/summary.csv"))?;
    let mut phases_csv = File::create(format!("{out_dir}/phases.csv"))?;
    let mut event_csv = File::create(format!("{out_dir}/event_metrics.csv"))?;
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
        event_csv,
        "seed,policy,false_global_benches,false_context_benches,outage_detection_tasks,recovery_delay_tasks,phase2_context_cost,phase4_full_outage_cost"
    )?;
    writeln!(
        scenario_csv,
        "seed,scenario,policy,count,mean_cost,first_ok_pct,oracle_match_pct"
    )?;

    println!("=========================================================================================");
    println!("ZERO CONTEXT-AWARE RECOVERY BENCHMARK 0.1 (Global Health vs Context Health)");
    println!("Seeds: {seeds} (Fresh independent evaluation seeds, distinct from dev tuning)");
    println!("Tasks per seed: {TRAIN_TASKS} train / {TEST_TASKS} test (90,000 paired decisions per strategy)");
    println!("Events: P1(Normal) -> P2(ContextualOutage) -> P3(FlakyShift) -> P4(FullOutage) -> P5(Recovery)");
    println!("=========================================================================================");

    const NUM_POLICIES: usize = 5;
    let policy_names = [
        "zero_baseline",
        "zero_global_cb",
        "zero_context_cb",
        "zero_dual_health",
        "oracle",
    ];

    for s in 0..seeds {
        let seed = 20280001u64 + (s as u64) * 7919;
        let train_data = collect_training_samples(seed ^ 0xAAAA_1111, TRAIN_TASKS);

        // Build the 4 ZERO variants
        let mut r_base = ZeroRouter::new(&train_data, HealthStrategy::None);
        let mut r_global = ZeroRouter::new(&train_data, HealthStrategy::Global(GlobalCb::default()));
        let mut r_context = ZeroRouter::new(&train_data, HealthStrategy::Context(ContextCb::default()));
        let mut r_dual = ZeroRouter::new(&train_data, HealthStrategy::Dual(DualHealth::default()));

        // Pre-generate paired test tasks
        let mut test_rng = Rng::new(seed ^ 0xCCCC_3333);
        let test_tasks: Vec<Task> = (0..TEST_TASKS)
            .map(|_| generate_task(&mut test_rng, true))
            .collect();

        let mut pol_metrics: Vec<Metric> = vec![Metric::default(); NUM_POLICIES];
        let mut phase_metrics: Vec<Vec<Metric>> = vec![vec![Metric::default(); 5]; NUM_POLICIES];
        let mut scen_metrics: Vec<Vec<Metric>> = vec![vec![Metric::default(); 5]; NUM_POLICIES];

        // Outage detection & recovery tracking
        let mut first_outage_avoided_task = [None; NUM_POLICIES];
        let mut first_recovery_reused_task = [None; NUM_POLICIES];

        for (task_idx, &task) in test_tasks.iter().enumerate() {
            let env = EnvState::for_task(task_idx);
            let opt = oracle_choice(task, &env);
            let phase_idx = env.phase - 1;
            let scen_idx = task.kind_tag;

            for (p_idx, &pol) in policy_names.iter().enumerate() {
                let mut exec_rng = Rng::new(seed ^ 0x5555_4444 ^ ((task_idx as u64) << 16));

                let t0 = Instant::now();
                let chosen = black_box(match pol {
                    "zero_baseline" => r_base.choose(black_box(task)),
                    "zero_global_cb" => r_global.choose(black_box(task)),
                    "zero_context_cb" => r_context.choose(black_box(task)),
                    "zero_dual_health" => r_dual.choose(black_box(task)),
                    "oracle" => opt,
                    _ => 0,
                });
                let elapsed = t0.elapsed().as_nanos();

                let fb = run_feedback(task, chosen, &env, &mut exec_rng);

                // Ground-truth health context (used ONLY for measuring false positive metrics, NOT given to router)
                let is_tool_globally_dead = env.phase == 4 && chosen == 3;
                let is_context_actually_dead = is_tool_globally_dead
                    || (env.phase == 2 && chosen == 2 && task.a > 200);

                match pol {
                    "zero_baseline" => r_base.observe_feedback(chosen, task, fb, is_tool_globally_dead, is_context_actually_dead),
                    "zero_global_cb" => r_global.observe_feedback(chosen, task, fb, is_tool_globally_dead, is_context_actually_dead),
                    "zero_context_cb" => r_context.observe_feedback(chosen, task, fb, is_tool_globally_dead, is_context_actually_dead),
                    "zero_dual_health" => r_dual.observe_feedback(chosen, task, fb, is_tool_globally_dead, is_context_actually_dead),
                    _ => {}
                }

                // Measure outage detection: how many tasks into Phase 4 (1800..) until tool 3 is avoided?
                if env.phase == 4 && opt != 3 && chosen != 3 && first_outage_avoided_task[p_idx].is_none() {
                    first_outage_avoided_task[p_idx] = Some(task_idx - 1800);
                }

                // Measure recovery delay: how many tasks into Phase 5 (2400..) until tool 3 is re-utilized?
                if env.phase == 5 && opt == 3 && chosen == 3 && first_recovery_reused_task[p_idx].is_none() {
                    first_recovery_reused_task[p_idx] = Some(task_idx - 2400);
                }

                pol_metrics[p_idx].add(chosen, fb, opt, elapsed);
                phase_metrics[p_idx][phase_idx].add(chosen, fb, opt, elapsed);
                scen_metrics[p_idx][scen_idx].add(chosen, fb, opt, elapsed);
            }
        }

        // Write outputs for this seed
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
            let phase_names = ["1_Normal", "2_ContextualOutage", "3_FlakyShift", "4_FullOutage", "5_Recovery"];
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

            // Event and health metrics
            let (false_g, false_c) = match pol {
                "zero_global_cb" => {
                    if let HealthStrategy::Global(ref g) = r_global.health {
                        (g.false_global_benches, 0)
                    } else {
                        (0, 0)
                    }
                }
                "zero_context_cb" => {
                    if let HealthStrategy::Context(ref c) = r_context.health {
                        (0, c.false_context_benches)
                    } else {
                        (0, 0)
                    }
                }
                "zero_dual_health" => {
                    if let HealthStrategy::Dual(ref d) = r_dual.health {
                        (d.false_global_benches, d.false_context_benches)
                    } else {
                        (0, 0)
                    }
                }
                _ => (0, 0),
            };

            let outage_det = first_outage_avoided_task[p_idx].unwrap_or(600);
            let rec_delay = first_recovery_reused_task[p_idx].unwrap_or(600);
            let p2_cost = phase_metrics[p_idx][1].mean_cost();
            let p4_cost = phase_metrics[p_idx][3].mean_cost();

            writeln!(
                event_csv,
                "{seed},{pol},{},{},{},{},{:.4},{:.4}",
                false_g,
                false_c,
                outage_det,
                rec_delay,
                p2_cost,
                p4_cost
            )?;
        }

        if s % 5 == 0 || s == seeds - 1 {
            println!(
                "Seed {:>2}/{} evaluated | Base={:.4} | GlobalCB={:.4} | ContextCB={:.4} | DualHealth={:.4} (Oracle={:.4})",
                s + 1,
                seeds,
                pol_metrics[0].mean_cost(),
                pol_metrics[1].mean_cost(),
                pol_metrics[2].mean_cost(),
                pol_metrics[3].mean_cost(),
                pol_metrics[4].mean_cost(),
            );
        }
    }

    println!("=========================================================================================");
    println!("Context-Aware Recovery benchmark finished. Results written to: {out_dir}/");
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("Execution error: {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_id_partitions_domain() {
        let t1 = Task { a: 10, b: 10, kind_tag: 0 };
        let t2 = Task { a: 300, b: 300, kind_tag: 0 };
        assert_eq!(t1.context_id(), 0);
        assert_eq!(t2.context_id(), CONTEXTS - 1);
    }

    #[test]
    fn contextual_cb_isolates_failures_to_specific_context() {
        let mut cb = ContextCb::default();
        let ctx_a = 0;
        let ctx_b = 5;

        // Fail twice in ctx_a -> tool 2 benched ONLY in ctx_a
        cb.observe(2, ctx_a, false, false);
        cb.observe(2, ctx_a, false, false);

        assert!(cb.is_benched(2, ctx_a), "Tool 2 should be benched in failing ctx_a");
        assert!(!cb.is_benched(2, ctx_b), "Tool 2 MUST remain available in other healthy contexts!");
    }

    #[test]
    fn dual_health_escalates_to_global_when_failures_span_distinct_contexts() {
        let mut dh = DualHealth::default();
        let ctx1 = 1;
        let ctx2 = 8;

        // Failure in first context
        dh.observe(3, ctx1, false, false, false);
        assert!(!dh.is_benched(3, ctx2), "Tool 3 not globally benched after 1 context failure");

        // Failure in a DIFFERENT context -> escalation to Global Outage
        dh.observe(3, ctx2, false, true, true);
        assert!(dh.is_benched(3, ctx1), "Tool 3 should be globally benched in ctx1");
        assert!(dh.is_benched(3, ctx2), "Tool 3 should be globally benched in ctx2");
        assert!(dh.is_benched(3, 0), "Tool 3 should be globally benched in ANY context");
    }
}
