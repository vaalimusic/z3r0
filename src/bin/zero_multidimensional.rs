//! ZERO Multidimensional & Flaky Tools Benchmark.
//! Evaluates decision making under 2D non-linear tool competence boundaries
//! and smooth probabilistic failure rates (flakiness).
//! Zero external dependencies; fully reproducible with independent seeds.

use std::fs::{self, File};
use std::io::{self, Write};
use std::time::Instant;

const TOOLS: usize = 5;
const FEATURES: usize = 7;
const TRAIN_TASKS: usize = 3000;
const VALID_TASKS: usize = 600;
const TEST_TASKS: usize = 3000;
const MIN_LEAF: usize = 30;

const TOOL_NAMES: [&str; TOOLS] = [
    "X0_universal",
    "X1_narrow_box",
    "X2_asymmetric",
    "X3_flaky_sigmoid",
    "X4_high_load",
];

#[allow(dead_code)]
const FEATURE_NAMES: [&str; FEATURES] = [
    "a", "b", "min", "max", "sum", "abs_diff", "product",
];

#[derive(Clone, Copy, Debug)]
pub struct Task {
    pub a: u32,
    pub b: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct Feedback {
    pub tool_cost: f64,
    pub first_ok: bool,
    pub total_cost: f64,
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
    if test_shift && r.next() % 5 == 0 {
        // Sample around non-linear boundary transitions
        let a = r.range(60, 160);
        let b = (a as i64 + r.range(0, 70) as i64 - 35).clamp(20, 350) as u32;
        Task { a, b }
    } else {
        let (lo, hi) = if test_shift { (20, 360) } else { (15, 330) };
        Task {
            a: r.range(lo, hi),
            b: r.range(lo, hi),
        }
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
        (t.a as u64 * t.b as u64 / 100).min(u32::MAX as u64) as u32,
    ]
}

pub fn call_cost(tool: usize) -> f64 {
    match tool {
        0 => 9.5, // Universal: reliable but heavy
        1 => 0.8, // X1: cheap specialist for balanced small tasks
        2 => 1.4, // X2: asymmetric specialist
        3 => 1.1, // X3: flaky sigmoid tool
        4 => 2.2, // X4: high-load specialist
        _ => 10.0,
    }
}

pub fn reference_answer(t: Task) -> i64 {
    t.a as i64 * 37 + t.b as i64 * 53
}

pub fn execute_tool(t: Task, tool: usize, rng: &mut Rng) -> Option<i64> {
    let ans = reference_answer(t);
    match tool {
        0 => Some(ans), // Universal always works
        1 => {
            // Non-linear 2D boundary: sum <= 160 AND |a - b| <= 35
            if (t.a + t.b <= 160) && (t.a.abs_diff(t.b) <= 35) {
                Some(ans)
            } else {
                None
            }
        }
        2 => {
            // Asymmetric boundary: a >= 1.5 * b AND a <= 260
            if (t.a as f64 >= 1.5 * t.b as f64) && t.a <= 260 {
                Some(ans)
            } else {
                None
            }
        }
        3 => {
            // Flaky tool: success probability follows sigmoid over total load (sum)
            // Inflection at sum = 185, scale = 25.
            let sum = (t.a + t.b) as f64;
            let logit = (sum - 185.0) / 25.0;
            let p_fail = 1.0 / (1.0 + (-logit).exp());
            if rng.next_f64() >= p_fail {
                Some(ans)
            } else {
                None
            }
        }
        4 => {
            // High load specialist: sum > 200 AND abs_diff <= 120
            if (t.a + t.b > 200) && (t.a.abs_diff(t.b) <= 120) {
                Some(ans)
            } else {
                None
            }
        }
        _ => None,
    }
}

pub fn run_feedback(t: Task, tool: usize, rng: &mut Rng) -> Feedback {
    let cost = call_cost(tool);
    let ok = execute_tool(t, tool, rng) == Some(reference_answer(t));
    let fallback_cost = if ok { 0.0 } else { call_cost(0) };
    Feedback {
        tool_cost: cost,
        first_ok: ok,
        total_cost: cost + fallback_cost,
    }
}

pub fn true_expected_cost(t: Task, tool: usize) -> f64 {
    let base = call_cost(tool);
    let p_fail = match tool {
        0 => 0.0,
        1 => {
            if (t.a + t.b <= 160) && (t.a.abs_diff(t.b) <= 35) {
                0.0
            } else {
                1.0
            }
        }
        2 => {
            if (t.a as f64 >= 1.5 * t.b as f64) && t.a <= 260 {
                0.0
            } else {
                1.0
            }
        }
        3 => {
            let sum = (t.a + t.b) as f64;
            let logit = (sum - 185.0) / 25.0;
            1.0 / (1.0 + (-logit).exp())
        }
        4 => {
            if (t.a + t.b > 200) && (t.a.abs_diff(t.b) <= 120) {
                0.0
            } else {
                1.0
            }
        }
        _ => 1.0,
    };
    base + p_fail * call_cost(0)
}

pub fn oracle_choice(t: Task) -> usize {
    let mut best_i = 0;
    let mut best_cost = f64::INFINITY;
    for i in 0..TOOLS {
        let c = true_expected_cost(t, i);
        if c < best_cost {
            best_cost = c;
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

pub fn collect_samples(seed: u64, n: usize, test_shift: bool) -> Vec<Sample> {
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|_| {
            let t = generate_task(&mut rng, test_shift);
            let outcomes = std::array::from_fn(|i| {
                // Each tool observation has independent execution noise
                let mut tool_rng = Rng::new(rng.next());
                run_feedback(t, i, &mut tool_rng)
            });
            Sample { task: t, outcomes }
        })
        .collect()
}

// ---------------- Policy 1: KindOnly (Global Mean) ----------------
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

// ---------------- Policy 2: Fixed 1D Bins ----------------
#[derive(Clone)]
pub struct FixedBins {
    grid_means: [[f64; TOOLS]; 8],
}
impl FixedBins {
    pub fn train(data: &[Sample]) -> Self {
        let mut sums = [[0.0; TOOLS]; 8];
        let mut counts = [0usize; 8];
        let prior = KindOnly::train(data);

        for s in data {
            let bin = Self::bucket(s.task);
            counts[bin] += 1;
            for i in 0..TOOLS {
                sums[bin][i] += s.outcomes[i].total_cost;
            }
        }
        let mut grid = [[0.0; TOOLS]; 8];
        for b in 0..8 {
            for i in 0..TOOLS {
                grid[b][i] = if counts[b] < 10 {
                    prior.means[i]
                } else {
                    sums[b][i] / counts[b] as f64
                };
            }
        }
        Self { grid_means: grid }
    }
    pub fn bucket(t: Task) -> usize {
        ((t.a + t.b) as usize / 80).min(7)
    }
    pub fn choose(&self, t: Task) -> usize {
        let b = Self::bucket(t);
        (0..TOOLS)
            .min_by(|&i, &j| self.grid_means[b][i].total_cmp(&self.grid_means[b][j]))
            .unwrap_or(0)
    }
}

// ---------------- Policy 3: Single 1D Decision Stump ----------------
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

// ---------------- Policy 4: ZERO Multidimensional 2D-Tree Router ----------------
// Fits a depth-2 decision tree (non-linear 2D boundary discovery) per tool,
// learning joint conditions like (sum <= 160 AND abs_diff <= 35).
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

fn fit_stump_on_indices(
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

    // Step 1: Root split
    let (root_f, root_cut, _, _, _) = fit_stump_on_indices(&all_idx, &feat_matrix, &targets);

    let (left_idx, right_idx): (Vec<usize>, Vec<usize>) = all_idx
        .into_iter()
        .partition(|&i| feat_matrix[i][root_f] <= root_cut);

    // Step 2: Left child split
    let (left_f2, left_cut2, left_left, left_right, _) =
        fit_stump_on_indices(&left_idx, &feat_matrix, &targets);

    // Step 3: Right child split
    let (right_f2, right_cut2, right_left, right_right, _) =
        fit_stump_on_indices(&right_idx, &feat_matrix, &targets);

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

#[derive(Default)]
pub struct Metric {
    pub n: usize,
    pub first_ok: usize,
    pub total_cost: f64,
    pub oracle_matches: usize,
    pub elapsed_nanos: u128,
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
}

pub fn run() -> io::Result<()> {
    let mut seeds = 8usize;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--seeds" => seeds = args.next().and_then(|x| x.parse().ok()).unwrap_or(8),
            _ => {}
        }
    }

    let out_dir = "reports/zero_multidimensional";
    fs::create_dir_all(out_dir)?;

    let mut summary = File::create(format!("{out_dir}/summary.csv"))?;
    let mut decisions = File::create(format!("{out_dir}/decisions_sample.csv"))?;

    writeln!(
        summary,
        "seed,split,policy,n,first_success_pct,mean_total_cost,oracle_match_pct,mean_decision_ns"
    )?;
    writeln!(
        decisions,
        "seed,split,task_idx,a,b,policy,chosen_tool,first_ok,cost,oracle_tool"
    )?;

    println!("===============================================================================");
    println!("ZERO MULTIDIMENSIONAL & FLAKY BENCHMARK | 2D BOUNDARIES & PROBABILISTIC NOISE");
    println!("Seeds: {seeds} | Tasks per seed: {TRAIN_TASKS} train / {VALID_TASKS} valid / {TEST_TASKS} test");
    println!("===============================================================================");
    println!("policy               test_cost  first_ok%   oracle%  ns/decision");
    println!("-------------------------------------------------------------------------------");

    let policies = [
        "universal",
        "kind_only",
        "fixed_bins",
        "single_stump",
        "zero_multidim",
        "oracle",
    ];

    for s in 0..seeds {
        let seed = 20261107u64 + (s as u64) * 7919;
        let train_data = collect_samples(seed ^ 0xAAAA_1111, TRAIN_TASKS, false);
        let valid_data = collect_samples(seed ^ 0xBBBB_2222, VALID_TASKS, false);
        let test_data = collect_samples(seed ^ 0xCCCC_3333, TEST_TASKS, true);

        // Train policies
        let p_kind = KindOnly::train(&train_data);
        let p_fixed = FixedBins::train(&train_data);
        let p_stump = SingleStumpPolicy::train(&train_data);
        let p_zero = ZeroMultidimensional::train(&train_data);

        for (split, dataset) in [("validation", &valid_data), ("test", &test_data)] {
            for &pol in &policies {
                let mut metric = Metric::default();
                let mut eval_rng = Rng::new(seed ^ 0x9999_8888);

                for (idx, sample) in dataset.iter().enumerate() {
                    let t = sample.task;
                    let opt = oracle_choice(t);

                    let t0 = Instant::now();
                    let chosen = match pol {
                        "universal" => 0,
                        "kind_only" => p_kind.choose(),
                        "fixed_bins" => p_fixed.choose(t),
                        "single_stump" => p_stump.choose(t),
                        "zero_multidim" => p_zero.choose(t),
                        "oracle" => opt,
                        _ => 0,
                    };
                    let elapsed = t0.elapsed().as_nanos();

                    let fb = run_feedback(t, chosen, &mut eval_rng);
                    metric.add(chosen, fb, opt, elapsed);

                    if s == 0 && split == "test" && idx < 50 {
                        writeln!(
                            decisions,
                            "{seed},{split},{idx},{},{},{pol},{},{},{:.4},{}",
                            t.a,
                            t.b,
                            TOOL_NAMES[chosen],
                            fb.first_ok,
                            fb.total_cost,
                            TOOL_NAMES[opt]
                        )?;
                    }
                }

                writeln!(
                    summary,
                    "{seed},{split},{pol},{},{:.2},{:.4},{:.2},{}",
                    metric.n,
                    metric.first_pct(),
                    metric.mean_cost(),
                    metric.oracle_pct(),
                    metric.elapsed_nanos / metric.n as u128
                )?;

                if split == "test" {
                    println!(
                        "{:<20} {:>9.4} {:>9.2}% {:>8.2}% {:>12}",
                        format!("{pol} (s={s})"),
                        metric.mean_cost(),
                        metric.first_pct(),
                        metric.oracle_pct(),
                        metric.elapsed_nanos / metric.n as u128
                    );
                }
            }
            if split == "test" && s + 1 < seeds {
                println!("-------------------------------------------------------------------------------");
            }
        }
    }

    println!("===============================================================================");
    println!("Reports generated in: {out_dir}/summary.csv, decisions_sample.csv");
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
    fn reference_matches_formula() {
        let t = Task { a: 10, b: 20 };
        assert_eq!(reference_answer(t), 10 * 37 + 20 * 53);
    }

    #[test]
    fn universal_always_succeeds() {
        let mut rng = Rng::new(42);
        for _ in 0..50 {
            let t = generate_task(&mut rng, true);
            assert_eq!(execute_tool(t, 0, &mut rng), Some(reference_answer(t)));
        }
    }

    #[test]
    fn narrow_box_obeys_2d_condition() {
        let mut rng = Rng::new(77);
        // Valid: sum <= 160 and |a - b| <= 35
        let ok_t = Task { a: 70, b: 80 };
        assert_eq!(execute_tool(ok_t, 1, &mut rng), Some(reference_answer(ok_t)));

        // Bad sum
        let bad_sum = Task { a: 100, b: 100 };
        assert_eq!(execute_tool(bad_sum, 1, &mut rng), None);

        // Bad diff
        let bad_diff = Task { a: 20, b: 90 };
        assert_eq!(execute_tool(bad_diff, 1, &mut rng), None);
    }

    #[test]
    fn flaky_tool_degrades_with_load() {
        let mut rng = Rng::new(1234);
        let easy_task = Task { a: 30, b: 40 }; // sum = 70 << 185
        let hard_task = Task { a: 150, b: 150 }; // sum = 300 >> 185

        let easy_successes = (0..500)
            .filter(|_| execute_tool(easy_task, 3, &mut rng).is_some())
            .count();
        let hard_successes = (0..500)
            .filter(|_| execute_tool(hard_task, 3, &mut rng).is_some())
            .count();

        assert!(easy_successes > 450, "Easy success rate should be > 90%");
        assert!(hard_successes < 50, "Hard success rate should be < 10%");
    }

    #[test]
    fn zero_multidim_tree_fits_and_predicts() {
        let data = collect_samples(42, 500, false);
        let tree = fit_tree2d(&data, 1);
        let pred = tree.predict(Task { a: 50, b: 50 });
        assert!(pred > 0.0);
    }
}
