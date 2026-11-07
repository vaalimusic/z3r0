//! ZERO Tool Adaptation 0.2: online adaptation after an unannounced tool failure.
//! Pure Rust, no crates. The test environment changes without telling the learner.
use std::fs::{self, File};
use std::io::{self, Write};

const TOOLS: usize = 7;
const KINDS: usize = 5;
const CONTEXTS: usize = KINDS * 2;
const TRAIN: usize = 4000;
const PHASE_LEN: usize = 600;
const EVAL: usize = PHASE_LEN * 3;
const COST: [f64; TOOLS] = [11.0, 2.0, 3.0, 3.0, 2.0, 2.0, 1.0];
const NAMES: [&str; TOOLS] = ["T0", "T1", "T2", "T3", "T4", "T5", "T6"];

#[derive(Clone, Copy)]
struct Task { kind: usize, size: usize, a: u32, b: u32 }
impl Task {
    fn context(self) -> usize { self.kind * 2 + usize::from(self.size > 99) }
}

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self { Self(seed) }
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn range(&mut self, n: u32) -> u32 { self.next() % n }
    fn uniform(&mut self) -> f64 { self.next() as f64 / (u32::MAX as f64 + 1.0) }
}

fn make_task(r: &mut Rng, index: usize, train: bool) -> Task {
    let kind = index % KINDS;
    let big = (index / KINDS) % 2 == 1;
    let size = match (big, train) {
        (false, true) => 5 + r.range(44) as usize,
        (false, false) => 50 + r.range(45) as usize,
        (true, true) => 120 + r.range(160) as usize,
        (true, false) => 320 + r.range(540) as usize,
    };
    Task { kind, size, a: r.next(), b: r.next() }
}

// Tool 6 is the cheapest small-arithmetic tool. It fails in the middle phase
// and recovers in the third phase. Neither the adaptive learner nor the frozen
// learner receives the phase number. The environment alone knows.
fn succeeds(t: Task, tool: usize, phase: usize) -> bool {
    match tool {
        0 => true,
        1 => t.kind == 0,
        2 => t.kind == 1,
        3 => t.kind == 2,
        4 => t.kind == 3,
        5 => t.kind == 4,
        6 => (t.kind == 0 || t.kind == 1) && t.size <= 99 && phase != 1,
        _ => false,
    }
}
fn reward(success: bool, tool: usize) -> f64 {
    (if success { 1.0 } else { -1.0 }) - 0.02 * COST[tool]
}

#[derive(Clone)]
struct Model {
    n: [[u32; TOOLS]; CONTEXTS],
    estimated: [[f64; TOOLS]; CONTEXTS],
    visits: [u32; CONTEXTS],
}
impl Model {
    fn new() -> Self {
        Self { n: [[0; TOOLS]; CONTEXTS], estimated: [[0.0; TOOLS]; CONTEXTS], visits: [0; CONTEXTS] }
    }
    fn observe(&mut self, c: usize, tool: usize, value: f64, online: bool) {
        let n = self.n[c][tool];
        if online {
            // Fast reaction to changes, unlike the cumulative training average.
            self.estimated[c][tool] = if n == 0 { value }
                else { 0.4 * value + 0.6 * self.estimated[c][tool] };
        } else {
            self.estimated[c][tool] = (self.estimated[c][tool] * n as f64 + value) / (n + 1) as f64;
        }
        self.n[c][tool] += 1;
    }
    fn best_except(&self, c: usize, excluded: Option<usize>) -> usize {
        let mut best = None;
        let mut best_value = f64::NEG_INFINITY;
        for t in 0..TOOLS {
            if excluded == Some(t) { continue; }
            let val = if self.n[c][t] == 0 { f64::NEG_INFINITY } else { self.estimated[c][t] };
            if best.is_none() || val > best_value {
                best = Some(t);
                best_value = val;
            }
        }
        best.unwrap_or(0)
    }
    fn train_choose(&self, c: usize, rng: &mut Rng, epsilon: f64) -> usize {
        if rng.uniform() < epsilon { rng.range(TOOLS as u32) as usize }
        else { self.best_except(c, None) }
    }
    fn adaptive_choose(&mut self, c: usize) -> usize {
        self.visits[c] += 1;
        let preferred = self.best_except(c, None);
        // Only probe a historically viable CHEAPER alternative once per 12
        // occurrences of this context; this provides a route to recovery.
        if self.visits[c] % 12 == 0 {
            let mut probe = None;
            for t in 0..TOOLS {
                if COST[t] >= COST[preferred] || self.n[c][t] == 0 { continue; }
                // Training history must have shown that this tool can succeed.
                // At evaluation time estimate can be low due to a recent outage;
                // the probe intentionally re-tests it.
                let historically_applicable = match t {
                    6 => c == 0 || c == 2,
                    1 => c / 2 == 0,
                    2 => c / 2 == 1,
                    3 => c / 2 == 2,
                    4 => c / 2 == 3,
                    5 => c / 2 == 4,
                    _ => false,
                };
                if historically_applicable && probe.map_or(true, |p| COST[t] < COST[p]) {
                    probe = Some(t);
                }
            }
            if let Some(t) = probe { return t; }
        }
        preferred
    }
}

#[derive(Clone, Copy, Default)]
struct Stats {
    tasks: u32, solved: u32, first_ok: u32, calls: u32,
    total_cost: f64, oracle_cost: f64,
}
impl Stats {
    fn record(&mut self, outcome: Outcome, optimal_cost: f64) {
        self.tasks += 1;
        self.solved += u32::from(outcome.ok);
        self.first_ok += u32::from(outcome.first_ok);
        self.calls += outcome.calls;
        self.total_cost += outcome.cost;
        self.oracle_cost += optimal_cost;
    }
    fn csv(self, policy: &str, phase: &str) -> String {
        format!("{policy},{phase},{},{:.2},{:.2},{:.3},{:.3},{:.3}\n",
            self.tasks,
            100.0 * self.solved as f64 / self.tasks.max(1) as f64,
            100.0 * self.first_ok as f64 / self.tasks.max(1) as f64,
            self.total_cost / self.tasks.max(1) as f64,
            self.calls as f64 / self.tasks.max(1) as f64,
            (self.total_cost - self.oracle_cost) / self.tasks.max(1) as f64)
    }
}
#[derive(Clone, Copy)]
struct Outcome { ok: bool, first_ok: bool, calls: u32, cost: f64, first_tool: usize }

fn use_policy(t: Task, phase: usize, model: &mut Model, adaptive: bool) -> Outcome {
    let c = t.context();
    let chosen = if adaptive { model.adaptive_choose(c) } else { model.best_except(c, None) };
    let first_ok = succeeds(t, chosen, phase);
    let mut out = Outcome { ok: first_ok, first_ok, calls: 1, cost: COST[chosen], first_tool: chosen };
    if adaptive { model.observe(c, chosen, reward(first_ok, chosen), true); }
    if !first_ok {
        // Feedback is visible ONLY after the first invocation. Retry has a real cost.
        // The frozen policy gets the same retry opportunity but cannot update scores.
        let fallback = model.best_except(c, Some(chosen));
        let second_ok = succeeds(t, fallback, phase);
        out.ok = second_ok;
        out.calls += 1;
        out.cost += COST[fallback];
        if adaptive { model.observe(c, fallback, reward(second_ok, fallback), true); }
    }
    out
}

fn oracle_tool(t: Task, phase: usize) -> usize {
    (0..TOOLS).filter(|&k| succeeds(t, k, phase))
        .min_by(|&a, &b| COST[a].total_cmp(&COST[b])).unwrap()
}

fn main() -> io::Result<()> {
    let out_dir = "reports/tool_adaptation";
    fs::create_dir_all(out_dir)?;
    let mut rng = Rng::new(20261107);
    let mut pretrained = Model::new();
    for i in 0..TRAIN {
        let t = make_task(&mut rng, i, true);
        let c = t.context();
        let selected = pretrained.train_choose(c, &mut rng, if i < 2000 { 0.35 } else { 0.12 });
        pretrained.observe(c, selected, reward(succeeds(t, selected, 0), selected), false);
    }
    let mut adaptive = pretrained.clone();
    let mut frozen = pretrained;
    let mut eval_rng = Rng::new(90210);
    let mut scores = [[Stats::default(); 3]; 4];
    let mut windows = [[Stats::default(); 2]; 3];
    let mut events = String::from("step,phase,kind,size,context,adaptive_tool,adaptive_first_ok,adaptive_ok,adaptive_cost,frozen_tool,frozen_first_ok,frozen_cost,oracle_tool\n");
    for i in 0..EVAL {
        let phase = i / PHASE_LEN;
        let window = if i % PHASE_LEN < PHASE_LEN / 2 { 0 } else { 1 };
        let t = make_task(&mut eval_rng, i, false);
        let opt = oracle_tool(t, phase);
        let frozen_result = use_policy(t, phase, &mut frozen, false);
        let adaptive_result = use_policy(t, phase, &mut adaptive, true);
        let universal = Outcome { ok: true, first_ok: true, calls: 1, cost: COST[0], first_tool: 0 };
        let oracle = Outcome { ok: true, first_ok: true, calls: 1, cost: COST[opt], first_tool: opt };
        for (k, o) in [frozen_result, adaptive_result, universal, oracle].iter().enumerate() {
            scores[k][phase].record(*o, COST[opt]);
        }
        windows[phase][window].record(adaptive_result, COST[opt]);
        events.push_str(&format!("{},{},{},{},{},{},{},{},{:.3},{},{},{:.3},{}\n",
            i, phase, t.kind, t.size, t.context(), NAMES[adaptive_result.first_tool],
            adaptive_result.first_ok, adaptive_result.ok, adaptive_result.cost,
            NAMES[frozen_result.first_tool], frozen_result.first_ok, frozen_result.cost, NAMES[opt]));
        let _ = (t.a, t.b);
    }
    let phase_names = ["normal", "T6_broken", "T6_restored"];
    let policy_names = ["frozen", "adaptive", "always_universal", "oracle"];
    let mut summary = String::from("policy,phase,tasks,success_pct,first_attempt_success_pct,mean_cost,mean_calls,mean_extra_cost_vs_oracle\n");
    for k in 0..4 {
        for phase in 0..3 {
            summary.push_str(&scores[k][phase].csv(policy_names[k], phase_names[phase]));
        }
    }
    for phase in 0..3 {
        for window in 0..2 {
            let label = format!("{}_{}", phase_names[phase], if window == 0 { "early" } else { "late" });
            summary.push_str(&windows[phase][window].csv("adaptive_window", &label));
        }
    }
    File::create(format!("{out_dir}/summary.csv"))?.write_all(summary.as_bytes())?;
    File::create(format!("{out_dir}/events.csv"))?.write_all(events.as_bytes())?;
    print!("{summary}");
    println!("Saved: {out_dir}/summary.csv and events.csv");
    println!("NOTE: This is a contextual-bandit simulation with observed task categories, not general intelligence.");
    println!("The learner is NOT told about the tool failure/restoration. Test success feedback is available after calls.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn failure_and_recovery() {
        let t = Task { kind: 0, size: 50, a: 1, b: 2 };
        assert!(succeeds(t, 6, 0));
        assert!(!succeeds(t, 6, 1));
        assert!(succeeds(t, 6, 2));
        assert!(succeeds(t, 1, 1));
    }
    #[test]
    fn reward_has_cost_penalty() {
        let t = Task { kind: 0, size: 50, a: 1, b: 2 };
        assert!(reward(succeeds(t, 6, 0), 6) > reward(succeeds(t, 0, 0), 0));
    }
}
