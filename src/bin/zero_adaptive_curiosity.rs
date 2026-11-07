//! ZERO 0.3 — adaptive curiosity under nonstationary tool availability.
//! Standalone binary, std only. Copy into src/bin of an existing Cargo project.
use std::fs::{self, File};
use std::io::{self, Write};

const TOOL_COUNT: usize = 7;
const KINDS: usize = 5;
const CONTEXTS: usize = KINDS * 2;
const STEPS: usize = 10_000;
const CHUNK: usize = 500;
const UNIVERSAL_COST: f64 = 11.0;
const INITIAL_COST: [f64; TOOL_COUNT] = [11.0, 2.0, 3.0, 3.0, 2.0, 2.0, 1.0];
const LABELS: [&str; TOOL_COUNT] = ["T0", "T1", "T2", "T3", "T4", "T5", "T6"];

#[derive(Clone)]
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self { Self(seed) }
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn range(&mut self, n: usize) -> usize { self.next() as usize % n }
}

#[derive(Clone, Copy)]
struct Task { kind: usize, large: bool }
impl Task {
    fn context(self) -> usize { self.kind * 2 + usize::from(self.large) }
}
fn tasks(seed: u64) -> Vec<Task> {
    let mut rng = Rng::new(seed);
    let mut result = Vec::with_capacity(STEPS);
    for i in 0..STEPS {
        // Uniform coverage of all contexts, randomized within 10-task blocks.
        let mut c = (i % CONTEXTS + rng.range(CONTEXTS)) % CONTEXTS;
        if i % CONTEXTS == 0 { c = rng.range(CONTEXTS); }
        result.push(Task { kind: c / 2, large: c % 2 == 1 });
    }
    result
}

#[derive(Clone, Copy)]
struct World { enabled: [bool; TOOL_COUNT], cost: [f64; TOOL_COUNT] }
impl World {
    fn baseline() -> Self { Self { enabled: [true; TOOL_COUNT], cost: INITIAL_COST } }
    fn succeeds(self, task: Task, tool: usize) -> bool {
        if !self.enabled[tool] { return false; }
        match tool {
            0 => true,
            1..=5 => task.kind == tool - 1,
            6 => task.kind <= 1 && !task.large,
            _ => false,
        }
    }
    fn oracle(self, task: Task) -> usize {
        (0..TOOL_COUNT).filter(|&tool| self.succeeds(task, tool))
            .min_by(|&a, &b| self.cost[a].total_cmp(&self.cost[b]))
            .expect("T0 must remain available")
    }
}

#[derive(Clone)]
struct Change { step: usize, tool: usize, enabled: bool, price: f64, kind: &'static str }
fn generate_changes(seed: u64) -> Vec<Change> {
    let mut rng = Rng::new(seed);
    let mut changes = Vec::new();
    let mut world = World::baseline();
    let mut step = 900;
    let mut n = 0;
    while step < STEPS - 250 {
        // Alternate changing T6's reliability and changing other tools, with
        // occasional directly observable price changes. Event times randomized.
        let tool = if n % 2 == 0 { 6 } else { 1 + rng.range(5) };
        let kind;
        if n % 5 == 4 {
            world.cost[tool] = if (world.cost[tool] - INITIAL_COST[tool]).abs() < 0.001 {
                INITIAL_COST[tool] + 2.5
            } else { INITIAL_COST[tool] };
            kind = "price_change";
        } else {
            world.enabled[tool] = !world.enabled[tool];
            kind = if world.enabled[tool] { "restored" } else { "broken" };
        }
        changes.push(Change { step, tool, enabled: world.enabled[tool], price: world.cost[tool], kind });
        n += 1;
        step += 440 + rng.range(280);
    }
    changes
}

#[derive(Clone, Copy)]
struct Belief { p: f64, samples: u32, ever_succeeded: bool }
impl Belief {
    const fn unknown() -> Self { Self { p: 0.5, samples: 0, ever_succeeded: false } }
    fn update(&mut self, success: bool, fast: bool) {
        let y = if success { 1.0 } else { 0.0 };
        if self.samples == 0 { self.p = y; }
        else {
            let alpha = if fast { 0.42 } else { 1.0 / (self.samples + 1) as f64 };
            self.p += alpha * (y - self.p);
        }
        self.samples += 1;
        if success { self.ever_succeeded = true; }
    }
}
#[derive(Clone)]
struct Router {
    beliefs: [[Belief; TOOL_COUNT]; CONTEXTS],
    seen: [u32; CONTEXTS],
    disturbance: [u8; CONTEXTS],
    probes: u32,
}
impl Router {
    fn pretrained() -> Self {
        let mut model = Self {
            beliefs: [[Belief::unknown(); TOOL_COUNT]; CONTEXTS],
            seen: [0; CONTEXTS], disturbance: [0; CONTEXTS], probes: 0,
        };
        // Training: every tool tried 12 times on each context, success observed.
        // The evaluation world's changes are never disclosed during training.
        let baseline = World::baseline();
        for c in 0..CONTEXTS {
            let task = Task { kind: c / 2, large: c % 2 == 1 };
            for tool in 0..TOOL_COUNT {
                for _ in 0..12 {
                    model.beliefs[c][tool].update(baseline.succeeds(task, tool), false);
                }
            }
        }
        model
    }
    fn estimated_total_cost(&self, c: usize, tool: usize, world: World) -> f64 {
        // A failed call falls back to reliable T0 and costs both calls.
        // Do not use the world success function here: only previous observations.
        let p = self.beliefs[c][tool].p;
        world.cost[tool] + (1.0 - p) * UNIVERSAL_COST
    }
    fn best(&self, c: usize, world: World) -> usize {
        (0..TOOL_COUNT).min_by(|&a, &b| {
            self.estimated_total_cost(c, a, world)
                .total_cmp(&self.estimated_total_cost(c, b, world))
        }).unwrap()
    }
    fn probe_candidate(&self, c: usize, chosen: usize, world: World) -> Option<usize> {
        // Probe an alternative with potential lower cost, including previously
        // broken tools. Historical success is a *learned* prerequisite.
        (1..TOOL_COUNT)
            .filter(|&tool| tool != chosen && self.beliefs[c][tool].ever_succeeded)
            .filter(|&tool| world.cost[tool] < self.estimated_total_cost(c, chosen, world))
            .min_by(|&a, &b| world.cost[a].total_cmp(&world.cost[b]))
    }
    fn choose(&mut self, c: usize, world: World, policy: Policy, rng: &mut Rng) -> (usize, bool) {
        let best = self.best(c, world);
        if policy == Policy::Frozen || policy == Policy::Universal { return (best, false); }
        self.seen[c] += 1;
        let candidate = self.probe_candidate(c, best, world);
        let Some(other) = candidate else { return (best, false); };
        let probe = match policy {
            Policy::Fixed => self.seen[c] % 10 == 0,
            Policy::Adaptive => {
                // Rare checks during calm. After surprising failure, temporarily
                // increase exploration, then decay back to quiescence.
                if self.disturbance[c] > 0 { self.seen[c] % 4 == 0 }
                else { self.seen[c] % 38 == 0 }
            }
            _ => false,
        };
        let _ = rng; // Policies are deterministic given history; seed changes tasks/events.
        if probe { self.probes += 1; (other, true) } else { (best, false) }
    }
    fn observe(&mut self, c: usize, tool: usize, ok: bool, policy: Policy, probed: bool) {
        if policy == Policy::Frozen || policy == Policy::Universal { return; }
        let old = self.beliefs[c][tool].p;
        self.beliefs[c][tool].update(ok, true);
        if !ok && old > 0.75 { self.disturbance[c] = 28; }
        else if ok && old < 0.35 { self.disturbance[c] = 20; }
        else if self.disturbance[c] > 0 { self.disturbance[c] -= 1; }
        let _ = probed;
    }
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum Policy { Frozen, Fixed, Adaptive, Universal, Oracle }
impl Policy {
    fn name(self) -> &'static str {
        match self {
            Self::Frozen => "frozen", Self::Fixed => "fixed_exploration",
            Self::Adaptive => "adaptive_curiosity", Self::Universal => "always_universal",
            Self::Oracle => "oracle",
        }
    }
}
const POLICIES: [Policy; 5] = [Policy::Frozen, Policy::Fixed, Policy::Adaptive, Policy::Universal, Policy::Oracle];

#[derive(Clone, Copy, Default)]
struct Stats {
    n: u32, solved: u32, first_ok: u32, calls: u32,
    probes: u32, failures: u32, cost: f64, oracle_cost: f64,
}
impl Stats {
    fn record(&mut self, o: Outcome, min_cost: f64) {
        self.n += 1;
        self.solved += u32::from(o.solved);
        self.first_ok += u32::from(o.first_ok);
        self.calls += o.calls;
        self.probes += u32::from(o.probed);
        self.failures += u32::from(!o.first_ok);
        self.cost += o.cost;
        self.oracle_cost += min_cost;
    }
    fn row(self, policy: &str, label: &str) -> String {
        let n = self.n.max(1) as f64;
        format!("{policy},{label},{},{:.3},{:.3},{:.4},{:.4},{:.4},{:.4},{}\n",
            self.n, 100.0 * self.solved as f64 / n,
            100.0 * self.first_ok as f64 / n, self.cost / n,
            self.oracle_cost / n, (self.cost - self.oracle_cost) / n,
            self.calls as f64 / n, self.probes)
    }
}
#[derive(Clone, Copy)]
struct Outcome { solved: bool, first_ok: bool, calls: u32, cost: f64, first_tool: usize, probed: bool }
fn act(model: &mut Router, policy: Policy, task: Task, world: World, rng: &mut Rng) -> Outcome {
    let context = task.context();
    let oracle = world.oracle(task);
    let (tool, probed) = match policy {
        Policy::Universal => (0, false),
        Policy::Oracle => (oracle, false),
        _ => model.choose(context, world, policy, rng),
    };
    let first_ok = world.succeeds(task, tool);
    let mut result = Outcome {
        solved: first_ok, first_ok, calls: 1, cost: world.cost[tool],
        first_tool: tool, probed,
    };
    model.observe(context, tool, first_ok, policy, probed);
    if !first_ok {
        // Same guaranteed fallback for all non-oracle strategies.
        result.calls += 1;
        result.cost += world.cost[0];
        result.solved = world.succeeds(task, 0);
        model.observe(context, 0, result.solved, policy, false);
    }
    result
}

fn simulate(seed: u64, outdir: Option<&str>) -> io::Result<[Stats; 5]> {
    let dataset = tasks(seed ^ 0xDADA);
    let changes = generate_changes(seed ^ 0xABCDEF);
    let mut world = World::baseline();
    let baseline = Router::pretrained();
    let mut routers = [baseline.clone(), baseline.clone(), baseline.clone(), baseline.clone(), baseline];
    let mut randoms = [Rng::new(1), Rng::new(2), Rng::new(3), Rng::new(4), Rng::new(5)];
    let mut totals = [Stats::default(); 5];
    let mut chunks = [[Stats::default(); STEPS / CHUNK]; 5];
    let mut events = String::from("step,tool,change,enabled,price\n");
    let mut decisions = String::from("step,context,policy,tool,first_success,success,calls,cost,oracle_cost,was_probe\n");
    let mut change_idx = 0;
    for (step, &task) in dataset.iter().enumerate() {
        while change_idx < changes.len() && changes[change_idx].step == step {
            let e = &changes[change_idx];
            world.enabled[e.tool] = e.enabled;
            world.cost[e.tool] = e.price;
            events.push_str(&format!("{},{},{},{},{:.3}\n", step, LABELS[e.tool], e.kind, e.enabled, e.price));
            change_idx += 1;
        }
        let cheapest = world.cost[world.oracle(task)];
        for (p, &policy) in POLICIES.iter().enumerate() {
            let outcome = act(&mut routers[p], policy, task, world, &mut randoms[p]);
            totals[p].record(outcome, cheapest);
            chunks[p][step / CHUNK].record(outcome, cheapest);
            if outdir.is_some() {
                decisions.push_str(&format!("{step},{},{},{},{},{},{},{:.3},{:.3},{}\n",
                    task.context(), policy.name(), LABELS[outcome.first_tool], outcome.first_ok,
                    outcome.solved, outcome.calls, outcome.cost, cheapest, outcome.probed));
            }
        }
    }
    if let Some(dir) = outdir {
        fs::create_dir_all(dir)?;
        let mut summary = String::from("policy,window,tasks,success_pct,first_attempt_pct,mean_cost,oracle_mean_cost,regret_per_task,mean_calls,exploration_calls\n");
        for (p, &policy) in POLICIES.iter().enumerate() {
            summary.push_str(&totals[p].row(policy.name(), "all"));
            for (i, stats) in chunks[p].iter().enumerate() {
                let label = format!("{}-{}", i * CHUNK, (i + 1) * CHUNK - 1);
                summary.push_str(&stats.row(policy.name(), &label));
            }
        }
        File::create(format!("{dir}/summary.csv"))?.write_all(summary.as_bytes())?;
        File::create(format!("{dir}/events.csv"))?.write_all(events.as_bytes())?;
        File::create(format!("{dir}/decisions.csv"))?.write_all(decisions.as_bytes())?;
        println!("policy,tasks,success_pct,first_attempt_pct,mean_cost,oracle_mean_cost,regret_per_task,mean_calls,exploration_calls");
        for (p, &policy) in POLICIES.iter().enumerate() {
            let s = totals[p];
            let n = s.n as f64;
            println!("{},{},{:.2},{:.2},{:.3},{:.3},{:.3},{:.3},{}",
                policy.name(), s.n, s.solved as f64 * 100.0 / n,
                s.first_ok as f64 * 100.0 / n, s.cost / n, s.oracle_cost / n,
                (s.cost - s.oracle_cost) / n, s.calls as f64 / n, s.probes);
        }
        println!("Saved {dir}/summary.csv, events.csv, decisions.csv");
        println!("NOTE: Tool families/context are known; success/failures are observed only AFTER invocation.");
        println!("NOTE: Listed costs are observable; hidden failures/restorations are not. The oracle sees full state.");
    }
    Ok(totals)
}

fn main() -> io::Result<()> {
    let mut seed = 20261107u64;
    let args: Vec<String> = std::env::args().collect();
    if let Some(pos) = args.iter().position(|s| s == "--seed") {
        if pos + 1 >= args.len() {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "--seed requires a u64"));
        }
        seed = args[pos + 1].parse().map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "invalid seed"))?;
    }
    simulate(seed, Some("reports/zero_curiosity"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn all_contexts_have_a_valid_tool() {
        let w = World::baseline();
        for c in 0..CONTEXTS {
            let t = Task { kind: c / 2, large: c % 2 == 1 };
            assert!(w.succeeds(t, w.oracle(t)));
        }
    }
    #[test]
    fn broken_tool_cannot_succeed() {
        let mut w = World::baseline();
        let small_math = Task { kind: 0, large: false };
        assert!(w.succeeds(small_math, 6));
        w.enabled[6] = false;
        assert!(!w.succeeds(small_math, 6));
        assert!(w.succeeds(small_math, 1));
    }
    #[test]
    fn baseline_prefers_cheapest_tool() {
        let r = Router::pretrained();
        let w = World::baseline();
        assert_eq!(r.best(0, w), 6);
        assert_eq!(r.best(1, w), 1);
    }
    #[test]
    fn reproducible_and_all_solved() {
        let a = simulate(20261107, None).unwrap();
        let b = simulate(20261107, None).unwrap();
        for i in 0..5 {
            assert_eq!(a[i].cost, b[i].cost);
            assert_eq!(a[i].solved, STEPS as u32);
            assert!(a[i].cost >= a[4].cost);
        }
    }
    #[test]
    fn reward_counts_failed_attempt_cost() {
        let mut w = World::baseline();
        w.enabled[6] = false;
        let mut r = Router::pretrained();
        let t = Task { kind: 0, large: false };
        let out = act(&mut r, Policy::Frozen, t, w, &mut Rng::new(1));
        assert!(out.solved && !out.first_ok);
        assert_eq!(out.calls, 2);
        assert!((out.cost - 12.0).abs() < 1e-6);
    }
}
