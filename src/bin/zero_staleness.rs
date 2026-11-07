//! ZERO 0.4: compare exploration driven by stale knowledge with time-based exploration.
//! Self-contained Rust binary. No external crates, no pretrained models.
//! Copy this file to src/bin/zero_staleness.rs in the existing Cargo project.
use std::fs::{self, File};
use std::io::{self, Write};

const TOOLS: usize = 7;
const CONTEXTS: usize = 10;
const STEPS: usize = 10_000;
const TOOL_COST: [f64; TOOLS] = [11.0, 2.0, 3.0, 3.0, 2.0, 2.0, 1.0];
const POLICY_COUNT: usize = 6;
const N_DEV: usize = 6;
const N_TEST: usize = 16;

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
    fn from_context(c: usize) -> Self { Self { kind: c / 2, large: c % 2 == 1 } }
    fn context(self) -> usize { self.kind * 2 + usize::from(self.large) }
}

#[derive(Clone, Copy)]
struct World { enabled: [bool; TOOLS], cost: [f64; TOOLS] }
impl World {
    fn new() -> Self { Self { enabled: [true; TOOLS], cost: TOOL_COST } }
    fn succeeds(self, t: Task, tool: usize) -> bool {
        if !self.enabled[tool] { return false; }
        match tool {
            0 => true,
            1..=5 => t.kind == tool - 1,
            6 => t.kind <= 1 && !t.large,
            _ => false,
        }
    }
    fn oracle(self, task: Task) -> usize {
        (0..TOOLS).filter(|&t| self.succeeds(task, t))
            .min_by(|&a, &b| self.cost[a].total_cmp(&self.cost[b]))
            .unwrap()
    }
}

#[derive(Clone, Copy)]
struct Event { step: usize, tool: usize, enabled: bool, cost: f64, kind: &'static str }
fn make_events(seed: u64) -> Vec<Event> {
    let mut rng = Rng::new(seed);
    let mut world = World::new();
    let mut events = Vec::new();
    let mut step = 900;
    let mut n = 0;
    while step < STEPS - 250 {
        let tool = if n % 2 == 0 { 6 } else { 1 + rng.range(5) };
        let kind;
        if n % 5 == 4 {
            world.cost[tool] = if (world.cost[tool] - TOOL_COST[tool]).abs() < 1e-9 {
                TOOL_COST[tool] + 2.5
            } else { TOOL_COST[tool] };
            kind = "price_change";
        } else {
            world.enabled[tool] = !world.enabled[tool];
            kind = if world.enabled[tool] { "restored" } else { "broken" };
        }
        events.push(Event { step, tool, enabled: world.enabled[tool], cost: world.cost[tool], kind });
        n += 1;
        step += 440 + rng.range(280);
    }
    events
}

#[derive(Clone, Copy)]
struct Belief {
    p: f64,
    count: u32,
    ever_success: bool,
    last_checked: usize,
}
impl Belief {
    const fn unknown() -> Self {
        Self { p: 0.5, count: 0, ever_success: false, last_checked: 0 }
    }
    fn observe(&mut self, success: bool, now: usize, fast: bool) {
        let y = if success { 1.0 } else { 0.0 };
        if self.count == 0 { self.p = y; }
        else {
            let alpha = if fast { 0.42 } else { 1.0 / (self.count + 1) as f64 };
            self.p += alpha * (y - self.p);
        }
        self.count += 1;
        if success { self.ever_success = true; }
        self.last_checked = now;
    }
}

#[derive(Clone)]
struct Router {
    beliefs: [[Belief; TOOLS]; CONTEXTS],
    seen: [u32; CONTEXTS],
    disturbance: [u8; CONTEXTS],
}
impl Router {
    fn pretrained() -> Self {
        let mut r = Self {
            beliefs: [[Belief::unknown(); TOOLS]; CONTEXTS],
            seen: [0; CONTEXTS],
            disturbance: [0; CONTEXTS],
        };
        let w = World::new();
        // All policies start with the same baseline knowledge; no test-world state leaks.
        for c in 0..CONTEXTS {
            for tool in 0..TOOLS {
                for _ in 0..12 {
                    r.beliefs[c][tool].observe(w.succeeds(Task::from_context(c), tool), 0, false);
                }
            }
        }
        r
    }
    fn estimate(&self, c: usize, tool: usize, world: World) -> f64 {
        world.cost[tool] + (1.0 - self.beliefs[c][tool].p) * world.cost[0]
    }
    fn best(&self, c: usize, world: World) -> usize {
        (0..TOOLS).min_by(|&a, &b| self.estimate(c, a, world)
            .total_cmp(&self.estimate(c, b, world))).unwrap()
    }
    fn cheap_alternative(&self, c: usize, selected: usize, world: World) -> Option<usize> {
        (1..TOOLS)
            .filter(|&t| t != selected && self.beliefs[c][t].ever_success)
            .filter(|&t| world.cost[t] < self.estimate(c, selected, world))
            .min_by(|&a, &b| world.cost[a].total_cmp(&world.cost[b]))
    }
    fn choose(&mut self, c: usize, world: World, policy: Policy, step: usize, cfg: Config) -> (usize, bool) {
        let selected = self.best(c, world);
        if matches!(policy, Policy::Frozen) { return (selected, false); }
        self.seen[c] += 1;
        let Some(candidate) = self.cheap_alternative(c, selected, world) else { return (selected, false); };
        let investigate = match policy {
            Policy::Fixed => self.seen[c] % 10 == 0,
            Policy::Adaptive => {
                if self.disturbance[c] > 0 { self.seen[c] % 4 == 0 }
                else { self.seen[c] % 38 == 0 }
            }
            Policy::Staleness => {
                let b = self.beliefs[c][candidate];
                let age = step.saturating_sub(b.last_checked) as f64;
                // A tool that used to work may have recovered. Age increases the
                // probability that our most recent negative observation is stale.
                // This is a heuristic hazard model, not knowledge of actual state.
                let change_probability = 1.0 - (-age / cfg.tau).exp();
                let potential_saving = (self.estimate(c, selected, world) - world.cost[candidate]).max(0.0);
                let expected_future_benefit = change_probability * potential_saving * cfg.horizon;
                let conservative_probe_penalty = world.cost[candidate] + world.cost[0]
                    - self.estimate(c, selected, world);
                age >= cfg.min_age as f64 && expected_future_benefit > conservative_probe_penalty.max(0.0)
            }
            _ => false,
        };
        if investigate { (candidate, true) } else { (selected, false) }
    }
    fn observe(&mut self, c: usize, tool: usize, success: bool, policy: Policy, step: usize) {
        if policy == Policy::Frozen { return; }
        let old = self.beliefs[c][tool].p;
        self.beliefs[c][tool].observe(success, step, true);
        if !success && old > 0.75 { self.disturbance[c] = 28; }
        else if success && old < 0.35 { self.disturbance[c] = 20; }
        else if self.disturbance[c] > 0 { self.disturbance[c] -= 1; }
    }
}

#[derive(Clone, Copy, Debug)]
struct Config { min_age: usize, tau: f64, horizon: f64 }
const CONFIGS: [Config; 12] = [
    Config { min_age: 45, tau: 120.0, horizon: 5.0 },
    Config { min_age: 70, tau: 120.0, horizon: 5.0 },
    Config { min_age: 100, tau: 120.0, horizon: 5.0 },
    Config { min_age: 140, tau: 120.0, horizon: 5.0 },
    Config { min_age: 70, tau: 220.0, horizon: 5.0 },
    Config { min_age: 100, tau: 220.0, horizon: 5.0 },
    Config { min_age: 140, tau: 220.0, horizon: 5.0 },
    Config { min_age: 180, tau: 220.0, horizon: 5.0 },
    Config { min_age: 70, tau: 300.0, horizon: 10.0 },
    Config { min_age: 100, tau: 300.0, horizon: 10.0 },
    Config { min_age: 140, tau: 300.0, horizon: 10.0 },
    Config { min_age: 180, tau: 300.0, horizon: 10.0 },
];

#[derive(Clone, Copy, PartialEq, Eq)]
enum Policy { Frozen, Fixed, Adaptive, Staleness, Universal, Oracle }
impl Policy {
    fn name(self) -> &'static str {
        match self {
            Self::Frozen => "frozen", Self::Fixed => "fixed_exploration",
            Self::Adaptive => "adaptive_curiosity", Self::Staleness => "staleness_aware",
            Self::Universal => "always_universal", Self::Oracle => "oracle",
        }
    }
}
const POLICIES: [Policy; POLICY_COUNT] = [
    Policy::Frozen, Policy::Fixed, Policy::Adaptive,
    Policy::Staleness, Policy::Universal, Policy::Oracle,
];

#[derive(Default, Clone, Copy)]
struct Stats {
    tasks: usize,
    correct_first: usize,
    solved: usize,
    probes: usize,
    calls: usize,
    cost: f64,
    oracle_cost: f64,
}
impl Stats {
    fn add(&mut self, outcome: Outcome, oracle_cost: f64) {
        self.tasks += 1;
        self.correct_first += usize::from(outcome.first_success);
        self.solved += usize::from(outcome.solved);
        self.probes += usize::from(outcome.probe);
        self.calls += outcome.calls;
        self.cost += outcome.cost;
        self.oracle_cost += oracle_cost;
    }
    fn merge(&mut self, other: Stats) {
        self.tasks += other.tasks;
        self.correct_first += other.correct_first;
        self.solved += other.solved;
        self.probes += other.probes;
        self.calls += other.calls;
        self.cost += other.cost;
        self.oracle_cost += other.oracle_cost;
    }
    fn mean_cost(self) -> f64 { self.cost / self.tasks.max(1) as f64 }
    fn success_pct(self) -> f64 { 100.0 * self.correct_first as f64 / self.tasks.max(1) as f64 }
    fn regret(self) -> f64 { (self.cost - self.oracle_cost) / self.tasks.max(1) as f64 }
    fn csv(self, policy: &str, seed: u64, group: &str) -> String {
        format!("{policy},{seed},{group},{},{:.4},{:.4},{:.4},{:.4},{}\n",
            self.tasks, self.success_pct(), 100.0 * self.solved as f64 / self.tasks.max(1) as f64,
            self.mean_cost(), self.regret(), self.probes)
    }
}

#[derive(Clone, Copy)]
struct Outcome { first_success: bool, solved: bool, probe: bool, calls: usize, cost: f64, tool: usize }
fn run_action(router: &mut Router, policy: Policy, task: Task, world: World, step: usize, cfg: Config) -> Outcome {
    let c = task.context();
    let (tool, probe) = match policy {
        Policy::Universal => (0, false),
        Policy::Oracle => (world.oracle(task), false),
        _ => router.choose(c, world, policy, step, cfg),
    };
    let first_success = world.succeeds(task, tool);
    let mut outcome = Outcome { first_success, solved: first_success, probe, calls: 1, cost: world.cost[tool], tool };
    if !matches!(policy, Policy::Oracle | Policy::Universal) {
        router.observe(c, tool, first_success, policy, step);
    }
    if !first_success {
        // Failed attempts consume real cost. Reliable universal tool is fallback.
        outcome.calls += 1;
        outcome.cost += world.cost[0];
        outcome.solved = world.succeeds(task, 0);
        if !matches!(policy, Policy::Oracle | Policy::Universal) {
            router.observe(c, 0, outcome.solved, policy, step);
        }
    }
    outcome
}

#[derive(Clone, Copy, Default)]
struct Transition { n: usize, regret: f64, first_success: usize }
impl Transition {
    fn record(&mut self, outcome: Outcome, oracle_cost: f64) {
        self.n += 1;
        self.regret += outcome.cost - oracle_cost;
        self.first_success += usize::from(outcome.first_success);
    }
}

struct SimResult {
    totals: [Stats; POLICY_COUNT],
    by_event: [[Transition; 3]; POLICY_COUNT],
    events_csv: String,
    decisions_csv: String,
}
fn simulate(seed: u64, cfg: Config, details: bool) -> SimResult {
    let events = make_events(seed ^ 0xABCDEF);
    let mut rng = Rng::new(seed ^ 0xDADA);
    let mut world = World::new();
    let base = Router::pretrained();
    let mut routers: [Router; POLICY_COUNT] = std::array::from_fn(|_| base.clone());
    let mut totals = [Stats::default(); POLICY_COUNT];
    let mut transition = [[Transition::default(); 3]; POLICY_COUNT];
    let mut events_csv = String::from("step,tool,event,enabled,cost\n");
    let mut decisions_csv = String::from("step,context,policy,tool,first_success,solved,cost,oracle_cost,probe,best_last_checked_before\n");
    let mut event_idx = 0;
    let mut active_event: Option<(usize, usize)> = None;
    for step in 0..STEPS {
        while event_idx < events.len() && events[event_idx].step == step {
            let e = events[event_idx];
            world.enabled[e.tool] = e.enabled;
            world.cost[e.tool] = e.cost;
            let kind = match e.kind { "broken" => 0, "restored" => 1, _ => 2 };
            active_event = Some((step, kind));
            if details {
                events_csv.push_str(&format!("{step},T{},{},{},{:.3}\n", e.tool, e.kind, e.enabled, e.cost));
            }
            event_idx += 1;
        }
        let context = rng.range(CONTEXTS);
        let task = Task::from_context(context);
        let oracle_cost = world.cost[world.oracle(task)];
        for (i, &policy) in POLICIES.iter().enumerate() {
            let original_best = routers[i].best(context, world);
            let before = routers[i].beliefs[context][original_best].last_checked;
            let out = run_action(&mut routers[i], policy, task, world, step, cfg);
            totals[i].add(out, oracle_cost);
            if let Some((since, kind)) = active_event {
                if step - since < 150 {
                    transition[i][kind].record(out, oracle_cost);
                }
            }
            if details {
                decisions_csv.push_str(&format!("{step},{context},{},{},{},{},{:.3},{:.3},{},{}\n",
                    policy.name(), out.tool, out.first_success, out.solved, out.cost,
                    oracle_cost, out.probe, before));
            }
        }
    }
    SimResult { totals, by_event: transition, events_csv, decisions_csv }
}

fn dev_seed(i: usize) -> u64 { 0xDEA0_0000 + (i as u64) * 104729 }
fn test_seed(i: usize, offset: u64) -> u64 { 0xBEEF_0000 + (i as u64) * 65537 + offset.wrapping_mul(1009) }

fn main() -> io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mut offset: u64 = 0;
    if let Some(p) = args.iter().position(|s| s == "--seed-offset") {
        offset = args.get(p + 1).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "missing --seed-offset value"))?
            .parse().map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "seed offset must be an integer"))?;
    }
    let dir = "reports/zero_staleness";
    fs::create_dir_all(dir)?;
    let mut tuning = String::from("min_age,tau,horizon,dev_mean_cost,dev_mean_regret,dev_first_attempt_pct\n");
    let mut best_cfg = CONFIGS[0];
    let mut best_cost = f64::INFINITY;
    for cfg in CONFIGS {
        let mut aggregate = Stats::default();
        for i in 0..N_DEV {
            let run = simulate(dev_seed(i), cfg, false);
            aggregate.merge(run.totals[3]);
        }
        tuning.push_str(&format!("{},{:.0},{:.0},{:.5},{:.5},{:.3}\n",
            cfg.min_age, cfg.tau, cfg.horizon, aggregate.mean_cost(),
            aggregate.regret(), aggregate.success_pct()));
        if aggregate.mean_cost() < best_cost {
            best_cost = aggregate.mean_cost();
            best_cfg = cfg;
        }
    }
    File::create(format!("{dir}/development_tuning.csv"))?.write_all(tuning.as_bytes())?;
    println!("Chosen on DEVELOPMENT only: min_age={} tau={} horizon={} (mean_cost {:.4})",
        best_cfg.min_age, best_cfg.tau, best_cfg.horizon, best_cost);

    let mut seed_csv = String::from("policy,seed,split,tasks,first_attempt_pct,final_success_pct,mean_cost,regret_per_task,exploration_calls\n");
    let mut aggregates = [Stats::default(); POLICY_COUNT];
    let mut transitions = [[Transition::default(); 3]; POLICY_COUNT];
    let mut per_seed_costs = [[0.0; N_TEST]; POLICY_COUNT];
    for i in 0..N_TEST {
        let seed = test_seed(i, offset);
        let run = simulate(seed, best_cfg, i == 0);
        for (j, &policy) in POLICIES.iter().enumerate() {
            let s = run.totals[j];
            aggregates[j].merge(s);
            per_seed_costs[j][i] = s.mean_cost();
            seed_csv.push_str(&s.csv(policy.name(), seed, "test"));
            for k in 0..3 {
                transitions[j][k].n += run.by_event[j][k].n;
                transitions[j][k].regret += run.by_event[j][k].regret;
                transitions[j][k].first_success += run.by_event[j][k].first_success;
            }
        }
        if i == 0 {
            File::create(format!("{dir}/example_events.csv"))?.write_all(run.events_csv.as_bytes())?;
            File::create(format!("{dir}/example_decisions.csv"))?.write_all(run.decisions_csv.as_bytes())?;
        }
    }
    File::create(format!("{dir}/per_seed.csv"))?.write_all(seed_csv.as_bytes())?;
    let mut summary = String::from("policy,seeds,tasks,first_attempt_pct,final_success_pct,mean_cost,regret_per_task,total_probes\n");
    println!("policy,seeds,tasks,first_attempt_pct,final_success_pct,mean_cost,regret_per_task,total_probes");
    for (i, &policy) in POLICIES.iter().enumerate() {
        let s = aggregates[i];
        let row = format!("{},{},{},{:.3},{:.3},{:.5},{:.5},{}\n",
            policy.name(), N_TEST, s.tasks, s.success_pct(),
            100.0 * s.solved as f64 / s.tasks as f64,
            s.mean_cost(), s.regret(), s.probes);
        print!("{row}");
        summary.push_str(&row);
    }
    File::create(format!("{dir}/summary.csv"))?.write_all(summary.as_bytes())?;
    let mut transition_csv = String::from("policy,event_type,observations_in_first_150_tasks,mean_regret,first_attempt_pct\n");
    for (i, &policy) in POLICIES.iter().enumerate() {
        for (k, event) in ["broken", "restored", "price_change"].iter().enumerate() {
            let t = transitions[i][k];
            transition_csv.push_str(&format!("{},{},{},{:.5},{:.3}\n",
                policy.name(), event, t.n, t.regret / t.n.max(1) as f64,
                100.0 * t.first_success as f64 / t.n.max(1) as f64));
        }
    }
    File::create(format!("{dir}/transitions.csv"))?.write_all(transition_csv.as_bytes())?;
    let mut paired = String::from("seed,staleness_minus_fixed_mean_cost,staleness_minus_adaptive_mean_cost\n");
    let mut better_fixed = 0;
    let mut better_adaptive = 0;
    for i in 0..N_TEST {
        let d_fixed = per_seed_costs[3][i] - per_seed_costs[1][i];
        let d_adapt = per_seed_costs[3][i] - per_seed_costs[2][i];
        if d_fixed < 0.0 { better_fixed += 1; }
        if d_adapt < 0.0 { better_adaptive += 1; }
        paired.push_str(&format!("{},{:.5},{:.5}\n", test_seed(i, offset), d_fixed, d_adapt));
    }
    File::create(format!("{dir}/paired.csv"))?.write_all(paired.as_bytes())?;
    println!("Staleness beat Fixed on {better_fixed}/{N_TEST} test seeds; beat Adaptive on {better_adaptive}/{N_TEST}.");
    println!("Reports: {dir}/summary.csv, per_seed.csv, paired.csv, transitions.csv, development_tuning.csv, example_*.csv");
    println!("Limits: simulated tools, known task families, feedback after each call; not a test of general intelligence.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reliable_fallback_and_contexts() {
        let w = World::new();
        for c in 0..CONTEXTS {
            let task = Task::from_context(c);
            assert!(w.succeeds(task, w.oracle(task)));
        }
        let mut r = Router::pretrained();
        let mut broken = w;
        broken.enabled[6] = false;
        let o = run_action(&mut r, Policy::Frozen, Task::from_context(0), broken, 0, CONFIGS[0]);
        assert!(!o.first_success && o.solved && o.calls == 2);
        assert!((o.cost - 12.0).abs() < 1e-9);
    }
    #[test]
    fn staleness_means_last_check_not_world_access() {
        let mut b = Belief::unknown();
        b.observe(false, 123, true);
        assert_eq!(b.last_checked, 123);
        b.observe(true, 456, true);
        assert_eq!(b.last_checked, 456);
        assert!(b.ever_success);
    }
    #[test]
    fn deterministic_seed_and_valid_regret() {
        let a = simulate(test_seed(0, 0), CONFIGS[0], false);
        let b = simulate(test_seed(0, 0), CONFIGS[0], false);
        for i in 0..POLICY_COUNT {
            assert_eq!(a.totals[i].cost, b.totals[i].cost);
            assert_eq!(a.totals[i].solved, STEPS);
            assert!(a.totals[i].cost + 1e-7 >= a.totals[5].cost);
        }
    }
    #[test]
    fn development_and_test_seeds_are_distinct() {
        for i in 0..N_DEV {
            for j in 0..N_TEST {
                assert_ne!(dev_seed(i), test_seed(j, 0));
            }
        }
    }
}
