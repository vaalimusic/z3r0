//! ZERO 0.6: real Rust operations behind anonymous tools.
//! No pretrained networks, network calls or external dependencies.
//! Correctness feedback comes from an external reference evaluator.
//!
//! Changes vs 0.5 (see decisions.csv analysis):
//!  1. Reward of a failed specialist now includes the cost of the universal fallback.
//!  2. A tool that never produced a value for a signature ("unsupported") is excluded from routing.
//!  3. Blind `tick % 29` probing is replaced by a per-(signature, tool) circuit breaker:
//!     2 consecutive failures bench the tool, exponential cooldown, targeted re-probe with the
//!     pre-failure estimate restored.
//!  4. decisions.csv header now matches the rows (added `tool_id`).
use std::fs::{self, File};
use std::io::Write;
use std::time::Instant;

const TOOLS: usize = 8;
const KINDS: usize = 6;
const TRAIN_KINDS: usize = 5;
const LAMBDA: f64 = 0.10;
const TEST_TASKS: usize = 1500;
const TOOL_LABELS: [&str; TOOLS] = ["X0", "X1", "X2", "X3", "X4", "X5", "X6", "X7"];

// Circuit breaker parameters (units: tasks of the same signature).
const STREAK_TO_BENCH: u32 = 2;
const BENCH_BASE: u32 = 4;
const BENCH_MAX: u32 = 32;

const DECISIONS_HEADER: &str = "seed,strategy,task_idx,kind,tool_name,sort_tool_broken,tool_id,first_ok,final_ok,charged_cost,calls,work";

#[derive(Clone, Debug, PartialEq, Eq)]
enum Value { Number(i64), Numbers(Vec<i64>), Text(String) }

#[derive(Clone)]
struct Task { kind: usize, a: u32, b: u32, values: Vec<i64>, text: String }

impl Task {
    fn signature(&self) -> usize { self.kind } // Visible operation signature, NOT inferred semantic intent.
}

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self { Self(seed) }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn range(&mut self, n: usize) -> usize { (self.next() % n as u64) as usize }
}

fn generate(r: &mut Rng, kind: usize, test: bool) -> Task {
    let n = if test { 24 + r.range(25) } else { 8 + r.range(26) };
    let values = (0..n).map(|_| r.range(2000) as i64 - 1000).collect();
    let text_len = if test { 30 + r.range(35) } else { 10 + r.range(30) };
    let alphabet = b"abcdefghijklmnopqrstuvwxyz";
    let text: String = (0..text_len).map(|_| alphabet[r.range(alphabet.len())] as char).collect();
    Task { kind, a: (10 + r.range(180)) as u32, b: (10 + r.range(180)) as u32, values, text }
}

fn reference(t: &Task) -> Value {
    match t.kind {
        0 => Value::Number(t.a as i64 + t.b as i64),
        1 => Value::Number(t.a as i64 * t.b as i64),
        2 => { let mut v = t.values.clone(); v.sort_unstable(); Value::Numbers(v) },
        3 => Value::Text(t.text.chars().rev().collect()),
        4 => Value::Number(t.text.chars().filter(|c| "aeiou".contains(*c)).count() as i64),
        5 => Value::Number(*t.values.iter().max().unwrap_or(&0)),
        _ => unreachable!(),
    }
}

#[derive(Clone, Debug)]
struct Call { value: Option<Value>, work: u64, elapsed_us: u128, cost: f64 }

fn execute(t: &Task, tool: usize, broken_sort: bool) -> Call {
    let start = Instant::now();
    let mut work: u64 = 1;
    let value = match (tool, t.kind) {
        // Universal tool deliberately uses simpler general algorithms, not a simulated success flag.
        (0, 0) => { let mut x = t.a as i64; for _ in 0..t.b { x += 1; work += 1; } Some(Value::Number(x)) },
        (0, 1) => { let mut x = 0i64; for _ in 0..t.b { x += t.a as i64; work += 1; } Some(Value::Number(x)) },
        (0, 2) => {
            let mut v = t.values.clone();
            for i in 0..v.len() { for j in (i + 1)..v.len() { work += 1; if v[i] > v[j] { v.swap(i, j); } } }
            Some(Value::Numbers(v))
        },
        (0, 3) => { let mut s = String::new(); for c in t.text.chars() { work += s.len() as u64 + 1; s.insert(0, c); } Some(Value::Text(s)) },
        (0, 4) => { let mut n = 0i64; for c in t.text.chars() { work += 1; if "aeiou".contains(c) { n += 1; } } Some(Value::Number(n)) },
        (0, 5) => { let mut m = i64::MIN; for &v in &t.values { work += 1; if v > m { m = v; } } Some(Value::Number(m)) },
        (1, 0) => { work += 1; Some(Value::Number(t.a as i64 + t.b as i64)) },
        (2, 1) => { work += 1; Some(Value::Number(t.a as i64 * t.b as i64)) },
        (3, 2) => {
            let mut v = t.values.clone();
            v.sort_unstable();
            work += (v.len() as u64) * (v.len().max(2).ilog2() as u64 + 1);
            if broken_sort { v.reverse(); }
            Some(Value::Numbers(v))
        },
        (4, 3) => { work += t.text.len() as u64; Some(Value::Text(t.text.chars().rev().collect())) },
        (5, 4) => { work += t.text.len() as u64; Some(Value::Number(t.text.bytes().filter(|c| b"aeiou".contains(c)).count() as i64)) },
        (6, 0) if t.a < 100 && t.b < 100 => { work += 1; Some(Value::Number(t.a as i64 + t.b as i64)) },
        (7, 5) => { work += t.values.len() as u64; Some(Value::Number(*t.values.iter().max().unwrap_or(&0))) },
        _ => None,
    };
    let elapsed_us = start.elapsed().as_micros();
    // Deterministic work accounting plus invocation overhead; actual elapsed time is reported separately.
    let cost = (if tool == 0 { 2.0 } else { 0.5 }) + work as f64 / 40.0;
    Call { value, work, elapsed_us, cost }
}

fn is_correct(call: &Call, expected: &Value) -> bool { call.value.as_ref() == Some(expected) }

/// Reward of a routing decision. A failed specialist also pays for the universal fallback,
/// because that is what the failure actually costs the system.
fn reward(call: &Call, expected: &Value, fallback_cost: f64) -> f64 {
    let ok = is_correct(call, expected);
    (if ok { 1.0 } else { -1.0 }) - LAMBDA * (call.cost + if ok { 0.0 } else { fallback_cost })
}

/// Cost of the fallback that a failure of `tool` would trigger (0 if it succeeded or is universal).
fn fallback_cost_for(t: &Task, tool: usize, ok: bool) -> f64 {
    if ok || tool == 0 { 0.0 } else { execute(t, 0, false).cost }
}

#[derive(Clone)]
struct Router {
    mean: [[f64; TOOLS]; KINDS],
    seen: [[u32; TOOLS]; KINDS],
    nones: [[u32; TOOLS]; KINDS],   // calls that returned no value at all
    oks: [[u32; TOOLS]; KINDS],     // calls that were verified correct
    streak: [[u32; TOOLS]; KINDS],  // consecutive failures
    prior: [[f64; TOOLS]; KINDS],   // estimate before the current failure streak began
    bench_until: [[u32; TOOLS]; KINDS], // 0 = not benched
    bench_len: [[u32; TOOLS]; KINDS],   // >0 while a tool is on probation (backoff state)
    clock: [u32; KINDS],            // per-signature task counter (online only)
}

impl Router {
    fn new() -> Self {
        Self {
            mean: [[0.0; TOOLS]; KINDS],
            seen: [[0; TOOLS]; KINDS],
            nones: [[0; TOOLS]; KINDS],
            oks: [[0; TOOLS]; KINDS],
            streak: [[0; TOOLS]; KINDS],
            prior: [[0.0; TOOLS]; KINDS],
            bench_until: [[0; TOOLS]; KINDS],
            bench_len: [[0; TOOLS]; KINDS],
            clock: [0; KINDS],
        }
    }

    /// Clears online-only state (called after offline training).
    fn reset_runtime(&mut self) {
        self.streak = [[0; TOOLS]; KINDS];
        self.bench_until = [[0; TOOLS]; KINDS];
        self.bench_len = [[0; TOOLS]; KINDS];
        self.clock = [0; KINDS];
    }

    fn unsupported(&self, kind: usize, tool: usize) -> bool {
        self.nones[kind][tool] >= 2 && self.oks[kind][tool] == 0
    }

    fn observe(&mut self, kind: usize, tool: usize, r: f64, produced: bool, correct: bool) {
        let n = self.seen[kind][tool];
        let old = self.mean[kind][tool];
        self.mean[kind][tool] = if n == 0 { r } else { 0.8 * old + 0.2 * r };
        self.seen[kind][tool] = n + 1;
        if !produced { self.nones[kind][tool] += 1; }
        if correct {
            self.oks[kind][tool] += 1;
            self.streak[kind][tool] = 0;
            self.bench_len[kind][tool] = 0;
        } else {
            if self.streak[kind][tool] == 0 { self.prior[kind][tool] = old; }
            self.streak[kind][tool] += 1;
            let proven = self.oks[kind][tool] > 0; // only tools that worked before get benched
            let on_probation = self.bench_len[kind][tool] > 0;
            if proven && (self.streak[kind][tool] >= STREAK_TO_BENCH || on_probation) {
                let len = if self.bench_len[kind][tool] == 0 {
                    BENCH_BASE
                } else {
                    (self.bench_len[kind][tool] * 2).min(BENCH_MAX)
                };
                self.bench_len[kind][tool] = len;
                self.bench_until[kind][tool] = self.clock[kind] + len;
            }
        }
    }

    /// Greedy choice among tools that were seen, are supported and are not benched.
    /// Falls back to the universal tool 0.
    fn best_known(&self, kind: usize) -> usize {
        let mut best = 0;
        let mut score = f64::NEG_INFINITY;
        for t in 0..TOOLS {
            if self.seen[kind][t] == 0 || self.unsupported(kind, t) || self.bench_until[kind][t] > 0 {
                continue;
            }
            if self.mean[kind][t] > score { score = self.mean[kind][t]; best = t; }
        }
        best
    }

    /// Online choice: explores an unseen signature, releases expired benches
    /// (targeted re-probe with the pre-failure estimate), then picks greedily.
    fn select(&mut self, kind: usize) -> usize {
        self.clock[kind] += 1;
        let now = self.clock[kind];
        for t in 0..TOOLS {
            if self.bench_until[kind][t] > 0 && now >= self.bench_until[kind][t] {
                self.bench_until[kind][t] = 0;
                self.mean[kind][t] = self.prior[kind][t];
                self.streak[kind][t] = 0;
            }
        }
        // No tool has ever worked for this signature: try each unseen anonymous tool once.
        if !(0..TOOLS).any(|t| self.oks[kind][t] > 0) {
            if let Some(t) = (1..TOOLS).find(|&i| self.seen[kind][i] == 0) { return t; }
        }
        self.best_known(kind)
    }
}

fn train(seed: u64) -> Router {
    let mut r = Rng::new(seed);
    let mut p = Router::new();
    for k in 0..TRAIN_KINDS {
        for tool in 0..TOOLS {
            for _ in 0..5 {
                let task = generate(&mut r, k, false);
                let expected = reference(&task);
                let call = execute(&task, tool, false);
                let ok = is_correct(&call, &expected);
                let fb = fallback_cost_for(&task, tool, ok);
                p.observe(k, tool, reward(&call, &expected, fb), call.value.is_some(), ok);
            }
        }
    }
    for i in 0..4000 {
        let k = i % TRAIN_KINDS;
        let task = generate(&mut r, k, false);
        let chosen = if i % 17 == 0 { r.range(TOOLS) } else { p.best_known(k) };
        let expected = reference(&task);
        let call = execute(&task, chosen, false);
        let ok = is_correct(&call, &expected);
        let fb = fallback_cost_for(&task, chosen, ok);
        p.observe(k, chosen, reward(&call, &expected, fb), call.value.is_some(), ok);
    }
    p.reset_runtime();
    p
}

#[derive(Clone, Default)]
struct Metrics {
    n: usize, first_ok: usize, final_ok: usize, calls: usize, total_cost: f64,
    total_work: u64, elapsed_us: u128, failures: usize, novel_first_ok: usize, novel_n: usize,
}
impl Metrics {
    fn add(&mut self, first: bool, final_ok: bool, calls: usize, cost: f64, work: u64, us: u128, novel: bool) {
        self.n += 1;
        self.first_ok += first as usize;
        self.final_ok += final_ok as usize;
        self.calls += calls;
        self.total_cost += cost;
        self.total_work += work;
        self.elapsed_us += us;
        self.failures += (!first) as usize;
        if novel { self.novel_n += 1; self.novel_first_ok += first as usize; }
    }
    fn csv(&self, seed: u64, policy: &str) -> String {
        format!("{seed},{policy},{},{},{},{:.5},{:.5},{:.5},{},{},{},{},{:.3}",
            self.n, self.first_ok, self.final_ok, self.total_cost/self.n as f64,
            self.calls as f64/self.n as f64, self.elapsed_us as f64/self.n as f64,
            self.total_work, self.failures, self.novel_first_ok, self.novel_n,
            100.0*self.first_ok as f64/self.n as f64)
    }
}

fn test_tasks(seed: u64) -> Vec<Task> {
    let mut r = Rng::new(seed ^ 0xDB4F_0B91_75AE_2165);
    (0..TEST_TASKS).map(|i| {
        // A sixth operation is entirely withheld from training.
        let kind = if i % 6 == 0 { 5 } else { i % 6 - 1 };
        generate(&mut r, kind, true)
    }).collect()
}

fn select_oracle(t: &Task, broken: bool) -> usize {
    let expected = reference(t);
    let mut best = 0;
    let mut cost = f64::INFINITY;
    for i in 0..TOOLS {
        let c = execute(t, i, broken);
        if is_correct(&c, &expected) && c.cost < cost { best = i; cost = c.cost; }
    }
    best
}

#[allow(clippy::too_many_arguments)]
fn decision_row(seed: u64, strategy: &str, idx: usize, kind: usize, chosen: usize, broken: bool,
                first_ok: bool, final_ok: bool, cost: f64, calls: usize, work: u64) -> String {
    format!("{seed},{strategy},{idx},{kind},{},{broken},{chosen},{first_ok},{final_ok},{:.5},{calls},{work}",
        TOOL_LABELS[chosen], cost)
}

fn evaluate(seed: u64, strategy: &str, trained: &Router, tasks: &[Task], log: &mut Option<File>, phases_csv: &mut File) -> Metrics {
    let mut p = trained.clone();
    let mut m = Metrics::default();
    let mut phases: [Metrics; 3] = std::array::from_fn(|_| Metrics::default());
    for (idx, task) in tasks.iter().enumerate() {
        let expected = reference(task);
        // Tool X3 breaks after 500 tasks, recovers after task 999.
        let broken = (500..1000).contains(&idx);
        let chosen = match strategy {
            "universal" => 0,
            "oracle" => select_oracle(task, broken),
            "frozen" => p.best_known(task.signature()),
            "online" => p.select(task.signature()),
            _ => unreachable!(),
        };
        let first = execute(task, chosen, broken);
        let success = is_correct(&first, &expected);
        let mut total_cost = first.cost;
        let mut total_work = first.work;
        let mut total_us = first.elapsed_us;
        let mut calls = 1usize;
        let mut final_ok = success;
        let mut fallback_cost = 0.0;
        if !success && chosen != 0 {
            let fallback = execute(task, 0, false);
            fallback_cost = fallback.cost;
            total_cost += fallback.cost;
            total_work += fallback.work;
            total_us += fallback.elapsed_us;
            calls += 1;
            final_ok = is_correct(&fallback, &expected);
        }
        if strategy == "online" {
            p.observe(task.signature(), chosen, reward(&first, &expected, fallback_cost),
                first.value.is_some(), success);
        }
        m.add(success, final_ok, calls, total_cost, total_work, total_us, task.kind == 5);
        phases[idx / 500].add(success, final_ok, calls, total_cost, total_work, total_us, task.kind == 5);
        if let Some(file) = log.as_mut() {
            writeln!(file, "{}", decision_row(seed, strategy, idx, task.kind, chosen, broken,
                success, final_ok, total_cost, calls, total_work)).unwrap();
        }
    }
    for (idx, phase) in ["normal", "sort_broken", "sort_restored"].iter().enumerate() {
        writeln!(phases_csv, "{},{}", phase, phases[idx].csv(seed, strategy)).unwrap();
    }
    m
}

fn run() -> std::io::Result<()> {
    let mut seeds = 8usize;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--seeds" { seeds = args.next().and_then(|s| s.parse().ok()).unwrap_or(8); }
        if arg == "--help" { println!("Usage: cargo run --release --bin zero_real_tools -- [--seeds 8]"); return Ok(()); }
    }
    if seeds == 0 || seeds > 100 { eprintln!("--seeds must be 1..100"); std::process::exit(2); }
    let out = "reports/zero_real_tools";
    fs::create_dir_all(out)?;
    let mut summary = File::create(format!("{out}/summary.csv"))?;
    let mut decisions = Some(File::create(format!("{out}/decisions.csv"))?);
    let mut phase_summary = File::create(format!("{out}/phases.csv"))?;
    let mut routes = File::create(format!("{out}/routes.csv"))?;
    writeln!(summary, "seed,strategy,tasks,first_successes,final_successes,mean_cost,mean_calls,mean_elapsed_us,total_work,failed_first,novel_first_ok,novel_n,first_success_pct")?;
    writeln!(decisions.as_mut().unwrap(), "{DECISIONS_HEADER}")?;
    writeln!(phase_summary, "phase,seed,strategy,tasks,first_successes,final_successes,mean_cost,mean_calls,mean_elapsed_us,total_work,failed_first,novel_first_ok,novel_n,first_success_pct")?;
    writeln!(routes, "seed,kind,best_tool,observations,expected_reward")?;
    println!("ZERO REAL TOOLS 0.6 | reference-verified real computations | {seeds} independent seeds");
    println!("Tool identities are hidden from router. Task operation signature is visible.");
    println!("Unseen operation: MAX (kind 5). Sort tool X3 fails during test tasks 500..999.");
    println!("The universal fallback is known and all failed calls are charged (also in the reward).");
    println!("strategy,mean_cost,first_success_pct,mean_calls");
    for j in 0..seeds {
        let seed = 20261107u64 + j as u64 * 7919;
        let trained = train(seed);
        let tasks = test_tasks(seed);
        for kind in 0..TRAIN_KINDS {
            let choice = trained.best_known(kind);
            writeln!(routes, "{seed},{kind},{},{},{:.6}",
                TOOL_LABELS[choice], trained.seen[kind][choice], trained.mean[kind][choice])?;
        }
        for strategy in ["universal", "frozen", "online", "oracle"] {
            let m = evaluate(seed, strategy, &trained, &tasks, &mut decisions, &mut phase_summary);
            println!("{strategy},{:.4},{:.2},{:.3} (seed={seed})", m.total_cost/m.n as f64,
                100.0*m.first_ok as f64/m.n as f64, m.calls as f64/m.n as f64);
            writeln!(summary, "{}", m.csv(seed, strategy))?;
        }
    }
    println!("Saved: {out}/summary.csv, phases.csv, routes.csv and decisions.csv");
    println!("NOTE: real Rust tool operations, but artificial work-cost units and reference feedback.");
    println!("This is routing of typed tasks, not open-ended natural-language tool discovery.");
    Ok(())
}

fn main() { if let Err(e) = run() { eprintln!("ZERO failed: {e}"); std::process::exit(1); } }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tools_produce_correct_answers() {
        let mut rng = Rng::new(7);
        for kind in 0..KINDS {
            for _ in 0..20 {
                let task = generate(&mut rng, kind, true);
                let want = reference(&task);
                assert!(is_correct(&execute(&task, 0, false), &want));
                let specialist = [1, 2, 3, 4, 5, 7][kind];
                assert!(is_correct(&execute(&task, specialist, false), &want));
            }
        }
    }
    #[test]
    fn broken_tool_detected_by_reference() {
        let mut rng = Rng::new(99);
        let task = generate(&mut rng, 2, true);
        assert!(!is_correct(&execute(&task, 3, true), &reference(&task)));
    }
    #[test]
    fn tool_capabilities_are_not_universal() {
        let mut rng = Rng::new(18);
        let task = generate(&mut rng, 4, true);
        assert!(execute(&task, 2, false).value.is_none());
    }
    #[test]
    fn split_has_unseen_family() {
        let tasks = test_tasks(5);
        assert_eq!(tasks.iter().filter(|t| t.kind == 5).count(), TEST_TASKS / 6);
    }
    #[test]
    fn failed_specialist_reward_includes_fallback() {
        let mut rng = Rng::new(3);
        let task = generate(&mut rng, 2, true);
        let want = reference(&task);
        let bad = execute(&task, 1, false); // X1 cannot sort -> None
        let fb = fallback_cost_for(&task, 1, false);
        assert!(fb > 5.0);
        assert!(reward(&bad, &want, fb) < reward(&bad, &want, 0.0));
        // a failing tool must look worse than the correct but expensive universal tool
        let uni = execute(&task, 0, false);
        assert!(reward(&bad, &want, fb) < reward(&uni, &want, 0.0));
    }
    #[test]
    fn unsupported_tool_is_never_routed() {
        let p = train(11);
        for kind in 0..TRAIN_KINDS {
            let t = p.best_known(kind);
            assert!(!p.unsupported(kind, t));
        }
        assert_eq!(p.best_known(2), 3);
        assert!(p.unsupported(2, 1));
    }
    #[test]
    fn circuit_breaker_benches_and_reprobes() {
        let mut p = Router::new();
        p.observe(2, 0, 0.3, true, true);
        for _ in 0..3 { p.observe(2, 3, 0.9, true, true); }
        assert_eq!(p.best_known(2), 3);
        p.observe(2, 3, -3.3, true, false);
        p.observe(2, 3, -3.3, true, false); // second consecutive failure -> benched
        assert_eq!(p.select(2), 0);          // clock 1: benched, universal used
        assert_eq!(p.select(2), 0);
        assert_eq!(p.select(2), 0);
        assert_eq!(p.select(2), 3);          // clock 4: cooldown over, targeted re-probe
        p.observe(2, 3, -3.3, true, false);  // re-probe fails -> longer bench
        assert_eq!(p.bench_len[2][3], BENCH_BASE * 2);
        assert_eq!(p.select(2), 0);
    }
    #[test]
    fn decisions_header_matches_rows() {
        let row = decision_row(1, "online", 0, 2, 3, false, true, true, 1.0, 1, 5);
        assert_eq!(row.matches(',').count(), DECISIONS_HEADER.matches(',').count());
    }
}
