//! ZERO Tool Discovery: a reproducible, dependency-free routing experiment.
//! This is a contextual-bandit simulation, NOT a general-intelligence test.
use std::fs::{self, File};
use std::io::Write;

const KINDS: usize = 5;
const TOOLS: usize = 7;
const CONTEXTS: usize = KINDS * 2;
const TRAIN: usize = 4000;
const TEST: usize = 1000;
const TOOL_NAMES: [&str; TOOLS] = ["T0", "T1", "T2", "T3", "T4", "T5", "T6"];
const COST: [f64; TOOLS] = [11.0, 2.0, 3.0, 3.0, 2.0, 2.0, 1.0];

#[derive(Clone, Copy)]
struct Task { kind: usize, size: usize, a: u32, b: u32, nonce: u32 }
impl Task {
    fn bucket(&self) -> usize { usize::from(self.size > 99) }
    fn context(&self) -> usize { self.kind * 2 + self.bucket() }
    fn description(&self) -> &'static str {
        ["add two numbers", "multiply two numbers", "sort an array", "reverse a string", "count vowels"][self.kind]
    }
}

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self { Self(seed) }
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn range(&mut self, n: u32) -> u32 { self.next() % n }
    fn uniform(&mut self) -> f64 { self.next() as f64 / (u32::MAX as f64 + 1.0) }
}

fn task(r: &mut Rng, idx: usize, test: bool) -> Task {
    let kind = idx % KINDS;
    let big = (idx / KINDS) % 2 == 1;
    // Held-out values: no training task is replayed; same observable classes.
    let size = if big {
        if test { 320 + r.range(540) as usize } else { 120 + r.range(160) as usize }
    } else if test { 50 + r.range(45) as usize } else { 5 + r.range(44) as usize };
    Task { kind, size, a: 1 + r.range(size as u32), b: 1 + r.range(size as u32), nonce: r.next() }
}

// Tool IDs are intentionally anonymous to the learner. A universal tool,
// five family specialists, and one cheap small-arithmetic specialist.
fn succeeds(t: Task, tool: usize) -> bool {
    match tool {
        0 => true,
        1 => t.kind == 0,
        2 => t.kind == 1,
        3 => t.kind == 2,
        4 => t.kind == 3,
        5 => t.kind == 4,
        6 => (t.kind == 0 || t.kind == 1) && t.size <= 99,
        _ => false,
    }
}

fn reward(t: Task, tool: usize) -> f64 {
    (if succeeds(t, tool) { 1.0 } else { -1.0 })
    - 0.02 * COST[tool]
}

#[derive(Clone)]
struct Learner { n: [[u32; TOOLS]; CONTEXTS], total: [[f64; TOOLS]; CONTEXTS] }
impl Learner {
    fn new() -> Self { Self { n: [[0; TOOLS]; CONTEXTS], total: [[0.0; TOOLS]; CONTEXTS] } }
    fn best(&self, context: usize) -> usize {
        let mut best = 0;
        let mut value = f64::NEG_INFINITY;
        for tool in 0..TOOLS {
            // Untried tools aren't preferred at evaluation time.
            let v = if self.n[context][tool] > 0 {
                self.total[context][tool] / self.n[context][tool] as f64
            } else { f64::NEG_INFINITY };
            if v > value { best = tool; value = v; }
        }
        best
    }
    fn choose_train(&self, context: usize, r: &mut Rng, epsilon: f64) -> usize {
        if r.uniform() < epsilon { r.range(TOOLS as u32) as usize }
        else { self.best(context) }
    }
    fn observe(&mut self, context: usize, tool: usize, score: f64) {
        self.n[context][tool] += 1;
        self.total[context][tool] += score;
    }
}

#[derive(Default, Clone, Copy)]
struct Score { good: usize, total: usize, cost: f64, utility: f64 }
impl Score {
    fn record(&mut self, t: Task, tool: usize) {
        self.total += 1;
        if succeeds(t, tool) { self.good += 1; }
        self.cost += COST[tool];
        self.utility += reward(t, tool);
    }
    fn row(&self, name: &str) -> String {
        format!("{name},{},{},{:.3},{:.3},{:.3}\n", self.total,
          self.good, self.good as f64 * 100.0 / self.total as f64,
          self.cost / self.total as f64, self.utility / self.total as f64)
    }
}

fn main() -> std::io::Result<()> {
    fs::create_dir_all("reports/tool_discovery")?;
    let mut rng = Rng::new(20261107);
    let mut model = Learner::new();
    let mut early = Score::default();
    let mut late = Score::default();
    for i in 0..TRAIN {
        let t = task(&mut rng, i, false);
        let epsilon = if i < 2000 { 0.35 } else { 0.12 };
        let selected = model.choose_train(t.context(), &mut rng, epsilon);
        model.observe(t.context(), selected, reward(t, selected));
        if i < 500 { early.record(t, selected); }
        if i >= TRAIN - 500 { late.record(t, selected); }
    }

    let mut evaluation_rng = Rng::new(90210);
    let mut random_rng = Rng::new(123456);
    let mut random = Score::default();
    let mut universal = Score::default();
    let mut learned = Score::default();
    let mut oracle = Score::default();
    let mut predictions = String::from("task,kind,size,context,chosen,correct,cost,oracle\n");
    for i in 0..TEST {
        let t = task(&mut evaluation_rng, i, true);
        let selected = model.best(t.context());
        let oracle_tool = (0..TOOLS).filter(|&k| succeeds(t, k))
            .min_by(|&a, &b| COST[a].total_cmp(&COST[b])).unwrap();
        random.record(t, random_rng.range(TOOLS as u32) as usize);
        universal.record(t, 0);
        learned.record(t, selected);
        oracle.record(t, oracle_tool);
        predictions.push_str(&format!("{},{},{},{},{},{},{},{}\n", t.description(),
            t.kind, t.size, t.context(), TOOL_NAMES[selected], succeeds(t, selected), COST[selected], TOOL_NAMES[oracle_tool]));
        let _ = (t.a, t.b, t.nonce); // Different task instances, opaque to router.
    }
    let mut table = String::from("strategy,n,correct,accuracy_percent,mean_cost,mean_utility\n");
    for (name, score) in [("training_early",early),("training_late",late),
        ("random",random),("always_universal",universal),
        ("learned",learned),("oracle",oracle)] {
        table.push_str(&score.row(name));
    }
    File::create("reports/tool_discovery/summary.csv")?.write_all(table.as_bytes())?;
    File::create("reports/tool_discovery/predictions.csv")?.write_all(predictions.as_bytes())?;
    let mut learned_map = String::from("context,kind,size_bucket,chosen_tool,calls_seen,mean_training_reward\n");
    for context in 0..CONTEXTS {
        let choice = model.best(context);
        learned_map.push_str(&format!("{},{},{},{},{},{:.4}\n",context,context/2,
            if context%2==0 {"small"} else {"large"}, TOOL_NAMES[choice],
            model.n[context][choice], model.total[context][choice]/model.n[context][choice].max(1) as f64));
    }
    File::create("reports/tool_discovery/learned_routes.csv")?.write_all(learned_map.as_bytes())?;
    print!("{}", table);
    println!("Saved: reports/tool_discovery/{{summary,predictions,learned_routes}}.csv");
    println!("IMPORTANT: Feedback gives the router task success during training. Test never feeds back.");
    println!("This tests costly tool selection for KNOWN task families, not intelligence or unseen abilities.");
    Ok(())
}
