//! ZERO Capability Bounds: learning a tool's conditional competence from real calls.
//! Std-only Rust. No pretrained model, no labels supplied to the router.
//! The evaluator knows the correct answer; the router only sees success/cost feedback.
use std::fs::{self, File};
use std::io::{self, Write};

const TOOLS: usize = 8;
const KINDS: usize = 6;
const BINS: usize = 8;
const TEST_TASKS: usize = 2400;
const LAMBDA: f64 = 0.10;
const LABELS: [&str; TOOLS] = ["X0", "X1", "X2", "X3", "X4", "X5", "X6", "X7"];

#[derive(Clone, PartialEq, Debug)]
enum Value { Num(i64), Vec(Vec<i64>), Str(String) }
#[derive(Clone)]
struct Task { kind: usize, a: u32, b: u32, nums: Vec<i64>, text: String }
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self { Self(seed.max(1)) }
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn range(&mut self, n: usize) -> usize { (self.next() % n as u64) as usize }
}
fn make_task(r: &mut Rng, kind: usize, stratum: usize, test: bool) -> Task {
    // Crucially the test set contains UNSEEN individual inputs, not unseen operation families.
    // For multiplication use 8 neutral, fixed input-size bins. The limits of X6/X7 are NOT fed to the router.
    let base = (stratum * 40 + 10) as u32;
    let (a,b) = (base + r.range(37) as u32, base + r.range(37) as u32);
    let n = if test { 25+r.range(18) } else { 12+r.range(20) };
    let nums = (0..n).map(|_| r.range(2000) as i64-1000).collect();
    let text: String = (0..(20+r.range(25))).map(|_| (b'a'+r.range(26) as u8) as char).collect();
    Task { kind, a, b, nums, text }
}
fn expected(t: &Task) -> Value {
    match t.kind {
        0 => Value::Num(t.a as i64 + t.b as i64),
        1 => Value::Num(t.a as i64 * t.b as i64),
        2 => { let mut v=t.nums.clone(); v.sort_unstable(); Value::Vec(v) },
        3 => Value::Str(t.text.chars().rev().collect()),
        4 => Value::Num(t.text.bytes().filter(|c| b"aeiou".contains(c)).count() as i64),
        5 => Value::Num(*t.nums.iter().max().unwrap()),
        _ => unreachable!(),
    }
}
struct Call { value: Option<Value>, cost: f64 }
fn execute(t: &Task, tool: usize, broken: bool) -> Call {
    let mut work = 1u64;
    let value = match (tool,t.kind) {
        (0,0) => { let mut x=t.a as i64; for _ in 0..t.b {x+=1; work+=1;} Some(Value::Num(x)) },
        (0,1) => { let mut x=0i64; for _ in 0..t.b {x+=t.a as i64; work+=1;} Some(Value::Num(x)) },
        (0,2) => { let mut v=t.nums.clone(); for i in 0..v.len(){for j in i+1..v.len(){work+=1;if v[i]>v[j]{v.swap(i,j)}}} Some(Value::Vec(v)) },
        (0,3) => {let mut s=String::new();for c in t.text.chars(){work+=s.len() as u64+1;s.insert(0,c);}Some(Value::Str(s))},
        (0,4) => {work+=t.text.len() as u64;Some(Value::Num(t.text.bytes().filter(|c| b"aeiou".contains(c)).count() as i64))},
        (0,5) => {work+=t.nums.len() as u64;Some(Value::Num(*t.nums.iter().max().unwrap()))},
        (1,0) => {work+=1;Some(Value::Num(t.a as i64+t.b as i64))},
        (2,1) => {work+=85;Some(Value::Num(t.a as i64*t.b as i64))},
        (3,2) => {let mut v=t.nums.clone();v.sort_unstable();work+=(v.len() as u64)*(v.len().max(2).ilog2() as u64+1);if broken{v.reverse();}Some(Value::Vec(v))},
        (4,3) => {work+=t.text.len() as u64;Some(Value::Str(t.text.chars().rev().collect()))},
        (5,4) => {work+=t.text.len() as u64;Some(Value::Num(t.text.bytes().filter(|c| b"aeiou".contains(c)).count() as i64))},
        // Hidden capability: X6 handles multiplication only when both operands are < 120.
        (6,1) if t.a < 120 && t.b < 120 => {work+=1;Some(Value::Num(t.a as i64*t.b as i64))},
        // Hidden capability: X7 is efficient for large operands; also supports MAX (old role).
        (7,1) if t.a >= 160 && t.b >= 160 => {work+=2;Some(Value::Num(t.a as i64*t.b as i64))},
        (7,5) => {work+=t.nums.len() as u64;Some(Value::Num(*t.nums.iter().max().unwrap()))},
        _ => None,
    };
    let cost = (if tool==0 {2.0} else {0.5}) + work as f64/40.0;
    Call { value, cost }
}
fn bucket(t: &Task) -> usize {
    // Inputs are observable; bin boundaries are generic, not tool-specific.
    // Deliberately different from the hidden X6<120 and X7>=160 thresholds.
    ((t.a.max(t.b) / 40) as usize).min(BINS-1)
}
fn context(t: &Task, contextual: bool) -> usize {
    if contextual {t.kind * BINS + bucket(t)} else {t.kind}
}
#[derive(Clone)]
struct Stats { n: u32, mean_reward: f64, ok: u32, none: u32 }
impl Stats {
    fn new() -> Self { Self { n:0, mean_reward:0.0, ok:0, none:0 } }
    fn record(&mut self, reward: f64, ok: bool, produced: bool) {
        self.n+=1; self.mean_reward+=(reward-self.mean_reward)/self.n as f64;
        self.ok+=ok as u32; self.none+=(!produced) as u32;
    }
}
#[derive(Clone)]
struct Router { table: Vec<Vec<Stats>>, contextual: bool }
impl Router {
    fn new(contextual: bool) -> Self {
        let count=if contextual {KINDS*BINS} else {KINDS};
        Self {table:(0..count).map(|_|(0..TOOLS).map(|_|Stats::new()).collect()).collect(),contextual}
    }
    fn observe(&mut self, t: &Task, tool: usize, call: &Call, truth: &Value) {
        let ok=call.value.as_ref()==Some(truth);
        let fallback=if ok || tool==0 {0.0} else {execute(t,0,false).cost};
        let total=call.cost+fallback;
        let reward=(if ok {1.0}else{-1.0})-LAMBDA*total;
        self.table[context(t,self.contextual)][tool].record(reward,ok,call.value.is_some());
    }
    fn best(&self,t:&Task) -> usize {
        let mut choice=0usize;
        let mut score=f64::NEG_INFINITY;
        for tool in 0..TOOLS {
            let stat=&self.table[context(t,self.contextual)][tool];
            if stat.n==0 {continue}
            if stat.ok==0 && stat.none>=2 {continue}
            if stat.mean_reward>score {score=stat.mean_reward;choice=tool}
        }
        choice
    }
}
fn train(seed:u64,contextual:bool) -> Router {
    let mut rng=Rng::new(seed);
    let mut router=Router::new(contextual);
    // Balanced exploration, identical tasks and probes across the two learning approaches.
    // There is NO reference data or feedback in test decision-making.
    for kind in 0..KINDS {
        let groups=BINS;
        for group in 0..groups {
            for _ in 0..12 {
                let task=make_task(&mut rng,kind,group,false);
                let truth=expected(&task);
                for tool in 0..TOOLS {
                    let call=execute(&task,tool,false);
                    router.observe(&task,tool,&call,&truth);
                }
            }
        }
    }
    router
}
fn test_tasks(seed:u64) -> Vec<Task> {
    let mut rng=Rng::new(seed^0x9EA8_E54C_1D5B_0431);
    (0..TEST_TASKS).map(|i|{
        let kind=i%KINDS;
        let stratum=(i/6)%BINS;
        make_task(&mut rng,kind,stratum,true)
    }).collect()
}
#[derive(Default)]
struct Metrics { tasks:usize,correct:usize,first_ok:usize,calls:usize,cost:f64,wrong_route:usize }
impl Metrics {
    fn record(&mut self,first:bool,final_ok:bool,calls:usize,cost:f64,oracle_tool:usize,tool:usize){
        self.tasks+=1; self.correct+=final_ok as usize; self.first_ok+=first as usize;
        self.calls+=calls;self.cost+=cost;self.wrong_route+=(tool!=oracle_tool) as usize;
    }
    fn line(&self,seed:u64,mode:&str)->String{
        format!("{seed},{mode},{},{},{},{:.6},{:.6},{:.6},{}",self.tasks,self.first_ok,self.correct,self.cost/self.tasks as f64,self.calls as f64/self.tasks as f64,100.0*self.first_ok as f64/self.tasks as f64,self.wrong_route)
    }
}
fn oracle(t:&Task,broken:bool)->usize {
    let truth=expected(t);
    let mut best=0;
    let mut min=f64::INFINITY;
    for i in 0..TOOLS {
        let c=execute(t,i,broken);
        if c.value.as_ref()==Some(&truth) && c.cost<min {best=i;min=c.cost;}
    }
    best
}
fn evaluate(seed:u64,mode:&str,plain:&Router,contextual:&Router,tasks:&[Task],log:&mut Option<File>)->io::Result<(Metrics,[Metrics;3])>{
    let mut all=Metrics::default();
    let mut phase:[Metrics;3]=std::array::from_fn(|_|Metrics::default());
    for (idx,t) in tasks.iter().enumerate(){
        let broken=(800..1600).contains(&idx);
        let best_oracle=oracle(t,broken);
        let choice=match mode{"universal"=>0,"kind_only"=>plain.best(t),"contextual"=>contextual.best(t),"oracle"=>best_oracle,_=>unreachable!()};
        let truth=expected(t);
        let first=execute(t,choice,broken);
        let first_ok=first.value.as_ref()==Some(&truth);
        let fb=if first_ok || choice==0 {None} else {Some(execute(t,0,false))};
        let cost=first.cost+fb.as_ref().map(|c|c.cost).unwrap_or(0.0);
        let calls=1+fb.is_some() as usize;
        let final_ok=first_ok || fb.as_ref().is_some_and(|c|c.value.as_ref()==Some(&truth));
        all.record(first_ok,final_ok,calls,cost,best_oracle,choice);
        phase[idx/800].record(first_ok,final_ok,calls,cost,best_oracle,choice);
        if let Some(f)=log.as_mut(){
            writeln!(f,"{seed},{mode},{idx},{},{},{},{choice},{},{first_ok},{final_ok},{calls},{cost:.6}",t.kind,t.a,t.b,broken as u8)?;
        }
    }
    Ok((all,phase))
}
fn run()->io::Result<()> {
    let mut seeds=8usize;
    let mut args=std::env::args().skip(1);
    while let Some(arg)=args.next(){
        match arg.as_str(){
            "--seeds"=>seeds=args.next().and_then(|s|s.parse().ok()).unwrap_or(8),
            "--help"=>{println!("Usage: cargo run --release --bin zero_capability_bounds -- [--seeds 8]");return Ok(())},
            _=>{eprintln!("Unknown argument: {arg}");std::process::exit(2)}
        }
    }
    if !(1..=100).contains(&seeds){eprintln!("--seeds must be 1..100");std::process::exit(2)}
    let dir="reports/zero_bounds";fs::create_dir_all(dir)?;
    let mut out=File::create(format!("{dir}/summary.csv"))?;
    let mut phases=File::create(format!("{dir}/phases.csv"))?;
    let mut routes=File::create(format!("{dir}/routes.csv"))?;
    let mut decisions=if seeds<=8 {Some(File::create(format!("{dir}/decisions.csv"))?)}else{None};
    writeln!(out,"seed,mode,tasks,first_successes,final_successes,mean_cost,mean_calls,first_success_pct,non_oracle_routes")?;
    writeln!(phases,"seed,mode,phase,tasks,first_successes,final_successes,mean_cost,mean_calls,first_success_pct,non_oracle_routes")?;
    writeln!(routes,"seed,mode,kind,input_bin,representative_a,representative_b,chosen_tool,oracle_tool")?;
    if let Some(f)=decisions.as_mut(){writeln!(f,"seed,mode,task_index,kind,a,b,broken_sort,tool_id,first_ok,final_ok,calls,total_cost")?;}
    println!("ZERO CAPABILITY BOUNDS | real tool calls, hidden input restrictions | seeds={seeds}");
    println!("mode,mean_cost,first_success_pct,non_oracle_routes");
    for si in 0..seeds {
        let seed=20261107u64+si as u64*7919;
        let p=train(seed,false);
        let q=train(seed,true);
        // Route table uses representative input values and is *diagnostic*, not used by router.
        for b in 0..BINS {
            let a=(b*40+25) as u32;
            let t=Task{kind:1,a,b:a,nums:vec![1,2,3],text:"abcd".into()};
            for (mode,r) in [("kind_only",&p),("contextual",&q)] {
                writeln!(routes,"{seed},{mode},1,{b},{a},{a},{},{}",LABELS[r.best(&t)],LABELS[oracle(&t,false)])?;
            }
        }
        let tasks=test_tasks(seed);
        for mode in ["universal","kind_only","contextual","oracle"] {
            let (m,sub)=evaluate(seed,mode,&p,&q,&tasks,&mut decisions)?;
            println!("{mode},{:.4},{:.2},{} (seed={seed})",m.cost/m.tasks as f64,100.0*m.first_ok as f64/m.tasks as f64,m.wrong_route);
            writeln!(out,"{}",m.line(seed,mode))?;
            for (i,phase) in sub.iter().enumerate(){
                writeln!(phases,"{seed},{mode},{},{},{},{},{:.6},{:.6},{:.6},{}",["normal","sort_broken","sort_restored"][i],phase.tasks,phase.first_ok,phase.correct,phase.cost/phase.tasks as f64,phase.calls as f64/phase.tasks as f64,100.0*phase.first_ok as f64/phase.tasks as f64,phase.wrong_route)?;
            }
        }
    }
    println!("Saved {dir}/summary.csv, phases.csv, routes.csv, decisions.csv (for <=8 seeds)");
    println!("IMPORTANT: input bins are pre-defined. This tests conditional competence learning, NOT autonomous discovery of features or general intelligence.");
    Ok(())
}
fn main(){if let Err(e)=run(){eprintln!("Experiment failed: {e}");std::process::exit(1)}}
#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn hidden_tool_limits() {
        let mut r=Rng::new(5);
        let small=make_task(&mut r,1,1,true);
        let middle=make_task(&mut r,1,3,true);
        let big=make_task(&mut r,1,6,true);
        assert_eq!(execute(&small,6,false).value,Some(expected(&small)));
        assert!(execute(&middle,6,false).value.is_none());
        assert_eq!(execute(&big,7,false).value,Some(expected(&big)));
        assert!(execute(&small,7,false).value.is_none());
    }
    #[test] fn routers_have_different_state_spaces(){assert_eq!(Router::new(false).table.len(),KINDS);assert_eq!(Router::new(true).table.len(),KINDS*BINS);}
    #[test] fn contextual_prefers_specialists_at_extremes(){
        let p=train(77,true);
        for (a,want) in [(45,6),(225,7)] {
            let t=Task{kind:1,a,b:a,nums:vec![1,2],text:"hello".into()};
            assert_eq!(p.best(&t),want);
        }
    }
    #[test] fn universal_is_correct_for_all_kinds(){
        let mut r=Rng::new(2);
        for kind in 0..KINDS {for _ in 0..12 {let t=make_task(&mut r,kind,3,true);assert_eq!(execute(&t,0,false).value,Some(expected(&t)));}}
    }
    #[test] fn csv_columns_consistent(){
        let m=Metrics{tasks:1,correct:1,first_ok:1,calls:1,cost:2.0,wrong_route:0};
        assert_eq!(m.line(1,"test").split(',').count(),9);
    }
}
