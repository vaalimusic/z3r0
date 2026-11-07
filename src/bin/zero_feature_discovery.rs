//! ZERO: discover useful input features for black-box tool selection.
//! Standalone std-only binary. The router sees raw a,b and feedback, never tool limits.
use std::fs::{self, File};
use std::io::{self, Write};
use std::time::Instant;

const TOOLS: usize = 4;
const FEATURES: usize = 7;
const TRAIN: usize = 2400;
const VALID: usize = 500;
const TEST: usize = 3000;
const MIN_LEAF: usize = 35;
const FEATURE_NAMES: [&str; FEATURES] = ["a", "b", "min", "max", "sum", "abs_diff", "product"];
const NAMES: [&str; TOOLS] = ["X0_universal", "X1_small", "X2_general", "X3_large"];

#[derive(Clone, Copy)]
struct Task { a: u32, b: u32 }
#[derive(Clone, Copy)]
struct Feedback { cost: f64, first_ok: bool, total_cost: f64 }

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self { Self(seed.max(1)) }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn range(&mut self, lo: u32, hi: u32) -> u32 { lo + (self.next() % (hi - lo + 1) as u64) as u32 }
}

fn task(r: &mut Rng, shifted: bool) -> Task {
    // Shifted test changes the distribution and deliberately samples near boundaries.
    if shifted && r.next() % 4 == 0 {
        const EDGE: [u32; 10] = [112, 116, 119, 120, 123, 154, 159, 160, 164, 169];
        Task { a: EDGE[(r.next() % EDGE.len() as u64) as usize], b: r.range(30, 355) }
    } else {
        let (lo, hi) = if shifted { (30, 355) } else { (20, 320) };
        Task { a: r.range(lo, hi), b: r.range(lo, hi) }
    }
}

fn features(t: Task) -> [u32; FEATURES] {
    [t.a, t.b, t.a.min(t.b), t.a.max(t.b), t.a + t.b, t.a.abs_diff(t.b), t.a * t.b]
}

// Hidden capability restrictions are inside tool implementations, not router features.
fn execute(t: Task, which: usize) -> Option<i64> {
    match which {
        0 => { let mut s=0i64; for _ in 0..t.b { s += t.a as i64; } Some(s) }
        1 if t.a < 120 && t.b < 120 => Some(t.a as i64 * t.b as i64),
        2 => Some(t.a as i64 * t.b as i64),
        3 if t.a >= 160 && t.b >= 160 => Some(t.a as i64 * t.b as i64),
        _ => None,
    }
}
fn call_cost(which: usize) -> f64 { [8.0, 0.65, 3.0, 0.75][which] }
fn reference(t: Task) -> i64 { t.a as i64 * t.b as i64 }
fn feedback(t: Task, which: usize) -> Feedback {
    let ok = execute(t, which) == Some(reference(t));
    let cost = call_cost(which);
    Feedback { cost, first_ok: ok, total_cost: cost + if ok { 0.0 } else { call_cost(0) } }
}
fn oracle(t: Task) -> usize {
    (0..TOOLS).filter(|&i| execute(t, i) == Some(reference(t)))
        .min_by(|&i, &j| call_cost(i).total_cmp(&call_cost(j))).unwrap_or(0)
}

#[derive(Clone, Copy)]
struct Sample { task: Task, outcomes: [Feedback; TOOLS] }
fn samples(seed: u64, n: usize, shifted: bool) -> Vec<Sample> {
    let mut rng=Rng::new(seed);
    (0..n).map(|_| { let t=task(&mut rng, shifted); Sample {
        task:t, outcomes:std::array::from_fn(|i| feedback(t,i))
    }}).collect()
}

#[derive(Clone, Copy)]
struct Stats { sum: f64, count: usize }
impl Stats {
    fn new() -> Self { Self { sum:0.0, count:0 } }
    fn add(&mut self, x:f64) { self.sum+=x; self.count+=1; }
    fn mean(&self) -> f64 { if self.count==0 { f64::INFINITY } else { self.sum/self.count as f64 } }
}

#[derive(Clone)]
struct KindOnly { means:[f64;TOOLS] }
impl KindOnly {
    fn train(data:&[Sample]) -> Self { Self { means:std::array::from_fn(|i| data.iter().map(|r|r.outcomes[i].total_cost).sum::<f64>() / data.len() as f64) } }
    fn choose(&self, _:Task)->usize { argmin(&self.means) }
}
fn argmin(v:&[f64;TOOLS])->usize {
    (0..TOOLS).min_by(|&i,&j|v[i].total_cmp(&v[j])).unwrap()
}
#[derive(Clone)]
struct Binned { means:[[f64;TOOLS];9] }
impl Binned {
    fn train(data:&[Sample]) -> Self {
        let prior=KindOnly::train(data);
        let mut sums=[[0.0;TOOLS];9];let mut counts=[0usize;9];
        for s in data { let b=bucket(s.task);counts[b]+=1;for i in 0..TOOLS {sums[b][i]+=s.outcomes[i].total_cost;} }
        for b in 0..9 {for i in 0..TOOLS {sums[b][i]=if counts[b]<10 {prior.means[i]}else{sums[b][i]/counts[b] as f64};}}
        Self{means:sums}
    }
    fn choose(&self,t:Task)->usize{argmin(&self.means[bucket(t)])}
}
fn bucket(t:Task)->usize { ((t.a.max(t.b)/40) as usize).min(8) }

// Single decision stump fitted independently for each tool. Feature and threshold are
// selected from *training observations*, not from tool implementation constants.
#[derive(Clone, Copy)]
struct Stump { feature:usize, cut:u32, below:f64, above:f64, score:f64 }
impl Stump {
    fn predict(&self,t:Task)->f64 {if features(t)[self.feature]<=self.cut {self.below}else{self.above}}
}
fn fit_tool(data:&[Sample], tool:usize)->Stump {
    let ys:Vec<f64>=data.iter().map(|s|s.outcomes[tool].total_cost).collect();
    let n=ys.len();let mean=ys.iter().sum::<f64>()/n as f64;
    let mut best=Stump{feature:0,cut:u32::MAX,below:mean,above:mean,score:ys.iter().map(|y|(y-mean).powi(2)).sum()};
    for feature in 0..FEATURES {
        let mut pairs:Vec<(u32,f64)>=data.iter().zip(&ys).map(|(s,y)|(features(s.task)[feature],*y)).collect();
        pairs.sort_unstable_by_key(|x|x.0);
        let sum: f64=pairs.iter().map(|p|p.1).sum();
        let sq: f64=pairs.iter().map(|p|p.1*p.1).sum();
        let mut left=0.0;let mut left_sq=0.0;
        for idx in 0..n-1 {
            let y=pairs[idx].1;left+=y;left_sq+=y*y;
            let a=idx+1; let b=n-a;
            if a<MIN_LEAF||b<MIN_LEAF||pairs[idx].0==pairs[idx+1].0 {continue;}
            let right=sum-left;
            let mse=left_sq-left*left/a as f64 + (sq-left_sq)-right*right/b as f64;
            if mse+1e-9<best.score {
                best=Stump{feature,cut:pairs[idx].0,below:left/a as f64,above:right/b as f64,score:mse};
            }
        }
    }
    best
}
#[derive(Clone)]
struct Discovered { stumps:[Stump;TOOLS] }
impl Discovered {
    fn train(data:&[Sample])->Self{Self{stumps:std::array::from_fn(|i|fit_tool(data,i))}}
    fn choose(&self,t:Task)->usize {
        let predicted: [f64;TOOLS]=std::array::from_fn(|i| self.stumps[i].predict(t));
        argmin(&predicted)
    }
}

fn choose(policy:&str,t:Task,kind:&KindOnly,binned:&Binned,discovered:&Discovered)->usize{
    match policy{"universal"=>0,"kind_only"=>kind.choose(t),"fixed_bins"=>binned.choose(t),"discovered"=>discovered.choose(t),"oracle"=>oracle(t),_=>unreachable!()}
}
#[derive(Default)]
struct Metric { n:usize,first:usize,cost:f64,optimal:usize,decision_ns:u128 }
impl Metric {
    fn record(&mut self,t:Task,policy:&str,k:&KindOnly,b:&Binned,d:&Discovered)->(usize,Feedback){
        let start=Instant::now();let i=choose(policy,t,k,b,d);
        self.decision_ns+=start.elapsed().as_nanos();
        let f=feedback(t,i);self.n+=1;self.first+=f.first_ok as usize;self.cost+=f.total_cost;
        self.optimal+=(i==oracle(t)) as usize;
        (i,f)
    }
    fn mean(&self)->f64{self.cost/self.n as f64}
}

fn run()->io::Result<()> {
    let mut seeds=8usize;
    let mut args=std::env::args().skip(1);
    while let Some(arg)=args.next(){match arg.as_str(){
        "--seeds"=>seeds=args.next().and_then(|x|x.parse().ok()).unwrap_or(8),
        "--help"=>{println!("cargo run --release --bin zero_feature_discovery -- --seeds 8");return Ok(())},
        _=>{eprintln!("Unknown argument: {arg}");std::process::exit(2)}
    }}
    if !(1..=100).contains(&seeds){eprintln!("--seeds must be 1..100");std::process::exit(2)}
    let dir="reports/zero_feature_discovery";fs::create_dir_all(dir)?;
    let mut summary=File::create(format!("{dir}/summary.csv"))?;
    let mut models=File::create(format!("{dir}/learned_features.csv"))?;
    let mut decisions=File::create(format!("{dir}/sample_decisions.csv"))?;
    writeln!(summary,"seed,split,policy,n,first_success_pct,mean_total_cost,oracle_choice_pct,mean_decision_ns,training_probe_cost,training_probe_calls")?;
    writeln!(models,"seed,tool,feature,threshold,estimated_cost_below,estimated_cost_above,train_sse")?;
    writeln!(decisions,"seed,split,a,b,policy,tool,first_ok,total_cost,oracle_tool")?;
    let policies=["universal","kind_only","fixed_bins","discovered","oracle"];
    println!("ZERO FEATURE DISCOVERY | {seeds} seeds | 2400 train / 500 validation / 3000 test per seed");
    for s in 0..seeds {
        let seed=20261107u64+(s as u64)*7919;
        let train=samples(seed^0xA11C_E001,TRAIN,false);
        let validation=samples(seed^0xB00C_F012,VALID,false);
        let test=samples(seed^0xD00D_B123,TEST,true);
        // All policies observe identical full-information probe calls. Cost is
        // reported separately, never silently erased from lifetime accounting.
        let probes=train.iter().flat_map(|x|x.outcomes.iter());
        let probe_cost:f64=probes.map(|f|f.total_cost).sum();
        let k=KindOnly::train(&train);let b=Binned::train(&train);let d=Discovered::train(&train);
        for i in 0..TOOLS {let st=d.stumps[i];
            writeln!(models,"{seed},{},{},{},{:.6},{:.6},{:.6}",NAMES[i],FEATURE_NAMES[st.feature],st.cut,st.below,st.above,st.score)?;
        }
        for (split,data) in [("validation",&validation),("test",&test)] {
            for policy in policies {
                let mut m=Metric::default();
                for (ix,sample) in data.iter().enumerate() {
                    let (selected,f)=m.record(sample.task,policy,&k,&b,&d);
                    if s==0 && ix<80 && split=="test" {
                        writeln!(decisions,"{seed},{split},{},{},{policy},{},{},{:.6},{}",sample.task.a,sample.task.b,NAMES[selected],f.first_ok,f.total_cost,NAMES[oracle(sample.task)])?;
                    }
                }
                writeln!(summary,"{seed},{split},{policy},{},{:.4},{:.6},{:.4},{:.2},{:.3},{}",m.n,
                    100.0*m.first as f64/m.n as f64,m.mean(),100.0*m.optimal as f64/m.n as f64,
                    m.decision_ns as f64/m.n as f64,probe_cost,TRAIN*TOOLS)?;
                if split=="test" {println!("seed={seed} {policy:12} cost={:.4} first={:.2}%",m.mean(),100.0*m.first as f64/m.n as f64);}
            }
        }
    }
    println!("Saved reports in {dir}/. Training probes are fully charged in separate fields.");
    println!("NOTE: this is supervised black-box feature selection with reference feedback, not general intelligence.");
    Ok(())
}
fn main(){if let Err(err)=run(){eprintln!("ZERO error: {err}");std::process::exit(1)}}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]fn tools_obey_hidden_constraints(){
        assert!(execute(Task{a:90,b:90},1).is_some());
        assert!(execute(Task{a:120,b:90},1).is_none());
        assert!(execute(Task{a:160,b:160},3).is_some());
        assert!(execute(Task{a:160,b:159},3).is_none());
    }
    #[test]fn input_features_contain_no_fixed_bins(){
        let x=features(Task{a:20,b:180});
        assert_eq!(x,[20,180,20,180,200,160,3600]);
    }
    #[test]fn training_finds_useful_feature(){
        let data=samples(505,TRAIN,false);let model=Discovered::train(&data);
        let good_low=model.choose(Task{a:60,b:70});
        let good_high=model.choose(Task{a:230,b:250});
        assert_eq!(good_low,1,"cheap small specialist should be selected");
        assert_eq!(good_high,3,"cheap large specialist should be selected");
    }
    #[test]fn oracle_is_best_for_examples(){
        assert_eq!(oracle(Task{a:30,b:40}),1);
        assert_eq!(oracle(Task{a:130,b:140}),2);
        assert_eq!(oracle(Task{a:200,b:201}),3);
    }
    #[test]fn all_failed_calls_charge_fallback(){
        let f=feedback(Task{a:200,b:200},1);
        assert!(!f.first_ok);assert!((f.total_cost-8.65).abs()<1e-9);
    }
    #[test]fn split_deterministic(){
        let a=samples(101,15,true);let b=samples(101,15,true);
        for (x,y) in a.iter().zip(&b){assert_eq!(x.task.a,y.task.a);assert_eq!(x.task.b,y.task.b);}
    }
}
