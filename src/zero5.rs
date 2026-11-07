//! ZERO 0.5: small confusable-character specialists and an honest frozen-brain test.
//! Calibration uses only Lab validation; the test split is read only for final scoring.
use super::*;
use serde_json::{json, Value};
use std::collections::BTreeMap;

const POLICY_FILE: &str = "zero_specialists_v05.json";
const REPORT_DIR: &str = "reports/zero5";
const MANIFEST: &str = "reports/lab/manifest.json";
const GROUPS: &[&str] = &["O0o", "Il1i", "Ss5", "Zz2", "Cc", "Vv", "Ww"];
const THRESHOLDS: &[f32] = &[0.0, 0.002, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16];

#[derive(Serialize, Deserialize)]
struct Policy {
    version: u32,
    classes: String,
    memory_len: usize,
    threshold: f32,
    prototypes: BTreeMap<char, Vec<f32>>,
    validation_images: usize,
    validation_cer: f64,
    validation_neural_cer: f64,
    validation_specialist_calls: usize,
}
struct Entry { id: String, image: String, truth: String }
fn options<'a>(args: &'a [String], key: &str) -> Option<&'a str> {
    args.iter().position(|a|a==key).and_then(|i|args.get(i+1)).map(String::as_str)
}
fn entries(split: &str) -> Result<Vec<Entry>> {
    let manifest:Value = serde_json::from_slice(&fs::read(MANIFEST).map_err(|_| 
        "Missing Lab manifest. Generate the dataset using zero_lab -- generate --count 240 --seed 2026")?)?;
    let mut result=Vec::new();
    for x in manifest["entries"].as_array().ok_or("Invalid Lab manifest")? {
        if x["split"].as_str()!=Some(split) {continue;}
        let get=|key:&str| -> Result<String> {
            Ok(x[key].as_str().ok_or_else(||format!("Manifest missing {key}"))?.to_string())
        };
        result.push(Entry {id:get("id")?,image:get("path")?,truth:get("ground_truth")?});
    }
    if result.is_empty() {return Err(format!("No samples in {split}").into());}
    Ok(result)
}
fn prototypes(brain:&Brain)->BTreeMap<char,Vec<f32>> {
    let mut result=BTreeMap::new();
    let mut counts=BTreeMap::<char,usize>::new();
    for item in &brain.memory {
        if !GROUPS.iter().any(|g|g.contains(item.ch)) || item.pixels.len()!=INPUT {continue;}
        let sum=result.entry(item.ch).or_insert_with(||vec![0f32;INPUT]);
        for (s,p) in sum.iter_mut().zip(&item.pixels) {*s+=*p;}
        *counts.entry(item.ch).or_default()+=1;
    }
    for (ch,sum) in result.iter_mut() {
        let n=*counts.get(ch).unwrap_or(&1) as f32;
        for v in sum.iter_mut() {*v/=n;}
    }
    result
}
fn group(ch:char)->Option<&'static str>{GROUPS.iter().copied().find(|s|s.contains(ch))}
// The specialist is deliberately cheap: at most four 24x24 centroid comparisons.
// Never use the correct text during prediction.
fn specialist(p:&Policy, x:&[f32], neural:char, threshold:f32)->Option<char> {
    let g=group(neural)?;
    let mut ranked=Vec::<(char,f32)>::new();
    for c in g.chars() {
        let Some(proto)=p.prototypes.get(&c) else {continue;};
        let mse=x.iter().zip(proto).map(|(a,b)| (a-b)*(a-b)).sum::<f32>() / x.len().max(1) as f32;
        ranked.push((c,mse));
    }
    if ranked.len()<2 {return None;}
    ranked.sort_by(|a,b|a.1.total_cmp(&b.1));
    // If the centroid cannot clearly distinguish the alternatives, do nothing.
    let margin=ranked[1].1-ranked[0].1;
    if margin>=threshold && ranked[0].0!=neural {Some(ranked[0].0)} else {None}
}
fn load(brain:&Brain)->Result<Policy>{
    let p:Policy=serde_json::from_slice(&fs::read(POLICY_FILE).map_err(|_| 
        "No ZERO 0.5 calibration: run `cargo run --release --bin zero_ocr -- zero5-calibrate`")?)?;
    if p.version!=5 || p.classes!=brain.chars.iter().collect::<String>() || p.memory_len!=brain.memory.len() {
        return Err("ZERO 0.5 policy incompatible with OCR brain".into());
    }
    Ok(p)
}
struct Item { pixels:Vec<f32>, result:char, neural:char, p:f32, m:f32,
    space:bool, line:usize, rank:f32 }
struct Scan { text:String, memory_calls:usize, specialist_calls:usize, specialist_changes:usize,
    duration_us:u128 }
fn scan(brain:&Brain, p:&Policy, old:Option<&Scheduler>, img:&GrayImage,
        budget:usize, threshold:f32, enable_specialists:bool)->Scan {
    let start=Instant::now();let mut items=Vec::<Item>::new();
    for (line_index,line) in segment(img).iter().enumerate() {
        let height=line.iter().map(|(r,_)|r.h).max().unwrap_or(24) as f32;
        let mut gaps:Vec<u32>=line.iter().skip(1).map(|(_,g)|*g).filter(|g|*g>0).collect();
        gaps.sort();let median=gaps.get(gaps.len()/2).copied().unwrap_or(1) as f32;
        for (i,(r,gap)) in line.iter().enumerate() {
            let space=i>0 && *gap as f32 >= (height*0.22).max(median*2.1).max(5.);
            let crop=image::imageops::crop_imm(img,r.x,r.y,r.w,r.h).to_image();
            let x=normalize(&crop);
            let (n,prob,margin)=brain.neural(&x);
            let answer=core3_neural_answer(n,prob,margin);
            let rank=if let Some(pol)=old {
                let bin=Scheduler::bin(prob,margin,n);
                if pol.should_check(prob,margin,n) {1.0+pol.gains[bin]} else {-1.}
            } else {-1.};
            items.push(Item {pixels:x,result:answer,neural:n,p:prob,m:margin,
                line:line_index,space,rank});
        }
    }
    let mut spec_calls=0;let mut spec_changes=0;
    if enable_specialists {
        for item in &mut items {
            if group(item.neural).is_some() {
                spec_calls+=1;
                if let Some(ch)=specialist(p,&item.pixels,item.neural,threshold) {
                    if ch!=item.result {spec_changes+=1;item.result=ch;}
                }
            }
        }
    }
    // The expensive Core 0.2 fallback runs on a fixed budget. It is intentionally
    // held constant in the specialist+Core02 comparison; no ground truth leaks in.
    let mut selected:Vec<usize>=(0..items.len()).filter(|&i|items[i].rank>0.).collect();
    selected.sort_by(|a,b|items[*b].rank.total_cmp(&items[*a].rank).then_with(||a.cmp(b)));
    let mut calls=0;
    for i in selected.into_iter().take(budget) {
        let (ch,s,m)=brain.template(&items[i].pixels);
        items[i].result=core3_template_answer(ch,s,m,items[i].result);
        calls+=1;
    }
    let mut text=String::new();let mut prev_line=None;
    for item in &items {
        if let Some(prev)=prev_line {
            if prev!=item.line {text.push('\n');}
            else if item.space {text.push(' ');}
        }
        prev_line=Some(item.line);text.push(item.result);
    }
    Scan{text,memory_calls:calls,specialist_calls:spec_calls,specialist_changes:spec_changes,
        duration_us:start.elapsed().as_micros()}
}
fn clean(s:&str)->String{s.trim_end_matches(['\r','\n']).replace("\r\n","\n")}
fn evaluate(brain:&Brain,p:&Policy,split:&str,threshold:f32,
    baseline:bool,budget:usize)->Result<(usize,usize,u128,usize,usize,usize)> {
    let old=if budget>0 {Some(load_scheduler(brain)?)} else {None};
    let mut total_err=0;let mut total_chars=0;let mut total_us=0;
    let mut mem=0;let mut spec=0;let mut changes=0;
    for e in entries(split)? {
        let img=image::open(&e.image)?.to_luma8();
        let truth=clean(&fs::read_to_string(&e.truth)?);
        let s=scan(brain,p,old.as_ref(),&img,budget,threshold,!baseline);
        total_err+=levenshtein(&truth,&s.text);
        total_chars+=truth.chars().count();total_us+=s.duration_us;
        mem+=s.memory_calls;spec+=s.specialist_calls;changes+=s.specialist_changes;
    }
    Ok((total_err,total_chars,total_us,mem,spec,changes))
}
fn summary_row(brain:&Brain,p:&Policy,split:&str,mode:&str,budget:usize,
    baseline:bool)->Result<Value>{
    let (err,chars,us,mem,spec,changes)=evaluate(brain,p,split,p.threshold,baseline,budget)?;
    let images=entries(split)?.len();
    Ok(json!({"split":split,"mode":mode,"budget":budget,"images":images,
        "errors":err,"characters":chars,"cer":err as f64/chars.max(1) as f64,
        "mean_ms":us as f64/images as f64/1000.,
        "mean_memory_calls":mem as f64/images as f64,
        "mean_specialist_calls":spec as f64/images as f64,
        "mean_specialist_changes":changes as f64/images as f64}))
}
pub(super) fn calibrate(_args:&[String])->Result<()> {
    let brain=load_brain()?;
    let mut policy=Policy{version:5,classes:brain.chars.iter().collect(),
        memory_len:brain.memory.len(),threshold:0.,prototypes:prototypes(&brain),
        validation_images:0,validation_cer:1.,validation_neural_cer:1.,validation_specialist_calls:0};
    let expected:Vec<char>=GROUPS.iter().flat_map(|g|g.chars()).collect();
    let present=expected.iter().filter(|c|policy.prototypes.contains_key(*c)).count();
    if present<12 {return Err("Too few prototypes for specialists; check OCR brain memory".into());}
    println!("ZERO 0.5 | prototypes {} / {} | frozen OCR",present,expected.len());
    let (base_err,chars,_,_,_,_)=evaluate(&brain,&policy,"validation",0.,true,0)?;
    let mut best=(usize::MAX,0f32,0usize);
    for &threshold in THRESHOLDS {
        let (errors,_,_,_,calls,changes)=evaluate(&brain,&policy,"validation",threshold,false,0)?;
        println!("threshold {:>5.3}: CER {:>6.2}% | specialists {} | changed {}",
            threshold,100.*errors as f64/chars.max(1) as f64,calls,changes);
        if errors<best.0 || (errors==best.0 && threshold>best.1) {best=(errors,threshold,calls);}
    }
    policy.threshold=best.1;
    policy.validation_images=entries("validation")?.len();
    policy.validation_cer=best.0 as f64/chars.max(1) as f64;
    policy.validation_neural_cer=base_err as f64/chars.max(1) as f64;
    policy.validation_specialist_calls=best.2;
    fs::write(POLICY_FILE,serde_json::to_vec_pretty(&policy)?)?;
    println!("Selected threshold {:.3} on VALIDATION only. CER {:.2}% vs Neural {:.2}%",
        policy.threshold,100.*policy.validation_cer,100.*policy.validation_neural_cer);
    println!("Saved {POLICY_FILE}. No OCR or Core2 weight modifications.");
    Ok(())
}
pub(super) fn bench(args:&[String])->Result<()> {
    let brain=load_brain()?;let policy=load(&brain)?;
    let split=options(args,"--split").unwrap_or("test");
    if split!="test" {return Err("Final bench only permits --split test; calibration must remain separate".into());}
    let mut rows=Vec::new();
    for budget in [0usize,2,4,6,8,12] {
        for (name,baseline,cap) in [
            ("neural",true,0),
            ("specialist",false,0),
            ("core02",true,budget),
            ("specialist_core02",false,budget),
        ] {
            let row=summary_row(&brain,&policy,split,name,cap,baseline)?;
            println!("cap {:>2} {:<18} CER {:>6.2}%  time {:>7.2}ms  mem {:>5.2}  specialist {:>5.2}",
                budget,name,row["cer"].as_f64().unwrap_or(0.)*100.,
                row["mean_ms"].as_f64().unwrap_or(0.),
                row["mean_memory_calls"].as_f64().unwrap_or(0.),
                row["mean_specialist_calls"].as_f64().unwrap_or(0.));
            rows.push(row);
        }
    }
    fs::create_dir_all(REPORT_DIR)?;
    let mut csv=String::from("split,mode,budget,images,characters,errors,cer,mean_ms,mean_memory_calls,mean_specialist_calls,mean_specialist_changes\n");
    for r in &rows {
        csv.push_str(&format!("{},{},{},{},{},{},{:.8},{:.4},{:.4},{:.4},{:.4}\n",
            r["split"].as_str().unwrap_or(""),r["mode"].as_str().unwrap_or(""),
            r["budget"].as_u64().unwrap_or(0),r["images"].as_u64().unwrap_or(0),
            r["characters"].as_u64().unwrap_or(0),r["errors"].as_u64().unwrap_or(0),
            r["cer"].as_f64().unwrap_or(0.),r["mean_ms"].as_f64().unwrap_or(0.),
            r["mean_memory_calls"].as_f64().unwrap_or(0.),
            r["mean_specialist_calls"].as_f64().unwrap_or(0.),
            r["mean_specialist_changes"].as_f64().unwrap_or(0.)));
    }
    fs::write(format!("{REPORT_DIR}/benchmark.csv"),csv)?;
    fs::write(format!("{REPORT_DIR}/benchmark.json"),serde_json::to_vec_pretty(&rows)?)?;
    println!("Saved {REPORT_DIR}/benchmark.csv + benchmark.json");
    println!("Interpret CER and time jointly. No claim of benefit until measured on TEST.");
    Ok(())
}
pub(super) fn run(args:&[String])->Result<()> {
    let brain=load_brain()?;let p=load(&brain)?;
    let budget:usize=options(args,"--budget-calls").unwrap_or("6").parse()?;
    let old=if budget>0 {Some(load_scheduler(&brain)?)} else {None};
    fs::create_dir_all("output/zero5")?;
    for path in image_paths()? {
        let image=image::open(&path)?.to_luma8();
        let result=scan(&brain,&p,old.as_ref(),&image,budget,p.threshold,true);
        let name=path.file_stem().unwrap_or_default().to_string_lossy();
        let output=format!("output/zero5/{}.txt",name);
        fs::write(&output,&result.text)?;
        println!("{} | {:.2} ms | memory {} | specialists {} (changes {})\n{}\n",
            path.display(),result.duration_us as f64/1000.,result.memory_calls,
            result.specialist_calls,result.specialist_changes,result.text);
    }
    Ok(())
}
