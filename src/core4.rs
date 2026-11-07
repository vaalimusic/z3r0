//! ZERO Core 0.4: two separately trained outcome predictors.
//! Frozen OCR; labels only from synthetic glyphs on original OCR-training fonts.
use super::*;
use serde_json::{json, Value};
use std::collections::BTreeMap;

const FILE: &str = "zero_scheduler_v04.json";
const DIR: &str = "reports/core4";
const FEATURES: usize = 15;

#[derive(Serialize, Deserialize)]
struct Policy {
    version: u32,
    classes: String,
    memory_len: usize,
    features: usize,
    fix_weights: Vec<f32>,
    harm_weights: Vec<f32>,
    penalty: f32,
    mean_memory_us: f32,
    train_fonts: Vec<String>,
    validation_fonts: Vec<String>,
    train_samples: usize,
    validation_samples: usize,
    validation_net_gain: f32,
    validation_calls: usize,
}
#[derive(Clone)]
struct Example { features: [f32;FEATURES], fix: f32, harm: f32 }

fn features(p:f32, m:f32, ch:char, x:&[f32]) -> [f32;FEATURES] {
    let density = x.iter().sum::<f32>()/x.len().max(1) as f32;
    let mut cx=0f32;let mut cy=0f32;let mut mass=0f32;
    let mut symmetry=0f32;
    for y in 0..SIDE {for xx in 0..SIDE {
        let a=x[y*SIDE+xx];mass+=a;
        cx+=a*xx as f32/SIDE as f32;cy+=a*y as f32/SIDE as f32;
        symmetry+=(a-x[y*SIDE+SIDE-1-xx]).abs();
    }}
    let mass=mass.max(0.00001);
    [1., p, m, 1.-p, 1.-m, (p*m).sqrt(),
        if "O0o".contains(ch){1.}else{0.},
        if "Il1i".contains(ch){1.}else{0.},
        if "Ss5".contains(ch){1.}else{0.},
        if "Zz2".contains(ch){1.}else{0.},
        if "Cc".contains(ch){1.}else{0.},
        density, cx/mass, cy/mass, symmetry/x.len().max(1) as f32]
}
fn sigmoid(x:f32)->f32 {let x=x.clamp(-18.,18.);1./(1.+(-x).exp())}
fn probability(w:&[f32], f:&[f32;FEATURES])->f32 {
    sigmoid(w.iter().zip(f).map(|(a,b)|a*b).sum())
}
// Independent Bernoulli heads. Not a calibrated 3-class model; their scores
// are empirical estimates and are explicitly assessed on validation data.
fn fit(examples:&[Example],fix:bool)->Vec<f32> {
    let mut w=vec![0f32;FEATURES];
    let pos=examples.iter().filter(|e|if fix{e.fix>0.}else{e.harm>0.}).count();
    let prior=((pos as f32+1.)/(examples.len() as f32+2.)).clamp(0.0001,0.9999);
    w[0]=(prior/(1.-prior)).ln();
    let mut rng=StdRng::seed_from_u64(if fix{40401}else{40402});
    let mut order:Vec<usize>=(0..examples.len()).collect();
    for epoch in 0..80 {
        order.shuffle(&mut rng);
        let rate=0.025/(1.+epoch as f32*0.04);
        for &i in &order {
            let e=&examples[i];let target=if fix{e.fix}else{e.harm};
            let diff=probability(&w,&e.features)-target;
            for j in 0..FEATURES {w[j]-=rate*(diff*e.features[j]+0.0005*w[j]);}
        }
    }
    w
}
fn collect(brain:&Brain, fonts:&[PathBuf], seed:u64)->Result<(Vec<Example>,f32)> {
    let mut rng=StdRng::seed_from_u64(seed);
    let mut out=Vec::new();let mut elapsed=0u128;let mut calls=0u128;
    for path in fonts {
        let font=load_font(path)?;
        let mut chars=brain.chars.clone();chars.shuffle(&mut rng);
        for ch in chars {for _ in 0..6 {
            let sz=rng.gen_range(14.0..56.0);let shear=rng.gen_range(-0.20..0.20);
            let Some(x)=sample_glyph(&font,ch,sz,shear,&mut rng) else {continue};
            let (nc,p,m)=brain.neural(&x);
            let base=core3_neural_answer(nc,p,m);
            let time=Instant::now();let (mc,s,d)=brain.template(&x);
            elapsed+=time.elapsed().as_micros();calls+=1;
            let alternate=core3_template_answer(mc,s,d,base);
            out.push(Example {features:features(p,m,nc,&x),
                fix:if base!=ch && alternate==ch{1.}else{0.},
                harm:if base==ch && alternate!=ch{1.}else{0.}});
        }}
    }
    Ok((out,elapsed as f32/calls.max(1) as f32))
}
fn benefit(p:&Policy,f:&[f32;FEATURES])->(f32,f32,f32) {
    let fix=probability(&p.fix_weights,f);
    let harm=probability(&p.harm_weights,f);
    (fix,harm,fix-harm-p.penalty)
}
fn score_validation(ex:&[Example],fw:&[f32],hw:&[f32],penalty:f32)->(f32,usize) {
    // Equal budget cap of 12 calls per synthetic block of 64 glyphs.
    let mut net=0f32;let mut used=0usize;
    for batch in ex.chunks(64) {
        let mut options:Vec<(usize,f32)>=batch.iter().enumerate().map(|(i,e)|
            (i,probability(fw,&e.features)-probability(hw,&e.features)-penalty)).filter(|(_,v)|*v>0.).collect();
        options.sort_by(|a,b|b.1.total_cmp(&a.1));
        for (i,_) in options.into_iter().take(12) {
            net+=batch[i].fix-batch[i].harm-0.02;
            used+=1;
        }
    }
    (net/ex.len().max(1) as f32,used)
}
fn load(brain:&Brain)->Result<Policy> {
    let p:Policy=serde_json::from_slice(&fs::read(FILE).map_err(|_|"Missing ZERO Core4 policy; run `cargo run --release --bin zero_ocr -- core4-train`")?)?;
    if p.version!=4 || p.classes!=brain.chars.iter().collect::<String>() ||
        p.memory_len!=brain.memory.len() || p.features!=FEATURES ||
        p.fix_weights.len()!=FEATURES || p.harm_weights.len()!=FEATURES {
        return Err("Core4 policy incompatible with OCR brain; rerun core4-train".into());
    }
    Ok(p)
}
pub(super) fn train(_args:&[String])->Result<()> {
    let brain=load_brain()?;
    let held=brain.heldout_font.as_deref().unwrap_or("");
    let mut fonts:Vec<PathBuf>=brain.trained_fonts.iter().map(PathBuf::from)
        .filter(|p|p.is_file() && p.to_string_lossy()!=held).collect();
    fonts.sort();fonts.dedup();
    if fonts.len()<5 {return Err("Need at least 5 original training font files".into());}
    fonts.shuffle(&mut StdRng::seed_from_u64(440041));
    let valid_n=(fonts.len()/4).max(1);
    let valid=fonts.split_off(fonts.len()-valid_n);
    let (train,mean_us)=collect(&brain,&fonts,440042)?;
    let (validation,_)=collect(&brain,&valid,440043)?;
    let fw=fit(&train,true);let hw=fit(&train,false);
    let mut best=(f32::NEG_INFINITY,0f32,0usize);
    for penalty in [0.,0.01,0.02,0.04,0.06,0.08,0.10,0.12,0.15] {
        let (gain,calls)=score_validation(&validation,&fw,&hw,penalty);
        if gain>best.0 {best=(gain,penalty,calls);}
    }
    let model=Policy {version:4,classes:brain.chars.iter().collect(),memory_len:brain.memory.len(),
        features:FEATURES,fix_weights:fw,harm_weights:hw,penalty:best.1,mean_memory_us:mean_us,
        train_fonts:fonts.iter().map(|p|p.display().to_string()).collect(),
        validation_fonts:valid.iter().map(|p|p.display().to_string()).collect(),
        train_samples:train.len(),validation_samples:validation.len(),
        validation_net_gain:best.0,validation_calls:best.2};
    fs::write(FILE,serde_json::to_vec_pretty(&model)?)?;
    fs::create_dir_all(DIR)?;
    let fixes=train.iter().filter(|e|e.fix>0.).count();
    let harms=train.iter().filter(|e|e.harm>0.).count();
    println!("ZERO CORE 0.4 trained {} samples on {} fonts; validation {} samples on {} other fonts",train.len(),fonts.len(),validation.len(),valid.len());
    println!("Training outcomes: fixes {} harms {} neutral {}",fixes,harms,train.len()-fixes-harms);
    println!("Validation net utility {:.5}, penalty {:.3}, calls {}, mean memory {:.0}us",best.0,best.1,best.2,mean_us);
    println!("Saved {FILE}. OCR brain unchanged. Held-out OCR font not used.");
    Ok(())
}

struct Item { pixels:Vec<f32>, r:GlyphReport, line:usize,space:bool,score:f32,fix:f32,harm:f32 }
struct Scan { text:String, calls:usize, time_us:u128, memory_us:u128, decisions:Vec<Value> }
fn scan(brain:&Brain,policy:&Policy,img:&GrayImage,budget:usize)->Scan {
    let start=Instant::now();let mut items=Vec::<Item>::new();
    for (li,line) in segment(img).iter().enumerate() {
        let height=line.iter().map(|(r,_)|r.h).max().unwrap_or(24) as f32;
        let mut gaps:Vec<u32>=line.iter().skip(1).map(|(_,g)|*g).filter(|g|*g>0).collect();
        gaps.sort();let median=gaps.get(gaps.len()/2).copied().unwrap_or(1) as f32;
        for (i,(r,gap)) in line.iter().enumerate() {
            let space=i>0 && *gap as f32 >= (height*0.22).max(median*2.1).max(5.);
            let crop=image::imageops::crop_imm(img,r.x,r.y,r.w,r.h).to_image();
            let pixels=normalize(&crop);
            let t=Instant::now();let (n,p,m)=brain.neural(&pixels);let nus=t.elapsed().as_micros();
            let result=core3_neural_answer(n,p,m);
            let r=GlyphReport {x:r.x,y:r.y,w:r.w,h:r.h,result,
                mode:if result=='?' {"rejected".into()}else{"neural".into()},
                template_candidate:'?',template_similarity:0.,template_margin:0.,
                neural_candidate:Some(n),neural_confidence:Some(p),neural_margin:Some(m),
                template_us:None,neural_us:Some(nus)};
            let (fix,harm,score)=benefit(policy,&features(p,m,n,&pixels));
            items.push(Item {pixels,r,line:li,space,score,fix,harm});
        }
    }
    let mut idx:Vec<usize>=(0..items.len()).collect();
    idx.sort_by(|a,b|items[*b].score.total_cmp(&items[*a].score).then(a.cmp(b)));
    let mut calls=0;let mut mem_us=0;
    for i in idx {
        if calls>=budget || items[i].score<=0. {break;}
        let t=Instant::now();let (c,s,m)=brain.template(&items[i].pixels);
        let cost=t.elapsed().as_micros();calls+=1;mem_us+=cost;
        let r=&mut items[i].r;
        r.template_candidate=c;r.template_similarity=s;r.template_margin=m;r.template_us=Some(cost);
        let next=core3_template_answer(c,s,m,r.result);
        if next!=r.result {r.mode="memory".into();r.result=next;}
    }
    let mut text=String::new();let mut prev=None;let mut decisions=Vec::new();
    for (i,item) in items.iter().enumerate() {
        if let Some(p)=prev {if p!=item.line{text.push('\n');}else if item.space{text.push(' ');}}
        prev=Some(item.line);text.push(item.r.result);
        decisions.push(json!({"index":i,"line":item.line,"x":item.r.x,"y":item.r.y,
            "predicted_fix":item.fix,"predicted_harm":item.harm,"utility":item.score,
            "called_memory":item.r.template_us.is_some(),"output":item.r.result,
            "neural":item.r.neural_candidate,"confidence":item.r.neural_confidence,
            "memory_candidate":item.r.template_candidate,"memory_us":item.r.template_us}));
    }
    Scan{text,calls,time_us:start.elapsed().as_micros(),memory_us:mem_us,decisions}
}
fn opt<'a>(args:&'a[String],name:&str)->Option<&'a str> {
    args.iter().position(|s|s==name).and_then(|i|args.get(i+1)).map(String::as_str)
}
fn budgets(args:&[String])->Result<Vec<usize>> {
    let raw=opt(args,"--budgets").unwrap_or("0,2,4,6,8,12");
    let mut b=Vec::new();for v in raw.split(',') {
        let n:usize=v.trim().parse()?;if n>500{return Err("Budget must be <=500".into());}b.push(n);
    }
    b.sort_unstable();b.dedup();Ok(b)
}
// Use Lab manifest, and score ONLY selected split. Lab train does not train this policy.
fn test_images(args:&[String])->Result<Vec<(String,GrayImage,String)>> {
    let split=opt(args,"--split").unwrap_or("test");
    if !["train","validation","test"].contains(&split) {return Err("--split must be train, validation, test".into());}
    let manifest:Value=serde_json::from_slice(&fs::read("reports/lab/manifest.json")?)?;
    let entries=manifest["entries"].as_array().ok_or("Invalid manifest entries")?;
    let mut out=Vec::new();
    for e in entries {
        if e["split"].as_str()!=Some(split){continue;}
        let path=e["path"].as_str().ok_or("Missing image path")?;
        let gt=e["ground_truth"].as_str().ok_or("Missing truth path")?;
        let image=image::open(path)?.to_luma8();
        let truth=fs::read_to_string(gt)?.trim_end_matches(['\r','\n']).replace("\r\n","\n");
        out.push((path.to_string(),image,truth));
    }
    if out.is_empty(){return Err("No dataset entries in requested split; generate ZERO Lab dataset first".into());}
    Ok(out)
}
pub(super) fn bench(args:&[String])->Result<()> {
    let brain=load_brain()?;let core4=load(&brain)?;
    let core3=load_core3(&brain)?;let core2=load_scheduler(&brain)?;
    let dataset=test_images(args)?;let caps=budgets(args)?;
    fs::create_dir_all(DIR)?;
    let mut rows=Vec::new();
    println!("ZERO Core4 benchmark | {} images | budgets {:?}",dataset.len(),caps);
    for &cap in &caps {
        let mut acc:BTreeMap<String,(usize,usize,u128,usize)>=BTreeMap::new();
        for (_name,img,truth) in &dataset {
            for mode in ["neural","confidence","core02","core03","core04"] {
                let (text,calls,time)=if mode=="core04" {
                    let _=scan(&brain,&core4,img,cap); // warm-up
                    let s=scan(&brain,&core4,img,cap);
                    (s.text,s.calls,s.time_us)
                } else {
                    let strategy=match mode {"confidence"=>Core3Mode::Confidence,"core02"=>Core3Mode::Core02,
                        "core03"=>Core3Mode::Core03,_=>Core3Mode::Random};
                    let b=if mode=="neural"{0}else{cap};
                    let _=core3_scan(&brain,&core3,&core2,img,strategy,b,None,715190);
                    let s=core3_scan(&brain,&core3,&core2,img,strategy,b,None,715190);
                    (s.text,s.memory_calls,s.duration_us)
                };
                let errors=levenshtein(truth,&text);let total=truth.chars().count();
                let a=acc.entry(mode.into()).or_default();a.0+=errors;a.1+=total;a.2+=time;a.3+=calls;
            }
        }
        for (mode,(err,chars,elapsed,calls)) in acc {
            let cer=err as f64/chars.max(1) as f64;
            let avg_ms=elapsed as f64/dataset.len() as f64/1000.;
            let mean_calls=calls as f64/dataset.len() as f64;
            println!("cap {:>2} {:<12} CER {:>6.2}%  avg {:>7.2}ms  calls {:>5.2}",cap,mode,cer*100.,avg_ms,mean_calls);
            rows.push(json!({"cap":cap,"mode":mode,"images":dataset.len(),"characters":chars,
                "errors":err,"micro_cer":cer,"mean_ms":avg_ms,"mean_calls":mean_calls}));
        }
    }
    let mut csv=String::from("cap,mode,images,characters,errors,micro_cer,mean_ms,mean_calls\n");
    for row in &rows {csv.push_str(&format!("{},{},{},{},{},{:.8},{:.4},{:.4}\n",
        row["cap"],row["mode"].as_str().unwrap_or(""),row["images"],row["characters"],
        row["errors"],row["micro_cer"].as_f64().unwrap_or(0.),
        row["mean_ms"].as_f64().unwrap_or(0.),row["mean_calls"].as_f64().unwrap_or(0.)));}
    fs::write(format!("{DIR}/benchmark.csv"),csv)?;
    fs::write(format!("{DIR}/benchmark.json"),serde_json::to_vec_pretty(&rows)?)?;
    println!("Reports: {DIR}/benchmark.csv + benchmark.json. Test ground truth used ONLY for scoring.");
    Ok(())
}
pub(super) fn run(args:&[String])->Result<()> {
    let brain=load_brain()?;let policy=load(&brain)?;
    let budget:usize=opt(args,"--budget-calls").unwrap_or("12").parse()?;
    fs::create_dir_all("output")?;fs::create_dir_all(DIR)?;
    for path in image_paths()? {
        let img=image::open(&path)?.to_luma8();
        let s=scan(&brain,&policy,&img,budget);
        let stem=path.file_stem().unwrap_or_default().to_string_lossy();
        fs::write(format!("output/{}_core4.txt",stem),&s.text)?;
        fs::write(format!("output/{}_core4.json",stem),serde_json::to_vec_pretty(&s.decisions)?)?;
        println!("{} [{:.2}ms, memory calls {}, {:.2}ms memory]:\n{}\n",path.display(),
            s.time_us as f64/1000.,s.calls,s.memory_us as f64/1000.,s.text);
    }
    Ok(())
}
