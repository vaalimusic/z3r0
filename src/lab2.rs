//! ZERO Lab 0.2: budget curves, per-glyph decision audit, dataset diagnostics.
//! This code never trains or modifies the OCR brain or schedulers.
use super::*;
use serde_json::{json, Value};
use std::collections::BTreeMap;

const MANIFEST: &str = "reports/lab/manifest.json";
const DIR: &str = "reports/lab2";

#[derive(Clone)]
struct Entry {
    id: String, split: String, path: String, truth: String, font: String,
    px: f64, noise: u64, bg: u64, lines: usize, chars: usize, trained: bool,
}
fn entries() -> Result<Vec<Entry>> {
    let v:Value=serde_json::from_slice(&fs::read(MANIFEST)
        .map_err(|_| "Missing reports/lab/manifest.json: run `cargo run --release --bin zero_lab -- generate --count 240 --seed 2026`")?)?;
    let list=v["entries"].as_array().ok_or("Invalid Lab manifest entries")?;
    let mut out=Vec::new();
    for e in list {
        let get=|k:&str| -> Result<&str> {Ok(e[k].as_str().ok_or_else(||format!("Missing {k} in manifest"))?)};
        out.push(Entry{
            id:get("id")?.to_string(), split:get("split")?.to_string(),
            path:get("path")?.to_string(), truth:get("ground_truth")?.to_string(),
            font:get("font")?.to_string(),
            px:e["point_size"].as_f64().unwrap_or(0.),
            noise:e["noise"].as_u64().unwrap_or(0),
            bg:e["background"].as_u64().unwrap_or(255),
            lines:e["lines"].as_u64().unwrap_or(0) as usize,
            chars:e["chars"].as_u64().unwrap_or(0) as usize,
            trained:e["font_trained_on_ocr"].as_bool().unwrap_or(false),
        });
    }
    Ok(out)
}
fn opt<'a>(args:&'a[String],name:&str)->Option<&'a str>{
    args.iter().position(|s|s==name).and_then(|i|args.get(i+1)).map(String::as_str)
}
fn split(args:&[String])->&str {opt(args,"--split").unwrap_or("test")}
fn select(args:&[String])->Result<Vec<Entry>> {
    let target=split(args);
    if !["train","validation","test","all"].contains(&target) {
        return Err("--split must be test, train, validation, or all".into());
    }
    let found:Vec<_>=entries()?.into_iter().filter(|e|target=="all"||e.split==target).collect();
    if found.is_empty(){return Err(format!("No images for split {target}").into());}
    Ok(found)
}
fn clean_truth(s:&str)->String{s.trim_end_matches(['\r','\n']).replace("\r\n","\n")}
fn load_image(e:&Entry)->Result<(GrayImage,String)> {
    let img=image::open(&e.path).map_err(|x|format!("{}: {x}",e.path))?.to_luma8();
    let truth=clean_truth(&fs::read_to_string(&e.truth)?);
    Ok((img,truth))
}
fn budgets(args:&[String])->Result<Vec<usize>> {
    let s=opt(args,"--budgets").unwrap_or("0,2,4,6,8,12");
    let mut out=Vec::new();
    for p in s.split(',') {let x:usize=p.trim().parse()?;if x>1000{return Err("Budget over 1000 not supported".into());}out.push(x);}
    out.sort_unstable();out.dedup();
    if out.is_empty(){return Err("No budgets".into());}
    Ok(out)
}
fn text_error(truth:&str,found:&str)->usize {levenshtein(truth,found)}
fn filename(s:&str)->String { s.replace('"',"\"\"") }

/// Tests all strategies against EXACTLY the same images and budget caps.
/// Reports both allowed and actually USED calls: Core03 may stop early.
pub(super) fn sweep(args:&[String])->Result<()> {
    let dataset=select(args)?;let caps=budgets(args)?;
    let brain=load_brain()?;let policy=load_core3(&brain)?;let old=load_scheduler(&brain)?;
    fs::create_dir_all(DIR)?;
    let repeats:usize=opt(args,"--repeats").unwrap_or("3").parse()?;
    if !(1..=20).contains(&repeats){return Err("--repeats must be 1..20".into());}
    let mut rows=Vec::<Value>::new();
    let mut summary:BTreeMap<(usize,String),(usize,usize,u128,u128,usize,usize)>=BTreeMap::new();
    println!("ZERO LAB 0.2 budget sweep: split {}, images {}, budgets {:?}, repeats {}",split(args),dataset.len(),caps,repeats);
    for (k,e) in dataset.iter().enumerate() {
        let (img,truth)=load_image(e)?;
        let chars=truth.chars().count().max(1);
        for &cap in &caps {
            for (name,mode) in [
                ("neural",Core3Mode::Random),
                ("random",Core3Mode::Random),
                ("confidence",Core3Mode::Confidence),
                ("core02",Core3Mode::Core02),
                ("core03",Core3Mode::Core03),
            ] {
                let max_calls=if name=="neural" {0} else {cap};
                let execute=||core3_scan(&brain,&policy,&old,&img,mode,max_calls,None,715190);
                let _=execute(); // warmup
                let mut times=Vec::new();let mut last=None;
                for _ in 0..repeats {let run=execute();times.push(run.duration_us);last=Some(run);}
                times.sort_unstable();let t=times[times.len()/2];let result=last.unwrap();
                let err=text_error(&truth,&result.text);
                let row=json!({"split":e.split,"id":e.id,"image":e.path,"font":e.font,
                    "budget_cap":cap,"mode":name,"errors":err,"chars":chars,
                    "cer":err as f64/chars as f64,"median_us":t,
                    "used_calls":result.memory_calls,"glyphs":result.glyphs,
                    "memory_us_last_run":result.extra_memory_us,"recognized":result.text});
                rows.push(row);
                let acc=summary.entry((cap,name.into())).or_default();
                acc.0+=err;acc.1+=chars;acc.2+=t;acc.3+=result.extra_memory_us;
                acc.4+=result.memory_calls;acc.5+=1;
            }
        }
        if (k+1)%10==0 || k+1==dataset.len(){println!("  scored {}/{} images",k+1,dataset.len());}
    }
    let mut report=Vec::new();
    let mut csv=String::from("split,budget_cap,mode,images,chars,errors,micro_cer,mean_ms,mean_used_calls,mean_memory_ms\n");
    for ((cap,mode),(err,chars,total_us,mem_us,calls,n)) in summary {
        let cer=err as f64/chars.max(1) as f64;
        let ms=total_us as f64/n as f64/1000.;
        let mem=mem_us as f64/n as f64/1000.;
        let used=calls as f64/n as f64;
        println!("cap {:>2} {:<11} CER {:>6.2}%  avg {:>7.2}ms  calls {:>5.2}",cap,mode,cer*100.,ms,used);
        csv.push_str(&format!("{},{},{},{},{},{},{:.8},{:.4},{:.4},{:.4}\n",split(args),cap,mode,n,chars,err,cer,ms,used,mem));
        report.push(json!({"split":split(args),"budget_cap":cap,"mode":mode,"images":n,
            "chars":chars,"errors":err,"micro_cer":cer,"mean_ms":ms,
            "mean_used_calls":used,"mean_memory_ms":mem}));
    }
    fs::write(format!("{DIR}/sweep.csv"),csv)?;
    fs::write(format!("{DIR}/sweep.json"),serde_json::to_vec_pretty(&report)?)?;
    fs::write(format!("{DIR}/per_image.json"),serde_json::to_vec_pretty(&rows)?)?;
    println!("Saved {DIR}/sweep.csv, sweep.json, per_image.json");
    println!("NOTE: cap = maximum allowed calls; actual calls may differ. Compare accuracy vs actual cost, not cap alone.");
    Ok(())
}

pub(super) fn check(_args:&[String])->Result<()> {
    let rows=entries()?;
    fs::create_dir_all(DIR)?;
    let mut grouped:BTreeMap<String,Vec<&Entry>>=BTreeMap::new();
    for e in &rows {grouped.entry(e.split.clone()).or_default().push(e);}
    let mut report=Vec::new();
    for (name,items) in grouped {
        let mut fonts=BTreeMap::<String,usize>::new();
        let mut sizes=BTreeMap::<String,usize>::new();
        let mut lines=BTreeMap::<String,usize>::new();
        let mut noise=BTreeMap::<String,usize>::new();
        let mut bg=BTreeMap::<String,usize>::new();
        let mut bad_truth=0usize;let mut unreadable=0usize;
        let mut char_total=0usize;let mut pixels=0u128;
        let mut untrained=0usize;
        for e in &items {
            *fonts.entry(e.font.clone()).or_default()+=1;
            *sizes.entry(e.px.to_string()).or_default()+=1;
            *lines.entry(e.lines.to_string()).or_default()+=1;
            *noise.entry(e.noise.to_string()).or_default()+=1;
            *bg.entry(e.bg.to_string()).or_default()+=1;
            if !e.trained {untrained+=1;}
            match fs::read_to_string(&e.truth) {
                Ok(t)=>{let len=clean_truth(&t).chars().count();if len!=e.chars {bad_truth+=1;}char_total+=len;},
                Err(_)=>bad_truth+=1,
            }
            match image::image_dimensions(&e.path){Ok((w,h))=>pixels+=w as u128*h as u128,Err(_)=>unreadable+=1}
        }
        println!("{}: {} images, {} chars, {:.1} chars/image; {} distinct fonts, {} OCR-unseen; invalid truth {}, unreadable images {}",
            name,items.len(),char_total,char_total as f64/items.len() as f64,fonts.len(),untrained,bad_truth,unreadable);
        println!("  sizes {:?}; lines {:?}; noise {:?}; backgrounds {:?}",sizes,lines,noise,bg);
        report.push(json!({"split":name,"images":items.len(),"characters":char_total,
            "fonts":fonts,"sizes":sizes,"lines":lines,"noise":noise,"backgrounds":bg,
            "untrained_font_images":untrained,"bad_ground_truth":bad_truth,"unreadable_images":unreadable,
            "mean_pixels":pixels as f64/items.len() as f64}));
    }
    fs::write(format!("{DIR}/dataset_check.json"),serde_json::to_vec_pretty(&report)?)?;
    println!("Saved {DIR}/dataset_check.json. Compare distributions before interpreting CER across splits.");
    Ok(())
}

// Monotonic Levenshtein alignment: pairs each neural output character with a
// corresponding truth character, if any. Insertions/deletions are marked None.
// Critical: alignment is computed AFTER inference; ground truth never enters decisions.
fn align(pred:&[char],truth:&[char])->Vec<Option<usize>> {
    let n=pred.len();let m=truth.len();
    let mut d=vec![vec![0usize;m+1];n+1];
    for i in 0..=n{d[i][0]=i;}
    for j in 0..=m{d[0][j]=j;}
    for i in 1..=n {for j in 1..=m {
        d[i][j]=(d[i-1][j-1]+usize::from(pred[i-1]!=truth[j-1]))
            .min(d[i-1][j]+1).min(d[i][j-1]+1);
    }}
    let mut mapped=vec![None;n];let (mut i,mut j)=(n,m);
    while i>0||j>0 {
        if i>0&&j>0&&d[i][j]==d[i-1][j-1]+usize::from(pred[i-1]!=truth[j-1]) {
            mapped[i-1]=Some(j-1);i-=1;j-=1;
        } else if i>0&&d[i][j]==d[i-1][j]+1 {i-=1;}
        else if j>0 {j-=1;}else{break;}
    }
    mapped
}

pub(super) fn audit(args:&[String])->Result<()> {
    let mut dataset=select(args)?;
    let limit:usize=opt(args,"--limit").unwrap_or("48").parse()?;
    if limit==0{return Err("--limit must be > 0".into());}
    dataset.truncate(limit);
    let brain=load_brain()?;let policy=load_core3(&brain)?;let old=load_scheduler(&brain)?;
    fs::create_dir_all(DIR)?;
    let mut data=Vec::<Value>::new();
    let mut total=0usize;let mut fixes=0usize;let mut harms=0usize;let mut same=0usize;
    let mut unmatched=0usize;let mut predicted=0usize;
    let mut bins:BTreeMap<usize,(usize,usize,usize)>=BTreeMap::new();
    for e in &dataset {
        let (img,truth)=load_image(e)?;
        let base=core3_scan(&brain,&policy,&old,&img,Core3Mode::Random,0,None,715190);
        let base_chars:Vec<char>=base.text.chars().collect();
        let actual_chars:Vec<char>=truth.chars().collect();
        let mapping=align(&base_chars,&actual_chars);
        let mut char_index=0usize;
        let mut previous_line=None;
        for (li,line) in segment(&img).iter().enumerate(){
            let height=line.iter().map(|(r,_)|r.h).max().unwrap_or(24) as f32;
            let mut gaps:Vec<u32>=line.iter().skip(1).map(|(_,g)|*g).filter(|g|*g>0).collect();
            gaps.sort();let median=gaps.get(gaps.len()/2).copied().unwrap_or(1) as f32;
            for (i,(rect,gap)) in line.iter().enumerate(){
                let space=i>0 && *gap as f32 >= (height*0.22).max(median*2.1).max(5.);
                if let Some(prev)=previous_line {if prev!=li||space {char_index+=1;}}
                previous_line=Some(li);
                let crop=image::imageops::crop_imm(&img,rect.x,rect.y,rect.w,rect.h).to_image();
                let pixels=normalize(&crop);
                let (nc,p,m)=brain.neural(&pixels);
                let n=core3_neural_answer(nc,p,m);
                let t0=Instant::now();let (mc,s,mg)=brain.template(&pixels);
                let us=t0.elapsed().as_micros();
                let mem=core3_template_answer(mc,s,mg,n);
                let bin=core3_bin(p,m,nc);let rank=policy.score(bin);
                let truth_pos=mapping.get(char_index).copied().flatten();
                let gt=truth_pos.and_then(|pos|actual_chars.get(pos)).copied();
                let outcome=match gt {
                    Some(g) if n!=g && mem==g => {fixes+=1;"fix"},
                    Some(g) if n==g && mem!=g => {harms+=1;"harm"},
                    Some(_) => {same+=1;"neutral"},
                    None => {unmatched+=1;"unaligned"},
                };
                if gt.is_some(){let entry=bins.entry(bin).or_default();entry.0+=1;if outcome=="fix"{entry.1+=1;}if outcome=="harm"{entry.2+=1;}}
                let record=json!({"image":e.path,"split":e.split,"font":e.font,
                    "x":rect.x,"y":rect.y,"w":rect.w,"h":rect.h,"line":li,
                    "neural":n.to_string(),"raw_neural":nc.to_string(),"confidence":p,"margin":m,
                    "memory":mem.to_string(),"memory_candidate":mc.to_string(),
                    "memory_similarity":s,"memory_margin":mg,"memory_us":us,
                    "truth":gt.map(|c|c.to_string()),"aligned":truth_pos.is_some(),
                    "outcome":outcome,"bin":bin,"predicted_utility":rank});
                data.push(record);total+=1;char_index+=1;
            }
        }
        predicted+=base_chars.len();
    }
    let mut bin_rows=Vec::new();
    for (bin,(n,f,h)) in bins {
        bin_rows.push(json!({"bin":bin,"samples":n,"fixes":f,"harms":h,
            "actual_gain_per_call":(f as f64-h as f64)/n as f64,
            "predicted_utility":policy.score(bin)}));
    }
    let mut csv=String::from("image,x,y,neural,memory,truth,outcome,confidence,margin,predicted_utility,memory_us\n");
    for v in &data {
        let t=|key:&str|v[key].as_str().unwrap_or("").to_string();
        csv.push_str(&format!("\"{}\",{},{},{},{},{},{},{},{},{},{}\n",filename(&t("image")),v["x"],v["y"],t("neural"),t("memory"),t("truth"),t("outcome"),v["confidence"],v["margin"],v["predicted_utility"],v["memory_us"]));
    }
    fs::write(format!("{DIR}/glyph_audit.csv"),csv)?;
    fs::write(format!("{DIR}/glyph_audit.json"),serde_json::to_vec_pretty(&data)?)?;
    fs::write(format!("{DIR}/utility_bins.json"),serde_json::to_vec_pretty(&bin_rows)?)?;
    println!("ZERO Lab 0.2 audit: {} images; {} glyphs; fixes {} harms {} neutral {} unaligned {}",dataset.len(),total,fixes,harms,same,unmatched);
    println!("Output: {DIR}/glyph_audit.csv/json and utility_bins.json");
    println!("CAUTION: alignment-based per-glyph labels are approximate where segmentation inserts/deletes characters; use only aligned samples for utility estimates.");
    println!("Scored baseline text characters: {predicted}");
    Ok(())
}
