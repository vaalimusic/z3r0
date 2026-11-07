//! ZERO Lab 0.1: reproducible OCR image corpus and benchmark analysis.
//! Does NOT train OCR or scheduler, and never passes ground truth to OCR.
use fontdue::{Font, FontSettings};
use image::{GrayImage, Luma};
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::{BTreeMap, HashSet}, env, error::Error, fs, path::{Path, PathBuf}};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const MANIFEST: &str = "reports/lab/manifest.json";
const REPORT: &str = "reports/core3/benchmark.json";
const TEXT: &[&str] = &[
    "A quick brown fox jumps over the lazy dog.",
    "ZERO reads letters and digits: 2026, 0185, 9034.",
    "Every computation should justify its cost.",
    "Learning is useful when new evidence arrives.",
    "Small fonts reveal segmentation mistakes.",
    "Memory may rescue an uncertain neural decision.",
    "The next experiment checks unseen typography.",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789",
    "abcdefghijklmnopqrstuvwxyz 0123456789",
    "Software can be fast without being accurate.",
    "A rejected answer is sometimes the safest choice.",
    "Confidence alone cannot guarantee correctness.",
    "Rust code handles images, words, and symbols.",
    "Measure errors before changing the scheduler.",
    "The result is either correct or incorrect.",
    "Recognizing 0 and O requires careful context.",
    "A longer sentence contains more possible errors.",
    "The system must stop when work is not worthwhile.",
];

#[derive(Clone, Serialize, Deserialize)]
struct Entry {
    id: String, split: String, font: String, font_trained_on_ocr: bool,
    point_size: f32, background: u8, noise: u8, lines: usize,
    path: String, ground_truth: String, chars: usize,
}
#[derive(Serialize, Deserialize)]
struct Manifest {
    version: u32, seed: u64, count: usize, entries: Vec<Entry>,
    note: String,
}
#[derive(Serialize)]
struct Summary {
    split: String, mode: String, images: usize, characters: usize,
    edit_errors: usize, micro_cer: f64, mean_cer: f64,
    mean_ms: f64, mean_memory_calls: f64, median_ms: f64,
}
struct FontCandidate { path: PathBuf, trained: bool }

fn main() -> Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("generate") => generate(&args),
        Some("analyze") => analyze(),
        _ => {
            println!("ZERO Lab 0.1\n  cargo run --release --bin zero_lab -- generate --count 240 --seed 2026\n  cargo run --release --bin zero_ocr -- core3-bench --budget-calls 12\n  cargo run --release --bin zero_lab -- analyze");
            Ok(())
        }
    }
}
fn option<T: std::str::FromStr>(args: &[String], flag: &str, default: T) -> Result<T>
where T::Err: Error + 'static {
    if let Some(i) = args.iter().position(|s| s == flag) {
        let val = args.get(i + 1).ok_or_else(|| format!("Missing value for {flag}"))?;
        return Ok(val.parse::<T>()?);
    }
    Ok(default)
}
fn norm(path: &str) -> String { path.replace('\\', "/").to_ascii_lowercase() }
fn known_fonts() -> Vec<PathBuf> {
    const FONTS: &[&str] = &[
        "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/calibri.ttf", "C:/Windows/Fonts/calibrib.ttf",
        "C:/Windows/Fonts/consola.ttf", "C:/Windows/Fonts/consolab.ttf",
        "C:/Windows/Fonts/verdana.ttf", "C:/Windows/Fonts/verdanab.ttf",
        "C:/Windows/Fonts/tahoma.ttf", "C:/Windows/Fonts/tahomabd.ttf",
        "C:/Windows/Fonts/times.ttf", "C:/Windows/Fonts/timesbd.ttf",
        "C:/Windows/Fonts/georgia.ttf", "C:/Windows/Fonts/georgiab.ttf",
        "C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/seguisb.ttf",
        "C:/Windows/Fonts/trebuc.ttf", "C:/Windows/Fonts/trebucbd.ttf",
        "C:/Windows/Fonts/cambria.ttc", "C:/Windows/Fonts/cour.ttf",
        "C:/Windows/Fonts/lucon.ttf", "C:/Windows/Fonts/constan.ttf",
        "C:/Windows/Fonts/gadugi.ttf", "C:/Windows/Fonts/pala.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ];
    let mut out: Vec<PathBuf> = FONTS.iter().map(PathBuf::from).filter(|p|p.is_file()).collect();
    if let Ok(it) = fs::read_dir("fonts") {
        for entry in it.flatten() {
            let p=entry.path();
            if p.extension().and_then(|e| e.to_str()).map(|e| e.eq_ignore_ascii_case("ttf")||e.eq_ignore_ascii_case("otf")).unwrap_or(false) {out.push(p);}
        }
    }
    out.sort(); out.dedup(); out
}
fn select_fonts() -> Result<(Vec<FontCandidate>,Vec<FontCandidate>,Vec<FontCandidate>)> {
    let brain:Value = serde_json::from_slice(&fs::read("zero_brain_v03.json")
        .map_err(|_| "Missing zero_brain_v03.json: keep the trained brain beside Cargo.toml")?)?;
    let trained:HashSet<String> = brain["trained_fonts"].as_array().ok_or("Brain missing trained_fonts")?
        .iter().filter_map(Value::as_str).map(norm).collect();
    let heldout = brain["heldout_font"].as_str().map(norm);
    let mut train=Vec::new();let mut val=Vec::new();let mut test=Vec::new();
    let mut candidates=known_fonts();
    if let Some(h)=brain["heldout_font"].as_str() {let p=PathBuf::from(h);if p.is_file(){candidates.push(p);}}
    let mut trained_index=0usize;
    for path in candidates {
        let id=norm(&path.to_string_lossy());
        // Ignore unparseable fonts without interrupting the whole generation.
        if Font::from_bytes(fs::read(&path)?,FontSettings::default()).is_err(){continue;}
        if trained.contains(&id) {
            if trained_index%4==3 { val.push(FontCandidate{path,trained:true}); }
            else { train.push(FontCandidate{path,trained:true}); }
            trained_index+=1;
        } else {
            // Never train scheduler against heldout samples.
            test.push(FontCandidate{path,trained:false});
        }
    }
    if train.is_empty() || val.is_empty() || test.is_empty() {
        return Err(format!("Insufficient fonts: train={} validation={} unseen_test={}. Add TTF/OTF to fonts/ (unseen fonts for test), or check brain.trained_fonts.",train.len(),val.len(),test.len()).into());
    }
    // Reserve the originally held-out font for test if installed; all test fonts are unseen by OCR.
    if let Some(h)=heldout {test.sort_by_key(|f| if norm(&f.path.to_string_lossy())==h {0}else{1});}
    Ok((train,val,test))
}
fn render(font:&Font,text:&str,px:f32,bg:u8,noise:u8,rng:&mut StdRng)->GrayImage {
    let lines:Vec<&str>=text.lines().collect();
    let pad=20i32; let line_height=(px*1.65).ceil() as i32;
    let mut width=240i32;
    for line in &lines {
        let advance: f32=line.chars().map(|c|font.metrics(c,px).advance_width.max(0.)).sum();
        width=width.max(advance.ceil() as i32 + 2*pad + 12);
    }
    let height=(lines.len() as i32*line_height + 2*pad + 12).max(70);
    let mut img=GrayImage::from_pixel(width as u32,height as u32,Luma([bg]));
    for (li,line) in lines.iter().enumerate() {
        let mut x=pad as f32; let baseline=pad+li as i32*line_height+px as i32;
        for c in line.chars() {
            let (m,bits)=font.rasterize(c,px);
            let left=x.round() as i32+m.xmin;
            let top=baseline-m.height as i32-m.ymin;
            for dy in 0..m.height {for dx in 0..m.width {
                let xx=left+dx as i32;let yy=top+dy as i32;
                if xx>=0&&yy>=0&&xx<width&&yy<height {
                    let ink=bits[dy*m.width+dx] as f32/255.;
                    let v=(bg as f32*(1.-ink)).round() as u8;
                    let old=img.get_pixel(xx as u32,yy as u32)[0];
                    img.put_pixel(xx as u32,yy as u32,Luma([old.min(v)]));
                }
            }}
            x+=m.advance_width.max(0.);
        }
    }
    if noise>0 {
        for pixel in img.pixels_mut() {
            let d=rng.gen_range(-(noise as i16)..=(noise as i16));
            pixel[0]=(pixel[0] as i16+d).clamp(0,255) as u8;
        }
    }
    img
}
fn generate(args:&[String])->Result<()> {
    let count=option(args,"--count",240usize)?;
    let seed=option(args,"--seed",2026u64)?;
    if count==0||count>5000{return Err("--count must be 1..5000".into());}
    let (train,val,test)=select_fonts()?;
    fs::create_dir_all("input")?;fs::create_dir_all("reports/lab")?;
    let mut rng=StdRng::seed_from_u64(seed);
    let mut entries=Vec::with_capacity(count);
    let mut font_cache: BTreeMap<String,Font>=BTreeMap::new();
    let mut groups=BTreeMap::<String,usize>::new();
    for i in 0..count {
        let split=if i%10<6 {"train"} else if i%10<8 {"validation"} else {"test"};
        let pool=match split {"train"=>&train,"validation"=>&val,_=>&test};
        let candidate=&pool[(i/10+ rng.gen_range(0..pool.len()))%pool.len()];
        let path=candidate.path.to_string_lossy().to_string();
        if !font_cache.contains_key(&path) {
            let font=Font::from_bytes(fs::read(&candidate.path)?,FontSettings::default())?;
            font_cache.insert(path.clone(),font);
        }
        let font=&font_cache[&path];
        let nlines=if i%5==0 {3}else if i%3==0 {2}else{1};
        let lines:Vec<&str>=(0..nlines).map(|_|TEXT[rng.gen_range(0..TEXT.len())]).collect();
        let text=lines.join("\n");
        let px=match i%5 {0=>14.,1=>19.,2=>26.,3=>34.,_=>46.};
        let bg=if i%7==0 {215}else{255};
        // Ink remains safely < 180 and background > 180 in clean conditions.
        let noise=if i%11==0 {10}else if i%4==0 {4}else{0};
        let id=format!("lab_{split}_{i:04}");
        let image_path=format!("input/{id}.png");
        let truth_path=format!("input/{id}.gt.txt");
        render(font,&text,px,bg,noise,&mut rng).save(&image_path)?;
        fs::write(&truth_path,&text)?;
        entries.push(Entry{id,split:split.to_string(),font:path,
            font_trained_on_ocr:candidate.trained,point_size:px,background:bg,noise,
            lines:nlines,path:image_path,ground_truth:truth_path,chars:text.chars().count()});
        *groups.entry(split.to_string()).or_default()+=1;
    }
    let manifest=Manifest{version:1,seed,count:entries.len(),entries,
        note:"TRAIN/VAL fonts may overlap OCR brain training; TEST fonts are absent from brain.trained_fonts. Ground truth is not used for OCR decisions. Training scheduler with this data requires a separate implementation; generator itself never trains.".into()};
    fs::write(MANIFEST,serde_json::to_vec_pretty(&manifest)?)?;
    println!("ZERO LAB generated {} images, seed {seed}",manifest.count);
    println!("Fonts: train {}, validation {}, unseen test {}",train.len(),val.len(),test.len());
    println!("Splits: {:?}. Images and .gt.txt in input/; manifest at {MANIFEST}",groups);
    println!("NEXT: cargo run --release --bin zero_ocr -- core3-bench --budget-calls 12");
    println!("THEN: cargo run --release --bin zero_lab -- analyze");
    Ok(())
}
fn analyze()->Result<()> {
    let manifest:Manifest=serde_json::from_slice(&fs::read(MANIFEST)
        .map_err(|_|"Missing manifest: run zero_lab generate")?)?;
    let raw:Vec<Value>=serde_json::from_slice(&fs::read(REPORT)
        .map_err(|_|"Missing benchmark: run zero_ocr core3-bench first")?)?;
    let mut by_path=BTreeMap::new();
    for entry in &manifest.entries {by_path.insert(norm(&entry.path),entry);}
    let mut data:BTreeMap<(String,String),Vec<(usize,usize,f64,f64,f64)>>=BTreeMap::new();
    let mut processed=HashSet::new();
    for row in &raw {
        let Some(image)=row["image"].as_str() else {continue};
        let Some(entry)=by_path.get(&norm(image)) else {continue};
        let Some(mode)=row["mode"].as_str() else {continue};
        let Some(errors)=row["errors"].as_u64() else {continue};
        let Some(time)=row["median_us"].as_u64() else {continue};
        let Some(calls)=row["calls"].as_u64() else {continue};
        let Some(cer)=row["cer"].as_f64() else {continue};
        processed.insert(entry.id.clone());
        data.entry((entry.split.clone(),mode.to_string())).or_default()
            .push((entry.chars,errors as usize,cer,time as f64/1000.,calls as f64));
    }
    if data.is_empty(){return Err("No matching lab images found in benchmark.json; generate and run benchmark after generation".into());}
    let mut summaries=Vec::new();
    let mut csv=String::from("split,mode,images,chars,edit_errors,micro_cer,mean_cer,mean_ms,median_ms,mean_memory_calls\n");
    println!("ZERO LAB aggregate | {} / {} images scored",processed.len(),manifest.count);
    for ((split,mode),v) in data {
        let n=v.len();
        let chars:usize=v.iter().map(|x|x.0).sum();
        let errors:usize=v.iter().map(|x|x.1).sum();
        let micro=errors as f64/chars.max(1) as f64;
        let mean_cer=v.iter().map(|x|x.2).sum::<f64>()/n as f64;
        let mean_ms=v.iter().map(|x|x.3).sum::<f64>()/n as f64;
        let mean_calls=v.iter().map(|x|x.4).sum::<f64>()/n as f64;
        let mut times:Vec<f64>=v.iter().map(|x|x.3).collect();
        times.sort_by(|a,b|a.total_cmp(b));
        let median_ms=times[n/2];
        let r=Summary {split,mode,images:n,characters:chars,edit_errors:errors,
            micro_cer:micro,mean_cer,mean_ms,mean_memory_calls:mean_calls,median_ms};
        println!("{:>10} {:<16} {:>3} images CER {:>6.2}% median {:>7.2}ms calls {:>5.1}",r.split,r.mode,n,micro*100.,median_ms,mean_calls);
        csv.push_str(&format!("{},{},{},{},{},{:.6},{:.6},{:.3},{:.3},{:.3}\n",r.split,r.mode,n,chars,errors,micro,mean_cer,mean_ms,median_ms,mean_calls));
        summaries.push(r);
    }
    fs::write("reports/lab/summary.csv",csv)?;
    fs::write("reports/lab/summary.json",serde_json::to_vec_pretty(&summaries)?)?;
    if processed.len()<manifest.count {eprintln!("WARNING: only {} of {} images present; rerun core3-bench.",processed.len(),manifest.count);}
    println!("Saved reports/lab/summary.csv + summary.json");
    println!("TEST split contains fonts not in OCR training list; only test split should be used for final claims.");
    Ok(())
}
