# ZERO (`z3r0`) — Autonomous Adaptive Decision Engine & Tool Orchestrator

[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg?style=flat-square&logo=rust)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)
[![Zero Dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen.svg?style=flat-square)](#architecture)
[![Decision Latency](https://img.shields.io/badge/latency-%3C60_ns-blueviolet.svg?style=flat-square)](#empirical-benchmarks)
[![Open Source](https://img.shields.io/badge/Open_Source-desk.everty.ru-emerald.svg?style=flat-square)](https://desk.everty.ru/opensource)

**ZERO** is a zero-dependency, ultra-low-latency (sub-60 ns) adaptive decision engine and tool orchestrator written in pure Rust. It enables autonomous AI agents, multi-path network routers, and distributed service meshes to learn conditional competence boundaries, isolate partial failures, and make cost-optimal execution choices under non-stationary, noisy environments.

Developed by **[Артур Валиев (Arthur Valiev)](https://desk.everty.ru/opensource)**.

---

## ⚡ Core Concept: The Tool Selection Problem

Modern agentic systems and multi-path orchestrators face fundamental failure modes:
1. **Blind Heuristic Routing**: Static rules fail when specialized tools or routes fail silently or have hidden competence boundaries.
2. **Contextual Outages vs Global Failures**: Standard circuit breakers disable a tool globally even when it only fails on a specific input partition (e.g. large payloads or diagonal 2D boundaries).
3. **Expensive Exploration**: Random exploration or continuous polling wastes bandwidth and budget.
4. **Latency Overhead**: Calling heavy LLMs or external classifiers for routing introduces hundreds of milliseconds of overhead.

### The ZERO Solution
ZERO formulates tool and route selection as an online **Contextual Decision Process** with strict cost minimization:
* **Zero External Dependencies**: Implemented in 100% standard library Rust (`std::time`, `std::fs`, bitwise operations).
* **Nanosecond-Scale Routing**: Routing decisions execute in **45–60 nanoseconds** per request.
* **Autonomous Feature & Boundary Discovery**: Discovers non-linear 2D competence thresholds without prior knowledge of tool internals.
* **Dual-Level Health Tracking (Global + Contextual)**: Separates global process crashes from context-specific degradation.
* **Economic Lifecycle Accounting**: Balances exploration cost vs runtime cost with Out-of-Distribution (OOD) guards and exponential backoff circuit breakers.

---

## 🔬 Research Roadmap & Benchmark Suite

The repository contains 10 standalone, reproducible research binaries exploring each phase of adaptive decision making:

```
src/bin/
├── zero_context_recovery.rs      # Phase 5: Dual Health (Global vs Contextual Circuit Breakers)
├── zero_multidim_composition.rs  # Phase 4: Multidim Trees + Circuit Breakers + OOD Guards
├── zero_multidim_robustness.rs   # Phase 4: 30+ seeds stress test under non-linear shifts
├── zero_multidimensional.rs      # Phase 3: 2D non-linear bounds & flaky probabilistic tools
├── zero_feature_discovery.rs     # Phase 1: Autonomous feature discovery (a, b, min, max, sum)
├── zero_capability_bounds.rs     # Phase 1: Conditional competence learning from execution feedback
├── zero_adaptive_curiosity.rs    # Phase 2: Curiosity-driven probing under non-stationary tools
├── zero_staleness.rs             # Phase 2: Stale knowledge decay vs time-based exploration
├── zero_adaptation.rs            # Phase 2: Online adaptation after unannounced tool failure
└── zero_real_tools.rs            # Production: Real Rust operations behind anonymous interfaces
```

### Binary Overview

| Binary | Scientific Objective | Key Mechanisms | Status |
|---|---|---|---|
| **`zero_context_recovery`** | Isolate sub-region outages without breaking unaffected traffic | Dual-Level Health, Global CB, Context CB | Benchmark Complete |
| **`zero_multidim_composition`** | Prevent catastrophic failure during out-of-distribution shifts | Multidim Decision Trees, OOD Guard, Circuit Breaker | Benchmark Complete |
| **`zero_multidim_robustness`** | 30+ independent seeds evaluation across lifecycle horizons | Empirical Bayesian Shrinkage, Breakeven Analysis | Benchmark Complete |
| **`zero_multidimensional`** | 2D conjunctions, diagonals, and flaky tool handling | Piecewise 2D splits, Probabilistic failure smoothing | Benchmark Complete |
| **`zero_feature_discovery`** | Discover which input metric predicts tool competence | Decision Stumps over derived features (`min`, `max`, `diff`) | Benchmark Complete |
| **`zero_capability_bounds`** | Learn numeric and categorical operational bounds | Feedback-driven binning, no ground-truth supervision | Benchmark Complete |
| **`zero_adaptive_curiosity`** | Maintain fresh knowledge of fluctuating tools | Information-gain probing under budget constraints | Benchmark Complete |
| **`zero_staleness`** | Measure cost of stale competence models | Age-weighted belief decay, targeted re-probing | Benchmark Complete |
| **`zero_adaptation`** | Detect unannounced tool dropouts | Rapid fallback triggering, penalty learning | Benchmark Complete |
| **`zero_real_tools`** | Evaluate real arithmetic and string pipelines | Anonymous polymorphic execution, zero-cost abstractions | Benchmark Complete |

---

## 📊 Empirical Benchmarks

### 1. Dual Health vs Standard Circuit Breakers (`zero_context_recovery`)
Evaluated across 30 independent test seeds under simultaneous global outages, regional contextual failures, and unannounced recoveries:

| Architecture | Mean Cost | p95 Cost | First-Call Success | Decision Latency |
|---|---|---|---|---|
| **Oracle (Theoretical Optimum)** | **3.71** | **10.70** | **91.8%** | 36 ns |
| **ZERO Dual-Health (Global + Context)** | **4.19** | **11.10** | **83.2%** | **55 ns** |
| ZERO Global Circuit Breaker | 4.21 | 11.10 | 82.7% | 53 ns |
| ZERO Context Circuit Breaker | 4.26 | 11.50 | 77.5% | 53 ns |
| ZERO Frozen Baseline | 4.31 | 11.50 | 73.5% | 49 ns |

> **Key Finding**: Dual-Health achieves the highest first-call success rate and lowest total cost by preventing the false quarantine of tools that remain healthy outside of degraded sub-regions.

### 2. Decision Speed
All routing decisions are computed synchronously in user-space:
* Single Stump Decision: **~35 ns**
* Multidimensional 2D Tree with OOD Guard: **~48 ns**
* Full Composition with Dual Circuit Breakers: **~55 ns**

---

## 🚀 Quick Start & Reproducibility

### Prerequisites
- [Rust toolchain](https://rustup.rs/) (edition 2021, Rust 1.75+ recommended).
- No external crate dependencies required for running benchmarks.

### Building & Checking
```bash
# Verify all binaries compile cleanly
cargo check --bins

# Run all unit and integration tests
cargo test --bins
```

### Running Key Benchmarks

#### 1. Context-Aware Recovery (Dual Health)
```bash
cargo run --release --bin zero_context_recovery
```
Outputs detailed per-event transitions and summary CSVs to `reports/zero_context_recovery/`.

#### 2. Multidimensional Robustness Stress Test (30 Seeds)
```bash
cargo run --release --bin zero_multidim_robustness
```
Generates p50/p95 distributions, timing diagnostics, and breakeven horizons in `reports/zero_multidim_robustness/`.

#### 3. Multidimensional Composition & OOD Guard
```bash
cargo run --release --bin zero_multidim_composition
```
Evaluates behavior when incoming traffic drifts outside of the training domain.

#### 4. Feature & Boundary Discovery
```bash
cargo run --release --bin zero_feature_discovery -- --seeds 8
```

---

## 📁 Repository Structure

```
z3r0/
├── Cargo.toml                  # Package configuration with standalone binaries
├── LICENSE                     # MIT License
├── README.md                   # Project documentation & benchmark overview
├── reports/                    # Experimental results & benchmark CSV data
│   ├── zero_context_recovery/  # Event logs, scenario breakdowns, and seed summaries
│   ├── zero_multidim_composition/
│   ├── zero_multidim_robustness/
│   ├── zero_feature_discovery/
│   └── ...
└── src/
    ├── bin/                    # 10 pure-Rust research binaries
    │   ├── zero_context_recovery.rs
    │   ├── zero_multidim_composition.rs
    │   ├── zero_multidim_robustness.rs
    │   ├── zero_multidimensional.rs
    │   ├── zero_feature_discovery.rs
    │   ├── zero_capability_bounds.rs
    │   ├── zero_adaptive_curiosity.rs
    │   ├── zero_staleness.rs
    │   ├── zero_adaptation.rs
    │   └── zero_real_tools.rs
    └── ...
```

---

## 👤 Author & Open Source Initiative

* **Author:** Артур Валиев (Arthur Valiev)
* **Open Source Portal:** [https://desk.everty.ru/opensource](https://desk.everty.ru/opensource)
* **GitHub:** [@vaalimusic](https://github.com/vaalimusic)

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
