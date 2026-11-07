"""Static and dataset-design checks for environments without Rust toolchain.
Does not replace `cargo test` or verify Rust compilation.
"""
from pathlib import Path
import re
s = (Path(__file__).parent / 'src/bin/zero_real_tools.rs').read_text(encoding='utf-8')
checks = {
    'no external dependencies': '[dependencies]' not in (Path(__file__).parent / 'Cargo.toml').read_text(),
    'required binaries and main': 'fn main()' in s and 'name = "zero_real_tools"' in (Path(__file__).parent / 'Cargo.toml').read_text(),
    'reference correctness oracle': 'fn reference(' in s and 'fn is_correct(' in s,
    'real task handlers': all(x in s for x in ('v.sort_unstable()', 's.insert(0, c)', 'x += t.a as i64', '"aeiou".contains')),
    'failure cost included': 'total_cost += fallback.cost;' in s,
    'recovery scenario': '(500..1000).contains(&idx)' in s,
    'independent test seeds': 'j as u64 * 7919' in s,
    'phase metrics': 'phases[idx / 500].add(' in s,
    'unit tests': s.count('#[test]') >= 4,
    'unknown family held out': 'for k in 0..TRAIN_KINDS' in s and 'if i % 6 == 0 { 5 }' in s,
    'all three outputs': all(x in s for x in ('summary.csv', 'phases.csv', 'decisions.csv')),
}
for name, good in checks.items():
    print(f"{'PASS' if good else 'FAIL'} {name}")
assert all(checks.values()), 'Static checks failed'
# Actual generation logic of test families: 250 unseen and 250 per trained kind.
kinds = [5 if i % 6 == 0 else i % 6 - 1 for i in range(1500)]
assert {k: kinds.count(k) for k in range(6)} == {k:250 for k in range(6)}
print('PASS test distribution: 250 per family, including 250 held out')
print('NOTE: Rust compiler unavailable here; compile and runtime tests need cargo.')
