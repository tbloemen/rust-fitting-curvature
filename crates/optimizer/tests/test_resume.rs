//! End-to-end test for `--mode pareto --resume`.
//!
//! The 5000-sample sweeps run far longer than the cluster's 24h wall-clock cap,
//! so each (dataset, experiment, geometry) job is split into chained chunks that
//! resume from the JSONL the previous chunk wrote. Because qParEGO is sequential
//! (every batch is proposed from a GP refit on all prior trials), resume works by
//! replaying the exact same `suggest_batch` calls and substituting the recorded
//! evaluation for trials already on disk — so a chunked run must produce results
//! bit-identical to a single uninterrupted run.
//!
//! This test asserts exactly that: a one-shot run and a (stop, resume, finish)
//! run yield the same JSONL up to the non-deterministic per-trial `time_ms`.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Common, deterministic optimizer settings. A small synthetic dataset and a
/// fixed `--threads` (which is the qParEGO batch size) keep every invocation
/// reproducible and fast. Hyperbolic `all_off` optimizes 4 params, so the LHS
/// init phase is 11·4−1 = 43 trials; the GP phase then adds `--n-trials` more.
const DATASET: &str = "tree";
const GEOMETRY: &str = "hyperbolic";
const EXPERIMENT: &str = "all_off";
const N_SAMPLES: &str = "60";
const N_SEEDS: &str = "1";
const THREADS: &str = "2";

fn unique_dir() -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("resume_test_{}_{}", std::process::id(), nanos));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Run the optimizer binary once, returning its captured stdout.
fn run_pareto(output: &Path, n_trials: usize, resume: bool) -> String {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_optimizer"));
    cmd.args([
        "--mode",
        "pareto",
        "--dataset",
        DATASET,
        "--experiment",
        EXPERIMENT,
        "--geometry",
        GEOMETRY,
        "--n-samples",
        N_SAMPLES,
        "--n-seeds",
        N_SEEDS,
        "--threads",
        THREADS,
        "--data-path",
        "/tmp", // ignored for synthetic datasets
    ]);
    cmd.arg("--n-trials").arg(n_trials.to_string());
    cmd.arg("--output").arg(output);
    if resume {
        cmd.arg("--resume");
    }
    let out = cmd.output().expect("failed to launch optimizer binary");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "optimizer exited with {:?}\nstdout:\n{stdout}\nstderr:\n{stderr}",
        out.status.code(),
    );
    // Progress/status lines go to stderr (indicatif); top-level prints to stdout.
    // Callers inspect either, so return both.
    format!("{stdout}{stderr}")
}

/// Parse a JSONL results file into one `serde_json::Value` per trial, with the
/// non-deterministic `time_ms` field stripped so two runs can be compared.
fn canonical_trials(path: &Path) -> Vec<serde_json::Value> {
    let contents = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("could not read {}: {e}", path.display()));
    contents
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            let mut v: serde_json::Value =
                serde_json::from_str(line).expect("results line is not valid JSON");
            v.as_object_mut().unwrap().remove("time_ms");
            v
        })
        .collect()
}

#[test]
fn resume_reproduces_uninterrupted_run() {
    let dir = unique_dir();
    let full = dir.join("full.jsonl");
    let chunked = dir.join("chunked.jsonl");

    // Full GP budget, and the point at which the first chunk is "interrupted".
    // 3 of 4 is deliberately an odd boundary: with batch size 2 the full run's
    // final GP batch is [trial 3, trial 4], so resume must reuse trial 3 and
    // freshly evaluate trial 4 *within the same batch* — exercising the
    // partial-batch reuse path, not just whole-batch reuse.
    const N_FULL: usize = 4;
    const N_CHUNK1: usize = 3;

    // 1. Uninterrupted reference run.
    run_pareto(&full, N_FULL, false);
    let full_trials = canonical_trials(&full);

    // 2. First chunk: stops early. `--resume` on an absent file == fresh start,
    //    matching how every job in a chained sweep is launched.
    run_pareto(&chunked, N_CHUNK1, true);
    let chunk1_trials = canonical_trials(&chunked);
    assert!(
        chunk1_trials.len() < full_trials.len(),
        "first chunk ({}) should record fewer trials than the full run ({}) so resume has work to do",
        chunk1_trials.len(),
        full_trials.len()
    );

    // 3. Second chunk: resumes the same file and finishes the budget.
    run_pareto(&chunked, N_FULL, true);
    let resumed_trials = canonical_trials(&chunked);

    // The chunked run must reproduce the uninterrupted run exactly (sans time_ms).
    assert_eq!(
        resumed_trials.len(),
        full_trials.len(),
        "resumed run trial count differs from uninterrupted run"
    );
    for (i, (a, b)) in full_trials.iter().zip(&resumed_trials).enumerate() {
        assert_eq!(
            a, b,
            "trial {i} differs between uninterrupted and resumed runs:\n full:    {a}\n resumed: {b}"
        );
    }

    // The first chunk's trials must be a byte-stable prefix: resume reuses them
    // verbatim (it never rewrites already-recorded lines).
    for (i, (a, b)) in chunk1_trials.iter().zip(&resumed_trials).enumerate() {
        assert_eq!(a, b, "resume rewrote or altered already-recorded trial {i}");
    }

    // 4. Idempotency: resuming an already-complete run is a no-op early exit.
    let before = std::fs::read_to_string(&chunked).unwrap();
    let stdout = run_pareto(&chunked, N_FULL, true);
    let after = std::fs::read_to_string(&chunked).unwrap();
    assert_eq!(before, after, "resuming a complete run must not change the JSONL");
    assert!(
        stdout.contains("already complete"),
        "expected an 'already complete' early-exit message, got:\n{stdout}"
    );

    // The pareto-front sidecar should also have been written.
    let front = dir.join(format!("chunked_pareto_{DATASET}_{GEOMETRY}.json"));
    assert!(front.exists(), "pareto front file {} missing", front.display());

    std::fs::remove_dir_all(&dir).ok();
}
