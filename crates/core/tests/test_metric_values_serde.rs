//! The `MetricValues` wire format.
//!
//! Three properties carry real weight here. The column names and their order
//! must match what pre-registry builds wrote, or `results/` splits into two
//! incompatible schemas. Old files must keep loading, retired columns and all —
//! there are ~350 MB of them and they are the thesis data. And float parsing
//! must stay correctly rounded through `#[serde(flatten)]`.

#![cfg(feature = "serde")]

use fitting_core::metrics::{Metric, MetricValues, ALL};
use serde::{Deserialize, Serialize};

fn values(f: impl Fn(Metric) -> f64) -> MetricValues {
    let mut v = MetricValues::MISSING;
    for m in ALL {
        v.set(*m, f(*m));
    }
    v
}

#[test]
fn the_column_names_and_their_order_are_the_wire_schema() {
    let json = serde_json::to_string(&values(|_| 0.5)).unwrap();
    let keys: Vec<&str> = json
        .trim_matches(|c| c == '{' || c == '}')
        .split(',')
        .map(|kv| kv.split(':').next().unwrap().trim_matches('"'))
        .collect();
    let want: Vec<&str> = ALL.iter().map(|m| m.name()).collect();
    assert_eq!(keys, want, "serialisation order must be ALL order");
}

#[test]
fn absent_serialises_as_null_and_reads_back_as_absent() {
    let json = serde_json::to_string(&MetricValues::MISSING).unwrap();
    assert!(
        !json.contains("NaN") && json.contains("null"),
        "MISSING must be an all-null block, got {json}"
    );
    // `--mode scan` writes a result it never scored; this is that line.
    let back: MetricValues = serde_json::from_str(&json).unwrap();
    for m in ALL {
        assert_eq!(back.get(*m), None, "{} came back present", m.name());
    }
}

#[test]
fn a_diverged_trials_non_finite_value_is_absent_not_an_error() {
    let json = serde_json::to_string(&values(|_| f64::INFINITY)).unwrap();
    let back: MetricValues = serde_json::from_str(&json).unwrap();
    assert_eq!(back.get(fitting_core::metrics::TRUSTWORTHINESS), None);
}

#[test]
fn round_trip_preserves_every_value() {
    let v = values(|m| m.index() as f64 * 0.125);
    let back: MetricValues = serde_json::from_str(&serde_json::to_string(&v).unwrap()).unwrap();
    assert_eq!(back, v);
}

#[test]
fn retired_columns_are_ignored_and_missing_ones_stay_absent() {
    // Shaped like a real pre-registry line: three retired metrics, no
    // `r_gyration`, and a mix of non-numeric columns that a
    // `BTreeMap<String, f64>` flatten target would have choked on.
    let line = r#"{
        "dataset_name": "antipodal_clusters",
        "geometry": null,
        "n_samples": 1000,
        "knn_overlap": 0.4,
        "knn_overlap_manifold": 0.41,
        "class_density_measure": 12.5,
        "trustworthiness": 0.97,
        "normalized_stress": 0.31
    }"#;
    let v: MetricValues = serde_json::from_str(line).unwrap();
    assert_eq!(v.get(fitting_core::metrics::TRUSTWORTHINESS), Some(0.97));
    assert_eq!(v.get(fitting_core::metrics::NORMALIZED_STRESS), Some(0.31));
    assert_eq!(v.get(fitting_core::metrics::R_GYRATION), None);
    assert_eq!(v.get(fitting_core::metrics::CONTINUITY), None);
}

// ─── Embedded in a record, the way the trial records use it ──────────────────

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct Record {
    dataset_name: String,
    learning_rate: f64,
    #[serde(flatten)]
    metrics: MetricValues,
    time_ms: u64,
}

#[test]
fn flatten_puts_the_metrics_at_the_top_level_in_field_order() {
    let r = Record {
        dataset_name: "tree".into(),
        learning_rate: 0.5,
        metrics: values(|m| m.index() as f64),
        time_ms: 7,
    };
    let json = serde_json::to_string(&r).unwrap();
    // Named fields keep their positions; the metric block lands between them.
    assert!(json.starts_with(r#"{"dataset_name":"tree","learning_rate":0.5,"trustworthiness":0.0,"#));
    assert!(json.ends_with(r#""r_gyration":15.0,"time_ms":7}"#));
    assert_eq!(serde_json::from_str::<Record>(&json).unwrap(), r);
}

#[test]
fn flatten_does_not_claim_the_records_own_columns() {
    let json = r#"{"dataset_name":"tree","learning_rate":0.5,"trustworthiness":0.9,"time_ms":7}"#;
    let r: serde_json::Result<Record> = serde_json::from_str(json);
    let r = r.expect("a record with no metric block must still load");
    assert_eq!(r.dataset_name, "tree");
    assert_eq!(r.learning_rate, 0.5);
    assert_eq!(r.time_ms, 7);
    assert_eq!(
        r.metrics.get(fitting_core::metrics::TRUSTWORTHINESS),
        Some(0.9)
    );
}

#[test]
fn float_parsing_stays_correctly_rounded_through_flatten() {
    // `fitting-analysis` enables serde_json's `float_roundtrip` because the
    // Pareto sort is a chain of exact `<=` comparisons and one ulp was enough
    // to drop a real front point in 8 of 176 cells. `flatten` buffers the
    // object through serde's `Content` before this type ever sees it; this
    // pins that the buffering does not fall back to the fast, lossy parser.
    let text = "0.30863871419973954";
    let want: f64 = text.parse().unwrap();
    let json = format!(r#"{{"dataset_name":"t","learning_rate":0.0,"normalized_stress":{text},"time_ms":0}}"#);
    let r: Record = serde_json::from_str(&json).unwrap();
    let got = r.metrics.get(fitting_core::metrics::NORMALIZED_STRESS).unwrap();
    assert_eq!(
        got.to_bits(),
        want.to_bits(),
        "flatten lost precision: {got:?} != {want:?}"
    );
}
