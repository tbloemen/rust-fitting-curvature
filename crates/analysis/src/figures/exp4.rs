//! Experiment 4 (`ablation-results`) — the stage tables its figures read.
//!
//! The figures themselves are [`super::exp4_gain_dots::GainDots`],
//! [`super::exp4_epsilon_dots::EpsilonDots`] and
//! [`super::exp4_tradeoff::TradeoffScatter`]; this module loads the `r2`
//! tables they are drawn from.

use std::path::Path;

use crate::aggregate::DeltaRow;
use crate::error::{Error, Result};
use crate::records::load_jsonl;

/// Load the R2 rows written by `r2 aggregate --deltas`.
///
/// An **absent** table is not an error: it is a separate `r2` run, and the
/// figures that read it are simply not drawn without it. A table that is there
/// and will not parse still fails.
///
/// # Errors
///
/// Returns `Err` if the file is present but malformed. A missing file returns
/// an empty `Vec`.
pub fn load_deltas(path: &Path) -> Result<Vec<DeltaRow>> {
    load_table(path)
}

/// Any stage table the figures read back: absent is an empty `Vec`, since
/// every such table is a separate `r2` run and the figures that need it are
/// then simply not written; present but malformed is an error.
///
/// # Errors
///
/// Returns `Err` if the file is present but malformed.
pub fn load_table<T: serde::de::DeserializeOwned>(path: &Path) -> Result<Vec<T>> {
    match load_jsonl(path) {
        Ok(rows) => Ok(rows),
        Err(Error::Io { source, .. }) if source.kind() == std::io::ErrorKind::NotFound => {
            Ok(Vec::new())
        }
        Err(e) => Err(e),
    }
}
