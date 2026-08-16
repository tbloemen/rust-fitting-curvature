//! The crate's error type.
//!
//! Everything fallible here is a file the sweeps wrote — a results JSONL, a
//! stage-1 indicator table, a κ_data table — or a CLI argument that names
//! something the data does not contain. Loading is **strict**: an unreadable
//! file and an unparseable line are both errors, carrying the path (and line)
//! that caused them, so a sweep killed mid-write fails the analysis instead of
//! silently contributing a short cell.

use std::fmt;
use std::path::PathBuf;

/// `Result` with this crate's error type.
pub type Result<T> = std::result::Result<T, Error>;

pub enum Error {
    /// A file could not be opened, read or written.
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    /// A JSONL line did not deserialise. `line` is 1-based.
    Parse {
        path: PathBuf,
        line: usize,
        source: serde_json::Error,
    },
    /// A record could not be serialised on the way out.
    Serialize(serde_json::Error),
    /// Nothing in the results directory (or the stage-1 table) parsed as an
    /// experiment cell.
    NoCells(PathBuf),
    /// `--region` named a preference region the indicator table does not carry.
    UnknownRegion {
        region: String,
        available: Vec<String>,
    },
    /// Friedman needs at least three treatments to be worth running.
    TooFewSettings(Vec<String>),
    /// The post-hoc comparisons are against a control, so it has to be listed.
    MissingBaseline {
        baseline: &'static str,
        settings: Vec<String>,
    },
    /// A (N, geometry, dataset) block carries settings to compare but no
    /// baseline cell to compare them against. The ε-indicator is only ever
    /// formed against the control, so there is nothing sensible to emit.
    NoBaselineCell {
        baseline: &'static str,
        n: usize,
        geometry: String,
        dataset: String,
    },
    /// A figure failed to render. plotters' error type is generic over the
    /// backend, so it is flattened to its message here, which keeps plotters
    /// out of the non-`plots` build.
    Plot(String),
}

impl Error {
    /// An I/O failure against *path*.
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Error::Io {
            path: path.into(),
            source,
        }
    }

    /// A parse failure at 1-based *line* of *path*.
    pub fn parse(path: impl Into<PathBuf>, line: usize, source: serde_json::Error) -> Self {
        Error::Parse {
            path: path.into(),
            line,
            source,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io { path, source } => write!(f, "{}: {source}", path.display()),
            Error::Parse { path, line, source } => {
                write!(f, "{}:{line}: {source}", path.display())
            }
            Error::Serialize(source) => write!(f, "serialising JSON: {source}"),
            Error::NoCells(path) => {
                write!(f, "no result cells found in {}", path.display())
            }
            Error::UnknownRegion { region, available } => write!(
                f,
                "no preference region {region:?} in the table; have {}",
                available.join(", ")
            ),
            Error::TooFewSettings(settings) => write!(
                f,
                "--settings needs at least three settings to rank; got {}",
                settings.join(", ")
            ),
            Error::MissingBaseline { baseline, settings } => write!(
                f,
                "--settings must include the {baseline} control; got {}",
                settings.join(", ")
            ),
            Error::NoBaselineCell {
                baseline,
                n,
                geometry,
                dataset,
            } => write!(
                f,
                "no {baseline} cell for ({dataset}, {geometry}, N={n}) to compare against"
            ),
            Error::Plot(msg) => write!(f, "rendering figure: {msg}"),
        }
    }
}

/// Same text as [`Display`](fmt::Display), because this is what the user sees:
/// `main` returns `Result`, and `Termination` renders the error with `Debug`.
/// The derived form would print `Io { path: "results", source: Os { code: 2, ..`
/// where `results: No such file or directory (os error 2)` is the whole story.
impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io { source, .. } => Some(source),
            Error::Parse { source, .. } => Some(source),
            Error::Serialize(source) => Some(source),
            _ => None,
        }
    }
}

/// Attach the path to an [`std::io::Result`]: `File::open(p).at(p)?`. A bare
/// `No such file or directory` is useless when a run touches a few hundred files.
pub trait IoContext<T> {
    fn at(self, path: impl Into<PathBuf>) -> Result<T>;
}

impl<T> IoContext<T> for std::io::Result<T> {
    fn at(self, path: impl Into<PathBuf>) -> Result<T> {
        self.map_err(|source| Error::io(path, source))
    }
}
