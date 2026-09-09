use fitting_core::cast::count_to_f64;
use wasm_bindgen::prelude::*;
use web_sys::HtmlCanvasElement;

use fitting_core::config::{ScalingLossType, TrainingConfig};
use fitting_core::embedding::EmbeddingState;
use fitting_core::metrics;
use fitting_core::synthetic_data;
use fitting_core::visualisation::{self, SphericalProjection};

mod plot;

#[cfg(target_arch = "wasm32")]
use lol_alloc::{AssumeSingleThreaded, FreeListAllocator};

// SAFETY: WASM is single-threaded
#[cfg(target_arch = "wasm32")]
#[global_allocator]
static ALLOCATOR: AssumeSingleThreaded<FreeListAllocator> =
    unsafe { AssumeSingleThreaded::new(FreeListAllocator::new()) };

fn parse_scaling_loss(s: &str) -> ScalingLossType {
    match s {
        "rms" => ScalingLossType::Rms,
        "hard_barrier" => ScalingLossType::HardBarrier,
        "softplus_barrier" => ScalingLossType::SoftplusBarrier,
        "mean_distance" => ScalingLossType::MeanDistance,
        _ => ScalingLossType::None,
    }
}

fn parse_projection(s: &str) -> SphericalProjection {
    match s {
        "orthographic" => SphericalProjection::Orthographic,
        "stereographic" => SphericalProjection::Stereographic,
        // "azimuthal_equidistant", and anything unrecognised, lands here.
        _ => SphericalProjection::AzimuthalEquidistant,
    }
}

/// Step-based embedding runner for animated rendering.
#[wasm_bindgen]
pub struct EmbeddingRunner {
    state: EmbeddingState,
    canvas: HtmlCanvasElement,
    /// Human-readable names for each integer label, indexed by label value.
    /// When set, the legend shows these names instead of "Label N".
    label_names: Option<Vec<String>>,
    /// Current viewport: (`center_x`, `center_y`, `half_extent`). None = auto-fit.
    view: Option<(f64, f64, f64)>,
    /// Auto-fit half-extent from the last render, used to anchor zoom interactions.
    auto_half: f64,
}

#[wasm_bindgen]
impl EmbeddingRunner {
    /// Create a runner from a named synthetic dataset.
    #[allow(clippy::too_many_arguments)]
    pub fn from_synthetic(
        canvas_id: &str,
        dataset_name: &str,
        n_points: usize,
        curvature: f64,
        n_iterations: usize,
        perplexity: f64,
        learning_rate: f64,
        early_exaggeration_factor: f64,
        early_exaggeration_iterations: usize,
        centering_weight: f64,
        scaling_loss: &str,
        global_loss_weight: f64,
        norm_loss_weight: f64,
        projection: &str,
    ) -> Result<EmbeddingRunner, JsValue> {
        let synth = synthetic_data::load_synthetic(dataset_name, n_points, 42)
            .map_err(|e| JsValue::from_str(&e))?;

        let n_features = synth.ambient_dim;

        let config = TrainingConfig {
            n_points: synth.n_points,
            curvature,
            n_iterations,
            perplexity,
            learning_rate,
            early_exaggeration_factor,
            early_exaggeration_iterations,
            centering_weight,
            scaling_loss_type: parse_scaling_loss(scaling_loss),
            global_loss_weight,
            norm_loss_weight,
            ..Default::default()
        };

        let proj = parse_projection(projection);
        let state = EmbeddingState::new(&synth.x, n_features, &config)
            .with_labels(synth.labels.clone())
            .with_projection(proj);
        let canvas = get_canvas(canvas_id)?;

        Ok(EmbeddingRunner {
            state,
            canvas,
            label_names: None,
            view: None,
            auto_half: 1.0,
        })
    }

    /// Create a runner from external data with labels (e.g., MNIST).
    #[allow(clippy::too_many_arguments)]
    pub fn from_data_with_labels(
        canvas_id: &str,
        data: &[f64],
        labels: &[u32],
        n_points: usize,
        n_features: usize,
        curvature: f64,
        n_iterations: usize,
        perplexity: f64,
        learning_rate: f64,
        early_exaggeration_factor: f64,
        early_exaggeration_iterations: usize,
        centering_weight: f64,
        scaling_loss: &str,
        global_loss_weight: f64,
        norm_loss_weight: f64,
        projection: &str,
    ) -> Result<EmbeddingRunner, JsValue> {
        let config = TrainingConfig {
            n_points,
            curvature,
            n_iterations,
            perplexity,
            learning_rate,
            early_exaggeration_factor,
            early_exaggeration_iterations,
            centering_weight,
            scaling_loss_type: parse_scaling_loss(scaling_loss),
            global_loss_weight,
            norm_loss_weight,
            ..Default::default()
        };

        let proj = parse_projection(projection);
        let state = EmbeddingState::new(data, n_features, &config)
            .with_labels(labels.to_vec())
            .with_projection(proj);
        let canvas = get_canvas(canvas_id)?;

        Ok(EmbeddingRunner {
            state,
            canvas,
            label_names: None,
            view: None,
            auto_half: 1.0,
        })
    }

    /// Create a runner from a pre-computed pairwise distance matrix (e.g., `WordNet` tree distances).
    ///
    /// `distances` is a flat n × n row-major `Float64Array` of pairwise distances.
    /// `labels` is a `Uint32Array` of integer class labels of length n.
    #[allow(clippy::too_many_arguments)]
    pub fn from_distances(
        canvas_id: &str,
        distances: &[f64],
        labels: &[u32],
        n_points: usize,
        curvature: f64,
        n_iterations: usize,
        perplexity: f64,
        learning_rate: f64,
        early_exaggeration_factor: f64,
        early_exaggeration_iterations: usize,
        centering_weight: f64,
        scaling_loss: &str,
        global_loss_weight: f64,
        norm_loss_weight: f64,
        projection: &str,
    ) -> Result<EmbeddingRunner, JsValue> {
        let config = TrainingConfig {
            n_points,
            curvature,
            n_iterations,
            perplexity,
            learning_rate,
            early_exaggeration_factor,
            early_exaggeration_iterations,
            centering_weight,
            scaling_loss_type: parse_scaling_loss(scaling_loss),
            global_loss_weight,
            norm_loss_weight,
            ..Default::default()
        };

        let proj = parse_projection(projection);
        let state = EmbeddingState::from_distances(distances, n_points, &config)
            .with_labels(labels.to_vec())
            .with_projection(proj);
        let canvas = get_canvas(canvas_id)?;

        Ok(EmbeddingRunner {
            state,
            canvas,
            label_names: None,
            view: None,
            auto_half: 1.0,
        })
    }

    /// Set human-readable names for integer labels.
    ///
    /// `names_tsv` is a tab-separated list of names, one per label value in
    /// ascending order (e.g. `"B cells\tCD4 T\tCD14 Monocytes"`).
    /// When set, the legend uses these names instead of "Label N".
    pub fn set_label_names(&mut self, names_tsv: &str) {
        self.label_names = Some(
            names_tsv
                .split('\t')
                .map(std::string::ToString::to_string)
                .collect(),
        );
    }

    /// Run N iterations and render the current state.
    /// Returns true if there are more iterations to run.
    pub fn step(&mut self, n_steps: usize) -> bool {
        for _ in 0..n_steps {
            if self.state.is_done() {
                return false;
            }
            self.state.step();
        }
        // Diagnostic: log the norm-loss gradient magnitude vs the full gradient so
        // its effect (or lack thereof) is visible in the browser dev console.
        let norm_rms = self.state.last_norm_grad_rms;
        let total_rms = self.state.last_total_grad_rms;
        let ratio = if total_rms > 0.0 {
            norm_rms / total_rms
        } else {
            0.0
        };
        web_sys::console::log_1(&JsValue::from_str(&format!(
            "iter {} | norm-loss grad RMS {:.3e} | total grad RMS {:.3e} | ratio {:.2}%",
            self.state.iteration,
            norm_rms,
            total_rms,
            ratio * 100.0
        )));
        true
    }

    /// Render the current state to canvas.
    /// Stores the auto-fit half-extent so zoom/pan can use it as a reference.
    pub fn render(&mut self) -> Result<(), JsValue> {
        let auto_half = plot::draw_embedding(
            &self.canvas,
            &plot::PlotParams {
                points: &self.state.points,
                n_points: self.state.n_points,
                ambient_dim: self.state.ambient_dim,
                curvature: self.state.config().curvature,
                labels: self.state.labels.as_deref(),
                label_names: self.label_names.as_deref(),
                projection: self.state.projection,
                view: self.view,
            },
        )?;
        self.auto_half = auto_half;
        Ok(())
    }

    /// Render the current state as a square SVG string of `size`×`size` pixels.
    pub fn render_svg(&self, size: u32) -> Result<String, JsValue> {
        plot::draw_embedding_svg(
            size,
            &plot::PlotParams {
                points: &self.state.points,
                n_points: self.state.n_points,
                ambient_dim: self.state.ambient_dim,
                curvature: self.state.config().curvature,
                labels: self.state.labels.as_deref(),
                label_names: self.label_names.as_deref(),
                projection: self.state.projection,
                view: self.view,
            },
        )
    }

    /// Return the 2D projected coordinates of all points as a flat `Float64Array` [x0,y0,x1,y1,...].
    /// Coordinates are in the same plot space used by `render()`.
    #[must_use]
    pub fn get_projected_coords(&self) -> Vec<f64> {
        visualisation::project_to_2d(
            &self.state.points,
            self.state.n_points,
            self.state.ambient_dim,
            self.state.config().curvature,
            self.state.projection,
        )
        .coords
    }

    /// Current viewport state as [`center_x`, `center_y`, `half`, `auto_half`].
    /// With no explicit viewport, the centre is the origin and `half = auto_half`.
    #[must_use]
    pub fn get_viewport(&self) -> Vec<f64> {
        let (center_x, center_y, half) = self.view.unwrap_or((0.0, 0.0, self.auto_half));
        vec![center_x, center_y, half, self.auto_half]
    }

    /// Zoom the viewport around a normalized canvas position (0..1, 0..1).
    /// `factor > 1` zooms in, `factor < 1` zooms out.
    pub fn zoom_at(&mut self, norm_x: f64, norm_y: f64, factor: f64) {
        let (center_x, center_y, half) = self.view.unwrap_or((0.0, 0.0, self.auto_half));
        let aspect = f64::from(self.canvas.width()) / f64::from(self.canvas.height().max(1));
        let half_x = half * aspect;
        // Canvas coordinate → plot coordinate
        let plot_x = center_x + (norm_x - 0.5) * 2.0 * half_x;
        let plot_y = center_y - (norm_y - 0.5) * 2.0 * half; // y axis is flipped
        let new_half = (half / factor).clamp(1e-6, self.auto_half * 20.0);
        let new_half_x = new_half * aspect;
        // Keep plot_x/plot_y under the cursor fixed
        let new_center_x = plot_x - (norm_x - 0.5) * 2.0 * new_half_x;
        let new_center_y = plot_y + (norm_y - 0.5) * 2.0 * new_half;
        self.view = Some((new_center_x, new_center_y, new_half));
    }

    /// Pan the viewport by a normalized canvas delta.
    pub fn pan_by(&mut self, norm_delta_x: f64, norm_delta_y: f64) {
        let (center_x, center_y, half) = self.view.unwrap_or((0.0, 0.0, self.auto_half));
        let aspect = f64::from(self.canvas.width()) / f64::from(self.canvas.height().max(1));
        let half_x = half * aspect;
        let plot_delta_x = -norm_delta_x * 2.0 * half_x;
        let plot_delta_y = norm_delta_y * 2.0 * half; // y axis is flipped
        self.view = Some((center_x + plot_delta_x, center_y + plot_delta_y, half));
    }

    /// Reset the viewport to auto-fit.
    pub fn reset_view(&mut self) {
        self.view = None;
    }

    /// Get current iteration number.
    #[must_use]
    pub fn iteration(&self) -> usize {
        self.state.iteration
    }

    /// Get current loss value.
    #[must_use]
    pub fn loss(&self) -> f64 {
        self.state.loss
    }

    /// Whether training is complete.
    #[must_use]
    pub fn is_done(&self) -> bool {
        self.state.is_done()
    }

    /// Total number of iterations configured.
    #[must_use]
    pub fn total_iterations(&self) -> usize {
        self.state.config().n_iterations
    }

    /// Compute all quality metrics after training.
    ///
    /// Returns a JS object keyed by [`web_name`], so a metric with two
    /// readings appears as `{base}_manifold` and `{base}_2d`, plus the three
    /// spread diagnostics under their own names. A metric that is
    /// undefined for this state — every label-aware one, on unlabelled data —
    /// is omitted rather than sent as NaN, which is what lets the panel filter
    /// on `undefined`.
    pub fn compute_metrics(&self) -> Result<JsValue, JsValue> {
        let (values, spread) = self.state.compute_metrics();
        let obj = js_sys::Object::new();
        for m in metrics::ALL {
            if let Some(v) = values.get(*m) {
                set_prop(&obj, &web_name(*m), v)?;
            }
        }
        // The spread diagnostics are not metrics and do not come off the
        // registry; three named quantities, listed once here and once in the
        // panel's Spread group.
        for (key, v) in [
            ("r_max", spread.r_max()),
            ("r_rms", spread.r_rms()),
            ("r_gyration", spread.r_gyration()),
        ] {
            if let Some(v) = v {
                set_prop(&obj, key, v)?;
            }
        }
        Ok(obj.into())
    }
}

/// The metric registry, for the UI to build its tables from.
///
/// Quality metrics only — the spread diagnostics are not metrics and the panel
/// lists those three explicitly. Returns one entry per metric in
/// `metrics::ALL` order:
///
/// ```js
/// { key, base, label, short, family, space, dir, objective, dual }
/// ```
///
/// `key` is what [`EmbeddingRunner::compute_metrics`] puts on its result
/// object, and `key`/`dir`/`label` are what the metrics panel and the Pareto
/// selector used to hard-code. They fell out of date the moment a metric was
/// added or removed — `www/index.js` was still listing a `knn_overlap` row long
/// after the metric was deleted — which is the reason this exists.
///
/// A free function, not a method: the Pareto selector is populated from
/// front JSON before any `EmbeddingRunner` has been constructed.
#[wasm_bindgen]
pub fn metric_registry() -> Result<JsValue, JsValue> {
    let arr = js_sys::Array::new();
    for m in metrics::ALL {
        let o = js_sys::Object::new();
        set_str(&o, "key", &web_name(*m))?;
        set_str(&o, "base", m.base())?;
        set_str(&o, "label", m.label())?;
        set_str(&o, "short", m.short())?;
        set_str(&o, "family", m.family().name())?;
        set_str(
            &o,
            "space",
            match m.space() {
                metrics::Space::Projected => "projected",
                metrics::Space::Manifold => "manifold",
            },
        )?;
        // The arrow the panel prints beside the value.
        set_str(
            &o,
            "dir",
            match m.direction() {
                metrics::Direction::Maximize => "\u{2191}",
                metrics::Direction::Minimize => "\u{2193}",
            },
        )?;
        set_bool(&o, "objective", m.is_objective())?;
        set_bool(&o, "dual", m.has_twin())?;
        arr.push(&o);
    }
    Ok(arr.into())
}

fn set_str(obj: &js_sys::Object, key: &str, value: &str) -> Result<(), JsValue> {
    js_sys::Reflect::set(obj, &JsValue::from_str(key), &JsValue::from_str(value))?;
    Ok(())
}

fn set_bool(obj: &js_sys::Object, key: &str, value: bool) -> Result<(), JsValue> {
    js_sys::Reflect::set(obj, &JsValue::from_str(key), &JsValue::from_bool(value))?;
    Ok(())
}

/// The browser's name for a metric.
///
/// The browser labels the projected reading `_2d` where the JSONL calls it by
/// the bare name. Both conventions are load-bearing — one is baked into 350 MB
/// of results files, the other into the metrics panel's column layout — so the
/// registry carries the wire name and this is the single place the browser's
/// differs from it.
fn web_name(m: metrics::Metric) -> String {
    match m.space() {
        // Only a metric with two readings needs them told apart. The rest keep
        // their wire name, which is what the metrics panel's single-value rows
        // read.
        metrics::Space::Projected if m.has_twin() => format!("{}_2d", m.base()),
        metrics::Space::Manifold => format!("{}_manifold", m.base()),
        metrics::Space::Projected => m.name().to_string(),
    }
}

/// Return default `TrainingConfig` values as a JS object, so the frontend
/// can populate its inputs from a single source of truth.
#[wasm_bindgen]
pub fn get_default_config() -> Result<JsValue, JsValue> {
    let cfg = TrainingConfig::default();
    let obj = js_sys::Object::new();
    set_prop(&obj, "curvature", cfg.curvature)?;
    set_prop(&obj, "perplexity", cfg.perplexity)?;
    set_prop(&obj, "n_iterations", count_to_f64(cfg.n_iterations))?;
    set_prop(&obj, "learning_rate", cfg.learning_rate)?;
    set_prop(
        &obj,
        "early_exaggeration_factor",
        cfg.early_exaggeration_factor,
    )?;
    set_prop(
        &obj,
        "early_exaggeration_iterations",
        count_to_f64(cfg.early_exaggeration_iterations),
    )?;
    set_prop(&obj, "centering_weight", cfg.centering_weight)?;
    set_prop(&obj, "global_loss_weight", cfg.global_loss_weight)?;
    set_prop(&obj, "norm_loss_weight", cfg.norm_loss_weight)?;
    let scaling_loss_str = match cfg.scaling_loss_type {
        ScalingLossType::Rms => "rms",
        ScalingLossType::HardBarrier => "hard_barrier",
        ScalingLossType::SoftplusBarrier => "softplus_barrier",
        ScalingLossType::MeanDistance => "mean_distance",
        ScalingLossType::None => "none",
    };
    js_sys::Reflect::set(
        &obj,
        &JsValue::from_str("scaling_loss"),
        &JsValue::from_str(scaling_loss_str),
    )?;
    Ok(obj.into())
}

fn set_prop(obj: &js_sys::Object, key: &str, val: f64) -> Result<(), JsValue> {
    js_sys::Reflect::set(obj, &JsValue::from_str(key), &JsValue::from_f64(val))?;
    Ok(())
}

fn get_canvas(canvas_id: &str) -> Result<HtmlCanvasElement, JsValue> {
    let document = web_sys::window()
        .ok_or("no window")?
        .document()
        .ok_or("no document")?;
    Ok(document
        .get_element_by_id(canvas_id)
        .ok_or_else(|| JsValue::from_str(&format!("canvas '{canvas_id}' not found")))?
        .dyn_into::<HtmlCanvasElement>()?)
}
