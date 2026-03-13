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
        "azimuthal_equidistant" => SphericalProjection::AzimuthalEquidistant,
        "orthographic" => SphericalProjection::Orthographic,
        "stereographic" => SphericalProjection::Stereographic,
        _ => SphericalProjection::AzimuthalEquidistant,
    }
}

/// Step-based embedding runner for animated rendering.
#[wasm_bindgen]
pub struct EmbeddingRunner {
    state: EmbeddingState,
    canvas: HtmlCanvasElement,
    labels: Option<Vec<u32>>,
    projection: SphericalProjection,
    /// Current viewport: (center_x, center_y, half_extent). None = auto-fit.
    view: Option<(f64, f64, f64)>,
    /// Auto-fit half-extent from the last render, used to anchor zoom interactions.
    auto_half: f64,
}

#[wasm_bindgen]
impl EmbeddingRunner {
    /// Create a runner from raw data (Random mode).
    #[wasm_bindgen(constructor)]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        canvas_id: &str,
        data: &[f64],
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
            ..Default::default()
        };

        let state = EmbeddingState::new(data, n_features, &config);
        let canvas = get_canvas(canvas_id)?;

        Ok(EmbeddingRunner {
            state,
            canvas,
            labels: None,
            projection: parse_projection(projection),
            view: None,
            auto_half: 1.0,
        })
    }

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
            ..Default::default()
        };

        let state = EmbeddingState::new(&synth.x, n_features, &config);
        let canvas = get_canvas(canvas_id)?;

        Ok(EmbeddingRunner {
            state,
            canvas,
            labels: Some(synth.labels),
            projection: parse_projection(projection),
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
            ..Default::default()
        };

        let state = EmbeddingState::new(data, n_features, &config);
        let canvas = get_canvas(canvas_id)?;

        Ok(EmbeddingRunner {
            state,
            canvas,
            labels: Some(labels.to_vec()),
            projection: parse_projection(projection),
            view: None,
            auto_half: 1.0,
        })
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
        true
    }

    /// Render the current state to canvas.
    /// Stores the auto-fit half-extent so zoom/pan can use it as a reference.
    pub fn render(&mut self) -> Result<(), JsValue> {
        let auto_half = plot::draw_embedding(
            &self.canvas,
            &self.state.points,
            self.state.n_points,
            self.state.ambient_dim,
            self.state.config().curvature,
            self.labels.as_deref(),
            self.projection,
            self.view,
        )?;
        self.auto_half = auto_half;
        Ok(())
    }

    /// Zoom the viewport around a normalized canvas position (0..1, 0..1).
    /// `factor > 1` zooms in, `factor < 1` zooms out.
    pub fn zoom_at(&mut self, norm_x: f64, norm_y: f64, factor: f64) {
        let (cx, cy, half) = self.view.unwrap_or((0.0, 0.0, self.auto_half));
        // Canvas coordinate → plot coordinate
        let plot_x = cx + (norm_x - 0.5) * 2.0 * half;
        let plot_y = cy - (norm_y - 0.5) * 2.0 * half; // y axis is flipped
        let new_half = (half / factor).clamp(1e-6, self.auto_half * 20.0);
        // Keep plot_x/plot_y under the cursor fixed
        let new_cx = plot_x - (norm_x - 0.5) * 2.0 * new_half;
        let new_cy = plot_y + (norm_y - 0.5) * 2.0 * new_half;
        self.view = Some((new_cx, new_cy, new_half));
    }

    /// Pan the viewport by a normalized canvas delta.
    pub fn pan_by(&mut self, norm_dx: f64, norm_dy: f64) {
        let (cx, cy, half) = self.view.unwrap_or((0.0, 0.0, self.auto_half));
        let dx = -norm_dx * 2.0 * half;
        let dy = norm_dy * 2.0 * half; // y axis is flipped
        self.view = Some((cx + dx, cy + dy, half));
    }

    /// Reset the viewport to auto-fit.
    pub fn reset_view(&mut self) {
        self.view = None;
    }

    /// Get current iteration number.
    pub fn iteration(&self) -> usize {
        self.state.iteration
    }

    /// Get current loss value.
    pub fn loss(&self) -> f64 {
        self.state.loss
    }

    /// Whether training is complete.
    pub fn is_done(&self) -> bool {
        self.state.is_done()
    }

    /// Total number of iterations configured.
    pub fn total_iterations(&self) -> usize {
        self.state.config().n_iterations
    }

    /// Compute all quality metrics after training.
    /// Returns a JS object with metric names as keys.
    pub fn compute_metrics(&self) -> Result<JsValue, JsValue> {
        let n = self.state.n_points;
        let high_dim_dist = self.state.high_dim_distances();
        let embed_dist = self.state.embedded_distances();

        // Project to 2D for visualization-space metrics
        let proj = visualisation::project_to_2d(
            &self.state.points,
            n,
            self.state.ambient_dim,
            self.state.config().curvature,
            self.projection,
        );

        let k = (self.state.config().perplexity as usize).min(n - 2).max(1);

        // A. Local structure preservation (on manifold distances)
        let trust = metrics::trustworthiness(&high_dim_dist, &embed_dist, n, k);
        let cont = metrics::continuity(&high_dim_dist, &embed_dist, n, k);
        let knn = metrics::knn_overlap(&high_dim_dist, &embed_dist, n, k);

        let obj = js_sys::Object::new();
        set_prop(&obj, "trustworthiness", trust)?;
        set_prop(&obj, "continuity", cont)?;
        set_prop(&obj, "knn_overlap", knn)?;

        // D. Perceptual evaluation (on 2D projected coordinates)
        if let Some(labels) = &self.labels {
            let cdm = metrics::class_density_measure(&proj.coords, labels, n);
            let cldm = metrics::cluster_density_measure(&proj.coords, labels, n);
            let db_ratio = metrics::davies_bouldin_ratio(&high_dim_dist, &proj.coords, labels, n);
            set_prop(&obj, "class_density_measure", cdm)?;
            set_prop(&obj, "cluster_density_measure", cldm)?;
            set_prop(&obj, "davies_bouldin_ratio", db_ratio)?;
        }

        Ok(obj.into())
    }
}

/// Return default TrainingConfig values as a JS object, so the frontend
/// can populate its inputs from a single source of truth.
#[wasm_bindgen]
pub fn get_default_config() -> Result<JsValue, JsValue> {
    let cfg = TrainingConfig::default();
    let obj = js_sys::Object::new();
    set_prop(&obj, "curvature", cfg.curvature)?;
    set_prop(&obj, "perplexity", cfg.perplexity)?;
    set_prop(&obj, "n_iterations", cfg.n_iterations as f64)?;
    set_prop(&obj, "learning_rate", cfg.learning_rate)?;
    set_prop(
        &obj,
        "early_exaggeration_factor",
        cfg.early_exaggeration_factor,
    )?;
    set_prop(
        &obj,
        "early_exaggeration_iterations",
        cfg.early_exaggeration_iterations as f64,
    )?;
    set_prop(&obj, "centering_weight", cfg.centering_weight)?;
    set_prop(&obj, "global_loss_weight", cfg.global_loss_weight)?;
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

/// Generate sample data (Gaussian blob) for testing.
#[wasm_bindgen]
pub fn generate_sample_data(n_points: usize, n_features: usize, seed: u32) -> Vec<f64> {
    let mut rng = synthetic_data::Rng::new(seed as u64);
    (0..n_points * n_features).map(|_| rng.normal()).collect()
}

/// Get available synthetic dataset names.
#[wasm_bindgen]
pub fn get_dataset_names() -> Vec<String> {
    synthetic_data::DATASET_NAMES
        .iter()
        .map(|s| s.to_string())
        .collect()
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
