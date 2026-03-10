use wasm_bindgen::prelude::*;
use web_sys::HtmlCanvasElement;

use fitting_core::config::{ScalingLossType, TrainingConfig};
use fitting_core::embedding::EmbeddingState;
use fitting_core::synthetic_data;
use fitting_core::visualisation::SphericalProjection;

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
        _ => SphericalProjection::Stereographic,
    }
}

/// Step-based embedding runner for animated rendering.
#[wasm_bindgen]
pub struct EmbeddingRunner {
    state: EmbeddingState,
    canvas: HtmlCanvasElement,
    labels: Option<Vec<u32>>,
    projection: SphericalProjection,
}

#[wasm_bindgen]
impl EmbeddingRunner {
    /// Create a runner from raw data (Random mode).
    #[wasm_bindgen(constructor)]
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
            ..Default::default()
        };

        let state = EmbeddingState::new(data, n_features, &config);
        let canvas = get_canvas(canvas_id)?;

        Ok(EmbeddingRunner {
            state,
            canvas,
            labels: None,
            projection: parse_projection(projection),
        })
    }

    /// Create a runner from a named synthetic dataset.
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
            ..Default::default()
        };

        let state = EmbeddingState::new(&synth.x, n_features, &config);
        let canvas = get_canvas(canvas_id)?;

        Ok(EmbeddingRunner {
            state,
            canvas,
            labels: Some(synth.labels),
            projection: parse_projection(projection),
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
    pub fn render(&self) -> Result<(), JsValue> {
        plot::draw_embedding(
            &self.canvas,
            &self.state.points,
            self.state.n_points,
            self.state.ambient_dim,
            self.state.config().curvature,
            self.labels.as_deref(),
            self.projection,
        )
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
