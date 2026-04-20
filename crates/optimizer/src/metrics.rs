use crate::search_space::OptimizeDirection;
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct AllMetrics {
    // Local structure — 2D (after projection) and manifold (before projection)
    pub trustworthiness: f64,
    pub trustworthiness_manifold: f64,
    pub continuity: f64,
    pub continuity_manifold: f64,
    pub knn_overlap: f64,
    pub knn_overlap_manifold: f64,
    pub neighborhood_hit: f64,
    pub neighborhood_hit_manifold: f64,
    // Distance preservation — 2D and manifold
    pub normalized_stress: f64,
    pub normalized_stress_manifold: f64,
    pub shepard_goodness: f64,
    pub shepard_goodness_manifold: f64,
    // Class separation (2D only)
    pub davies_bouldin_ratio: f64,
    pub dunn_index: f64,
    pub class_density_measure: f64,
    pub cluster_density_measure: f64,
}

impl AllMetrics {
    /// Component-wise mean over a non-empty slice of samples.
    pub fn mean(samples: &[AllMetrics]) -> AllMetrics {
        let n = samples.len() as f64;
        let avg = |f: fn(&AllMetrics) -> f64| samples.iter().map(f).sum::<f64>() / n;
        AllMetrics {
            trustworthiness: avg(|m| m.trustworthiness),
            trustworthiness_manifold: avg(|m| m.trustworthiness_manifold),
            continuity: avg(|m| m.continuity),
            continuity_manifold: avg(|m| m.continuity_manifold),
            knn_overlap: avg(|m| m.knn_overlap),
            knn_overlap_manifold: avg(|m| m.knn_overlap_manifold),
            neighborhood_hit: avg(|m| m.neighborhood_hit),
            neighborhood_hit_manifold: avg(|m| m.neighborhood_hit_manifold),
            normalized_stress: avg(|m| m.normalized_stress),
            normalized_stress_manifold: avg(|m| m.normalized_stress_manifold),
            shepard_goodness: avg(|m| m.shepard_goodness),
            shepard_goodness_manifold: avg(|m| m.shepard_goodness_manifold),
            davies_bouldin_ratio: avg(|m| m.davies_bouldin_ratio),
            dunn_index: avg(|m| m.dunn_index),
            class_density_measure: avg(|m| m.class_density_measure),
            cluster_density_measure: avg(|m| m.cluster_density_measure),
        }
    }
}

/// One of the 16 DR quality metrics the optimizer can measure or optimise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Metric {
    Trustworthiness,
    TrustworthinessManifold,
    Continuity,
    ContinuityManifold,
    KnnOverlap,
    KnnOverlapManifold,
    NeighborhoodHit,
    NeighborhoodHitManifold,
    NormalizedStress,
    NormalizedStressManifold,
    ShepardGoodness,
    ShepardGoodnessManifold,
    DaviesBouldinRatio,
    DunnIndex,
    ClassDensityMeasure,
    ClusterDensityMeasure,
}

impl Metric {
    pub const ALL: &'static [Metric] = &[
        Metric::Trustworthiness,
        Metric::TrustworthinessManifold,
        Metric::Continuity,
        Metric::ContinuityManifold,
        Metric::KnnOverlap,
        Metric::KnnOverlapManifold,
        Metric::NeighborhoodHit,
        Metric::NeighborhoodHitManifold,
        Metric::NormalizedStress,
        Metric::NormalizedStressManifold,
        Metric::ShepardGoodness,
        Metric::ShepardGoodnessManifold,
        Metric::DaviesBouldinRatio,
        Metric::DunnIndex,
        Metric::ClassDensityMeasure,
        Metric::ClusterDensityMeasure,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Metric::Trustworthiness => "trustworthiness",
            Metric::TrustworthinessManifold => "trustworthiness_manifold",
            Metric::Continuity => "continuity",
            Metric::ContinuityManifold => "continuity_manifold",
            Metric::KnnOverlap => "knn_overlap",
            Metric::KnnOverlapManifold => "knn_overlap_manifold",
            Metric::NeighborhoodHit => "neighborhood_hit",
            Metric::NeighborhoodHitManifold => "neighborhood_hit_manifold",
            Metric::NormalizedStress => "normalized_stress",
            Metric::NormalizedStressManifold => "normalized_stress_manifold",
            Metric::ShepardGoodness => "shepard_goodness",
            Metric::ShepardGoodnessManifold => "shepard_goodness_manifold",
            Metric::DaviesBouldinRatio => "davies_bouldin_ratio",
            Metric::DunnIndex => "dunn_index",
            Metric::ClassDensityMeasure => "class_density_measure",
            Metric::ClusterDensityMeasure => "cluster_density_measure",
        }
    }

    pub fn direction(self) -> OptimizeDirection {
        match self {
            Metric::NormalizedStress | Metric::NormalizedStressManifold => {
                OptimizeDirection::Minimize
            }
            _ => OptimizeDirection::Maximize,
        }
    }

    pub fn value(self, m: &AllMetrics) -> f64 {
        match self {
            Metric::Trustworthiness => m.trustworthiness,
            Metric::TrustworthinessManifold => m.trustworthiness_manifold,
            Metric::Continuity => m.continuity,
            Metric::ContinuityManifold => m.continuity_manifold,
            Metric::KnnOverlap => m.knn_overlap,
            Metric::KnnOverlapManifold => m.knn_overlap_manifold,
            Metric::NeighborhoodHit => m.neighborhood_hit,
            Metric::NeighborhoodHitManifold => m.neighborhood_hit_manifold,
            Metric::NormalizedStress => m.normalized_stress,
            Metric::NormalizedStressManifold => m.normalized_stress_manifold,
            Metric::ShepardGoodness => m.shepard_goodness,
            Metric::ShepardGoodnessManifold => m.shepard_goodness_manifold,
            Metric::DaviesBouldinRatio => m.davies_bouldin_ratio,
            Metric::DunnIndex => m.dunn_index,
            Metric::ClassDensityMeasure => m.class_density_measure,
            Metric::ClusterDensityMeasure => m.cluster_density_measure,
        }
    }

    /// Parse from the CLI string representation. Returns `None` for unknown names.
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "trustworthiness" => Some(Metric::Trustworthiness),
            "trustworthiness_manifold" => Some(Metric::TrustworthinessManifold),
            "continuity" => Some(Metric::Continuity),
            "continuity_manifold" => Some(Metric::ContinuityManifold),
            "knn_overlap" => Some(Metric::KnnOverlap),
            "knn_overlap_manifold" => Some(Metric::KnnOverlapManifold),
            "neighborhood_hit" => Some(Metric::NeighborhoodHit),
            "neighborhood_hit_manifold" => Some(Metric::NeighborhoodHitManifold),
            "normalized_stress" => Some(Metric::NormalizedStress),
            "normalized_stress_manifold" => Some(Metric::NormalizedStressManifold),
            "shepard_goodness" => Some(Metric::ShepardGoodness),
            "shepard_goodness_manifold" => Some(Metric::ShepardGoodnessManifold),
            "davies_bouldin_ratio" => Some(Metric::DaviesBouldinRatio),
            "dunn_index" => Some(Metric::DunnIndex),
            "class_density_measure" => Some(Metric::ClassDensityMeasure),
            "cluster_density_measure" => Some(Metric::ClusterDensityMeasure),
            _ => None,
        }
    }

    pub fn valid_names() -> String {
        Metric::ALL
            .iter()
            .map(|m| m.name())
            .collect::<Vec<_>>()
            .join(", ")
    }
}
