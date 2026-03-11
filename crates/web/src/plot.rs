use plotters::prelude::*;
use plotters_canvas::CanvasBackend;
use wasm_bindgen::JsValue;
use web_sys::HtmlCanvasElement;

use fitting_core::visualisation::{Projection2D, SphericalProjection, project_to_2d, tab10_color};

const GRID_COLOR: RGBColor = RGBColor(200, 200, 200);
const AXIS_COLOR: RGBColor = RGBColor(140, 140, 140);
const BOUNDARY_COLOR: RGBColor = RGBColor(80, 80, 80);

/// Draw embedding points on a canvas using plotters.
pub fn draw_embedding(
    canvas: &HtmlCanvasElement,
    points: &[f64],
    n_points: usize,
    ambient_dim: usize,
    curvature: f64,
    labels: Option<&[u32]>,
    projection: SphericalProjection,
) -> Result<(), JsValue> {
    let Projection2D {
        coords: projected,
        scale,
    } = project_to_2d(points, n_points, ambient_dim, curvature, projection);

    // Fixed bounds centered at origin
    let half = if curvature < 0.0 {
        1.15 // Poincaré disk fits in [-1,1], add margin
    } else if curvature > 0.0 {
        match projection {
            SphericalProjection::Stereographic => 1.15,
            SphericalProjection::AzimuthalEquidistant => 1.15,
            SphericalProjection::Orthographic => 1.15,
        }
    } else {
        // Euclidean: use data extent but keep centered at origin
        let max_r = (0..n_points)
            .map(|i| {
                let x = projected[i * 2];
                let y = projected[i * 2 + 1];
                if x.is_finite() && y.is_finite() {
                    (x * x + y * y).sqrt()
                } else {
                    0.0
                }
            })
            .fold(0.0f64, f64::max);
        (max_r * 1.2).max(0.5)
    };

    let backend = CanvasBackend::with_canvas_object(canvas.clone())
        .ok_or("failed to create canvas backend")?;
    let root = backend.into_drawing_area();
    root.fill(&WHITE)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let proj_name = match projection {
        SphericalProjection::Stereographic => "Stereographic",
        SphericalProjection::AzimuthalEquidistant => "Azimuthal equidistant",
        SphericalProjection::Orthographic => "Orthographic",
    };

    let title = match curvature {
        k if k < 0.0 => format!("Hyperbolic (k={k:.2}) \u{2014} Poincar\u{e9} disk"),
        k if k > 0.0 => format!("Spherical (k={k:.2}) \u{2014} {proj_name}"),
        _ => "Euclidean".to_string(),
    };

    let mut chart = ChartBuilder::on(&root)
        .caption(&title, ("sans-serif", 16).into_font())
        .margin(5)
        .x_label_area_size(20)
        .y_label_area_size(20)
        .build_cartesian_2d(-half..half, -half..half)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Disable default mesh — we draw our own grid
    chart
        .configure_mesh()
        .disable_mesh()
        .disable_axes()
        .draw()
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Draw custom grid
    if curvature < 0.0 {
        draw_hyperbolic_grid(&mut chart)?;
    } else if curvature > 0.0 {
        draw_spherical_grid(&mut chart, projection)?;
    } else {
        draw_euclidean_grid(&mut chart, half, scale)?;
    }

    // Draw data points
    if let Some(labels) = labels {
        let mut label_set: Vec<u32> = labels.to_vec();
        label_set.sort();
        label_set.dedup();

        for &label in &label_set {
            let (r, g, b) = tab10_color(label);
            let color = RGBColor(r, g, b).mix(0.7);

            let point_data: Vec<(f64, f64)> = (0..n_points)
                .filter(|&i| labels[i] == label)
                .map(|i| (projected[i * 2], projected[i * 2 + 1]))
                .filter(|(x, y)| x.is_finite() && y.is_finite())
                .collect();

            chart
                .draw_series(
                    point_data
                        .iter()
                        .map(|&(x, y)| Circle::new((x, y), 3, color.filled())),
                )
                .map_err(|e| JsValue::from_str(&e.to_string()))?
                .label(format!("Label {label}"))
                .legend(move |(x, y)| Circle::new((x + 10, y), 3, RGBColor(r, g, b).filled()));
        }

        chart
            .configure_series_labels()
            .position(SeriesLabelPosition::UpperRight)
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK.mix(0.3))
            .draw()
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
    } else {
        let point_data: Vec<(f64, f64)> = (0..n_points)
            .map(|i| (projected[i * 2], projected[i * 2 + 1]))
            .filter(|(x, y)| x.is_finite() && y.is_finite())
            .collect();

        chart
            .draw_series(
                point_data
                    .iter()
                    .map(|&(x, y)| Circle::new((x, y), 3, BLUE.mix(0.7).filled())),
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }

    root.present()
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(())
}

type Chart<'a> = ChartContext<
    'a,
    CanvasBackend,
    Cartesian2d<plotters::coord::types::RangedCoordf64, plotters::coord::types::RangedCoordf64>,
>;

/// Draw Poincaré disk grid: geodesic arcs at tanh(n/2) and boundary circle.
fn draw_hyperbolic_grid(chart: &mut Chart) -> Result<(), JsValue> {
    let map_err = |e: DrawingAreaErrorKind<_>| JsValue::from_str(&e.to_string());

    // Boundary circle (unit disk)
    let boundary = circle_points(0.0, 0.0, 1.0, 128);
    chart
        .draw_series(LineSeries::new(boundary, BOUNDARY_COLOR.stroke_width(2)))
        .map_err(map_err)?;

    // Geodesic grid lines at hyperbolic distances n*r from origin
    // In the Poincaré disk, hyperbolic distance d maps to Euclidean radius tanh(d/(2r)).
    // For unit curvature r=1, tick at hyperbolic distance n => Euclidean a = tanh(n/2).
    let ticks: Vec<f64> = (1..=4).map(|n| (n as f64 / 2.0).tanh()).collect();

    for &a in &ticks {
        // Vertical geodesic at x = a (and x = -a)
        for &sign in &[1.0, -1.0] {
            let sa = sign * a;
            let arc = poincare_geodesic_arc(sa, true);
            chart
                .draw_series(LineSeries::new(arc, GRID_COLOR))
                .map_err(map_err)?;
        }
        // Horizontal geodesic at y = a (and y = -a)
        for &sign in &[1.0, -1.0] {
            let sa = sign * a;
            let arc = poincare_geodesic_arc(sa, false);
            chart
                .draw_series(LineSeries::new(arc, GRID_COLOR))
                .map_err(map_err)?;
        }
    }

    // Axes (straight lines through origin, which are geodesics in the Poincaré disk)
    chart
        .draw_series(LineSeries::new(vec![(-1.0, 0.0), (1.0, 0.0)], AXIS_COLOR))
        .map_err(map_err)?;
    chart
        .draw_series(LineSeries::new(vec![(0.0, -1.0), (0.0, 1.0)], AXIS_COLOR))
        .map_err(map_err)?;

    // Tick labels along x-axis
    let tick_labels = ["r", "2r", "3r", "4r"];
    for (i, &a) in ticks.iter().enumerate() {
        chart
            .draw_series(std::iter::once(Text::new(
                tick_labels[i].to_string(),
                (a + 0.02, -0.06),
                ("sans-serif", 11).into_font().color(&AXIS_COLOR),
            )))
            .map_err(map_err)?;
    }

    Ok(())
}

/// Compute a geodesic arc in the Poincaré disk.
///
/// A geodesic orthogonal to the boundary at Euclidean position `a` on one axis.
/// `vertical=true` means the geodesic crosses the x-axis at `a` and curves vertically;
/// `vertical=false` means it crosses the y-axis at `a` and curves horizontally.
///
/// The arc is a circular arc with:
///   center = (1+a²)/(2a)  on the relevant axis
///   radius = |1-a²|/(2|a|)
/// which is orthogonal to the unit boundary circle.
fn poincare_geodesic_arc(a: f64, vertical: bool) -> Vec<(f64, f64)> {
    if a.abs() < 1e-10 {
        return if vertical {
            vec![(0.0, -1.0), (0.0, 1.0)]
        } else {
            vec![(-1.0, 0.0), (1.0, 0.0)]
        };
    }

    let c = (1.0 + a * a) / (2.0 * a); // arc circle center coordinate
    let r = ((1.0 - a * a) / (2.0 * a)).abs(); // arc circle radius

    // Find the angular extent of the arc that lies inside the unit disk.
    // The arc circle intersects the unit circle at two points.
    // For the arc center at (c, 0), intersection angle on the arc circle:
    //   cos(θ) = (c² + r² - 1) / (2|c|r)
    let cos_theta = (c * c + r * r - 1.0) / (2.0 * c.abs() * r);
    let cos_theta = cos_theta.clamp(-1.0, 1.0);
    let theta = cos_theta.acos(); // half-angle of the arc

    // The arc sweeps from -theta to +theta around the center
    // (measuring from the line connecting center to origin)
    let n_seg = 64;
    let mut pts = Vec::with_capacity(n_seg + 1);
    for i in 0..=n_seg {
        let t = i as f64 / n_seg as f64;
        let ang = -theta + t * 2.0 * theta;

        let (px, py) = if vertical {
            // Arc center at (c, 0): angle measured from -x direction toward center
            // Arc parameterization: x = c - r*cos(ang), y = r*sin(ang)
            // but we need the correct reference direction.
            // The "inward" direction from center toward origin is at angle π (if c>0) or 0 (if c<0).
            if c > 0.0 {
                (c - r * ang.cos(), r * ang.sin())
            } else {
                (c + r * ang.cos(), -r * ang.sin())
            }
        } else {
            // Arc center at (0, c)
            if c > 0.0 {
                (r * ang.sin(), c - r * ang.cos())
            } else {
                (-r * ang.sin(), c + r * ang.cos())
            }
        };

        if px * px + py * py <= 1.01 {
            pts.push((px, py));
        }
    }

    pts
}

/// Draw spherical grid: concentric circles for parallels, radial meridians.
fn draw_spherical_grid(chart: &mut Chart, projection: SphericalProjection) -> Result<(), JsValue> {
    let map_err = |e: DrawingAreaErrorKind<_>| JsValue::from_str(&e.to_string());

    // Boundary circle
    let boundary = circle_points(0.0, 0.0, 1.0, 128);
    chart
        .draw_series(LineSeries::new(boundary, BOUNDARY_COLOR.stroke_width(2)))
        .map_err(map_err)?;

    // Concentric circles (parallels) at angular distances from center
    let step_deg = match projection {
        SphericalProjection::Orthographic => 15.0,
        _ => 30.0,
    };

    let mut theta_deg: f64 = step_deg;
    while theta_deg < 180.0 {
        let theta = theta_deg.to_radians();
        let r = match projection {
            SphericalProjection::Stereographic => (theta / 2.0).tan(),
            SphericalProjection::AzimuthalEquidistant => theta / std::f64::consts::PI,
            SphericalProjection::Orthographic => theta.sin(),
        };

        // Normalize: boundary at θ=π should map to r=1
        // For stereographic the boundary is at infinity, so we clip
        if r > 1.0 {
            theta_deg += step_deg;
            continue;
        }

        let circ = circle_points(0.0, 0.0, r, 64);
        chart
            .draw_series(LineSeries::new(circ, GRID_COLOR))
            .map_err(map_err)?;

        // Label
        let label = format!("{}\u{b0}", theta_deg as i32);
        chart
            .draw_series(std::iter::once(Text::new(
                label,
                (r * 0.72 + 0.02, r * 0.72 + 0.02), // ~45° direction
                ("sans-serif", 10).into_font().color(&AXIS_COLOR),
            )))
            .map_err(map_err)?;

        theta_deg += step_deg;
    }

    // Radial meridians (12 lines, every 30°)
    for i in 0..12 {
        let ang = (i as f64) * std::f64::consts::PI / 6.0;
        let (dx, dy) = (ang.cos(), ang.sin());
        chart
            .draw_series(LineSeries::new(vec![(0.0, 0.0), (dx, dy)], GRID_COLOR))
            .map_err(map_err)?;
    }

    // Axes (thicker)
    chart
        .draw_series(LineSeries::new(vec![(-1.0, 0.0), (1.0, 0.0)], AXIS_COLOR))
        .map_err(map_err)?;
    chart
        .draw_series(LineSeries::new(vec![(0.0, -1.0), (0.0, 1.0)], AXIS_COLOR))
        .map_err(map_err)?;

    Ok(())
}

/// Draw Euclidean grid: straight lines with tick labels in original coordinates.
fn draw_euclidean_grid(chart: &mut Chart, half: f64, scale: f64) -> Result<(), JsValue> {
    let map_err = |e: DrawingAreaErrorKind<_>| JsValue::from_str(&e.to_string());

    // Choose nice tick spacing based on original (unscaled) data range
    let half_original = half * scale;
    let spacing_original = nice_spacing(half_original);
    let spacing = spacing_original / scale;

    let mut v = spacing;
    while v < half {
        // Vertical grid lines
        chart
            .draw_series(LineSeries::new(vec![(v, -half), (v, half)], GRID_COLOR))
            .map_err(map_err)?;
        chart
            .draw_series(LineSeries::new(vec![(-v, -half), (-v, half)], GRID_COLOR))
            .map_err(map_err)?;
        // Horizontal grid lines
        chart
            .draw_series(LineSeries::new(vec![(-half, v), (half, v)], GRID_COLOR))
            .map_err(map_err)?;
        chart
            .draw_series(LineSeries::new(vec![(-half, -v), (half, -v)], GRID_COLOR))
            .map_err(map_err)?;

        // Tick labels in original coordinate space
        let orig_v = v * scale;
        let label = format_tick(orig_v);
        let neg_label = format_tick(-orig_v);
        let font = ("sans-serif", 10).into_font().color(&AXIS_COLOR);
        // x-axis labels (below axis)
        chart
            .draw_series(std::iter::once(Text::new(
                label.clone(),
                (v, -spacing * 0.3),
                font.clone(),
            )))
            .map_err(map_err)?;
        chart
            .draw_series(std::iter::once(Text::new(
                neg_label.clone(),
                (-v, -spacing * 0.3),
                font.clone(),
            )))
            .map_err(map_err)?;
        // y-axis labels (left of axis)
        chart
            .draw_series(std::iter::once(Text::new(
                label,
                (spacing * 0.15, v),
                font.clone(),
            )))
            .map_err(map_err)?;
        chart
            .draw_series(std::iter::once(Text::new(
                neg_label,
                (spacing * 0.15, -v),
                font,
            )))
            .map_err(map_err)?;

        v += spacing;
    }

    // Axes
    chart
        .draw_series(LineSeries::new(vec![(-half, 0.0), (half, 0.0)], AXIS_COLOR))
        .map_err(map_err)?;
    chart
        .draw_series(LineSeries::new(vec![(0.0, -half), (0.0, half)], AXIS_COLOR))
        .map_err(map_err)?;

    Ok(())
}

/// Generate points along a circle.
fn circle_points(cx: f64, cy: f64, r: f64, n: usize) -> Vec<(f64, f64)> {
    (0..=n)
        .map(|i| {
            let ang = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
            (cx + r * ang.cos(), cy + r * ang.sin())
        })
        .collect()
}

/// Format a tick value, dropping unnecessary trailing zeros.
fn format_tick(v: f64) -> String {
    let rounded = v.round();
    if (v - rounded).abs() < 1e-9 {
        format!("{}", rounded as i64)
    } else {
        format!("{:.1}", v)
    }
}

/// Choose a "nice" grid spacing for a given data half-range.
fn nice_spacing(half: f64) -> f64 {
    let raw = half / 4.0;
    let magnitude = 10.0f64.powf(raw.log10().floor());
    let normalized = raw / magnitude;
    let nice = if normalized < 1.5 {
        1.0
    } else if normalized < 3.5 {
        2.0
    } else if normalized < 7.5 {
        5.0
    } else {
        10.0
    };
    nice * magnitude
}
