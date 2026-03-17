use plotters::prelude::*;
use plotters_canvas::CanvasBackend;
use wasm_bindgen::JsValue;
use web_sys::HtmlCanvasElement;

use fitting_core::visualisation::{Projection2D, SphericalProjection, project_to_2d, tab10_color};

const GRID_COLOR: RGBColor = RGBColor(200, 200, 200);
const AXIS_COLOR: RGBColor = RGBColor(140, 140, 140);
const BOUNDARY_COLOR: RGBColor = RGBColor(80, 80, 80);

/// Parameters for drawing an embedding plot.
pub struct PlotParams<'a> {
    pub points: &'a [f64],
    pub n_points: usize,
    pub ambient_dim: usize,
    pub curvature: f64,
    pub labels: Option<&'a [u32]>,
    pub projection: SphericalProjection,
    /// Override the auto-fit viewport as `(center_x, center_y, half_extent)`.
    pub view: Option<(f64, f64, f64)>,
}

/// Draw embedding points on a canvas using plotters.
///
/// Returns the auto-fit half-extent so callers can anchor zoom/pan interactions.
pub fn draw_embedding(canvas: &HtmlCanvasElement, params: &PlotParams) -> Result<f64, JsValue> {
    let Projection2D {
        coords: projected,
        scale,
    } = project_to_2d(
        params.points,
        params.n_points,
        params.ambient_dim,
        params.curvature,
        params.projection,
    );
    let auto_half = calculate_auto_half(params.curvature, params.n_points, &projected);

    let (cx, cy, half) = params.view.unwrap_or((0.0, 0.0, auto_half));
    let x_min = cx - half;
    let x_max = cx + half;
    let y_min = cy - half;
    let y_max = cy + half;

    let backend = CanvasBackend::with_canvas_object(canvas.clone())
        .ok_or("failed to create canvas backend")?;
    let root = backend.into_drawing_area();
    root.fill(&WHITE)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let title = make_title(params.curvature, params.projection);

    let mut chart = ChartBuilder::on(&root)
        .caption(&title, ("sans-serif", 16).into_font())
        .margin(5)
        .x_label_area_size(20)
        .y_label_area_size(20)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Disable default mesh — we draw our own grid
    chart
        .configure_mesh()
        .disable_mesh()
        .disable_axes()
        .draw()
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Draw custom grid
    if params.curvature < 0.0 {
        let radius = 1.0 / (-params.curvature).sqrt();
        draw_hyperbolic_grid(&mut chart, radius, cx, cy, half)?;
    } else if params.curvature > 0.0 {
        draw_spherical_grid(&mut chart, params.projection, cx, cy, half)?;
    } else {
        draw_euclidean_grid(&mut chart, cx, cy, half, scale)?;
    }

    draw_points(&mut chart, &projected, params.n_points, params.labels)?;

    root.present()
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(auto_half)
}

fn make_title(curvature: f64, projection: SphericalProjection) -> String {
    match curvature {
        k if k < 0.0 => format!(
            "Hyperbolic (k={}) \u{2014} Poincar\u{e9} disk",
            format_curvature(k)
        ),
        k if k > 0.0 => {
            let proj_name = match projection {
                SphericalProjection::Stereographic => "Stereographic",
                SphericalProjection::AzimuthalEquidistant => "Azimuthal equidistant",
                SphericalProjection::Orthographic => "Orthographic",
            };
            format!("Spherical (k={}) \u{2014} {proj_name}", format_curvature(k))
        }
        _ => "Euclidean".to_string(),
    }
}

fn draw_points(
    chart: &mut Chart,
    projected: &[f64],
    n_points: usize,
    labels: Option<&[u32]>,
) -> Result<(), JsValue> {
    let map_err = |e: DrawingAreaErrorKind<_>| JsValue::from_str(&e.to_string());

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
                .map_err(map_err)?
                .label(format!("Label {label}"))
                .legend(move |(x, y)| Circle::new((x + 10, y), 3, RGBColor(r, g, b).filled()));
        }

        chart
            .configure_series_labels()
            .position(SeriesLabelPosition::UpperRight)
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK.mix(0.3))
            .draw()
            .map_err(map_err)?;
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
            .map_err(map_err)?;
    }

    Ok(())
}

fn calculate_auto_half(curvature: f64, n_points: usize, projected: &[f64]) -> f64 {
    if curvature < 0.0 {
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
        (max_r * 1.3).clamp(0.05, 1.15)
    } else if curvature > 0.0 {
        1.15
    } else {
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
    }
}

type Chart<'a> = ChartContext<
    'a,
    CanvasBackend,
    Cartesian2d<plotters::coord::types::RangedCoordf64, plotters::coord::types::RangedCoordf64>,
>;

/// Draw Poincaré disk grid, adapting ticks to the current zoom level.
///
/// `radius` is the hyperbolic radius r = 1/sqrt(-K).
/// `half` is the current chart half-extent in Poincaré disk coordinates.
fn draw_hyperbolic_grid(
    chart: &mut Chart,
    radius: f64,
    cx: f64,
    cy: f64,
    half: f64,
) -> Result<(), JsValue> {
    let map_err = |e: DrawingAreaErrorKind<_>| JsValue::from_str(&e.to_string());

    // Boundary circle — plotters clips to viewport automatically
    let boundary = circle_points(0.0, 0.0, 1.0, 128);
    chart
        .draw_series(LineSeries::new(boundary, BOUNDARY_COLOR.stroke_width(2)))
        .map_err(map_err)?;

    // Tick spacing based on zoom level (half-extent of the view)
    let edge_disk_r = (half * 0.9).min(0.9999);
    let max_geo_dist = 2.0 * radius * edge_disk_r.atanh();
    let geo_spacing = nice_spacing(max_geo_dist);

    let mut ticks: Vec<(f64, f64)> = Vec::new(); // (disk_radius, geo_distance)
    let mut geo_d = geo_spacing;
    while geo_d < max_geo_dist * 1.2 {
        let disk_r = (geo_d / (2.0 * radius)).tanh();
        if disk_r < 0.999 {
            ticks.push((disk_r, geo_d));
        }
        geo_d += geo_spacing;
    }

    // Draw geodesic arcs — plotters clips to viewport
    for &(a, _) in &ticks {
        for &sign in &[1.0, -1.0] {
            let sa = sign * a;
            let arc = poincare_geodesic_arc(sa, true);
            chart
                .draw_series(LineSeries::new(arc, GRID_COLOR))
                .map_err(map_err)?;
            let arc = poincare_geodesic_arc(sa, false);
            chart
                .draw_series(LineSeries::new(arc, GRID_COLOR))
                .map_err(map_err)?;
        }
    }

    // Axes spanning full viewport
    let (x_min, x_max) = (cx - half, cx + half);
    let (y_min, y_max) = (cy - half, cy + half);
    chart
        .draw_series(LineSeries::new(
            vec![(x_min, 0.0), (x_max, 0.0)],
            AXIS_COLOR,
        ))
        .map_err(map_err)?;
    chart
        .draw_series(LineSeries::new(
            vec![(0.0, y_min), (0.0, y_max)],
            AXIS_COLOR,
        ))
        .map_err(map_err)?;

    // Tick labels, skipping when too dense
    let font = ("sans-serif", 10).into_font().color(&AXIS_COLOR);
    let min_gap = half * 0.08;
    let mut last_label_pos = f64::NEG_INFINITY;
    for &(a, geo_d) in &ticks {
        if a - last_label_pos < min_gap {
            continue;
        }
        last_label_pos = a;
        let label = format_tick(geo_d);
        chart
            .draw_series(std::iter::once(Text::new(
                label,
                (a + half * 0.02, -half * 0.06),
                font.clone(),
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
fn draw_spherical_grid(
    chart: &mut Chart,
    projection: SphericalProjection,
    cx: f64,
    cy: f64,
    half: f64,
) -> Result<(), JsValue> {
    let map_err = |e: DrawingAreaErrorKind<_>| JsValue::from_str(&e.to_string());

    // Boundary circle — plotters clips to viewport
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

        if r > 1.0 {
            theta_deg += step_deg;
            continue;
        }

        let circ = circle_points(0.0, 0.0, r, 64);
        chart
            .draw_series(LineSeries::new(circ, GRID_COLOR))
            .map_err(map_err)?;

        let label = format!("{}\u{b0}", theta_deg as i32);
        chart
            .draw_series(std::iter::once(Text::new(
                label,
                (r * 0.72 + 0.02, r * 0.72 + 0.02),
                ("sans-serif", 10).into_font().color(&AXIS_COLOR),
            )))
            .map_err(map_err)?;

        theta_deg += step_deg;
    }

    // Radial meridians as full diameters, properly clipped to the viewport so that
    // plotters' coordinate clamping doesn't distort their angles.
    let (x_min, x_max) = (cx - half, cx + half);
    let (y_min, y_max) = (cy - half, cy + half);
    for i in 0..6 {
        let ang = (i as f64) * std::f64::consts::PI / 6.0;
        let (dx, dy) = (ang.cos(), ang.sin());
        if let Some((a, b)) = clip_line((-dx, -dy), (dx, dy), x_min, y_min, x_max, y_max) {
            chart
                .draw_series(LineSeries::new(vec![a, b], GRID_COLOR))
                .map_err(map_err)?;
        }
    }

    // Axes spanning full viewport
    chart
        .draw_series(LineSeries::new(
            vec![(x_min, 0.0), (x_max, 0.0)],
            AXIS_COLOR,
        ))
        .map_err(map_err)?;
    chart
        .draw_series(LineSeries::new(
            vec![(0.0, y_min), (0.0, y_max)],
            AXIS_COLOR,
        ))
        .map_err(map_err)?;

    Ok(())
}

/// Draw Euclidean grid: straight lines with tick labels in original coordinates.
fn draw_euclidean_grid(
    chart: &mut Chart,
    cx: f64,
    cy: f64,
    half: f64,
    scale: f64,
) -> Result<(), JsValue> {
    let map_err = |e: DrawingAreaErrorKind<_>| JsValue::from_str(&e.to_string());

    let (x_min, x_max) = (cx - half, cx + half);
    let (y_min, y_max) = (cy - half, cy + half);

    // Choose nice tick spacing based on original (unscaled) data range
    let half_original = half * scale;
    let spacing_original = nice_spacing(half_original);
    let spacing = spacing_original / scale;

    let font = ("sans-serif", 10).into_font().color(&AXIS_COLOR);

    // Vertical grid lines — iterate over all tick positions within the viewport
    let first_x = (x_min / spacing).ceil() * spacing;
    let mut x = first_x;
    while x <= x_max + spacing * 0.01 {
        if x.abs() > spacing * 0.01 {
            chart
                .draw_series(LineSeries::new(vec![(x, y_min), (x, y_max)], GRID_COLOR))
                .map_err(map_err)?;
            let label = format_tick(x * scale);
            chart
                .draw_series(std::iter::once(Text::new(
                    label,
                    (x, y_min + spacing * 0.3),
                    font.clone(),
                )))
                .map_err(map_err)?;
        }
        x += spacing;
    }

    // Horizontal grid lines
    let first_y = (y_min / spacing).ceil() * spacing;
    let mut y = first_y;
    while y <= y_max + spacing * 0.01 {
        if y.abs() > spacing * 0.01 {
            chart
                .draw_series(LineSeries::new(vec![(x_min, y), (x_max, y)], GRID_COLOR))
                .map_err(map_err)?;
            let label = format_tick(y * scale);
            chart
                .draw_series(std::iter::once(Text::new(
                    label,
                    (x_min + spacing * 0.1, y),
                    font.clone(),
                )))
                .map_err(map_err)?;
        }
        y += spacing;
    }

    // Axes
    chart
        .draw_series(LineSeries::new(
            vec![(x_min, 0.0), (x_max, 0.0)],
            AXIS_COLOR,
        ))
        .map_err(map_err)?;
    chart
        .draw_series(LineSeries::new(
            vec![(0.0, y_min), (0.0, y_max)],
            AXIS_COLOR,
        ))
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

/// Format curvature for display, using scientific notation for very small values.
fn format_curvature(k: f64) -> String {
    if k.abs() >= 0.01 {
        format!("{k:.2}")
    } else {
        format!("{k:.1e}")
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

/// Clip a line segment to a rectangle using Liang-Barsky. Returns the clipped endpoints,
/// or `None` if the segment is entirely outside.
fn clip_line(
    a: (f64, f64),
    b: (f64, f64),
    x_min: f64,
    y_min: f64,
    x_max: f64,
    y_max: f64,
) -> Option<((f64, f64), (f64, f64))> {
    let dx = b.0 - a.0;
    let dy = b.1 - a.1;
    let mut t0 = 0.0f64;
    let mut t1 = 1.0f64;

    let edges = [
        (-dx, a.0 - x_min), // left
        (dx, x_max - a.0),  // right
        (-dy, a.1 - y_min), // bottom
        (dy, y_max - a.1),  // top
    ];

    for &(p, q) in &edges {
        if p.abs() < 1e-12 {
            if q < 0.0 {
                return None;
            }
        } else {
            let t = q / p;
            if p < 0.0 {
                t0 = t0.max(t);
            } else {
                t1 = t1.min(t);
            }
        }
    }

    if t0 > t1 {
        return None;
    }

    Some((
        (a.0 + t0 * dx, a.1 + t0 * dy),
        (a.0 + t1 * dx, a.1 + t1 * dy),
    ))
}
