//! Seeded AOI generation shared by desktop and WebAssembly.
use super::{
    data::{self, Dataset},
    geometry::{self, Grid, RADIUS},
    preview::{Preview, Record},
};
use geo::{
    algorithm::{orient::Direction, unary_union},
    Buffer, Coord, LineString, MapCoords, MultiPolygon, Orient, Polygon,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::{
    cmp::Ordering,
    collections::{BTreeMap, BinaryHeap, HashSet},
    f64::consts::PI,
    sync::Arc,
};

#[derive(Clone)]
pub(super) struct Gdp {
    pub count: usize,
    pub seed: u32,
    pub mean_area: f64,
    pub size_ratio: f64,
    pub uniform: bool,
    pub hotspot_count: usize,
    pub hotspot_spread: f64,
    pub hotspot_fraction: f64,
    pub shape: String,
    pub random_rotation: bool,
    pub max_aspect: f64,
}

#[derive(Clone, serde::Serialize)]
pub(super) struct Environment {
    pub kind: &'static str,
    pub count: usize,
    pub seed: u32,
    pub mean_area: f64,
    pub size_ratio: f64,
    pub mean_length: f64,
    pub length_ratio: f64,
    pub buffer: f64,
    pub inland: f64,
    pub offshore: f64,
    pub flow: f64,
    pub max_elongation: f64,
    pub outline_variation: f64,
    pub forest_adherence: f64,
    pub latitude: f64,
    pub tile_edge: f64,
}

#[derive(Clone)]
pub(super) enum Settings {
    Gdp(Gdp),
    Environment(Environment),
}

pub(super) fn bounds(mean: f64, ratio: f64) -> Result<(f64, f64), String> {
    if !mean.is_finite() || mean <= 0.0 || !ratio.is_finite() || ratio < 1.0 {
        return Err("Average size must be positive and the size ratio at least one".into());
    }
    let minimum = if ratio == 1.0 {
        mean
    } else {
        mean * ratio.ln() / (ratio - 1.0)
    };
    Ok((minimum, minimum * ratio))
}

fn draw_size(rng: &mut ChaCha8Rng, mean: f64, ratio: f64) -> f64 {
    let (a, b) = bounds(mean, ratio).unwrap();
    (a.ln() + rng.random::<f64>() * (b.ln() - a.ln())).exp()
}

fn range(value: f64, lo: f64, hi: f64) -> Result<(), String> {
    if value.is_finite() && (lo..=hi).contains(&value) {
        Ok(())
    } else {
        Err(format!("AOI setting must be finite and in [{lo}, {hi}]"))
    }
}

impl Settings {
    pub fn source_kind(&self) -> &'static str {
        match self {
            Self::Gdp(_) => "gdp",
            Self::Environment(c) => c.kind,
        }
    }
    pub fn count(&self) -> usize {
        match self {
            Self::Gdp(c) => c.count,
            Self::Environment(c) => c.count,
        }
    }
    pub fn seed(&self) -> u32 {
        match self {
            Self::Gdp(c) => c.seed,
            Self::Environment(c) => c.seed,
        }
    }
    pub fn uniform(&self) -> bool {
        matches!(self, Self::Gdp(Gdp { uniform: true, .. }))
    }
    pub fn validate(&self) -> Result<(), String> {
        if !(1..=50_000).contains(&self.count()) {
            return Err("AOI count must be in [1, 50000]".into());
        }
        match self {
            Self::Gdp(c) => {
                range(c.mean_area, 1.0, 1_000_000.0)?;
                range(c.size_ratio, 1.0, 10_000.0)?;
                range(c.hotspot_fraction, 0.0, 1.0)?;
                range(c.hotspot_spread, 0.0, 2000.0)?;
                if c.hotspot_count > 10_000 {
                    return Err("Too many AOI hotspots".into());
                }
                range(c.max_aspect, 1.0, 10.0)?;
                if !["rectangle", "oriented_rectangle"].contains(&c.shape.as_str()) {
                    return Err("Unknown GDP AOI shape".into());
                }
                if c.shape == "rectangle"
                    && bounds(c.mean_area, c.size_ratio)?.0 < Grid::new(0.1).area()
                {
                    return Err("Smallest AOI is below one tile; increase its area or reduce the size ratio".into());
                }
            }
            Self::Environment(c) => {
                if !["forest", "river", "coast"].contains(&c.kind) {
                    return Err("Unknown environmental source".into());
                }
                for (v, lo, hi) in [
                    (c.mean_area, 10.0, 100_000.0),
                    (c.size_ratio, 1.0, 1000.0),
                    (c.mean_length, 1.0, 1000.0),
                    (c.length_ratio, 1.0, 100.0),
                    (c.buffer, 0.1, 50.0),
                    (c.inland, 0.1, 50.0),
                    (c.offshore, 0.1, 50.0),
                    (c.flow, 1.0, 1e6),
                    (c.max_elongation, 1.0, 10.0),
                    (c.outline_variation, 0.0, 1.0),
                    (c.forest_adherence, 0.0, 1.0),
                    (c.latitude, 1.0, 85.0),
                    (c.tile_edge, 0.05, 10.0),
                ] {
                    range(v, lo, hi)?;
                }
                let (mean, ratio, max) = if c.kind == "forest" {
                    (c.mean_area, c.size_ratio, 1_000_000.0)
                } else {
                    (c.mean_length, c.length_ratio, 3000.0)
                };
                if bounds(mean, ratio)?.1 > max {
                    return Err(format!(
                        "Requested size range exceeds {max}; reduce the average or size ratio"
                    ));
                }
            }
        }
        Ok(())
    }
}

struct Weights {
    cumulative: Vec<f64>,
    total: f64,
}
impl Weights {
    fn new(weights: impl IntoIterator<Item = f64>) -> Result<Self, String> {
        let mut total = 0.0;
        let cumulative = weights
            .into_iter()
            .map(|w| {
                total += w;
                total
            })
            .collect::<Vec<_>>();
        if !total.is_finite() || total <= 0.0 {
            return Err("No geographic features match these filters".into());
        }
        Ok(Self { cumulative, total })
    }
    fn sample(&self, rng: &mut ChaCha8Rng) -> usize {
        let target = (rng.random::<f64>() * self.total).min(self.total.next_down());
        self.cumulative.partition_point(|&w| w <= target)
    }
}

enum Sampler {
    Demand {
        weights: Option<Weights>,
        parents: Vec<[f64; 2]>,
    },
    Forest {
        weights: Weights,
        area: Arc<Vec<f64>>,
        columns: BTreeMap<usize, Vec<usize>>,
    },
    Network {
        weights: Weights,
        portions: Vec<Vec<(f64, f64)>>,
    },
}

pub(super) struct Job {
    settings: Settings,
    dataset: Option<Arc<Dataset>>,
    sampler: Sampler,
    locations: ChaCha8Rng,
    shapes: ChaCha8Rng,
    rotations: ChaCha8Rng,
    grid: Grid,
    pub completed: usize,
    preview: Option<Preview>,
}

fn direct(dataset: Option<&Dataset>, weights: Option<&Weights>, rng: &mut ChaCha8Rng) -> [f64; 2] {
    if let Some(Dataset::Demand(d)) = dataset {
        let index = d.cells[weights.unwrap().sample(rng)].0;
        Grid {
            columns: d.columns as i64,
            rows: d.rows as i64,
        }
        .lonlat(
            (index % d.columns) as f64 + rng.random::<f64>(),
            (index / d.columns) as f64 + rng.random::<f64>(),
        )
    } else {
        [
            rng.random::<f64>() * 360.0 - 180.0,
            (rng.random::<f64>() * 2.0 - 1.0).asin().to_degrees(),
        ]
    }
}

fn portions(points: &[[f64; 2]], band: f64) -> Vec<(f64, f64)> {
    let mut offset = 0.0;
    let mut result = Vec::new();
    for pair in points.windows(2) {
        let length = geometry::distance(pair[0], pair[1]);
        let delta = pair[1][1] - pair[0][1];
        let (begin, end) = if delta.abs() < 1e-12 {
            if pair[0][1].abs() > band {
                (0.0, 0.0)
            } else {
                (0.0, 1.0)
            }
        } else {
            let a = (-band - pair[0][1]) / delta;
            let b = (band - pair[0][1]) / delta;
            (a.min(b).clamp(0.0, 1.0), a.max(b).clamp(0.0, 1.0))
        };
        if end > begin && length > 0.0 {
            result.push((offset + begin * length, (end - begin) * length));
        }
        offset += length;
    }
    result
}

impl Job {
    pub fn new(settings: Settings, dataset: Option<Arc<Dataset>>) -> Result<Self, String> {
        settings.validate()?;
        let seed = settings.seed() as u64;
        let sampler = match &settings {
            Settings::Gdp(c) => {
                if !c.uniform && !matches!(dataset.as_deref(), Some(Dataset::Demand(_))) {
                    return Err("Choose a prepared demand dataset".into());
                }
                let weights = match dataset.as_deref() {
                    Some(Dataset::Demand(d)) if !c.uniform => {
                        Some(Weights::new(d.cells.iter().map(|c| c.1))?)
                    }
                    _ => None,
                };
                let mut parents_rng = ChaCha8Rng::seed_from_u64(seed ^ 0xc91e_831a_4d02_76f5);
                let parents = (0..c.hotspot_count)
                    .map(|_| {
                        direct(
                            if c.uniform { None } else { dataset.as_deref() },
                            weights.as_ref(),
                            &mut parents_rng,
                        )
                    })
                    .collect();
                Sampler::Demand { weights, parents }
            }
            Settings::Environment(c) => match dataset.as_deref() {
                Some(Dataset::Forest(d)) if c.kind == "forest" => {
                    let step = 360.0 / d.columns as f64;
                    let area: Vec<_> = (0..d.rows)
                        .map(|r| {
                            RADIUS.powi(2)
                                * step.to_radians()
                                * ((90.0 - r as f64 * step).to_radians().sin()
                                    - (90.0 - (r + 1) as f64 * step).to_radians().sin())
                        })
                        .collect();
                    let weights = Weights::new((0..d.rows).map(|r| {
                        if (90.0 - (r as f64 + 0.5) * step).abs() > c.latitude {
                            return 0.0;
                        }
                        (0..d.columns)
                            .filter(|&x| data::bit(&d.mask, r * d.columns + x))
                            .count() as f64
                            * area[r]
                    }))?;
                    Sampler::Forest {
                        weights,
                        area: Arc::new(area),
                        columns: BTreeMap::new(),
                    }
                }
                Some(Dataset::River(d)) if c.kind == "river" => network_sampler(d, c)?,
                Some(Dataset::Coast(d)) if c.kind == "coast" => network_sampler(d, c)?,
                _ => return Err(format!("Choose a prepared {} dataset", c.kind)),
            },
        };
        let edge = match &settings {
            Settings::Gdp(_) => 0.1,
            Settings::Environment(c) => c.tile_edge,
        };
        let preview = Some(Preview::empty(settings.count(), settings.seed()));
        Ok(Self {
            settings,
            dataset,
            sampler,
            locations: ChaCha8Rng::seed_from_u64(seed ^ 0x3af4_972d_182b_6c05),
            shapes: ChaCha8Rng::seed_from_u64(seed ^ 0x857b_63f0_a291_4ecd),
            rotations: ChaCha8Rng::seed_from_u64(seed ^ 0x19d3_5a40_e81b_723c),
            grid: Grid::new(edge),
            completed: 0,
            preview,
        })
    }

    // A time budget also keeps the browser event loop responsive during large batches.
    #[cfg(any(test, target_arch = "wasm32"))]
    pub fn step(&mut self) -> Result<Option<Preview>, String> {
        let start = web_time::Instant::now();
        while self.completed < self.settings.count() {
            let record = self.record()?;
            self.preview.as_mut().unwrap().push_record(record)?;
            self.completed += 1;
            if start.elapsed().as_millis() >= 4 {
                return Ok(None);
            }
        }
        Ok(Some(self.preview.take().unwrap().finish()))
    }

    #[cfg(any(test, target_arch = "wasm32"))]
    fn record(&mut self) -> Result<Record, String> {
        self.sample()?.build()
    }

    /// Sampling remains serial so random streams and existing seed outputs stay
    /// unchanged. Only these small requests are queued; source datasets are shared.
    fn sample(&mut self) -> Result<Request, String> {
        let shape = match (&self.settings, &mut self.sampler) {
            (Settings::Gdp(c), Sampler::Demand { weights, parents }) => {
                let data = if c.uniform {
                    None
                } else {
                    self.dataset.as_deref()
                };
                let anchor = if parents.is_empty()
                    || self.locations.random::<f64>() >= c.hotspot_fraction
                {
                    direct(data, weights.as_ref(), &mut self.locations)
                } else {
                    let parent = parents[self.locations.random_range(0..parents.len())];
                    let mut anchor = None;
                    for _ in 0..256 {
                        let radial = (-2.0 * (1.0 - self.locations.random::<f64>()).ln()).sqrt()
                            * c.hotspot_spread;
                        let angle = self.locations.random::<f64>() * 2.0 * PI;
                        let p = geometry::geographic(
                            Coord {
                                x: radial * angle.cos(),
                                y: radial * angle.sin(),
                            },
                            parent,
                        );
                        let p = [(p[0] + 180.0).rem_euclid(360.0) - 180.0, p[1]];
                        let eligible = match data {
                            Some(Dataset::Demand(d)) => {
                                let index = ((1.0 - p[1].to_radians().sin()) / 2.0 * d.rows as f64)
                                    .floor()
                                    .min((d.rows - 1) as f64)
                                    as usize
                                    * d.columns
                                    + (((p[0] + 180.0).rem_euclid(360.0)) / 360.0
                                        * d.columns as f64)
                                        .floor() as usize;
                                d.eligible.as_ref().is_none_or(|m| data::bit(m, index))
                            }
                            _ => true,
                        };
                        if eligible {
                            anchor = Some(p);
                            break;
                        }
                    }
                    anchor.ok_or(
                        "Hotspot jitter cannot reach eligible cells; reduce the hotspot spread",
                    )?
                };
                let area = draw_size(&mut self.shapes, c.mean_area, c.size_ratio);
                let aspect = ((2.0 * self.shapes.random::<f64>() - 1.0) * c.max_aspect.ln()).exp();
                let oriented = c.shape == "oriented_rectangle";
                let bearing = if oriented {
                    self.rotations.random::<f64>() * PI
                } else {
                    0.0
                };
                Shape::Rectangle {
                    anchor,
                    area,
                    aspect,
                    oriented,
                    bearing: if c.random_rotation { bearing } else { 0.0 },
                }
            }
            (
                Settings::Environment(c),
                Sampler::Forest {
                    weights,
                    area,
                    columns,
                },
            ) => {
                let Some(Dataset::Forest(d)) = self.dataset.as_deref() else {
                    unreachable!()
                };
                let row = weights.sample(&mut self.locations);
                let xs = columns.entry(row).or_insert_with(|| {
                    (0..d.columns)
                        .filter(|&x| data::bit(&d.mask, row * d.columns + x))
                        .collect()
                });
                let col = xs[self.locations.random_range(0..xs.len())];
                let step = 360.0 / d.columns as f64;
                let anchor = [
                    (col as f64 + 0.5) * step - 180.0,
                    90.0 - (row as f64 + 0.5) * step,
                ];
                let requested = draw_size(&mut self.shapes, c.mean_area, c.size_ratio);
                Shape::Forest {
                    settings: c.clone(),
                    dataset: Arc::clone(self.dataset.as_ref().unwrap()),
                    areas: Arc::clone(area),
                    anchor,
                    cell: (row, col),
                    requested,
                    seed: self.shapes.random(),
                }
            }
            (Settings::Environment(c), Sampler::Network { weights, portions }) => {
                let index = weights.sample(&mut self.locations);
                let allowed = &portions[index];
                let mut distance =
                    self.locations.random::<f64>() * allowed.iter().map(|p| p.1).sum::<f64>();
                let mut offset = 0.0;
                for &(start, length) in allowed {
                    if distance < length {
                        offset = start + distance;
                        break;
                    }
                    distance -= length;
                }
                let requested = draw_size(&mut self.shapes, c.mean_length, c.length_ratio);
                Shape::Network {
                    settings: c.clone(),
                    dataset: Arc::clone(self.dataset.as_ref().unwrap()),
                    index,
                    offset,
                    requested,
                }
            }
            _ => unreachable!(),
        };
        Ok(Request {
            index: self.completed,
            grid: self.grid,
            shape,
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn run_parallel(
        mut self,
        workers: usize,
        mut progress: impl FnMut(&'static str, usize, usize),
    ) -> Result<Preview, String> {
        let total = self.settings.count();
        let mut requests = Vec::with_capacity(total);
        progress("Sampling AOI locations", 0, total);
        while self.completed < total {
            requests.push(self.sample()?);
            self.completed += 1;
            if self.completed % 256 == 0 || self.completed == total {
                progress("Sampling AOI locations", self.completed, total);
            }
        }
        let prepared = leo_capture::parallel::map(
            requests,
            workers,
            |request| {
                let index = request.index;
                Preview::prepare_record(request.build()?, index)
            },
            |done, total| progress("Building AOI geometry", done, total),
        )?;
        let mut preview = self.preview.take().unwrap();
        for aoi in prepared {
            preview.push_prepared(aoi);
        }
        progress("Preparing AOI display buffers", total, total);
        Ok(preview.finish())
    }
}

struct Request {
    index: usize,
    grid: Grid,
    shape: Shape,
}
enum Shape {
    Rectangle {
        anchor: [f64; 2],
        area: f64,
        aspect: f64,
        oriented: bool,
        bearing: f64,
    },
    Forest {
        settings: Environment,
        dataset: Arc<Dataset>,
        areas: Arc<Vec<f64>>,
        anchor: [f64; 2],
        cell: (usize, usize),
        requested: f64,
        seed: u64,
    },
    Network {
        settings: Environment,
        dataset: Arc<Dataset>,
        index: usize,
        offset: f64,
        requested: f64,
    },
}
impl Request {
    fn build(self) -> Result<Record, String> {
        let Self { index, grid, shape } = self;
        let (anchor, rects, kind, path) = match shape {
            Shape::Rectangle {
                anchor,
                area,
                aspect,
                oriented,
                bearing,
            } => {
                if oriented {
                    return Ok(Record {
                        reduction: None,
                        id: format!("aoi-{index:08}"),
                        anchor_lon: anchor[0],
                        anchor_lat: anchor[1],
                        area_km2: area,
                        geometry: geometry::oriented_rectangle(anchor, area, aspect, bearing)?,
                        source_kind: "gdp".into(),
                        path_length_km: None,
                        path_truncated: false,
                    });
                }
                let tiles = (area / grid.area()).round().max(1.0) as i64;
                (anchor, grid.rectangle(anchor, tiles, aspect)?, "gdp", None)
            }
            Shape::Forest {
                settings,
                dataset,
                areas,
                anchor,
                cell,
                requested,
                seed,
            } => {
                let Dataset::Forest(d) = dataset.as_ref() else {
                    unreachable!()
                };
                let mut morphology = ChaCha8Rng::seed_from_u64(seed);
                let polygon = forest_patch(
                    d,
                    &settings,
                    anchor,
                    cell,
                    &areas,
                    requested,
                    &mut morphology,
                )?;
                let projected = polygon.map_coords(|p| grid.project([p.x, p.y], anchor[0]));
                (anchor, grid.cover(&projected)?, "forest", None)
            }
            Shape::Network {
                settings: c,
                dataset,
                index: network_index,
                offset,
                requested,
            } => {
                let network = match dataset.as_ref() {
                    Dataset::River(n) | Dataset::Coast(n) => n,
                    _ => unreachable!(),
                };
                let path = follow(network, network_index, offset, requested, c.kind == "coast")?;
                let length: f64 = path
                    .windows(2)
                    .map(|w| geometry::distance(w[0], w[1]))
                    .sum();
                let anchor = [(path[0][0] + 180.0).rem_euclid(360.0) - 180.0, path[0][1]];
                let line =
                    LineString::new(path.iter().map(|&p| geometry::local(p, anchor)).collect());
                let footprint = if c.kind == "coast" {
                    coast_buffer(&line, c.inland, c.offshore)
                } else {
                    line.buffer(c.buffer)
                };
                let projected = footprint
                    .map_coords(|p| grid.project(geometry::geographic(p, anchor), anchor[0]));
                (
                    anchor,
                    grid.cover(&projected)?,
                    c.kind,
                    Some((length, length < requested * 0.99)),
                )
            }
        };
        let area = rects
            .iter()
            .map(|r| (r.width * r.height) as f64)
            .sum::<f64>()
            * grid.area();
        let prefix = if kind == "gdp" { "aoi" } else { kind };
        super::compact::record(Record {
            reduction: None,
            id: format!("{prefix}-{:08}", index),
            anchor_lon: anchor[0],
            anchor_lat: anchor[1],
            area_km2: area,
            geometry: grid.geometry(&rects),
            source_kind: kind.into(),
            path_length_km: path.map(|p| p.0),
            path_truncated: path.is_some_and(|p| p.1),
        })
    }
}

fn network_sampler(d: &data::Network, c: &Environment) -> Result<Sampler, String> {
    let portions: Vec<_> = (0..d.length.len())
        .map(|i| {
            if c.kind == "river" && d.flow[i] < c.flow {
                Vec::new()
            } else {
                portions(d.points_for(i), c.latitude)
            }
        })
        .collect();
    let weights = Weights::new(portions.iter().map(|p| p.iter().map(|p| p.1).sum()))?;
    Ok(Sampler::Network { weights, portions })
}

#[derive(PartialEq)]
struct Candidate {
    priority: f64,
    row: usize,
    col: usize,
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .priority
            .total_cmp(&self.priority)
            .then_with(|| other.row.cmp(&self.row))
            .then_with(|| other.col.cmp(&self.col))
    }
}

fn forest_patch(
    d: &data::Forest,
    c: &Environment,
    anchor: [f64; 2],
    (row, col): (usize, usize),
    areas: &[f64],
    requested: f64,
    rng: &mut ChaCha8Rng,
) -> Result<MultiPolygon<f64>, String> {
    let step = 360.0 / d.columns as f64;
    let dy = RADIUS * step.to_radians();
    let dx = dy * anchor[1].to_radians().cos();
    let radius = (requested / PI).sqrt().max(dy);
    let stretch = (rng.random::<f64>() * c.max_elongation.ln() * 0.5).exp();
    let (sine, cosine) = (rng.random::<f64>() * PI).sin_cos();
    let strength = c.outline_variation * rng.random_range(0.65..1.0);
    let waves: [(f64, f64, f64); 6] = std::array::from_fn(|_| {
        let angle = rng.random::<f64>() * 2.0 * PI;
        let frequency = rng.random_range(2.0..6.0);
        (
            frequency * angle.cos(),
            frequency * angle.sin(),
            rng.random::<f64>() * 2.0 * PI,
        )
    });
    let nonforest_penalty = 3.0 * c.forest_adherence / (1.0 - 0.7 * c.forest_adherence);
    let mut pending = BinaryHeap::from([Candidate {
        priority: 0.0,
        row,
        col,
    }]);
    let mut seen = HashSet::from([(row, col)]);
    let mut selected: BTreeMap<usize, Vec<i64>> = BTreeMap::new();
    let (mut area, mut count) = (0.0, 0);
    while area < requested {
        let Some(cell) = pending.pop() else { break };
        let x = col as i64
            + (cell.col as i64 - col as i64 + d.columns as i64 / 2).rem_euclid(d.columns as i64)
            - d.columns as i64 / 2;
        selected.entry(cell.row).or_default().push(x);
        area += areas[cell.row];
        count += 1;
        if count > 50_000 {
            return Err("Forest patch exceeds 50,000 sampling cells; reduce its area".into());
        }
        for (yy, xx) in [
            (cell.row as i64 - 1, cell.col),
            (cell.row as i64 + 1, cell.col),
            (cell.row as i64, (cell.col + d.columns - 1) % d.columns),
            (cell.row as i64, (cell.col + 1) % d.columns),
        ] {
            if yy < 0 || yy >= d.rows as i64 || !seen.insert((yy as usize, xx)) {
                continue;
            }
            let east = (xx as i64 - col as i64 + d.columns as i64 / 2).rem_euclid(d.columns as i64)
                - d.columns as i64 / 2;
            let east = east as f64 * dx;
            let north = (row as i64 - yy) as f64 * dy;
            let u = (east * cosine + north * sine) / (radius * stretch);
            let v = (-east * sine + north * cosine) * stretch / radius;
            if u.hypot(v) > 4.0 {
                continue;
            }
            let noise = waves
                .iter()
                .map(|(x, y, phase)| (x * u + y * v + phase).sin())
                .sum::<f64>()
                / 3.0_f64.sqrt();
            let penalty = if data::bit(&d.mask, yy as usize * d.columns + xx) {
                0.0
            } else {
                nonforest_penalty
            };
            pending.push(Candidate {
                priority: u.hypot(v) * (strength * noise.clamp(-1.5, 1.5)).exp() + penalty,
                row: yy as usize,
                col: xx,
            });
        }
    }
    if area < requested {
        return Err(
            "Forest growth exhausted its bounded region before reaching the target area".into(),
        );
    }
    let mut pieces = Vec::new();
    for (y, mut xs) in selected {
        xs.sort_unstable();
        let (mut start, mut previous) = (xs[0], xs[0]);
        for x in xs
            .iter()
            .copied()
            .skip(1)
            .chain(std::iter::once(xs[xs.len() - 1] + 2))
        {
            if x != previous + 1 {
                pieces.push(geometry::rectangle(
                    start as f64,
                    y as f64,
                    (previous + 1) as f64,
                    (y + 1) as f64,
                ));
                start = x;
            }
            previous = x;
        }
    }
    let patch = unary_union(&pieces);
    Ok(patch.map_coords(|p| Coord {
        x: p.x * step - 180.0,
        y: 90.0 - p.y * step,
    }))
}

fn follow(
    d: &data::Network,
    index: usize,
    initial_offset: f64,
    requested: f64,
    closed: bool,
) -> Result<Vec<[f64; 2]>, String> {
    let initial = index;
    let mut index = Some(index);
    let (mut offset, mut remaining) = (initial_offset, requested);
    let mut seen = HashSet::new();
    let mut path: Vec<[f64; 2]> = Vec::new();
    while let Some(i) = index {
        let repeat = !seen.insert(i);
        if repeat && (!closed || i != initial) {
            break;
        }
        let mut available = if repeat {
            initial_offset.min(remaining)
        } else {
            remaining
        };
        if available <= 1e-6 {
            break;
        }
        for pair in d.points_for(i).windows(2) {
            let length = geometry::distance(pair[0], pair[1]);
            if length <= 0.0 {
                continue;
            }
            if offset >= length {
                offset -= length;
                continue;
            }
            let a = pair[0];
            let b = [
                a[0] + (pair[1][0] - a[0] + 180.0).rem_euclid(360.0) - 180.0,
                pair[1][1],
            ];
            let use_length = available.min(length - offset);
            let lerp = |t: f64| {
                [
                    a[0] + (b[0] - a[0]) * t / length,
                    a[1] + (b[1] - a[1]) * t / length,
                ]
            };
            let (mut start, mut end) = (lerp(offset), lerp(offset + use_length));
            if let Some(&last) = path.last() {
                let shift = 360.0 * ((last[0] - start[0]) / 360.0).round();
                start[0] += shift;
                end[0] += shift;
                if geometry::distance(last, start) > 0.5 {
                    return Ok(path);
                }
            } else {
                path.push(start);
            }
            path.push(end);
            remaining -= use_length;
            available -= use_length;
            offset = 0.0;
            if available <= 1e-6 {
                break;
            }
        }
        if repeat || remaining <= 1e-6 {
            break;
        }
        index = usize::try_from(d.next_index[i]).ok();
    }
    if path.len() < 2 {
        Err("Could not form a geographic corridor".into())
    } else {
        Ok(path)
    }
}

fn coast_buffer(line: &LineString<f64>, inland: f64, offshore: f64) -> MultiPolygon<f64> {
    let mut pieces = Vec::new();
    let mut previous: Option<(Coord<f64>, f64)> = None;
    for edge in line.0.windows(2) {
        let dx = edge[1].x - edge[0].x;
        let dy = edge[1].y - edge[0].y;
        let length = dx.hypot(dy);
        if length <= 1e-9 {
            continue;
        }
        let normal = Coord {
            x: -dy / length,
            y: dx / length,
        };
        let offset = |p: Coord<f64>, distance: f64| Coord {
            x: p.x + normal.x * distance,
            y: p.y + normal.y * distance,
        };
        pieces.push(
            Polygon::new(
                LineString(vec![
                    offset(edge[0], inland),
                    offset(edge[0], -offshore),
                    offset(edge[1], -offshore),
                    offset(edge[1], inland),
                    offset(edge[0], inland),
                ]),
                vec![],
            )
            .orient(Direction::Default),
        );
        let angle = normal.y.atan2(normal.x);
        if let Some((point, old)) = previous {
            let turn = (angle - old + PI).rem_euclid(2.0 * PI) - PI;
            let (start, radius) = if turn > 0.0 {
                (old + PI, offshore)
            } else {
                (old, inland)
            };
            let steps = ((turn.abs() / 0.2).ceil() as usize).max(1);
            let mut ring = vec![point];
            for i in 0..=steps {
                let theta = start + turn * i as f64 / steps as f64;
                ring.push(Coord {
                    x: point.x + theta.cos() * radius,
                    y: point.y + theta.sin() * radius,
                });
            }
            ring.push(point);
            pieces.push(Polygon::new(LineString(ring), vec![]).orient(Direction::Default));
        }
        previous = Some((edge[1], angle));
    }
    unary_union(&pieces)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn gdp() -> Gdp {
        Gdp {
            count: 300,
            seed: 42,
            mean_area: 1000.0,
            size_ratio: 100.0,
            uniform: true,
            hotspot_count: 0,
            hotspot_spread: 100.0,
            hotspot_fraction: 0.4,
            shape: "rectangle".into(),
            random_rotation: true,
            max_aspect: 2.0,
        }
    }
    fn environment(kind: &'static str) -> Environment {
        Environment {
            kind,
            count: 5,
            seed: 42,
            mean_area: 1000.0,
            size_ratio: 100.0,
            mean_length: 100.0,
            length_ratio: 20.0,
            buffer: 2.0,
            inland: 2.0,
            offshore: 1.0,
            flow: 10.0,
            max_elongation: 3.0,
            outline_variation: 0.7,
            forest_adherence: 0.8,
            latitude: 70.0,
            tile_edge: 0.1,
        }
    }
    fn run(settings: Settings, data: Option<Arc<Dataset>>) -> Preview {
        let mut job = Job::new(settings, data).unwrap();
        loop {
            if let Some(p) = job.step().unwrap() {
                return p;
            }
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn parallel_workers_preserve_seeded_geometry_metadata_and_render_buffers() {
        let mut cases = vec![];
        for shape in ["rectangle", "oriented_rectangle"] {
            cases.push((
                Settings::Gdp(Gdp {
                    count: 24,
                    hotspot_count: 4,
                    shape: shape.into(),
                    ..gdp()
                }),
                None,
            ));
        }
        for kind in ["forest", "river", "coast"] {
            let dataset =
                Arc::new(Dataset::decode(data::fixture(kind).bytes.as_ref().unwrap()).unwrap());
            cases.push((
                Settings::Environment(Environment {
                    count: 12,
                    ..environment(kind)
                }),
                Some(dataset),
            ));
        }
        for (settings, dataset) in cases {
            let serial = run(settings.clone(), dataset.clone());
            for workers in [1, 2, 4] {
                let mut progress = Vec::new();
                let parallel = Job::new(settings.clone(), dataset.clone())
                    .unwrap()
                    .run_parallel(workers, |stage, done, total| {
                        if stage == "Building AOI geometry" {
                            progress.push(done);
                        }
                        assert_eq!(total, settings.count());
                    })
                    .unwrap();
                assert_eq!(serial.aois, parallel.aois);
                assert_eq!(serial.mean_area, parallel.mean_area);
                assert_eq!(serial.mean_path_length, parallel.mean_path_length);
                assert_eq!(serial.original_edges, parallel.original_edges);
                for i in 0..serial.count {
                    assert_eq!(
                        serde_json::to_vec(&serial.record(i)).unwrap(),
                        serde_json::to_vec(&parallel.record(i)).unwrap()
                    );
                }
                let (a, b) = (serial.gpu_geometry(), parallel.gpu_geometry());
                assert_eq!(
                    bytemuck::cast_slice::<_, u8>(&a.segments),
                    bytemuck::cast_slice::<_, u8>(&b.segments)
                );
                assert_eq!(
                    bytemuck::cast_slice::<_, u8>(&a.triangles),
                    bytemuck::cast_slice::<_, u8>(&b.triangles)
                );
                assert_eq!(
                    bytemuck::cast_slice::<_, u8>(&a.dots),
                    bytemuck::cast_slice::<_, u8>(&b.dots)
                );
                assert_eq!(progress.first(), Some(&0));
                assert_eq!(progress.last(), Some(&serial.count));
                assert!(progress.windows(2).all(|w| w[0] <= w[1]));
            }
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    #[ignore = "CPU scaling benchmark with bundled datasets; generates in memory"]
    fn benchmark_parallel_generation() {
        let mut cases = vec![(
            Settings::Gdp(Gdp {
                count: 2000,
                shape: "oriented_rectangle".into(),
                ..gdp()
            }),
            None,
        )];
        for kind in ["forest", "river", "coast"] {
            cases.push((
                Settings::Environment(Environment {
                    count: 128,
                    ..environment(kind)
                }),
                Some(data::load(&data::Input::default(), kind).unwrap()),
            ));
        }
        for (settings, dataset) in cases {
            let mut reference = None;
            for workers in [1, leo_capture::parallel::worker_count(settings.count())] {
                let start = std::time::Instant::now();
                let preview = Job::new(settings.clone(), dataset.clone())
                    .unwrap()
                    .run_parallel(workers, |_, _, _| {})
                    .unwrap();
                eprintln!(
                    "{} {} AOIs · {workers} workers: {:?}",
                    settings.count(),
                    settings.source_kind(),
                    start.elapsed()
                );
                if let Some(original) = &reference {
                    assert_eq!(original, &preview.aois);
                }
                reference = Some(preview.aois);
            }
        }
    }
    #[test]
    fn oriented_rotation_preserves_locations_sizes_and_seeded_prefix() {
        let settings = Gdp {
            count: 40,
            shape: "oriented_rectangle".into(),
            max_aspect: 3.0,
            ..gdp()
        };
        let a = run(Settings::Gdp(settings.clone()), None);
        let b = run(
            Settings::Gdp(Gdp {
                count: 50,
                ..settings.clone()
            }),
            None,
        );
        let aligned = run(
            Settings::Gdp(Gdp {
                random_rotation: false,
                ..settings
            }),
            None,
        );
        for ((a, b), aligned) in a.aois.iter().zip(&b.aois).zip(&aligned.aois) {
            assert_eq!(a.geometry_hash, b.geometry_hash);
            assert_eq!(
                (a.longitude, a.latitude, a.area_km2),
                (aligned.longitude, aligned.latitude, aligned.area_km2)
            );
            assert_ne!(a.geometry_hash, aligned.geometry_hash);
        }
    }

    #[test]
    fn forest_shape_controls_preserve_clearings_and_connected_area() {
        use geo::{Area, ConvexHull, Validation};
        let mut forest = data::Forest {
            rows: 1800,
            columns: 3600,
            mask: vec![255; 1800 * 3600 / 8],
        };
        for y in 897..903 {
            for x in 1805..1811 {
                let index = y * forest.columns + x;
                forest.mask[index / 8] &= !(1 << (index % 8));
            }
        }
        let step = 0.1_f64;
        let areas: Vec<_> = (0..forest.rows)
            .map(|r| {
                RADIUS.powi(2)
                    * step.to_radians()
                    * ((90.0 - r as f64 * step).to_radians().sin()
                        - (90.0 - (r + 1) as f64 * step).to_radians().sin())
            })
            .collect();
        let settings = Environment {
            max_elongation: 1.0,
            outline_variation: 0.0,
            forest_adherence: 1.0,
            ..environment("forest")
        };
        let sample = |settings: &Environment| {
            forest_patch(
                &forest,
                settings,
                [0.05, -0.05],
                (900, 1800),
                &areas,
                100_000.0,
                &mut ChaCha8Rng::seed_from_u64(42),
            )
            .unwrap()
        };
        let regular = sample(&settings);
        assert_eq!(regular.0.len(), 1);
        assert!(regular.is_valid());
        assert!(
            !regular.0[0].interiors().is_empty(),
            "nonforest clearings must be retained"
        );
        let varied = sample(&Environment {
            outline_variation: 1.0,
            max_elongation: 4.0,
            ..settings
        });
        assert_eq!(varied.0.len(), 1);
        assert!(varied.is_valid());
        assert_ne!(varied, regular);
        assert!(varied.unsigned_area() / varied.convex_hull().unsigned_area() < 0.9);
        assert!((varied.unsigned_area() / regular.unsigned_area() - 1.0).abs() < 0.01);
    }
    #[test]
    fn gdp_generation_is_seeded_and_keeps_area_bounds() {
        let settings = gdp();
        let a = run(Settings::Gdp(settings.clone()), None);
        let b = run(Settings::Gdp(settings.clone()), None);
        assert_eq!(a.count, 300);
        assert!(a
            .aois
            .iter()
            .zip(&b.aois)
            .all(|(a, b)| a.geometry_hash == b.geometry_hash));
        let (lo, hi) = bounds(settings.mean_area, settings.size_ratio).unwrap();
        assert!(
            a.min_area >= lo - 0.1 * hi.sqrt() - 0.01 && a.max_area <= hi + 0.1 * hi.sqrt() + 0.01
        );
        let c = run(
            Settings::Gdp(Gdp {
                seed: 43,
                ..settings
            }),
            None,
        );
        assert_ne!(a.aois[0].geometry_hash, c.aois[0].geometry_hash);
    }
    #[test]
    fn random_streams_keep_locations_independent_of_area() {
        let a = run(Settings::Gdp(gdp()), None);
        let b = run(
            Settings::Gdp(Gdp {
                mean_area: 2000.0,
                ..gdp()
            }),
            None,
        );
        assert!(a
            .aois
            .iter()
            .zip(&b.aois)
            .all(|(a, b)| a.longitude == b.longitude && a.latitude == b.latitude));
    }
    #[test]
    fn uniform_globe_is_uniform_in_ground_area() {
        let preview = run(
            Settings::Gdp(Gdp {
                count: 4000,
                ..gdp()
            }),
            None,
        );
        let equatorial = preview
            .aois
            .iter()
            .filter(|a| a.latitude.abs() < 30.0)
            .count();
        assert!((1800..2200).contains(&equatorial));
    }
    #[test]
    fn all_sources_work_from_portable_bytes_without_files_or_python() {
        for kind in ["forest", "river", "coast"] {
            let input = data::fixture(kind);
            let data = Arc::new(Dataset::decode(input.bytes.as_ref().unwrap()).unwrap());
            let result = run(Settings::Environment(environment(kind)), Some(data));
            assert_eq!(result.count, 5);
            assert!(result.max_edges <= super::super::compact::EDGE_LIMIT);
            assert!(result
                .aois
                .iter()
                .all(|a| a.source == kind && a.area_km2 > 0.0));
            assert_eq!(result.mean_path_length.is_some(), kind != "forest");
        }
    }
    #[test]
    fn source_kind_filters_and_invalid_settings_return_errors() {
        assert!(Job::new(Settings::Gdp(Gdp { count: 0, ..gdp() }), None).is_err());
        assert!(Job::new(
            Settings::Gdp(Gdp {
                mean_area: f64::NAN,
                ..gdp()
            }),
            None
        )
        .is_err());
        assert!(Job::new(
            Settings::Gdp(Gdp {
                uniform: false,
                ..gdp()
            }),
            None
        )
        .is_err());
        let input = data::fixture("river");
        let data = Arc::new(Dataset::decode(input.bytes.as_ref().unwrap()).unwrap());
        assert!(Job::new(
            Settings::Environment(environment("forest")),
            Some(data.clone())
        )
        .is_err());
        assert!(Job::new(
            Settings::Environment(Environment {
                flow: 1000.0,
                ..environment("river")
            }),
            Some(data)
        )
        .is_err());
    }
    #[test]
    fn network_anchor_eligibility_clips_crossing_latitude_band() {
        let p = portions(&[[0.0, -80.0], [0.0, 80.0]], 10.0);
        assert_eq!(p.len(), 1);
        assert!((p[0].1 - geometry::distance([0.0, -10.0], [0.0, 10.0])).abs() < 1e-8);
        assert!(portions(&[[0.0, 80.0], [1.0, 80.0]], 10.0).is_empty());
    }
    #[test]
    fn corridors_follow_connections_but_never_bridge_gaps() {
        let input = data::fixture("river");
        let Dataset::River(mut d) = Dataset::decode(input.bytes.as_ref().unwrap()).unwrap() else {
            panic!()
        };
        let p = follow(&d, 0, 0.0, 150.0, false).unwrap();
        assert!(p.last().unwrap()[0] > 1.0);
        d.points[2][0] = 10.0;
        d.points[3][0] = 11.0;
        let p = follow(&d, 0, 0.0, 150.0, false).unwrap();
        assert_eq!(p.last().unwrap()[0], 1.0);
    }
    #[test]
    fn closed_coastline_stops_at_sampled_start_after_one_circuit() {
        let d = data::Network {
            points: vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]],
            offsets: vec![0, 4],
            next_index: vec![0],
            length: vec![400.0],
            ids: vec![1],
            flow: vec![0.0],
        };
        let p = follow(&d, 0, 20.0, 1000.0, true).unwrap();
        assert!(geometry::distance(p[0], *p.last().unwrap()) < 1e-8);
        assert!(
            p.windows(2)
                .map(|w| geometry::distance(w[0], w[1]))
                .sum::<f64>()
                < 400.0
        );
    }
    #[test]
    fn coast_buffers_respect_land_on_left_and_water_on_right() {
        use geo::BoundingRect;
        let polygon = coast_buffer(&LineString::from(vec![(0.0, 0.0), (10.0, 0.0)]), 3.0, 1.0);
        let bounds = polygon.bounding_rect().unwrap();
        assert!((bounds.min().y + 1.0).abs() < 1e-5);
        assert!((bounds.max().y - 3.0).abs() < 1e-5);
    }
    #[test]
    fn weighted_sampling_never_selects_zero_weight_cells() {
        let weights = Weights::new([0.0, 1.0, 0.0, 0.0]).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        for _ in 0..100 {
            assert_eq!(weights.sample(&mut rng), 1);
        }
    }
}
