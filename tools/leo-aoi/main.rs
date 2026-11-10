//! Convert existing prepared geographic caches without a Python runtime.
use base64::{engine::general_purpose::STANDARD, Engine};
use npyz::npz::NpzArchive;
use serde_json::{json, Value};
use std::{error::Error, fs::File, io::BufReader, path::Path};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
type Archive = NpzArchive<BufReader<File>>;

fn array<T: npyz::Deserialize>(archive: &mut Archive, name: &str) -> Result<(Vec<u64>, Vec<T>)> {
    let npy = archive
        .by_name(name)?
        .ok_or_else(|| format!("Missing array: {name}"))?;
    if npy.order() != npyz::Order::C || npy.len() > 64_000_000 {
        return Err(format!("Invalid ordering or oversized array: {name}").into());
    }
    let shape = npy.shape().to_vec();
    Ok((shape, npy.into_vec::<T>()?))
}

fn grid(shape: &[u64]) -> Result<(u64, u64)> {
    match shape {
        &[rows, columns] if rows > 0 && columns > 0 => Ok((columns, rows)),
        _ => Err("Expected a nonempty two-dimensional grid".into()),
    }
}

fn bits(values: impl ExactSizeIterator<Item = bool>) -> String {
    let mut packed = vec![0u8; values.len().div_ceil(8)];
    for (i, value) in values.enumerate() {
        if value {
            packed[i / 8] |= 1 << (i % 8);
        }
    }
    STANDARD.encode(packed)
}

fn packed<const N: usize, T>(values: &[T], bytes: impl Fn(&T) -> [u8; N]) -> String {
    STANDARD.encode(values.iter().flat_map(bytes).collect::<Vec<_>>())
}

fn document(source: &Path) -> Result<Value> {
    let mut archive = NpzArchive::open(source)?;
    let (_, metadata) = array::<String>(&mut archive, "metadata")?;
    if metadata.len() != 1 {
        return Err("Expected one metadata document".into());
    }
    let metadata: Value = serde_json::from_str(&metadata[0])?;
    if metadata["schema_version"] != 1 {
        return Err("Unsupported cache schema".into());
    }
    let mut doc = json!({"schema_version": 1, "metadata": metadata});
    let demand = archive.array_names().any(|name| name == "probability");
    if demand {
        let (shape, weights) = array::<f64>(&mut archive, "probability")?;
        let (columns, rows) = grid(&shape)?;
        if columns * rows > 20_000_000 {
            return Err("Demand grid exceeds app limits".into());
        }
        let (eligible_shape, eligible) = array::<bool>(&mut archive, "eligible")?;
        if eligible_shape != shape {
            return Err("Eligibility and probability grids differ".into());
        }
        let mut cells = Vec::new();
        for (i, (&weight, &allowed)) in weights.iter().zip(&eligible).enumerate() {
            if !weight.is_finite() || weight < 0.0 || (weight > 0.0 && !allowed) {
                return Err(format!("Invalid demand weight at cell {i}").into());
            }
            if weight > 0.0 {
                cells.push((i, weight));
            }
        }
        if cells.is_empty() {
            return Err("Demand grid has no positive weights".into());
        }
        doc["kind"] = json!("demand");
        doc["columns"] = json!(columns);
        doc["rows"] = json!(rows);
        doc["cells"] = json!(cells);
        if eligible.iter().any(|&v| !v) {
            doc["eligible"] = json!(bits(eligible.into_iter()));
        }
    } else if doc["metadata"]["kind"] == "forest" {
        let (shape, mask) = array::<u8>(&mut archive, "mask")?;
        let (columns, rows) = grid(&shape)?;
        if columns != rows * 2 || !mask.iter().any(|&v| v != 0) || mask.iter().any(|&v| v > 1) {
            return Err("Expected a nonempty global forest mask containing only 0 or 1".into());
        }
        doc["kind"] = json!("forest");
        doc["columns"] = json!(columns);
        doc["rows"] = json!(rows);
        doc["mask"] = json!(bits(mask.into_iter().map(|v| v != 0)));
    } else {
        let kind = doc["metadata"]["kind"]
            .as_str()
            .ok_or("Missing geographic kind")?
            .to_owned();
        if kind != "river" && kind != "coast" {
            return Err("Unknown geographic cache kind".into());
        }
        let (shape, points) = array::<f32>(&mut archive, "points")?;
        if shape.len() != 2
            || shape[1] != 2
            || points.chunks_exact(2).any(|p| {
                !p[0].is_finite()
                    || !p[1].is_finite()
                    || p[1].abs() > 90.0
                    || p[0].abs() > 1_000_000.0
            })
        {
            return Err("Invalid longitude/latitude points".into());
        }
        let (_, offsets) = array::<i64>(&mut archive, "offsets")?;
        let (_, next) = array::<i32>(&mut archive, "next_index")?;
        let (_, length) = array::<f64>(&mut archive, "length")?;
        let (_, ids) = array::<i64>(&mut archive, "ids")?;
        let (_, flow) = array::<f32>(&mut archive, "flow")?;
        let count = length.len();
        if count == 0
            || offsets.len() != count + 1
            || next.len() != count
            || ids.len() != count
            || flow.len() != count
            || offsets[0] != 0
            || offsets[count] != (points.len() / 2) as i64
            || offsets.windows(2).any(|w| w[1] < w[0].saturating_add(2))
            || next.iter().any(|&i| i < -1 || i as i64 >= count as i64)
            || length.iter().any(|v| !v.is_finite() || *v <= 0.0)
            || flow.iter().any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err("Invalid network arrays or connections".into());
        }
        doc["kind"] = json!(kind);
        doc["network"] = json!({
            "points": packed(&points, |v| v.to_le_bytes()),
            "offsets": packed(&offsets, |v| (*v as u64).to_le_bytes()),
            "next_index": packed(&next, |v| v.to_le_bytes()),
            "length": packed(&length, |v| v.to_le_bytes()),
            "ids": packed(&ids, |v| v.to_le_bytes()),
            "flow": packed(&flow, |v| v.to_le_bytes()),
        });
    }
    Ok(doc)
}

fn export(source: &Path, destination: &Path) -> Result<()> {
    let doc = document(source)?;
    let output = File::create_new(destination)?;
    let mut gzip = flate2::GzBuilder::new()
        .mtime(0)
        .write(output, flate2::Compression::best());
    serde_json::to_writer(&mut gzip, &doc)?;
    gzip.finish()?;
    println!("Exported {} to {}", doc["kind"], destination.display());
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() == 1 && (args[0] == "--help" || args[0] == "-h") {
        println!("leo-aoi-data SOURCE.npz DESTINATION.aoi-data.json.gz\nConverts prepared caches in Rust. Refuses to overwrite existing files.");
        return Ok(());
    }
    if args.len() != 2 {
        return Err("Usage: leo-aoi-data SOURCE.npz DESTINATION.aoi-data.json.gz".into());
    }
    export(Path::new(&args[0]), Path::new(&args[1]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn masks_use_little_bit_order_and_padding() {
        let values = [true, false, true, false, false, false, false, true, true];
        assert_eq!(
            STANDARD.decode(bits(values.into_iter())).unwrap(),
            vec![133, 1]
        );
    }

    #[test]
    fn arrays_use_little_endian_bytes() {
        assert_eq!(
            STANDARD
                .decode(packed(&[1u64, 256], |v| v.to_le_bytes()))
                .unwrap(),
            vec![1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        );
    }

    #[test]
    fn rejects_invalid_grid_shapes() {
        for shape in [vec![], vec![1], vec![1, 2, 3], vec![0, 1]] {
            assert!(grid(&shape).is_err());
        }
        assert_eq!(grid(&[360, 720]).unwrap(), (720, 360));
    }
}
