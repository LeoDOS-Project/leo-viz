//! Versioned compressed job specifications; geography remains in AOI datasets.
use super::workload::{Generator, Job};
use flate2::{read::GzDecoder, write::GzEncoder, Compression};
use leo_sim::acquisition::CaptureProfile;
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};

#[derive(Serialize, Deserialize)]
pub struct Document {
    pub schema_version: u32,
    pub generator: Generator,
    pub capture: CaptureProfile,
    pub jobs: Vec<Job>,
}
impl Document {
    pub fn validate(&self) -> Result<(), String> {
        if !(1..=2).contains(&self.schema_version)
            || self.jobs.is_empty()
            || self.jobs.len() > 200_000
        {
            return Err("Unsupported workload version or job count (1–200,000 required).".into());
        }
        self.generator.validate()?;
        let mut ids = std::collections::HashSet::new();
        for job in &self.jobs {
            job.estimate(self.capture)?;
            if !self
                .generator
                .bounds
                .contains(job.cadence_hours, job.samples)
            {
                return Err(format!(
                    "{} is outside the saved global bounds.",
                    job.aoi.key()
                ));
            }
            if !ids.insert(job.aoi.key()) {
                return Err("Duplicate job/AOI identifier in workload.".into());
            }
        }
        Ok(())
    }
    pub fn encode(&self) -> Result<Vec<u8>, String> {
        self.validate()?;
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        serde_json::to_writer(&mut encoder, self).map_err(|e| e.to_string())?;
        encoder.finish().map_err(|e| e.to_string())
    }
    pub fn decode(bytes: &[u8]) -> Result<Self, String> {
        let mut json = Vec::new();
        GzDecoder::new(bytes)
            .take(256 * 1024 * 1024 + 1)
            .read_to_end(&mut json)
            .map_err(|e| e.to_string())?;
        if json.len() > 256 * 1024 * 1024 {
            return Err("Workload exceeds the 256 MB decoded limit.".into());
        }
        let mut document: Self = serde_json::from_slice(&json).map_err(|e| e.to_string())?;
        if document.schema_version == 1 {
            // Legacy workloads had no acquisition semantics. Deterministically
            // add the new defaults without changing existing byte/pipeline draws.
            document.generator.validate()?;
            for job in &mut document.jobs {
                job.acquisition = document.generator.acquisition_for(&job.aoi);
            }
            document.schema_version = 2;
        }
        document.validate()?;
        Ok(document)
    }
    #[cfg(not(target_arch = "wasm32"))]
    pub fn save(&self, path: &std::path::Path) -> Result<(), String> {
        let bytes = self.encode()?;
        let temp = path.with_extension("json.gz.tmp");
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let mut file = std::fs::File::create(&temp).map_err(|e| e.to_string())?;
        file.write_all(&bytes).map_err(|e| e.to_string())?;
        file.sync_all().map_err(|e| e.to_string())?;
        std::fs::rename(temp, path).map_err(|e| e.to_string())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::workload::AoiRef;
    #[test]
    fn legacy_workload_migrates_only_new_acquisition_fields_and_v2_preserves_edits() {
        let generator = Generator::default();
        let jobs = generator
            .generate([AoiRef {
                source: "gdp".into(),
                id: "old".into(),
                area_km2: 10.0,
                longitude: 0.0,
                latitude: 0.0,
                geometry_hash: 42,
            }])
            .unwrap();
        let original = jobs[0].clone();
        let mut json = serde_json::to_value(Document {
            schema_version: 1,
            generator,
            capture: CaptureProfile::default(),
            jobs,
        })
        .unwrap();
        json["generator"]
            .as_object_mut()
            .unwrap()
            .remove("acquisition");
        json["jobs"][0]
            .as_object_mut()
            .unwrap()
            .remove("acquisition");
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        serde_json::to_writer(&mut encoder, &json).unwrap();
        let mut migrated = Document::decode(&encoder.finish().unwrap()).unwrap();
        assert_eq!(migrated.schema_version, 2);
        assert_eq!(migrated.jobs[0], original);
        migrated.jobs[0].acquisition.priority = 999;
        migrated.jobs[0].acquisition.solar_time_hours = 10.25;
        let decoded = Document::decode(&migrated.encode().unwrap()).unwrap();
        assert_eq!(decoded.jobs, migrated.jobs);
    }
    #[test]
    fn compressed_round_trip_preserves_assignments_and_capture_profile() {
        let generator = Generator::default();
        let jobs = generator
            .generate((0..100).map(|i| AoiRef {
                source: "gdp".into(),
                id: i.to_string(),
                area_km2: 1000.0,
                longitude: 12.0,
                latitude: 58.0,
                geometry_hash: i,
            }))
            .unwrap();
        let capture = CaptureProfile::default();
        let document = Document {
            schema_version: 1,
            generator,
            capture,
            jobs,
        };
        let bytes = document.encode().unwrap();
        assert_eq!(&bytes[..2], &[0x1f, 0x8b]);
        let decoded = Document::decode(&bytes).unwrap();
        assert_eq!(decoded.jobs, document.jobs);
        assert_eq!(decoded.generator, document.generator);
        assert_eq!(decoded.capture, document.capture);
        assert!(Document::decode(b"bad archive").is_err());
        let mut duplicate = decoded;
        duplicate.jobs.push(duplicate.jobs[0].clone());
        assert!(duplicate.validate().is_err());
    }
}
