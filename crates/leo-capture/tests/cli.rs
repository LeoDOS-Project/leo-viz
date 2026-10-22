use std::{path::Path, process::Command};

#[test]
fn headless_overlap_report_and_opportunity_stream_agree() {
    let dir = std::env::temp_dir().join(format!("leo-capture-cli-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let report = dir.join("report.json");
    let events = dir.join("events.jsonl");
    let config = Path::new(env!("CARGO_MANIFEST_DIR")).join("scenarios/overlap.json");
    let output = Command::new(env!("CARGO_BIN_EXE_leo-capture"))
        .arg(config)
        .arg(&report)
        .arg(&events)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: serde_json::Value = serde_json::from_slice(&std::fs::read(report).unwrap()).unwrap();
    let r = &value["report"];
    assert!(r["jobs"]
        .as_array()
        .unwrap()
        .iter()
        .all(|j| j["complete_due_updates"] == 1));
    let physical = r["retained_physical_bytes"].as_f64().unwrap();
    let logical = r["fulfilled_logical_bytes"].as_f64().unwrap();
    assert!(physical > 0.0 && (logical / physical - 2.0).abs() < 1e-6);
    let text = std::fs::read_to_string(events).unwrap();
    let mut lines = text.lines();
    let metadata: serde_json::Value = serde_json::from_str(lines.next().unwrap()).unwrap();
    assert_eq!(metadata["type"], "metadata");
    assert_eq!(metadata["workload"]["jobs"].as_array().unwrap().len(), 3);
    let records: Vec<serde_json::Value> = lines.map(|l| serde_json::from_str(l).unwrap()).collect();
    assert_eq!(
        records.len() as u64,
        r["opportunities"]["strip_patch_opportunities"]
            .as_u64()
            .unwrap()
    );
    assert!(records
        .iter()
        .all(|o| o["crops"]
            .as_array()
            .unwrap()
            .iter()
            .all(|c| !c["polygons_lon_lat"].as_array().unwrap().is_empty())));
    std::fs::remove_dir_all(dir).unwrap();
}
