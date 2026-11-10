//! Read-only checkpoint replay for diagnosing capture failures.
use leo_capture::session::Session;
fn main() -> Result<(), String> {
    let path = std::env::args()
        .nth(1)
        .ok_or("Usage: replay_checkpoint CHECKPOINT [END_SECONDS]")?;
    let end = std::env::args().nth(2).map(|s| s.parse::<f64>().unwrap());
    let mut percent = -1;
    let mut session = Session::load(std::path::Path::new(&path), |f| {
        let next = (f * 20.0) as i32;
        if next > percent {
            eprintln!("Preparing: {}%", next * 5);
            percent = next;
        }
        true
    })?;
    eprintln!(
        "Loaded t={}s, {} jobs, {} captures",
        session.elapsed_s(),
        session.inputs().jobs.len(),
        session.meters().captures_total
    );
    while !session.finished() && end.is_none_or(|end| session.elapsed_s() < end) {
        let before = session.elapsed_s();
        eprintln!("Advancing from {before}s");
        session.advance(12)?;
        let _ = session.meters();
        let _ = session.ground_positions()?;
        for capture in session
            .state()
            .captures
            .iter()
            .rev()
            .take_while(|c| c.end_s > session.elapsed_s() - 600.0)
            .take(2000)
        {
            let _ = session.footprint(capture);
        }
    }
    let m = session.meters();
    eprintln!(
        "Finished t={}s, {} captures, {} complete / {} due, {} partial, {:.3} TB retained",
        session.elapsed_s(),
        m.captures_total,
        m.complete_updates,
        m.due_updates,
        m.partial_updates,
        session.report().retained_physical_bytes / 1e12
    );
    Ok(())
}
