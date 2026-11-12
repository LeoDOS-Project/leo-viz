use super::{Mode, Options};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Action {
    pub tick: u32,
    pub mode: usize,
}

/// Action a has nodes (2a=start, 2a+1=end), with parallel capture and skip
/// edges. Along each mode, end -> next start is a waiting edge. From each end
/// we only need the earliest reachable start in every *other* mode: waiting
/// along that mode reaches every later action without acquiring intervening ones.
/// This is exact for the finite action set and the time-invariant mode slew law.
pub struct Topology {
    pub actions: Vec<Action>,
    pub next_same: Vec<Option<u32>>,
    by_mode: Vec<Vec<u32>>,
    lag: Vec<Vec<u32>>,
    initial_lag: Vec<u32>,
}
pub struct Hub {
    pub ready_tick: u32,
    pub targets: Vec<u32>,
}
pub struct Hubs {
    pub from_action: Vec<Option<usize>>,
    pub hubs: Vec<Hub>,
}
impl Topology {
    pub fn new(actions: Vec<Action>, modes: &[Mode], options: &Options) -> Result<Self, String> {
        if actions.len() > (u32::MAX as usize - 2) / 2 {
            return Err("Too many graph nodes".into());
        }
        let mut by_mode = vec![Vec::new(); modes.len()];
        for (i, a) in actions.iter().enumerate() {
            if a.mode >= modes.len() || a.tick >= options.ticks() {
                return Err("Invalid action".into());
            }
            let ids = &mut by_mode[a.mode];
            if ids
                .last()
                .is_some_and(|&p| actions[p as usize].tick >= a.tick)
            {
                return Err("Actions must be ordered, unique within mode".into());
            }
            ids.push(i as u32);
        }
        let mut next_same = vec![None; actions.len()];
        for ids in &by_mode {
            for pair in ids.windows(2) {
                next_same[pair[0] as usize] = Some(pair[1]);
            }
        }
        let ticks = |s: f64| ((s / options.step_s - 1e-10).ceil().max(0.0)) as u32;
        let lag = modes
            .iter()
            .map(|a| {
                modes
                    .iter()
                    .map(|b| ticks(a.transition_s(b, options)))
                    .collect()
            })
            .collect();
        let nadir = Mode::new(0.0, 0.0);
        let initial_lag = modes
            .iter()
            .map(|m| ticks(nadir.transition_s(m, options)))
            .collect();
        Ok(Self {
            actions,
            next_same,
            by_mode,
            lag,
            initial_lag,
        })
    }
    fn first(&self, mode: usize, ready: u32) -> Option<u32> {
        let ids = &self.by_mode[mode];
        ids.get(ids.partition_point(|&a| self.actions[a as usize].tick < ready))
            .copied()
    }
    pub fn source_targets(&self) -> Vec<u32> {
        (0..self.by_mode.len())
            .filter_map(|m| self.first(m, self.initial_lag[m]))
            .collect()
    }
    pub fn targets(&self, id: usize) -> Vec<u32> {
        let a = self.actions[id];
        (0..self.by_mode.len())
            .filter(|&m| m != a.mode)
            .filter_map(|m| self.first(m, a.tick + 1 + self.lag[a.mode][m]))
            .collect()
    }
    /// With coarse time steps all changes can have the same rounded lag. Factor
    /// their identical successor sets through a time-only choice node. The
    /// delayed same-mode choice is redundant with its wait spine and harmless.
    /// This preserves feasibility, not an additive exact physical slew cost.
    pub fn hubs(&self) -> Option<Hubs> {
        let lag = *self.lag.first()?.get(1)?;
        if self
            .lag
            .iter()
            .enumerate()
            .any(|(a, row)| row.iter().enumerate().any(|(b, &n)| a != b && n != lag))
        {
            return None;
        }
        let mut by_tick = std::collections::BTreeMap::new();
        for a in &self.actions {
            by_tick.entry(a.tick + 1 + lag).or_insert(None);
        }
        let mut hubs = Vec::new();
        for (&ready, index) in &mut by_tick {
            let targets: Vec<_> = (0..self.by_mode.len())
                .filter_map(|m| self.first(m, ready))
                .collect();
            if !targets.is_empty() {
                *index = Some(hubs.len());
                hubs.push(Hub {
                    ready_tick: ready,
                    targets,
                });
            }
        }
        Some(Hubs {
            from_action: self
                .actions
                .iter()
                .map(|a| by_tick[&(a.tick + 1 + lag)])
                .collect(),
            hubs,
        })
    }
    /// All candidate endpoints need a source-reachable path before receiving
    /// acquisition credit. The exported graph may also contain unreachable nodes.
    pub fn reachable(&self) -> Vec<bool> {
        let mut reachable = vec![false; self.actions.len()];
        for a in self.source_targets() {
            reachable[a as usize] = true;
        }
        let mut order: Vec<_> = (0..self.actions.len()).collect();
        order.sort_by_key(|&i| (self.actions[i].tick, i));
        for id in order {
            if !reachable[id] {
                continue;
            }
            if let Some(next) = self.next_same[id] {
                reachable[next as usize] = true;
            }
            for next in self.targets(id) {
                reachable[next as usize] = true;
            }
        }
        reachable
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn paths(t: &Topology, sparse: bool) -> BTreeSet<Vec<u32>> {
        fn visit(
            t: &Topology,
            id: usize,
            sparse: bool,
            taken: &mut Vec<u32>,
            out: &mut BTreeSet<Vec<u32>>,
        ) {
            for capture in [false, true] {
                if capture {
                    taken.push(id as u32);
                }
                out.insert(taken.clone());
                let a = t.actions[id];
                let next: Vec<_> = if sparse {
                    t.next_same[id].into_iter().chain(t.targets(id)).collect()
                } else {
                    t.actions
                        .iter()
                        .enumerate()
                        .filter(|(_, b)| b.tick >= a.tick + 1 + t.lag[a.mode][b.mode])
                        .map(|(i, _)| i as u32)
                        .collect()
                };
                for n in next {
                    visit(t, n as usize, sparse, taken, out);
                }
                if capture {
                    taken.pop();
                }
            }
        }
        let mut out = BTreeSet::from([Vec::new()]);
        let starts = if sparse {
            t.source_targets()
        } else {
            t.actions
                .iter()
                .enumerate()
                .filter(|(_, a)| a.tick >= t.initial_lag[a.mode])
                .map(|(i, _)| i as u32)
                .collect()
        };
        for i in starts {
            visit(t, i as usize, sparse, &mut Vec::new(), &mut out);
        }
        out
    }
    #[test]
    fn shared_hubs_have_exactly_the_same_reachable_next_actions_after_waiting() {
        let o = Options {
            duration_s: 360.0,
            ..Default::default()
        };
        let modes = vec![
            Mode::new(0.0, 0.0),
            Mode::new(0.0, 100.0),
            Mode::new(150.0, 0.0),
        ];
        let actions: Vec<_> = (0..12)
            .flat_map(|tick| {
                (0..3)
                    .filter(move |mode| (tick + mode) % 3 != 1)
                    .map(move |mode| Action {
                        tick: tick as u32,
                        mode,
                    })
            })
            .collect();
        let t = Topology::new(actions, &modes, &o).unwrap();
        let h = t.hubs().unwrap();
        for (id, a) in t.actions.iter().enumerate() {
            let mut reachable = std::collections::BTreeSet::new();
            let mut todo: Vec<_> = t.next_same[id]
                .into_iter()
                .chain(
                    h.from_action[id]
                        .into_iter()
                        .flat_map(|i| h.hubs[i].targets.iter().copied()),
                )
                .collect();
            while let Some(n) = todo.pop() {
                if reachable.insert(n) {
                    if let Some(next) = t.next_same[n as usize] {
                        todo.push(next);
                    }
                }
            }
            let dense: std::collections::BTreeSet<_> = t
                .actions
                .iter()
                .enumerate()
                .filter(|(_, b)| b.tick >= a.tick + 1 + t.lag[a.mode][b.mode])
                .map(|(i, _)| i as u32)
                .collect();
            assert_eq!(reachable, dense);
        }
    }
    #[test]
    fn sparse_wait_spines_preserve_every_dense_capture_sequence() {
        let o = Options {
            duration_s: 180.0,
            ..Default::default()
        };
        let modes = vec![Mode::new(0.0, 0.0), Mode::new(0.0, 100.0)];
        let actions = vec![
            Action { tick: 0, mode: 0 },
            Action { tick: 1, mode: 0 },
            Action { tick: 1, mode: 1 },
            Action { tick: 2, mode: 1 },
            Action { tick: 4, mode: 0 },
            Action { tick: 5, mode: 1 },
        ];
        let t = Topology::new(actions, &modes, &o).unwrap();
        assert_eq!(paths(&t, true), paths(&t, false));
        assert!(paths(&t, true).contains(&vec![0, 4])); // B need not be acquired.
        assert!(!paths(&t, true).contains(&vec![0, 2])); // Repointing is not free.
        assert!(t.reachable().iter().all(|x| *x));
    }
    #[test]
    fn initial_pointing_and_long_idle_gaps() {
        let o = Options {
            duration_s: 600.0,
            ..Default::default()
        };
        let modes = vec![Mode::new(0.0, 0.0), Mode::new(0.0, 200.0)];
        let t = Topology::new(
            vec![Action { tick: 0, mode: 1 }, Action { tick: 10, mode: 1 }],
            &modes,
            &o,
        )
        .unwrap();
        assert_eq!(t.reachable(), vec![false, true]);
        assert_eq!(t.source_targets(), vec![1]);
        assert_eq!(paths(&t, true), paths(&t, false));
    }
}
