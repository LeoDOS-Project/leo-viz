use crate::{Node, Object, Result};
use std::collections::{BTreeMap, BTreeSet};
#[derive(Default, Debug)]
pub struct Storage {
    objects: BTreeMap<String, u64>,
    copies: BTreeMap<Node, BTreeSet<String>>,
    reservations: BTreeSet<(Node, String)>,
    used: BTreeMap<Node, u64>,
    limits: BTreeMap<Node, u64>,
}
impl Storage {
    pub fn define(&mut self, o: &Object) -> Result<()> {
        if let Some(bytes) = self.objects.get(&o.id) {
            if *bytes != o.bytes {
                return Err(format!("conflicting sizes for {}", o.id));
            }
        } else {
            self.objects.insert(o.id.clone(), o.bytes);
        }
        Ok(())
    }
    pub fn knows(&self, id: &str) -> bool {
        self.objects.contains_key(id)
    }
    pub fn bytes(&self, id: &str) -> u64 {
        self.objects[id]
    }
    pub fn limit(&mut self, n: Node, limit: u64) {
        self.limits.insert(n, limit);
    }
    pub fn has(&self, n: Node, id: &str) -> bool {
        self.copies.get(&n).is_some_and(|s| s.contains(id))
    }
    pub fn used(&self, n: Node) -> u64 {
        self.used.get(&n).copied().unwrap_or(0)
    }
    pub fn reserved(&self, n: Node, id: &str) -> bool {
        self.reservations.contains(&(n, id.to_owned()))
    }
    pub fn reserve(&mut self, n: Node, id: &str) -> bool {
        if self.has(n, id) || self.reserved(n, id) {
            return false;
        }
        let Some(new) = self.used(n).checked_add(self.bytes(id)) else {
            return false;
        };
        if self.limits.get(&n).is_some_and(|l| new > *l) {
            return false;
        }
        self.used.insert(n, new);
        self.reservations.insert((n, id.to_owned()));
        true
    }
    pub fn commit(&mut self, n: Node, id: &str) {
        assert!(self.reservations.remove(&(n, id.to_owned())));
        self.copies.entry(n).or_default().insert(id.to_owned());
    }
    pub fn release(&mut self, n: Node, id: &str) {
        if self.reservations.remove(&(n, id.to_owned())) {
            *self.used.get_mut(&n).unwrap() -= self.bytes(id);
        }
    }
    pub fn delete(&mut self, n: Node, id: &str) -> bool {
        if self.copies.get_mut(&n).is_some_and(|c| c.remove(id)) {
            *self.used.get_mut(&n).unwrap() -= self.objects[id];
            true
        } else {
            false
        }
    }
    pub fn copies(&self) -> impl Iterator<Item = (Node, &str, u64)> {
        self.copies.iter().flat_map(move |(n, ids)| {
            ids.iter()
                .map(move |id| (*n, id.as_str(), self.objects[id]))
        })
    }
}
