//! Conservative sphere hierarchy for Earth-fixed AOI geometry.
use nalgebra::Vector3;

#[derive(Clone, Copy)]
pub(super) struct Sphere {
    pub center: Vector3<f64>,
    pub radius: f64,
}

struct Node {
    sphere: Sphere,
    start: usize,
    end: usize,
    children: Option<(usize, usize)>,
}

#[derive(Default)]
pub(super) struct Index {
    nodes: Vec<Node>,
    order: Vec<usize>,
}

impl Index {
    pub fn build(spheres: &[Sphere]) -> Self {
        let mut index = Self {
            nodes: Vec::new(),
            order: (0..spheres.len()).collect(),
        };
        if !spheres.is_empty() {
            index.branch(spheres, 0, spheres.len());
        }
        index
    }

    fn branch(&mut self, spheres: &[Sphere], start: usize, end: usize) -> usize {
        let mut lo = Vector3::repeat(f64::INFINITY);
        let mut hi = Vector3::repeat(f64::NEG_INFINITY);
        for &i in &self.order[start..end] {
            let sphere = spheres[i];
            for axis in 0..3 {
                lo[axis] = lo[axis].min(sphere.center[axis] - sphere.radius);
                hi[axis] = hi[axis].max(sphere.center[axis] + sphere.radius);
            }
        }
        let center = (lo + hi) * 0.5;
        let radius = self.order[start..end]
            .iter()
            .map(|&i| (spheres[i].center - center).norm() + spheres[i].radius)
            .fold(0.0, f64::max);
        let node = self.nodes.len();
        self.nodes.push(Node {
            sphere: Sphere { center, radius },
            start,
            end,
            children: None,
        });
        if end - start > 24 {
            let span = hi - lo;
            let axis = (0..3).max_by(|&a, &b| span[a].total_cmp(&span[b])).unwrap();
            let middle = (start + end) / 2;
            self.order[start..end].select_nth_unstable_by(middle - start, |&a, &b| {
                spheres[a].center[axis].total_cmp(&spheres[b].center[axis])
            });
            let left = self.branch(spheres, start, middle);
            let right = self.branch(spheres, middle, end);
            self.nodes[node].children = Some((left, right));
        }
        node
    }

    pub fn query(&self, mut intersects: impl FnMut(Sphere) -> bool) -> Vec<usize> {
        if self.nodes.is_empty() {
            return Vec::new();
        }
        let mut stack = vec![0];
        let mut found = Vec::new();
        while let Some(i) = stack.pop() {
            let node = &self.nodes[i];
            if !intersects(node.sphere) {
                continue;
            }
            if let Some((left, right)) = node.children {
                stack.push(right);
                stack.push(left);
            } else {
                found.extend_from_slice(&self.order[node.start..node.end]);
            }
        }
        // Stable ordering keeps colors/dot representatives deterministic.
        found.sort_unstable();
        found
    }
}
