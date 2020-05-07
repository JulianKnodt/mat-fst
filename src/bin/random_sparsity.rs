#![allow(unused)]

use rand::random;
use sparse_mat::{matrix::Matrix, output::FiniteFloat, util::compute_threshold};

const BOUNDS: [u16; 4] = [1024, 1024, 1024, 1024];

fn random_items(bounds: [u16; 4]) -> impl Iterator<Item = ([u16; 4], FiniteFloat<f32>)> {
  let [i, j, k, l] = bounds;
  (0..i).flat_map(move |x| {
    (0..j).flat_map(move |y| {
      (0..k).flat_map(move |z| (0..l).map(move |w| ([x, y, z, w], FiniteFloat::new(random()))))
    })
  })
}

fn load_mat(thresh: f32, v: &[([u16; 4], FiniteFloat<f32>)]) -> Matrix<Vec<u8>, u16, FiniteFloat<f32>, 4> {
  let entries = v.iter().filter_map(|&(i, v)| {
    if v.inner().abs() > thresh {
      Some((i, v))
    } else {
      None
    }
  });
  Matrix::new(BOUNDS, entries)
}

fn main() {
  let examples = (0..1)
    .map(|_| random_items(BOUNDS).collect::<Vec<_>>())
    .collect::<Vec<_>>();
  let thresholds = [
    0.9, 0.5, 0.4, 0.3, 0.2, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.015, 0.01,
    0.005, 0.0025, 0.001,
  ];
  for &t in &thresholds {
    let items = examples
      .iter()
      .map(|e| {
        let abs_thresh = compute_threshold(e.iter().map(|v| v.1), t).inner();
        let mat = load_mat(abs_thresh, e.as_slice());
        format!("{}", mat.nbytes())
      })
      .collect::<Vec<_>>();
    println!("{},{}", t, items.join(","));
  }
}
