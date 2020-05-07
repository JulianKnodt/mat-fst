#![allow(unused)]

use rand::random;
use sparse_mat::{matrix::Matrix, output::FiniteFloat, util::compute_threshold};

const BOUNDS: [u8; 4] = [64; 4];

fn random_items(bounds: [u8; 4]) -> impl Iterator<Item = ([u8; 4], FiniteFloat<f32>)> + Clone {
  let [i, j, k, l] = bounds;
  (0..i).flat_map(move |x| {
    (0..j).flat_map(move |y| {
      (0..k).flat_map(move |z| (0..l).map(move |w| ([x, y, z, w], FiniteFloat::new(random()))))
    })
  })
}

fn load_mat<I: Iterator<Item=([u8; 4], FiniteFloat<f32>)>>(thresh: f32, iter: I) -> Matrix<Vec<u8>, u8, FiniteFloat<f32>, 4> {
  let entries = iter.filter_map(|(i, v)| {
    if v.inner().abs() > thresh {
      Some((i, v))
    } else {
      None
    }
  });
  Matrix::new(BOUNDS, entries)
}

fn main() {
  let examples = (0..30)
    .map(|_| random_items(BOUNDS))
    .collect::<Vec<_>>();
  let thresholds = [
    0.9, 0.5, 0.4, 0.3, 0.2, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.015, 0.01,
    0.005, 0.0025, 0.001,
  ];
  for &t in &thresholds {
    let items = examples
      .iter()
      .cloned()
      .map(|e| {
        let mat = load_mat(1.0-t, e);
        format!("{}", mat.nbytes())
      })
      .collect::<Vec<_>>();
    println!("{},{}", t, items.join(","));
  }
}
