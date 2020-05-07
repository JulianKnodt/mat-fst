#![allow(unused)]

use sparse_mat::{matrix::Matrix, output::FiniteFloat, util::compute_threshold};
use std::{
  fs::File,
  io::{BufRead, BufReader},
};

fn items() -> Vec<FiniteFloat<f32>> {
  let file_name = "weights.txt";
  let f = File::open(file_name).unwrap();
  let buf = BufReader::new(f);
  buf
    .lines()
    .filter_map(|line| {
      let line = line.ok()?;
      let mut parts = line.split_whitespace();
      parts.next()?;
      parts.next()?;
      let v = parts.next().unwrap();
      let v = v.parse::<f32>().unwrap();
      Some(FiniteFloat::new(v.abs()))
    })
    .collect()
}

const THRESHOLD: f32 = 0.031_094_963;
// 0.005 works p well
// 0.05 hits a lot of edge cases

fn load_mat(thresh: f32) -> Matrix<Vec<u8>, u16, FiniteFloat<f32>, 2> {
  let file_name = "weights.txt";
  let file = File::open(file_name).unwrap();
  let buf = BufReader::new(file);
  let entries = buf.lines().filter_map(|line| {
    let line = line.unwrap();
    let mut parts = line.split_whitespace();
    let y = parts.next().unwrap();
    let y = y.parse::<u16>().unwrap();
    let x = parts.next().unwrap();
    let x = x.parse::<u16>().unwrap();
    let v = parts.next().unwrap();
    let v = v.parse::<f32>().unwrap();
    // return Some(FiniteFloat::new(v.abs()));
    if v.abs() > thresh {
      Some(([y, x], FiniteFloat::new(v)))
    } else {
      None
    }
  });
  Matrix::new([1024u16, 512], entries)
}

fn main() {
  let thresholds = [
    0.9, 0.5, 0.4, 0.3, 0.2, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.015, 0.01,
    0.005, 0.0025, 0.001,
  ];
  let is = items();
  for &t in &thresholds {
    let abs_thresh = compute_threshold(is.iter(), t).inner();
    let mat = load_mat(abs_thresh);
    let csr = mat.to_coo().to_csr();
    println!("{}, {}, {}", t, mat.nbytes(), csr.nbytes());
  }
}
