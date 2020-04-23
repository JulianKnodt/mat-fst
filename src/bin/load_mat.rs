#![allow(unused)]

use sparse_mat::{matrix::Matrix, output::FiniteFloat, util::compute_threshold};
use std::{
  fs::File,
  io::{BufRead, BufReader},
};

const THRESHOLD: f32 = 0.031_094_963;
// 0.005 works p well
// 0.05 hits a lot of edge cases

fn main() {
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
    if v.abs() > THRESHOLD {
      Some(([y, x], FiniteFloat::new(v)))
    } else {
      None
    }
  });
  /*
  let v = compute_threshold(entries, 0.90);
  println!("{}", v.inner());
  */
  let mat = Matrix::new([1024u16, 512], entries);
  println!("Sparsity {:#?}", mat.sparsity());
  println!("mat len {}", mat.count_nonzero());
  println!("mat shape {:?}", mat.shape());
  println!("mat bytes {}", mat.nbytes());
}
