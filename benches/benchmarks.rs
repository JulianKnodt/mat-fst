use criterion::{black_box, criterion_group, criterion_main, Criterion};
use num::One;
use sparse_mat::{dense::Dense, matrix::Matrix, output::FiniteFloat};
use std::{
  fs::File,
  io::{BufRead, BufReader},
  path::Path,
  time::Duration,
};

// 0.005 works p well
// 0.05 hits a lot of edge cases
const TEN_P_THRESH: f32 = 0.031094962;

fn file() -> File {
  let p = Path::new(file!())
    .parent()
    .unwrap()
    .parent()
    .unwrap()
    .join("src")
    .join("bin")
    .join("weights.txt");
  File::open(p).unwrap()
}

fn load_matrix(thresh: f32) -> Matrix<Vec<u8>, u16, FiniteFloat<f32>, 2> {
  let f = file();
  let buf = BufReader::new(f);
  let entries = buf.lines().filter_map(|line| {
    let line = line.unwrap();
    let mut parts = line.split_whitespace();
    let y = parts.next().unwrap();
    let y = y.parse::<u16>().unwrap();
    let x = parts.next().unwrap();
    let x = x.parse::<u16>().unwrap();
    let v = parts.next().unwrap();
    let v = v.parse::<f32>().unwrap();
    if v.abs() > thresh {
      Some(([y, x], FiniteFloat::new(v)))
    } else {
      None
    }
  });
  Matrix::new([1024u16, 512], entries)
}

pub fn fst(c: &mut Criterion) {
  let vec = [FiniteFloat::new(1.0); 512];
  c.bench_function("fst vecmul low nnz", |b| {
    let mat = load_matrix(0.05);
    let mut out = vec![FiniteFloat::new(0.0); 1024];
    b.iter(|| {
      mat.vecmul_into(black_box(&vec), &mut out);
    })
  });
  c.bench_function("fst vecmul high nnz", |b| {
    let mat = load_matrix(TEN_P_THRESH);
    let mut out = vec![FiniteFloat::new(0.0); 1024];
    b.iter(|| {
      mat.vecmul_into(black_box(&vec), &mut out);
    })
  });
  c.bench_function("fst convolve 5x5 kernel low nnz", |b| {
    let mat = load_matrix(0.05);
    let mut out = Dense::new(mat.dims);
    let kernel = [[FiniteFloat::one(); 5]; 5];
    b.iter(|| mat.convolve_2d_into(black_box(kernel), &mut out))
  });
}

criterion_group! {
  name = benches;
  config = Criterion::default().warm_up_time(Duration::from_secs(8));
  targets = fst,
}
criterion_main!(benches);
