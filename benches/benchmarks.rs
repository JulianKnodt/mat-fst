use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use sparse_mat::{matrix::Matrix, output::FiniteFloat, util::compute_threshold};
use std::{
  fs::File,
  io::{BufRead, BufReader},
  path::Path,
  time::Duration,
};

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

fn items() -> Vec<FiniteFloat<f32>> {
  let f = file();
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
      Some(FiniteFloat::new(v))
    })
    .collect()
}

pub fn fst(c: &mut Criterion) {
  let vec = [FiniteFloat::new(1.0); 512];
  let mut group = c.benchmark_group("vecmul");
  let is = items();
  let thresholds = [
    0.3, 0.2, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.015, 0.01, 0.005, 0.001,
  ];
  for &t in &thresholds {
    let abs_thresh = compute_threshold(is.iter(), t);
    let mat = load_matrix(abs_thresh.inner().abs());
    group.bench_with_input(BenchmarkId::new("fst", t), &t, |b, _| {
      let mut out = vec![FiniteFloat::new(0.0); 1024];
      b.iter(|| {
        mat.vecmul_into(black_box(&vec), &mut out);
      })
    });
    let mat = mat.to_coo().to_csr();
    group.bench_with_input(BenchmarkId::new("csr", t), &t, |b, _| {
      let mut out = vec![FiniteFloat::new(0.0); 1024];
      b.iter(|| {
        mat.vecmul_into(black_box(&vec), &mut out);
      })
    });
  }
  /*
  c.bench_function("fst convolve 5x5 kernel low nnz", |b| {
    let mat = load_matrix(0.05);
    let mut out = Dense::new(mat.dims);
    let kernel = [[FiniteFloat::one(); 5]; 5];
    b.iter(|| mat.convolve_2d_into(black_box(kernel), &mut out))
  });
  */
}

criterion_group! {
  name = benches;
  config = Criterion::default().warm_up_time(Duration::from_secs(8));
  targets = fst,
}
criterion_main!(benches);
