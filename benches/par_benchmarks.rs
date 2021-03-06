use criterion::{black_box, criterion_group, criterion_main, Criterion};
use sparse_mat::{matrix::Matrix, output::FiniteFloat};
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

pub fn par_fst(c: &mut Criterion) {
  let mat = load_matrix(0.05);
  let vec = [FiniteFloat::new(1.0); 512];
  let mut buf = [FiniteFloat::new(0.0); 1024];
  c.bench_function("par vecmul low nnz", |b| {
    b.iter(|| {
      mat.eager_par_vecmul_into(black_box(&vec), &mut buf);
    })
  });
  c.bench_function("par vecmul high nnz", |b| {
    b.iter(|| {
      mat.eager_par_vecmul_into(black_box(&vec), &mut buf);
    })
  });
}

criterion_group! {
  name = par_benches;
  config = Criterion::default().warm_up_time(Duration::from_secs(8));
  targets = par_fst,
}
criterion_main!(par_benches);
