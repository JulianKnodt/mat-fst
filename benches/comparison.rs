use criterion::{black_box, criterion_group, criterion_main, Criterion};
use sparse_mat::{coo::COO, output::FiniteFloat};
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

fn load_matrix(thresh: f32) -> COO<u16, FiniteFloat<f32>, 2> {
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
  COO::from_iter([1024u16, 512], entries)
}

pub fn low_nnz(c: &mut Criterion) {
  let mat = load_matrix(0.05).to_csr();
  let vec = [FiniteFloat::new(1.0); 512];
  let mut out = vec![FiniteFloat::new(0.0); 1024];
  c.bench_function("csr vecmul low nnz", |b| {
    b.iter(|| {
      mat.vecmul_into(black_box(&vec), &mut out);
    })
  });
}

criterion_group! {
  name = comparison;
  config = Criterion::default().warm_up_time(Duration::from_secs(8));
  targets = low_nnz,
}
criterion_main!(comparison);
