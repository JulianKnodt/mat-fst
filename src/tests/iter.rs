use crate::{
  matrix::Matrix,
  output::{FiniteFloat},
};
use num::{One, Zero};

#[test]
fn iter() {
  use crate::output::FiniteFloat;
  let n = 5u8;
  let items = (0..n).flat_map(move |i| {
    (0..n).flat_map(move |j| (0..n).map(move |k| ([i, j, k], FiniteFloat::new((i + j + k) as f32))))
  });
  let mat = Matrix::new([n, n, n], items);
  let expected = (0..n).flat_map(|i| {
    (0..n).flat_map(move |j| (0..n).map(move |k| ([i, j, k], FiniteFloat::new((i + j + k) as f32))))
  });
  assert!(
    mat.iter().eq(expected),
    "Got {:?}",
    mat.iter().collect::<Vec<_>>()
  );
}

#[test]
fn vec_matmul() {
  let mul = FiniteFloat::new(0.3);
  let mat: Matrix<_, _, _, 2> = Matrix::eye(16u8);
  let one = vec![mul; 16];
  let out = mat.vecmul(&one[..]);
  for i in out {
    assert_eq!(mul, i)
  }
}

#[test]
fn transpose() {
  let mat: Matrix<_, _, u32, 2> = Matrix::eye(16u8);
  // just check that it doesn't throw
  let _ = mat.transpose();
}

#[test]
fn matmul() {
  let a: Matrix<_, _, u32, 2> = Matrix::eye(16u8);
  let b: Matrix<_, _, u32, 2> = Matrix::eye(16u8);
  // just check that it doesn't throw
  let c = a.matmul(&b);
  assert!(c.iter().eq(a.iter()));
}

#[test]
fn convolve() {
  let cap = 6u8;
  let a: Matrix<_, _, FiniteFloat<f32>, 2> = Matrix::eye(cap);
  let l = FiniteFloat::one();
  let o = FiniteFloat::<f32>::zero();
  let h = FiniteFloat::new(0.5);
  /*
  let output = a.convolve_2d([[l]]);
  */
  /*
  let output = a.convolve_2d([
    [o, o, o],
    [o, l, o],
    [o, o, o],
  ]);
  */
  let output = a.convolve_2d([
    [o, o, o, o, o],
    [o, o, h, o, o],
    [o, h, l, h, o],
    [o, o, h, o, o],
    [o, o, o, o, o],
  ]);
  for i in 0..cap {
    for j in 0..cap {
      if i == j || i == j + 1 || j == i + 1 {
        assert!(output[[i, j]].is_one());
      } else {
        assert!(output[[i, j]].is_zero());
      }
    }
  }
}

#[test]
#[cfg(feature = "parallel")]
fn par_vec_matmul() {
  let mul = FiniteFloat::new(0.3);
  let mat: Matrix<_, _, _, 2> = Matrix::eye(16u8);
  let one = vec![mul; 16];
  let out = mat.par_vecmul(&one[..]);
  for i in out {
    assert_eq!(mul, i)
  }
}
