use crate::{build::Builder, bytes::*, fst::Fst, input::Input, output::Output};
use num::{One, Zero};
use std::{
  array::LengthAtMost32,
  ops::{Index, Mul, RangeBounds},
};

#[derive(Debug)]
pub struct Matrix<D, I, O, const N: usize>
where
  [I; N]: LengthAtMost32, {
  // row, col, etc
  pub dims: [I; N],
  pub(crate) data: Fst<D, I, O>,
}

impl<I, O, const N: usize> Matrix<Vec<u8>, I, O, N>
where
  I: Input,
  O: Output,
  Bytes<O>: Serialize + Deserialize,
  Bytes<I>: Serialize + Deserialize,
  [I; N]: LengthAtMost32,
{
  pub fn new<Iter: Iterator<Item = ([I; N], O)>>(dims: [I; N], i: Iter) -> Self {
    let mut builder = Builder::memory().unwrap();
    for (k, v) in i {
      builder.insert(k, v).expect("Failed to insert");
    }
    let data = builder.into_fst();
    Matrix { dims, data }
  }
}

impl<D, I, O> Matrix<D, I, O, 2>
where
  D: AsRef<[u8]>,
  I: Input,
  O: Output,
  O: Mul<Output = O>,
  Bytes<O>: Deserialize,
  Bytes<I>: Deserialize,
{
  #[inline]
  fn get(&self, idxs: [I; 2]) -> O { self.data.get(&idxs[..]).unwrap_or_else(O::zero) }
  /// Performs vector multiplication of this matrix with some dense vector
  #[inline]
  pub fn vecmul(&self, vec: &[O]) -> Vec<O> {
    let mut out = vec![O::zero(); self.dims[0].as_usize()];
    self.vecmul_into(vec, &mut out);
    out
  }
  /// Performs vector multiplication and puts the result into some destination buffer
  pub fn vecmul_into(&self, vec: &[O], out: &mut [O]) {
    assert_eq!(
      self.dims[1].as_usize(),
      vec.len(),
      "Dimension mismatch, expected vector of len {}",
      self.dims[1]
    );
    let rows = self.dims[0].as_usize();
    assert!(
      rows <= out.len(),
      "Dimension mismatch, expected output of size {}",
      self.dims[0]
    );
    self.iter().for_each(|([y, x], v)| {
      let y = y.as_usize();
      out[y] = out[y] + v * vec[x.as_usize()];
    });
  }
}

impl<D, I, O, const N: usize> Matrix<D, I, O, N>
where
  D: AsRef<[u8]>,
  I: Input,
  O: Output,
  O: Mul<Output = O>,
  Bytes<O>: Deserialize,
  Bytes<I>: Deserialize,
  [I; N]: LengthAtMost32,
{
  /// Returns the dimensions of this matrix
  pub fn shape(&self) -> [I; N] { self.dims }
  pub fn count_nonzero(&self) -> usize { self.data.len() }
  /// Returns the size of this matrix in bytes
  pub fn nbytes(&self) -> usize { self.data.nbytes() }
  pub fn sparsity(&self) -> f64 {
    let total = self.shape().iter().map(|l| l.as_usize()).product::<usize>();
    (self.count_nonzero() as f64) / (total as f64)
  }
}

impl<I, O, const N: usize> Matrix<Vec<u8>, I, O, N>
where
  I: Input,
  O: Output + One,
  O: Mul<Output = O>,
  Bytes<O>: Serialize + Deserialize,
  Bytes<I>: Serialize + Deserialize,
  [I; N]: LengthAtMost32,
{
  pub fn eye(n: I) -> Self {
    Matrix::new(
      [n; N],
      (0..n.as_usize()).map(|i| ([I::from_usize(i); N], O::one())),
    )
  }
}
