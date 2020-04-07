use crate::{bytes::*, fst::Fst, input::Input, output::Output};
use num::Zero;
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
  /*
  #[inline]
  pub fn slice<R: RangeBounds<I>>(&self, idxs: [R; 2]) -> impl Iterator<Item=O> {
    todo!()
  }
  */
  /// Returns a row of this matrix as an iterator with the column index
  pub fn row(&self, r: I) -> impl Iterator<Item = (I, O)> + '_ {
    // TODO this is a naive implementation
    self.iter().filter_map(move |([y, x], v)| {
      if y != r {
        return None;
      }
      Some((x, v))
    })
  }
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
    assert!(
      self.dims[0].as_usize() <= vec.len(),
      "Dimension mismatch, expected output of size {}",
      self.dims[0]
    );
    for i in 0..vec.len() {
      out[i] = self
        .row(I::from_usize(i))
        .map(|(j, v): (I, O)| v * vec[j.as_usize()])
        .fold(O::zero(), |acc, n: O| acc + n);
    }
  }
}
