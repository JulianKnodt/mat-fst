use crate::{bytes::*, input::Input, matrix::Matrix, output::Output, util::within};
use std::{array::LengthAtMost32, collections::BTreeMap, ops::Mul};

#[derive(Debug)]
pub struct CSR<I, O> {
  pub(crate) row_ptrs: Vec<usize>,
  pub(crate) cols: Vec<I>,
  pub(crate) values: Vec<O>,
  pub(crate) dims: [I; 2],
}

impl<I: Input, O: Output> CSR<I, O>
where
  I: Input,
  O: Output,
{
  pub fn row(&self, y: I) -> impl Iterator<Item = (I, O)> + '_ {
    let row_start = self.row_ptrs[y.as_usize()];
    let row_end = self.row_ptrs[y.as_usize() + 1];
    (row_start..row_end).map(move |j| (self.cols[j], self.values[j]))
  }
  pub fn vecmul(&self, vec: &[O]) -> Vec<O>
  where
    O: Mul<Output = O>, {
    let mut out = vec![O::zero(); self.dims[1].as_usize()];
    self.vecmul_into(vec, &mut out);
    out
  }
  pub fn vecmul_into(&self, vec: &[O], out: &mut [O])
  where
    O: Mul<Output = O>, {
    let [r, c] = self.dims;
    assert_eq!(c.as_usize(), vec.len());
    assert!(out.len() >= r.as_usize());
    for (i, v) in out.iter_mut().enumerate().take(c.as_usize()) {
      *v = self
        .row(I::from_usize(i))
        .fold(O::zero(), |acc, (x, o)| acc + vec[x.as_usize()] * o);
    }
  }
}
