use crate::{
  bytes::*,
  fst::{Fst, Transition},
  input::Input,
  matrix::Matrix,
  node::{par_immediate_iter, par_immediate_range_iter, Node},
  output::Output,
};
use num::Zero;
use rayon::{iter::plumbing::*, prelude::*};
use std::{
  array::LengthAtMost32,
  ops::{Mul, Range},
};

#[derive(Debug, Copy, Clone)]
pub struct ParSliceIter<'f, D, I, O, const N: usize, const P: usize>
where
  [I; N]: LengthAtMost32,
  [I; P]: LengthAtMost32, {
  matrix: &'f Matrix<D, I, O, N>,
  node: Node<'f, I>,
  /// The current output value
  curr_out: u32,
}

// TODO this really should be N, N-1, but waiting for const-generics to be stabilized
// also it should IndexedParallelIterator
impl<'f, D, I, O> ParallelIterator for ParSliceIter<'f, D, I, O, 2, 1>
where
  I: Input + Send + Sync,
  O: Output + Send + Sync,
  D: AsRef<[u8]> + Send + Sync,
  Bytes<O>: Deserialize,
  Bytes<I>: Deserialize,
{
  type Item = (I, O);
  fn drive_unindexed<C>(self, consumer: C) -> C::Result
  where
    C: UnindexedConsumer<Self::Item>, {
    let ParSliceIter {
      matrix,
      node,
      curr_out,
    } = self;
    (0..node.num_trans)
      .into_par_iter()
      .map(move |i| {
        let t = node.transition(i);
        (
          t.input,
          self.matrix.data.outputs[curr_out.cat(&t.num_out) as usize],
        )
      })
      .drive_unindexed(consumer)
  }
}

// TODO this really should be N, P < N, but waiting for const-generics to be stabilized
impl<'f, D, I, O> ParallelIterator for ParSliceIter<'f, D, I, O, 2, 0>
where
  I: Input + Send + Sync,
  O: Output + Send + Sync,
  D: AsRef<[u8]> + Send + Sync,
  Bytes<O>: Deserialize,
  Bytes<I>: Deserialize,
{
  // TODO this should be ParSliceIter<'f, D, I, O, N, P+1>
  type Item = (I, ParSliceIter<'f, D, I, O, 2, 1>);

  fn drive_unindexed<C>(self, consumer: C) -> C::Result
  where
    C: UnindexedConsumer<Self::Item>, {
    let ParSliceIter {
      matrix,
      node,
      curr_out,
    } = self;
    (0..node.num_trans)
      .into_par_iter()
      .map(move |i| {
        let t = node.transition(i);
        let next_node = self.matrix.data.node(t.addr);
        let iter = ParSliceIter {
          matrix: &self.matrix,
          node: next_node,
          curr_out: curr_out.cat(&t.num_out),
        };
        (t.input, iter)
      })
      .drive_unindexed(consumer)
  }
}

impl<D, I, O> Matrix<D, I, O, 2>
where
  I: Input + Send + Sync,
  O: Output + Send + Sync,
  D: AsRef<[u8]> + Send + Sync,
  Bytes<O>: Deserialize,
  Bytes<I>: Deserialize,
{
  // TODO convert to ParSliceIter<'_, D, I, O, N, 0> when this is more stable
  #[inline]
  pub fn par_iter(&self) -> ParSliceIter<'_, D, I, O, 2, 0> {
    let root = self.data.root();
    ParSliceIter {
      matrix: self,
      node: self.data.root(),
      curr_out: 0,
    }
  }
}

impl<D, I, O> Matrix<D, I, O, 2>
where
  I: Input + Send + Sync,
  O: Output + Send + Sync,
  O: Mul<Output = O>,
  D: AsRef<[u8]> + Send + Sync,
  Bytes<O>: Deserialize,
  Bytes<I>: Deserialize,
{
  pub fn par_vecmul(&self, vec: &[O]) -> Vec<O> {
    let mut out = vec![O::zero(); self.dims[0].as_usize()];
    self.par_vecmul_into(vec, &mut out[..]);
    out
  }
  /// performs vector multiplication but is required to allocate
  pub fn par_vecmul_into(&self, vec: &[O], out: &mut [O]) {
    assert_eq!(
      self.dims[1].as_usize(),
      vec.len(),
      "dimension mismatch, expected vector of len {}",
      self.dims[1]
    );
    assert_eq!(
      self.dims[0].as_usize(),
      out.len(),
      "dimension mismatch, expected output vector of len {}",
      self.dims[0]
    );
    let (idxs, vals): (Vec<usize>, Vec<O>) = self
      .par_iter()
      .map(|(y, row)| {
        let y = y.as_usize();
        let row_sum: O = row
          .fold(O::zero, |a, (x, b)| a + b * vec[x.as_usize()])
          .reduce(O::zero, |a, b| a + b);
        (y, row_sum)
      })
      .unzip();
    for (i, &y) in idxs.iter().enumerate() {
      out[y] = vals[i];
    }
  }
  pub fn eager_par_vecmul_into(&self, vec: &[O], out: &mut [O]) {
    assert_eq!(
      self.dims[1].as_usize(),
      vec.len(),
      "dimension mismatch, expected vector of len {}",
      self.dims[1]
    );
    assert_eq!(
      self.dims[0].as_usize(),
      out.len(),
      "dimension mismatch, expected output vector of len {}",
      self.dims[0]
    );
    let data = self.data.data.as_ref();
    let (idxs, vals): (Vec<I>, Vec<O>) = par_immediate_iter(self.data.meta.root_addr, data)
      .map(|t0: Transition<I>| {
        let row_sum = par_immediate_range_iter(t0.addr, data)
          .enumerate()
          .map(|(i, t1): (usize, I)| (t1, self.data.outputs[t0.num_out as usize + i]))
          .fold(O::zero, |a, (x, b)| a + b * vec[x.as_usize()])
          .reduce(O::zero, |a, b| a + b);
        (t0.input, row_sum)
      })
      .unzip();
    for (i, y) in idxs.iter().enumerate() {
      out[y.as_usize()] = vals[i];
    }
  }
}
