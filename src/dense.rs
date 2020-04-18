use crate::{input::Input, output::Output};
use std::ops::{Index, IndexMut, Mul};

pub struct Dense<I, O, const N: usize> {
  pub(crate) items: Box<[O]>,
  dims: [I; N],
}

fn compute_raw_index<I: Input, const N: usize>(idx: [I; N], dims: [I; N]) -> usize {
  let mut pos = 0;
  let mut mul = 1;
  for i in (0..N).rev() {
    pos += idx[i].as_usize() * mul;
    mul *= dims[i].as_usize();
  }
  pos
}

fn invert_raw_index<I: Input, const N: usize>(raw: usize, dims: [I; N]) -> [I; N] {
  let mut out = [I::zero(); N];
  let mut curr = raw;
  let mut mul = 1;
  for i in (0..N).rev() {
    out[i] = I::from_usize((curr / mul) % dims[i].as_usize());
    mul *= dims[i].as_usize();
  }
  out
}

impl<I: Input, O: Output, const N: usize> Dense<I, O, N> {
  pub fn new(dims: [I; N]) -> Self
  where
    I: Mul<Output = I>, {
    let len = dims.iter().fold(I::one(), |a, &n| a * n);
    Dense {
      items: vec![O::zero(); len.as_usize()].into_boxed_slice(),
      dims,
    }
  }
  // pub fn iter(&self) -> impl Iterator
}

impl<I: Input, O, const N: usize> Index<[I; N]> for Dense<I, O, N> {
  type Output = O;
  fn index(&self, idx: [I; N]) -> &Self::Output { &self.items[compute_raw_index(idx, self.dims)] }
}

impl<I: Input, O, const N: usize> IndexMut<[I; N]> for Dense<I, O, N> {
  fn index_mut(&mut self, idx: [I; N]) -> &mut Self::Output {
    let raw = compute_raw_index(idx, self.dims);
    &mut self.items[raw]
  }
}

#[cfg(test)]
quickcheck! {
  fn raw_index(v: (u16, u16, u16, u16)) -> bool {
    let (a, b, c, d) = v;
    let dims = [a.max(c)+1, b.max(d)+1];
    let pt = [a.min(c), b.min(d)];
    pt == invert_raw_index(compute_raw_index(pt, dims), dims)
  }
}
