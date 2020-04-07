use crate::{bytes::*, fst::Fst, input::Input, matrix::Matrix, node::Node, output::Output};
use num::Zero;
use std::{array::LengthAtMost32, ops::Range};

pub struct VIter<'f, D, I, O> {
  fst: &'f Fst<D, I, O>,
  items: Vec<(O, Range<usize>, Node<'f, O>)>,
}

impl<'f, D, I, O> Iterator for VIter<'f, D, I, O>
where
  I: Input,
  O: Output,
  D: AsRef<[u8]>,
  Bytes<O>: Deserialize,
  Bytes<I>: Deserialize,
{
  type Item = O;
  #[inline]
  fn next(&mut self) -> Option<Self::Item> {
    while !self.items.is_empty() {
      if let Some(ref mut last) = self.items.last_mut() {
        let (out, range, node) = last;
        let is_final = node.is_final;
        if let Some(i) = range.next() {
          let t = node.transition::<I>(i);
          let next_out = out.cat(&t.output);
          let next_node = self.fst.node(t.addr);
          self
            .items
            .push((next_out, 0..next_node.num_trans, next_node));
          if is_final {
            return Some(next_out);
          } else {
            continue;
          }
        }
        let (out, _, _) = self.items.pop().unwrap();
        if is_final {
          return Some(out);
        }
      }
    }
    None
  }
}

impl<D, I, O> Fst<D, I, O>
where
  I: Input,
  O: Output,
  D: AsRef<[u8]>,
  Bytes<O>: Deserialize,
{
  #[inline]
  pub fn values(&self) -> VIter<'_, D, I, O> {
    let root = self.root();
    VIter {
      fst: self,
      items: vec![(O::zero(), 0..root.num_trans, root)],
    }
  }
}

/// An iterator over keys and values for a matrix
#[derive(Debug)]
pub struct Iter<'f, D, I, O, const N: usize>
where
  [I; N]: LengthAtMost32, {
  matrix: &'f Matrix<D, I, O, N>,
  root: Node<'f, O>,
  root_range: Range<usize>,
  items: Vec<(I, O, Range<usize>, Node<'f, O>)>,
}

impl<'f, D, I, O, const N: usize> Iterator for Iter<'f, D, I, O, N>
where
  I: Input,
  O: Output,
  D: AsRef<[u8]>,
  Bytes<O>: Deserialize,
  Bytes<I>: Deserialize,
  [I; N]: LengthAtMost32,
{
  type Item = ([I; N], O);
  #[inline]
  fn next(&mut self) -> Option<Self::Item> {
    loop {
      while !self.items.is_empty() {
        if let Some(ref mut last) = self.items.last_mut() {
          let (_, out, range, node) = last;
          let is_final = node.is_final;
          if let Some(i) = range.next() {
            let t = node.transition(i);
            let next_out = out.cat(&t.output);
            let next_node = self.matrix.data.node(t.addr);
            self
              .items
              .push((t.input, next_out, 0..next_node.num_trans, next_node));
            if is_final {
              panic!("Maybe unreachable?");
              // is this branch reachable?
              let mut idxs = [Default::default(); N];
              for i in 0..(N - 1) {
                idxs[i] = self.items[i].0;
              }
              idxs[N - 1] = t.input;
              return Some((idxs, next_out));
            } else {
              continue;
            }
          }
          let (i_f, out, _, _) = self.items.pop().unwrap();
          if is_final {
            let mut idxs = [Default::default(); N];
            assert_eq!(self.items.len(), N, "{} != {}", self.items.len(), N);
            for i in 0..(N - 1) {
              idxs[i] = self.items[i].0;
            }
            idxs[N - 1] = i_f;
            return Some((idxs, out));
          }
        }
      }
      let i = self.root_range.next()?;
      let t = self.root.transition(i);
      let node = self.matrix.data.node(t.addr);
      self
        .items
        .push((t.input, t.output, 0..node.num_trans, node));
    }
  }
}

impl<D, I, O, const N: usize> Matrix<D, I, O, N>
where
  I: Input,
  O: Output,
  D: AsRef<[u8]>,
  Bytes<O>: Deserialize,
  Bytes<I>: Deserialize,
  [I; N]: LengthAtMost32,
{
  #[inline]
  pub fn iter(&self) -> Iter<'_, D, I, O, N> {
    let root = self.data.root();
    Iter {
      matrix: self,
      root: root,
      root_range: 0..root.num_trans,
      items: Vec::with_capacity(N),
    }
  }
}
