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
  Bytes<I>: Deserialize,
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
#[derive(Debug, Clone)]
pub struct Iter<'f, D, I, O, const N: usize>
where
  [I; N]: LengthAtMost32, {
  matrix: &'f Matrix<D, I, O, N>,
  root: Node<'f, O>,
  root_curr: usize,
  /// Which inputs have been seen thus far
  inputs: Vec<I>,
  items: Vec<(O, usize, Node<'f, O>)>,
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
          let (out, curr, node) = last;
          let is_final = node.is_final;
          if *curr < node.num_trans {
            *curr += 1;
            let t = node.transition(*curr - 1);
            let next_out = out.cat(&t.output);
            let next_node = self.matrix.data.node(t.addr);
            self.inputs.push(t.input);
            self.items.push((next_out, 0, next_node));
            debug_assert!(!is_final);
            continue;
          }
          let (out, _, last) = self.items.pop().unwrap();
          // TODO make this a check for item len which means we can remove the is_final
          // from the node itself
          if is_final {
            let mut idxs = [Default::default(); N];
            debug_assert_eq!(self.items.len(), N - 1); // because we popped one
            idxs.copy_from_slice(&self.inputs[..]);
            assert!(self.inputs.pop().is_some());
            return Some((idxs, out.cat(&last.final_output)));
          }
          self.inputs.pop();
        }
      }
      if self.root_curr >= self.root.num_trans {
        return None;
      }
      let t = self.root.transition(self.root_curr);
      let node = self.matrix.data.node(t.addr);
      self.inputs.push(t.input);
      self.items.push((t.output, 0, node));
      self.root_curr += 1;
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
      root,
      root_curr: 0,
      inputs: Vec::with_capacity(N),
      items: Vec::with_capacity(N),
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
  [Range<I>; N]: LengthAtMost32,
{
  #[inline]
  pub fn select(&self, ranges: [Range<I>; N]) -> SelectIter<'_, D, I, O, N> {
    let root = self.data.root();
    let root_range = root.find_input_range(&ranges[0]);
    let mut state = Vec::with_capacity(N);
    state.push((O::zero(), root_range, root));
    SelectIter {
      matrix: &self,
      ranges,
      inputs: [Default::default(); N],
      state,
    }
  }
}

/// An iterator over a select set of keys and values for a matrix
#[derive(Debug, Clone)]
pub struct SelectIter<'f, D, I, O, const N: usize>
where
  [I; N]: LengthAtMost32,
  [Range<I>; N]: LengthAtMost32, {
  matrix: &'f Matrix<D, I, O, N>,
  // TODO can actually shrink this by one
  ranges: [Range<I>; N],
  // can just use an array because it will have a fixed size
  inputs: [I; N],
  // Which nodes are currently being looked at?
  state: Vec<(O, Range<usize>, Node<'f, O>)>,
}

impl<'f, D, I, O, const N: usize> Iterator for SelectIter<'f, D, I, O, N>
where
  I: Input,
  O: Output,
  D: AsRef<[u8]>,
  Bytes<O>: Deserialize,
  Bytes<I>: Deserialize,
  [I; N]: LengthAtMost32,
  [Range<I>; N]: LengthAtMost32,
{
  type Item = ([I; N], O);
  fn next(&mut self) -> Option<Self::Item> {
    while let Some(ref mut last) = self.state.last_mut() {
      let i = if let Some(i) = last.1.next() {
        i
      } else {
        self.state.pop();
        continue;
      };
      let t = last.2.transition(i);
      self.inputs[N - 1] = t.input;
      let out = last.0.cat(&t.output);
      let len = self.state.len();
      assert!(len <= N);
      if len == N {
        return Some((self.inputs, out));
      } else {
        let node = self.matrix.data.node(t.addr);
        let range = node.find_input_range(&self.ranges[len]);
        self.state.push((out, range, node));
      }
    }
    None
  }
}
