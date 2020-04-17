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
  [I; N]: LengthAtMost32,
  [(O, usize, Node<'f, O>); N]: LengthAtMost32, {
  matrix: &'f Matrix<D, I, O, N>,
  root: Node<'f, O>,
  root_curr: usize,
  /// Which inputs have been seen thus far
  inputs: [I; N],
  items: [(O, usize, Node<'f, O>); N],
  curr_len: usize,
}

impl<'f, D, I, O, const N: usize> Iterator for Iter<'f, D, I, O, N>
where
  I: Input,
  O: Output,
  D: AsRef<[u8]>,
  Bytes<O>: Deserialize,
  Bytes<I>: Deserialize,
  [I; N]: LengthAtMost32,
  [(O, usize, Node<'f, O>); N]: LengthAtMost32,
{
  type Item = ([I; N], O);
  #[inline]
  fn next(&mut self) -> Option<Self::Item> {
    loop {
      while self.curr_len > 0 {
        assert!(self.curr_len <= N);
        let is_final = self.curr_len == N;
        let (out, curr, node) = &mut self.items[self.curr_len - 1];
        if *curr < node.num_trans {
          debug_assert!(!is_final);
          *curr += 1;
          let t = node.transition(*curr - 1);
          let next_out = out.cat(&t.output);
          let next_node = self.matrix.data.node(t.addr);
          self.inputs[self.curr_len] = t.input;
          self.items[self.curr_len] = (next_out, 0, next_node);
          self.curr_len += 1;
          continue;
        }
        self.curr_len -= 1;
        if !is_final {
          continue;
        }
        return Some((self.inputs, out.cat(&node.final_output)));
      }
      if self.root_curr >= self.root.num_trans {
        return None;
      }
      let t = self.root.transition(self.root_curr);
      let node = self.matrix.data.node(t.addr);
      assert_eq!(self.curr_len, 0);
      self.inputs[self.curr_len] = t.input;
      self.items[self.curr_len] = (t.output, 0, node);
      self.root_curr += 1;
      self.curr_len = 1;
    }
  }
}

impl<'f, D, I, O, const N: usize> Matrix<D, I, O, N>
where
  I: Input,
  O: Output,
  D: AsRef<[u8]>,
  Bytes<O>: Deserialize,
  Bytes<I>: Deserialize,
  [I; N]: LengthAtMost32,
  [(O, usize, Node<'f, O>); N]: LengthAtMost32,
{
  #[inline]
  pub fn iter(&'f self) -> Iter<'f, D, I, O, N> {
    let root = self.data.root();
    Iter {
      matrix: self,
      root,
      root_curr: 0,
      curr_len: 0,
      inputs: [I::zero(); N],
      items: [(O::zero(), 0, Node::placeholder()); N],
    }
  }
  #[inline]
  pub fn pred<F>(&'f self, pred: F) -> PredIter<'f, F, D, I, O, N>
  where
    F: Fn(&'_ [I]) -> bool, {
    let root = self.data.root();
    PredIter {
      matrix: self,
      root,
      pred,
      root_curr: 0,
      curr_len: 0,
      inputs: [I::zero(); N],
      items: [(O::zero(), 0, Node::placeholder()); N],
    }
  }
}

/// An iterator over keys and values for a matrix
#[derive(Debug, Clone)]
pub struct PredIter<'f, F, D, I, O, const N: usize>
where
  [I; N]: LengthAtMost32,
  [(O, usize, Node<'f, O>); N]: LengthAtMost32, {
  matrix: &'f Matrix<D, I, O, N>,
  root: Node<'f, O>,
  root_curr: usize,
  /// Predicate to be used while checking whether it should continue
  /// Will never be passed an empty slice
  pred: F,
  /// Which inputs have been seen thus far
  inputs: [I; N],
  items: [(O, usize, Node<'f, O>); N],
  curr_len: usize,
}

impl<'f, F, D, I, O, const N: usize> Iterator for PredIter<'f, F, D, I, O, N>
where
  I: Input,
  O: Output,
  D: AsRef<[u8]>,
  Bytes<O>: Deserialize,
  Bytes<I>: Deserialize,
  [I; N]: LengthAtMost32,
  [(O, usize, Node<'f, O>); N]: LengthAtMost32,
  F: Fn(&'_ [I]) -> bool,
{
  type Item = ([I; N], O);
  #[inline]
  fn next(&mut self) -> Option<Self::Item> {
    loop {
      while self.curr_len > 0 {
        while !(self.pred)(&self.inputs[..self.curr_len]) && self.curr_len > 0 {
          self.curr_len -= 1;
        }
        if self.curr_len == 0 {
          break;
        }
        assert!(self.curr_len <= N);
        let is_final = self.curr_len == N;
        let (out, curr, node) = &mut self.items[self.curr_len - 1];
        if *curr < node.num_trans {
          debug_assert!(!is_final);
          *curr += 1;
          let t = node.transition(*curr - 1);
          let next_out = out.cat(&t.output);
          let next_node = self.matrix.data.node(t.addr);
          self.inputs[self.curr_len] = t.input;
          self.items[self.curr_len] = (next_out, 0, next_node);
          self.curr_len += 1;
          continue;
        }
        self.curr_len -= 1;
        if !is_final {
          continue;
        }
        return Some((self.inputs, out.cat(&node.final_output)));
      }
      if self.root_curr >= self.root.num_trans {
        return None;
      }
      let t = self.root.transition(self.root_curr);
      let node = self.matrix.data.node(t.addr);
      assert_eq!(self.curr_len, 0);
      self.inputs[self.curr_len] = t.input;
      self.items[self.curr_len] = (t.output, 0, node);
      self.root_curr += 1;
      self.curr_len = 1;
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
