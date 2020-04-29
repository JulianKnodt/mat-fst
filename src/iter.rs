use crate::{
  bytes::*,
  input::Input,
  matrix::Matrix,
  node::{immediate_iter, immediate_range_iter, Node},
  output::Output,
};
use std::array::LengthAtMost32;

/// An iterator over keys and values for a matrix
#[derive(Debug, Clone)]
pub struct Iter<'f, D, I, O, const N: usize>
where
  [I; N]: LengthAtMost32,
  [(u32, usize, Node<'f, I>); N]: LengthAtMost32, {
  matrix: &'f Matrix<D, I, O, N>,
  root: Node<'f, I>,
  root_curr: usize,
  /// Which inputs have been seen thus far
  inputs: [I; N],
  items: [(u32, usize, Node<'f, I>); N],
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
  [(u32, usize, Node<'f, I>); N]: LengthAtMost32,
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
          let next_out = out.cat(&t.num_out);
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
        return Some((self.inputs, self.matrix.data.outputs[*out as usize]));
      }
      if self.root_curr >= self.root.num_trans {
        return None;
      }
      let t = self.root.transition(self.root_curr);
      let node = self.matrix.data.node(t.addr);
      assert_eq!(self.curr_len, 0);
      self.inputs[self.curr_len] = t.input;
      self.items[self.curr_len] = (t.num_out, 0, node);
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
  [(u32, usize, Node<'f, I>); N]: LengthAtMost32,
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
      items: [(0, 0, Node::placeholder()); N],
    }
  }
  /*
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
  */
}

/*
/// An iterator over keys and values for a matrix
#[derive(Debug, Clone)]
pub struct PredIter<'f, F, D, I, O, const N: usize>
where
  [I; N]: LengthAtMost32,
  [(u32, usize, Node<'f>); N]: LengthAtMost32, {
  matrix: &'f Matrix<D, I, O, N>,
  root: Node<'f>,
  root_curr: usize,
  /// Predicate to be used while checking whether it should continue
  /// Will never be passed an empty slice
  pred: F,
  /// Which inputs have been seen thus far
  inputs: [I; N],
  items: [(u32, usize, Node<'f>); N],
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
  [(u32, usize, Node<'f>); N]: LengthAtMost32,
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
          let next_out = out.cat(&t.num_out);
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
        return Some((self.inputs, out.cat(&node.num_out)));
      }
      if self.root_curr >= self.root.num_trans {
        return None;
      }
      let t = self.root.transition(self.root_curr);
      let node = self.matrix.data.node(t.addr);
      assert_eq!(self.curr_len, 0);
      self.inputs[self.curr_len] = t.input;
      self.items[self.curr_len] = (t.num_out, 0, node);
      self.root_curr += 1;
      self.curr_len = 1;
    }
  }
}
*/

/*
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
  state: Vec<(O, Range<usize>, Node<'f>)>,
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
      let out = last.0.cat(&t.num_out);
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
*/

impl<D, I, O> Matrix<D, I, O, 2>
where
  I: Input,
  O: Output,
  D: AsRef<[u8]>,
  Bytes<O>: Deserialize,
  Bytes<I>: Deserialize,
{
  pub fn eager_iter<F>(&self, mut f: F)
  where
    F: FnMut([I; 2], O), {
    let data = self.data.data.as_ref();
    for t0 in immediate_iter(self.data.meta.root_addr, data) {
      for (i, t1) in immediate_range_iter(t0.addr, data).enumerate() {
        f(
          [t0.input, t1],
          self.data.outputs[(t0.num_out + i as u32) as usize],
        )
      }
    }
  }
  pub fn iter2(&self) -> impl Iterator<Item = ([I; 2], O)> + '_ {
    let data = self.data.data.as_ref();
    immediate_iter(self.data.meta.root_addr, data).flat_map(move |t0| {
      immediate_range_iter(t0.addr, data)
        .enumerate()
        .map(move |(i, t1)| {
          (
            [t0.input, t1],
            self.data.outputs[(t0.num_out + i as u32) as usize],
          )
        })
    })
  }
}
