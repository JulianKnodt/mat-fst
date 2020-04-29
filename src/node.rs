use crate::{
  build::BuilderNode,
  bytes::*,
  fst::{u64_to_usize, CompiledAddr, Transition, END_ADDRESS},
  input::Input,
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::{
  io::{self, Write},
  marker::PhantomData,
  mem::size_of,
};

impl<I: Input> BuilderNode<I>
where
  Bytes<I>: Serialize,
{
  #[inline]
  pub fn compile<W: io::Write>(
    &self,
    dst: &mut W,
    is_final: bool,
    _last_addr: CompiledAddr,
    addr: CompiledAddr,
  ) -> io::Result<()> {
    assert!(self.transitions.len() <= I::max_value().as_usize());
    if self.transitions.is_empty() && is_final {
      Ok(())
    } else {
      Striated::compile(dst, addr, &self)
    }
  }
}

#[derive(Debug, Clone, Copy)]
pub struct Node<'a, I> {
  _input_type: PhantomData<I>,
  /// Slice over start to last addr of this node
  data: &'a [u8],
  // TODO make this state into one byte?
  // currently it occupies two because we have a byte inside an enum
  pub(crate) state: Striated,
  end: CompiledAddr,
  pub(crate) num_trans: usize,
  sizes: IOSize,
}

impl<'f, I: Input> Node<'f, I>
where
  Bytes<I>: Deserialize,
{
  pub fn new(addr: CompiledAddr, data: &[u8]) -> Node<'_, I> {
    if addr == END_ADDRESS {
      return Self::empty_final();
    }
    let s = Striated(data[addr]);
    let data = &data[..addr + 1];
    let sizes = s.sizes(data);
    let num_trans = s.num_trans::<I>(data);
    Node {
      _input_type: PhantomData,
      data,
      state: s,
      end: s.end_addr::<I>(data, sizes, num_trans),
      num_trans,
      sizes,
    }
  }
  /// Returns a placeholder node which is intended to be used as a default node
  /// with bad values
  fn empty_final() -> Node<'static, I> {
    Node {
      _input_type: PhantomData,
      data: &[],
      state: Striated(0),
      end: END_ADDRESS,
      num_trans: 0,
      sizes: IOSize::new(),
    }
  }
  pub fn placeholder() -> Node<'static, I> { Self::empty_final() }
  /// Gets the ith transition for this node
  pub(crate) fn transition(&self, i: usize) -> Transition<I> { self.state.transition(&self, i) }
  /// Returns which # input/transition this byte is if it exists or none otherwise
  #[inline]
  pub(crate) fn find_input(&self, b: I) -> Option<usize> {
    self.state.trans_iter::<I>(self).position(|t| t.input == b)
  }
  #[inline]
  pub fn trans_iter<'a>(&'a self) -> impl Iterator<Item = Transition<I>> + 'a
  where
    I: 'a, {
    self.state.trans_iter(&self)
  }
  #[inline]
  pub fn range_iter<'a>(&'a self) -> impl Iterator<Item = Transition<I>> + 'a
  where
    I: 'a, {
    self.state.range_iter(&self)
  }
  #[cfg(feature = "parallel")]
  pub fn par_trans_iter<'a>(&'a self) -> impl ParallelIterator<Item = Transition<I>> + 'a
  where
    I: 'a, {
    self.state.par_trans_iter(&self)
  }
  #[cfg(feature = "parallel")]
  pub fn par_range_iter<'a>(&'a self) -> impl ParallelIterator<Item = Transition<I>> + 'a
  where
    I: 'a, {
    self.state.par_range_iter(&self)
  }
  pub fn next_sibling(&self) -> Node<'f, I> { Node::new(self.end, self.data) }
}

fn is_range_iter<V: Iterator<Item = u32>>(vs: V) -> bool {
  vs.enumerate().all(|(i, v)| v == i as u32)
}

fn mismatches<I: Input, V: Iterator<Item = I>>(is: V) -> impl Iterator<Item = I> {
  let mut mismatch = 0;
  is.enumerate().filter_map(move |(i, v)| {
    if i + mismatch == v.as_usize() {
      None
    } else {
      mismatch = v.as_usize() - i;
      Some(v)
    }
  })
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Striated(u8);

const RANGE_MASK: u8 = 0b1000_0000;
const SIZE_MASK: u8 = 0b0110_0000;
const INPUT_RANGE_MASK: u8 = 0b0001_0000;

impl Striated {
  const fn new() -> Self { Self(0b00_000000) }
  fn compile<W: Write, I: Input>(
    mut wtr: W,
    addr: CompiledAddr,
    node: &BuilderNode<I>,
  ) -> io::Result<()>
  where
    Bytes<I>: Serialize,
    Bytes<I>: Serialize, {
    assert!(node.transitions.len() <= I::max_value().as_usize());
    let mut sink = io::sink();
    let (ibytes, tbytes, obytes, any_outs) =
      node
        .transitions
        .iter()
        .fold((1, 0, 0, false), |(ibytes, tbytes, obytes, any_outs), t| {
          (
            ibytes.max(Bytes(t.input).pack(&mut sink).unwrap()),
            tbytes.max(Pack(delta(addr, t.addr)).size()),
            obytes.max(Bytes(t.num_out).pack(&mut sink).unwrap()),
            any_outs || t.num_out > 0,
          )
        });
    let is_range = is_range_iter(node.transitions.iter().map(|v| v.num_out));
    // can only have one of is_range and is_input_range currently
    let is_input_range =
      !is_range && mismatches(node.transitions.iter().map(|v| v.input)).count() == 0;
    let ibytes = if is_input_range { 0 } else { ibytes };
    let obytes = if !is_range && any_outs { obytes } else { 0 };
    let iosize = IOSize::sizes(obytes, tbytes);
    let mut state = Self::new();
    state.set_input_bytes(ibytes);
    if is_range {
      debug_assert_eq!(node.transitions[0].num_out, 0);
      state.set_range();
    }
    if is_input_range {
      state.set_input_range();
    }
    for t in node.transitions.iter().rev() {
      let &Transition { input, num_out, .. } = t;
      if !is_input_range {
        Bytes(input).pack_to(ibytes, &mut wtr)?;
      }
      Bytes(num_out).pack_to(obytes, &mut wtr)?;
      Bytes(delta(addr, t.addr)).pack_to(tbytes, &mut wtr)?;
    }
    Bytes(iosize.encode()).write_le(&mut wtr)?;
    let s = (node.transitions.len() - 1) as u32;
    Bytes(s).write_le(&mut wtr)?;
    Bytes(state.0).write_le(&mut wtr)?;
    Ok(())
  }
  fn sizes(self, data: &[u8]) -> IOSize {
    let i = data.len() - 1 - self.num_trans_len() - 1;
    IOSize::decode(data[i])
  }
  // returns the size of the encoded number of transitions
  const fn num_trans_len(self) -> usize { size_of::<u32>() }
  // returns the number of transitions for a given amt of data
  fn num_trans<I: Input>(self, data: &[u8]) -> usize {
    let len = self.num_trans_len();
    Bytes::<u32>::read_le(&mut &data[data.len() - 1 - len..], len as u8)
      .unwrap()
      .inner() as usize
      + 1
  }
  fn end_addr<I: Input>(self, data: &[u8], sizes: IOSize, num_trans: usize) -> CompiledAddr {
    let trans_size = self.input_bytes() + sizes.output_bytes() + sizes.transition_bytes();
    data.len()
      - 1
      - self.num_trans_len()
      - 1 // IOSize
      - num_trans * trans_size
  }
  /// Used to denote that the input is repeating by some frequency
  fn set_range(&mut self) { self.0 |= RANGE_MASK; }
  const fn is_range(self) -> bool { self.0 & RANGE_MASK == RANGE_MASK }

  fn set_input_range(&mut self) { self.0 |= INPUT_RANGE_MASK; }
  const fn is_input_range(self) -> bool { self.0 & INPUT_RANGE_MASK == INPUT_RANGE_MASK }

  fn set_input_bytes(&mut self, size: u8) {
    assert!(size < 4);
    // assert!(size > 0, "Encoding one transition of size 0");
    self.0 |= (size << 5) & SIZE_MASK;
  }
  const fn input_bytes(self) -> usize { ((self.0 & SIZE_MASK) >> 5) as usize }
  /// gets the ith transition of this node
  pub fn transition<I: Input>(self, node: &Node<'_, I>, i: usize) -> Transition<I>
  where
    Bytes<I>: Deserialize, {
    let obytes = node.sizes.output_bytes();
    let tbytes = node.sizes.transition_bytes();
    let ibytes = self.input_bytes();
    let trans_size = ibytes + obytes + tbytes;
    let at = node.data.len()
      - 1
      - self.num_trans_len()
      - 1 // IOSize
      - trans_size * (i+1);
    let reader = &mut &node.data[at..];
    let input = if self.is_input_range() {
      assert_eq!(ibytes, 0);
      I::from_usize(i)
    } else {
      Bytes::<I>::read_le(reader, ibytes as u8).unwrap().inner()
    };
    let num_out = if self.is_range() {
      i as u32
    } else {
      Bytes::<u32>::read_le(reader, obytes as u8).unwrap().inner()
    };
    let delta = Bytes::<u64>::read_le(reader, tbytes as u8).unwrap().inner();
    Transition {
      input,
      num_out,
      addr: undo_delta(node.end, delta),
    }
  }
  #[inline]
  pub fn trans_iter<'a, I: Input>(
    self,
    node: &'a Node<'_, I>,
  ) -> impl Iterator<Item = Transition<I>> + 'a
  where
    Bytes<I>: Deserialize, {
    let obytes = node.sizes.output_bytes();
    let tbytes = node.sizes.transition_bytes();
    let ibytes = self.input_bytes();
    let trans_size = ibytes + obytes + tbytes;
    let mut at = node.data.len() - 1 - self.num_trans_len() - 1;
    let is_range = self.is_range();
    (0..node.num_trans).map(move |i| {
      at -= trans_size;
      let reader = &mut &node.data[at..];
      let input = if ibytes == 0 {
        I::from_usize(i)
      } else {
        Bytes::<I>::read_le(reader, ibytes as u8).unwrap().inner()
      };
      let output = if is_range {
        i as u32
      } else {
        Bytes::<u32>::read_le(reader, obytes as u8).unwrap().inner()
      };
      let addr = if tbytes == 0 {
        END_ADDRESS
      } else {
        let delta = Bytes::<u64>::read_le(reader, tbytes as u8).unwrap().inner();
        undo_delta(node.end, delta)
      };
      Transition {
        input,
        num_out: output,
        addr,
      }
    })
  }
  /// Iterates under the assumption that this node's outputs form a range
  /// Panics if this is not a range.
  #[inline]
  pub fn range_iter<'a, I: Input>(
    self,
    node: &'a Node<'_, I>,
  ) -> impl Iterator<Item = Transition<I>> + 'a
  where
    Bytes<I>: Deserialize, {
    assert_eq!(node.sizes.transition_bytes(), 0);
    assert_eq!(node.sizes.output_bytes(), 0);
    let ibytes = self.input_bytes();
    let trans_size = ibytes;
    let mut at = node.data.len() - 1 - self.num_trans_len() - 1;
    assert!(self.is_range());
    (0..node.num_trans).map(move |i| {
      at -= trans_size;
      let reader = &mut &node.data[at..];
      let input = if ibytes == 0 {
        I::from_usize(i)
      } else {
        Bytes::<I>::read_le(reader, ibytes as u8).unwrap().inner()
      };
      Transition {
        input,
        num_out: i as u32,
        addr: END_ADDRESS,
      }
    })
  }

  #[cfg(feature = "parallel")]
  pub fn par_trans_iter<'a, I: Input>(
    self,
    node: &'a Node<'_, I>,
  ) -> impl ParallelIterator<Item = Transition<I>> + 'a
  where
    Bytes<I>: Deserialize, {
    let obytes = node.sizes.output_bytes();
    let tbytes = node.sizes.transition_bytes();
    let ibytes = self.input_bytes();
    let trans_size = ibytes + obytes + tbytes;
    let mut start = node.data.len() - 1 - self.num_trans_len() - 1;
    let is_range = self.is_range();
    (0..node.num_trans).into_par_iter().map(move |i| {
      let at = start - (i + 1) * trans_size;
      let reader = &mut &node.data[at..];
      let input = Bytes::<I>::read_le(reader, ibytes as u8).unwrap().inner();
      let output = if is_range {
        i as u32
      } else {
        Bytes::<u32>::read_le(reader, obytes as u8).unwrap().inner()
      };
      let addr = if tbytes == 0 {
        END_ADDRESS
      } else {
        let delta = Bytes::<u64>::read_le(reader, tbytes as u8).unwrap().inner();
        undo_delta(node.end, delta)
      };
      Transition {
        input,
        num_out: output,
        addr,
      }
    })
  }
  #[cfg(feature = "parallel")]
  pub fn par_range_iter<'a, I: Input>(
    self,
    node: &'a Node<'_, I>,
  ) -> impl ParallelIterator<Item = Transition<I>> + 'a
  where
    Bytes<I>: Deserialize, {
    assert_eq!(node.sizes.transition_bytes(), 0);
    assert_eq!(node.sizes.output_bytes(), 0);
    let ibytes = self.input_bytes();
    let mut start = node.data.len() - 1 - self.num_trans_len() - 1;
    assert!(self.is_range());
    (0..node.num_trans).into_par_iter().map(move |i| {
      let at = start - (i + 1) * ibytes;
      let reader = &mut &node.data[at..];
      let input = Bytes::<I>::read_le(reader, ibytes as u8).unwrap().inner();
      Transition {
        input,
        num_out: i as u32,
        addr: END_ADDRESS,
      }
    })
  }
}

/// IOSize represents the input output size for a given node.
/// 4 Transition Addr Size | 4 Output Size | 1 Input Size
/// Ranges of values: [1, 8] | [0, 8] |1, 2]
/// All 0 indicates that there are no transitions or outputs.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) struct IOSize(u8);
impl IOSize {
  const TRANS_MASK: u8 = 0b1111_0000;
  const OUT_MASK: u8 = 0b000_1111;
  const fn new() -> Self { Self(0) }
  pub fn sizes(obytes: u8, tbytes: u8) -> Self {
    let mut out = Self::new();
    out.set_output_bytes(obytes);
    out.set_transition_bytes(tbytes);
    out
  }
  pub fn set_transition_bytes(&mut self, size: u8) {
    assert!(size <= 8, "Cannot encode transition larger than 8 bytes");
    self.0 = (self.0 & !IOSize::TRANS_MASK) | (size << 4);
  }
  pub fn transition_bytes(self) -> usize { ((self.0 & IOSize::TRANS_MASK) >> 4) as usize }
  pub fn set_output_bytes(&mut self, size: u8) {
    assert!(size <= 8, "Cannot encode output size larger than 8 bytes");
    self.0 = (self.0 & !IOSize::OUT_MASK) | size;
  }
  pub fn output_bytes(self) -> usize { (self.0 & IOSize::OUT_MASK) as usize }
  pub fn encode(self) -> u8 { self.0 }
  pub fn decode(v: u8) -> Self { Self(v) }
}

#[cfg(test)]
mod iosize_tests {
  use super::IOSize;
  #[test]
  fn basic() {
    for ts in 0..=8 {
      for os in 0..=8 {
        let mut ios = IOSize::new();
        ios.set_transition_bytes(ts);
        assert_eq!(ios.transition_bytes(), ts as usize);
        ios.set_output_bytes(os);
        assert_eq!(ios.output_bytes(), os as usize);
        assert_eq!(ios.transition_bytes(), ts as usize);
        ios.set_transition_bytes(ts);
        assert_eq!(ios.output_bytes(), os as usize);
      }
    }
  }
}

/// Returns the necessary amount to go from trans address to node address
#[inline]
pub(crate) fn delta(node_addr: CompiledAddr, trans_addr: CompiledAddr) -> usize {
  if trans_addr == END_ADDRESS {
    END_ADDRESS
  } else {
    node_addr - trans_addr
  }
}

/// Takes an address for a node and some delta to a transition and returns that transitions
/// address, or END_ADDRESS if there is no transition
#[inline]
pub(crate) fn undo_delta(node_addr: CompiledAddr, delta: u64) -> CompiledAddr {
  let delta: usize = u64_to_usize(delta);
  if delta == END_ADDRESS {
    END_ADDRESS
  } else {
    node_addr - delta
  }
}
// Instead of creating a node just create an iterator over transitions
// removing the intermediate node structure.
pub fn immediate_iter<I: Input>(
  addr: CompiledAddr,
  data: &[u8],
) -> impl Iterator<Item = Transition<I>> + '_
where
  Bytes<I>: Deserialize, {
  assert_ne!(addr, END_ADDRESS, "Cannot iterate over end address");
  let s = Striated(data[addr]);
  let data = &data[..addr + 1];
  let sizes = s.sizes(data);
  let num_trans = s.num_trans::<I>(data);
  let obytes = sizes.output_bytes();
  let tbytes = sizes.transition_bytes();
  let ibytes = s.input_bytes();
  let trans_size = ibytes + obytes + tbytes;
  let mut at = addr - s.num_trans_len() - 1;
  let end_addr = at - num_trans * trans_size;
  let is_range = s.is_range();
  let is_input_range = s.is_input_range();
  (0..num_trans).map(move |i| {
    at -= trans_size;
    let reader = &mut &data[at..];
    let input = if is_input_range {
      I::from_usize(i)
    } else {
      Bytes::<I>::read_le(reader, ibytes as u8).unwrap().inner()
    };
    let output = if is_range {
      i as u32
    } else {
      Bytes::<u32>::read_le(reader, obytes as u8).unwrap().inner()
    };
    let addr = if tbytes == 0 {
      END_ADDRESS
    } else {
      let delta = Bytes::<u64>::read_le(reader, tbytes as u8).unwrap().inner();
      undo_delta(end_addr, delta)
    };
    Transition {
      input,
      num_out: output,
      addr,
    }
  })
}

// Instead of creating a node just create an iterator over transitions
// removing the intermediate node structure.
pub fn immediate_range_iter<I: Input>(
  addr: CompiledAddr,
  data: &[u8],
) -> impl Iterator<Item = I> + '_
where
  Bytes<I>: Deserialize, {
  assert_ne!(addr, END_ADDRESS, "Cannot iterate over end address");
  let s = Striated(data[addr]);
  let data = &data[..addr + 1];
  let num_trans = s.num_trans::<I>(data);
  assert!(s.is_range());
  assert!(!s.is_input_range());
  let ibytes = s.input_bytes();
  let mut at = addr - s.num_trans_len() - 1;
  (0..num_trans).map(move |_| {
    at -= ibytes;
    let reader = &mut &data[at..];
    Bytes::<I>::read_le(reader, ibytes as u8).unwrap().inner()
  })
}

#[cfg(feature = "parallel")]
// Instead of creating a node just create an iterator over transitions
// removing the intermediate node structure.
pub fn par_immediate_iter<I: Input>(
  addr: CompiledAddr,
  data: &[u8],
) -> impl IndexedParallelIterator<Item = Transition<I>> + '_
where
  Bytes<I>: Deserialize, {
  assert_ne!(addr, END_ADDRESS, "Cannot iterate over end address");
  let s = Striated(data[addr]);
  let data = &data[..addr + 1];
  let sizes = s.sizes(data);
  let num_trans = s.num_trans::<I>(data);
  let obytes = sizes.output_bytes();
  let tbytes = sizes.transition_bytes();
  let ibytes = s.input_bytes();
  let trans_size = ibytes + obytes + tbytes;
  let start = addr - s.num_trans_len() - 1;
  let end_addr = start - num_trans * trans_size;
  let is_range = s.is_range();
  let is_input_range = s.is_input_range();
  (0..num_trans).into_par_iter().map(move |i| {
    let at = start - (i + 1) * trans_size;
    let reader = &mut &data[at..];
    let input = if is_input_range {
      I::from_usize(i)
    } else {
      Bytes::<I>::read_le(reader, ibytes as u8).unwrap().inner()
    };
    let output = if is_range {
      i as u32
    } else {
      Bytes::<u32>::read_le(reader, obytes as u8).unwrap().inner()
    };
    let addr = if tbytes == 0 {
      END_ADDRESS
    } else {
      let delta = Bytes::<u64>::read_le(reader, tbytes as u8).unwrap().inner();
      undo_delta(end_addr, delta)
    };
    Transition {
      input,
      num_out: output,
      addr,
    }
  })
}

#[cfg(feature = "parallel")]
pub fn par_immediate_range_iter<I: Input>(
  addr: CompiledAddr,
  data: &[u8],
) -> impl IndexedParallelIterator<Item = I> + '_
where
  Bytes<I>: Deserialize, {
  assert_ne!(addr, END_ADDRESS, "Cannot iterate over end address");
  let s = Striated(data[addr]);
  let data = &data[..addr + 1];
  let sizes = s.sizes(data);
  let num_trans = s.num_trans::<I>(data);
  assert!(s.is_range());
  assert!(!s.is_input_range());
  let ibytes = s.input_bytes();
  let start = addr - s.num_trans_len() - 1;
  (0..num_trans).into_par_iter().map(move |i| {
    let at = start - (i + 1) * ibytes;
    let reader = &mut &data[at..];
    Bytes::<I>::read_le(reader, ibytes as u8).unwrap().inner()
  })
}
