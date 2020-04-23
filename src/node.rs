use crate::{
  build::BuilderNode,
  bytes::*,
  fst::{u64_to_usize, CompiledAddr, Transition, END_ADDRESS},
  input::Input,
  output::Output,
};
use std::{
  convert::TryInto,
  io::{self, Write},
  iter::{once, Once},
  marker::PhantomData,
  mem::size_of,
  ops::Range,
};

impl<I: Input> BuilderNode<I>
where
  Bytes<I>: Serialize,
{
  pub fn compile<W: io::Write>(
    &self,
    dst: &mut W,
    is_final: bool,
    last_addr: CompiledAddr,
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
pub struct Node<'a> {
  /// Slice over start to last addr of this node
  data: &'a [u8],
  // TODO make this state into one byte?
  // currently it occupies two because we have a byte inside an enum
  pub(crate) state: Striated,
  end: CompiledAddr,
  pub(crate) num_trans: usize,
  sizes: IOSize,
}

impl<'f> Node<'f> {
  pub fn new<I: Input>(addr: CompiledAddr, data: &[u8]) -> Node<'_>
  where
    Bytes<I>: Deserialize, {
    if addr == END_ADDRESS {
      return Self::empty_final();
    }
    let s = Striated(data[addr]);
    let data = &data[..addr + 1];
    let sizes = s.sizes::<I>(data);
    let num_trans = s.num_trans::<I>(data);
    Node {
      data,
      state: s,
      end: s.end_addr::<I>(data, sizes, num_trans),
      num_trans,
      sizes,
    }
  }
  /// Returns a placeholder node which is intended to be used as a default node
  /// with bad values
  fn empty_final() -> Node<'static> {
    Node {
      data: &[],
      state: Striated(0),
      end: END_ADDRESS,
      num_trans: 0,
      sizes: IOSize::new(),
    }
  }
  pub fn placeholder() -> Node<'static> { Self::empty_final() }
  /// Gets the ith transition for this node
  pub(crate) fn transition<I: Input>(&self, i: usize) -> Transition<I>
  where
    Bytes<I>: Deserialize, {
    self.state.transition(&self, i)
  }
  // Returns which # input/transition this byte is if it exists or none otherwise
  pub(crate) fn find_input<I: Input>(&self, b: I) -> Option<usize>
  where
    Bytes<I>: Deserialize, {
    self
      .state
      .trans_iter::<I>(self)
      .position(|t| t.input == b)
  }
  pub fn trans_iter<'a, I: Input + 'a>(&'a self) -> impl Iterator<Item = Transition<I>> + 'a
  where
    Bytes<I>: Deserialize, {
    self.state.trans_iter(&self)
  }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Striated(u8);

impl Striated {
  const fn new() -> Self { Self(0b00_000000) }
  fn compile<W: Write, I: Input>(
    mut wtr: W,
    addr: CompiledAddr,
    node: &BuilderNode<I>,
  ) -> io::Result<()>
  where
    Bytes<I>: Serialize, {
    assert!(node.transitions.len() <= I::max_value().as_usize());
    let mut sink = io::sink();
    let (tbytes, obytes, any_outs) =
      node
        .transitions
        .iter()
        .fold((0, 0, false), |(tbytes, obytes, any_outs), trans| {
          let next_tsize = Pack(delta(addr, trans.addr)).size();
          (
            tbytes.max(next_tsize),
            obytes.max(Bytes(Pack(trans.num_out)).write_le(&mut sink).unwrap()),
            any_outs || trans.num_out > 0,
          )
        });
    let obytes = if any_outs { obytes } else { 0 };
    let iosize = IOSize::sizes(obytes, tbytes);
    let mut state = Self::new();
    state.set_state_num_trans(node.transitions.len());
    for t in node.transitions.iter().rev() {
      let &Transition { input, num_out, .. } = t;
      Bytes(input).write_le(&mut wtr)?;
      assert_eq!(obytes, Bytes(PackTo(num_out, obytes)).write_le(&mut wtr)?);
      let t_written = Bytes(PackTo(delta(addr, t.addr), tbytes)).write_le(&mut wtr)?;
      assert_eq!(t_written, tbytes);
    }
    Bytes(iosize.encode()).write_le(&mut wtr)?;
    if state.state_num_trans().is_none() {
      assert_ne!(node.transitions.len(), 0, "UNREACHABLE 0 should be encoded in state");
      let s = I::from_usize(node.transitions.len() - 1);
      Bytes(s).write_le(&mut wtr)?;
      assert_ne!(state.num_trans_len::<I>(), 0);
    } else {
      assert_eq!(state.num_trans_len::<I>(), 0);
    }
    Bytes(state.0).write_le(&mut wtr)?;
    Ok(())
  }
  fn sizes<I: Input>(self, data: &[u8]) -> IOSize {
    let i = data.len() - 1 - self.num_trans_len::<I>() - 1;
    IOSize::decode(data[i])
  }
  /// Attempts to encode number of transitions in the state
  fn set_state_num_trans(&mut self, n: usize) -> bool {
    if n >= 256 {
      return false;
    }
    let n = (n as u8).saturating_add(1);
    if n <= !0b11_000000 {
      self.0 = (self.0 & 0b11_000000) | n;
      return true;
    }
    false
  }
  /// Number of transitions encoded into the state
  /// If it is 0 it implies that the number of transitions is encoded elsewhere
  fn state_num_trans(self) -> Option<u8> {
    let n = self.0 & !0b11_000000;
    Some(n).filter(|&n| n != 0).map(|n| n - 1)
  }
  // returns the size of the encoded number of transitions
  fn num_trans_len<I: Input>(self) -> usize {
    if self.state_num_trans().is_none() {
      size_of::<I>()
    } else {
      0
    }
  }
  // returns the number of transitions for a given amt of data
  fn num_trans<I: Input>(self, data: &[u8]) -> usize
  where
    Bytes<I>: Deserialize, {
    if let Some(n) = self.state_num_trans() {
      return n as usize;
    }
    let input_len = size_of::<I>();
    Bytes::<I>::read_le(&mut &data[data.len() - 1 - input_len..], input_len as u8)
      .unwrap()
      .inner()
      .as_usize()
      + 1
  }
  fn end_addr<I: Input>(self, data: &[u8], sizes: IOSize, num_trans: usize) -> CompiledAddr {
    let trans_size = size_of::<I>() + sizes.output_bytes() + sizes.transition_bytes();
    data.len()
      - 1
      - self.num_trans_len::<I>()
      - 1 // IOSize
      - num_trans * trans_size
  }
  /// gets the ith transition of this node
  pub fn transition<I: Input>(self, node: &Node<'_>, i: usize) -> Transition<I>
  where
    Bytes<I>: Deserialize, {
    let obytes = node.sizes.output_bytes();
    let tbytes = node.sizes.transition_bytes();
    let ibytes = size_of::<I>();
    let trans_size = ibytes + obytes + tbytes;
    let mut at = node.data.len()
      - 1
      - self.num_trans_len::<I>()
      - 1 // IOSize
      - trans_size * (i+1);
    let input = Bytes::<I>::read_le(&mut &node.data[at..], ibytes as u8)
      .unwrap()
      .inner();
    at += ibytes;
    let num_out = Bytes::<u32>::read_le(&mut &node.data[at..], obytes as u8)
      .unwrap()
      .inner();
    at += obytes;
    let delta = Bytes::<u64>::read_le(&mut &node.data[at..], tbytes as u8)
      .unwrap()
      .inner();
    Transition {
      input,
      num_out,
      addr: undo_delta(node.end, delta),
    }
  }
  pub fn trans_iter<'a, I: Input>(
    self,
    node: &'a Node<'_>,
  ) -> impl Iterator<Item = Transition<I>> + 'a
  where
    Bytes<I>: Deserialize, {
    let obytes = node.sizes.output_bytes();
    let tbytes = node.sizes.transition_bytes();
    let ibytes = size_of::<I>();
    let trans_size = ibytes + obytes + tbytes;
    let mut at = node.data.len()
      - 1
      - self.num_trans_len::<I>()
      - 1;
    (0..node.num_trans).map(move |_| {
      at -= tbytes;
      let delta = Bytes::<u64>::read_le(&mut &node.data[at..], tbytes as u8)
        .unwrap()
        .inner();
      at -= obytes;
      let output = Bytes::<u32>::read_le(&mut &node.data[at..], obytes as u8)
        .unwrap()
        .inner();
      at -= ibytes;
      let input = Bytes::<I>::read_le(&mut &node.data[at..], ibytes as u8)
        .unwrap()
        .inner();
      let addr = undo_delta(node.end, delta);
      Transition {
        input,
        num_out: output,
        addr,
      }
    })
  }
}

/// IOSize represents the input output size for a given node.
/// 4 Transition Addr Size | 4 Output Size | 1 Input Size
/// Ranges of values: [1, 8] | [0, 8] |1, 2]
/// All 0 indicates that there are no transitions or outputs.
// TODO need to encode input bytes as well but maybe want to encode that in the state
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct IOSize(u8);
impl IOSize {
  const TRANS_MASK: u8 = 0b1111_0000;
  const OUT_MASK: u8 = 0b000_1111;
  const fn new() -> Self { Self(0) }
  fn sizes(obytes: u8, tbytes: u8) -> Self {
    let mut out = Self::new();
    out.set_output_bytes(obytes);
    out.set_transition_bytes(tbytes);
    out
  }
  fn set_transition_bytes(&mut self, size: u8) {
    assert!(size <= 8, "Cannot encode transition larger than 8 bytes");
    self.0 = (self.0 & !IOSize::TRANS_MASK) | (size << 4);
  }
  fn transition_bytes(self) -> usize { ((self.0 & IOSize::TRANS_MASK) >> 4) as usize }
  fn set_output_bytes(&mut self, size: u8) {
    assert!(size <= 8, "Cannot encode output size larger than 8 bytes");
    self.0 = (self.0 & !IOSize::OUT_MASK) | size;
  }
  fn output_bytes(self) -> usize { (self.0 & IOSize::OUT_MASK) as usize }
  fn encode(self) -> u8 { self.0 }
  fn decode(v: u8) -> Self { Self(v) }
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
// TODO also return packed size of both addresses here
#[inline]
fn delta(node_addr: CompiledAddr, trans_addr: CompiledAddr) -> usize {
  if trans_addr == END_ADDRESS {
    END_ADDRESS
  } else {
    node_addr - trans_addr
  }
}

/// Takes an address for a node and some delta to a transition and returns that transitions
/// address, or END_ADDRESS if there is no transition
fn undo_delta(node_addr: CompiledAddr, delta: u64) -> CompiledAddr {
  let delta: usize = u64_to_usize(delta);
  if delta == END_ADDRESS {
    END_ADDRESS
  } else {
    node_addr - delta
  }
}
