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

impl<I: Input, O: Output> BuilderNode<I, O>
where
  Bytes<I>: Serialize,
  Bytes<O>: Serialize,
{
  pub fn compile<W: io::Write>(
    &self,
    dst: &mut W,
    last_addr: CompiledAddr,
    addr: CompiledAddr,
  ) -> io::Result<()> {
    assert!(self.transitions.len() <= I::max_value().as_usize());
    if self.transitions.is_empty() && self.is_final && self.final_output.is_zero() {
      Ok(())
    } else if self.transitions.len() != 1 || self.is_final {
      StateAnyTrans::compile(dst, addr, &self)
    } else {
      let t = &self.transitions[0];
      if !self.is_final && t.addr == last_addr && t.output.is_zero() {
        StateOneTransNext::compile(dst, addr, t.input)
      } else {
        // TODO check if this is cloning t
        StateOneTrans::compile(dst, addr, *t)
      }
    }
  }
}

#[derive(Debug, Clone, Copy)]
pub struct Node<'a, O> {
  /// Slice over start to last addr of this node
  data: &'a [u8],
  pub(crate) state: State,
  start: CompiledAddr,
  end: CompiledAddr,
  // TODO trim this is final
  pub(crate) is_final: bool,
  pub(crate) num_trans: usize,
  sizes: IOSize,
  // TODO trim this final output
  pub(crate) final_output: O,
}

pub enum NodeTransIter<I, O, T> {
  OneTransNext(Once<(I, CompiledAddr)>),
  OneTrans(Once<Transition<I, O>>),
  AnyTrans(T),
}

impl<I, O, T> Iterator for NodeTransIter<I, O, T>
where
  I: Input,
  O: Output,
  T: Iterator<Item = Transition<I, O>>,
  Bytes<O>: Deserialize,
  Bytes<I>: Deserialize,
{
  type Item = Transition<I, O>;
  fn next(&mut self) -> Option<Self::Item> {
    match self {
      NodeTransIter::OneTransNext(o) => o.next().map(|(input, addr)| Transition {
        input,
        output: O::zero(),
        addr,
      }),
      NodeTransIter::OneTrans(o) => o.next(),
      NodeTransIter::AnyTrans(trans) => trans.next(),
    }
  }
}

impl<'f, O: Output> Node<'f, O>
where
  Bytes<O>: Deserialize,
{
  pub fn new<I: Input>(addr: CompiledAddr, data: &[u8]) -> Node<'_, O>
  where
    Bytes<I>: Deserialize, {
    let state = State::new(data, addr);
    match state {
      State::EmptyFinal => Self::empty_final(),
      State::OneTransNext(s) => {
        let data = &data[..addr + 1];
        Node {
          data,
          state,
          start: addr,
          end: s.end_addr::<I>(data),
          is_final: false,
          sizes: IOSize::new(),
          num_trans: 1,
          final_output: O::zero(),
        }
      },
      State::OneTrans(s) => {
        let data = &data[..addr + 1];
        let sizes = s.sizes::<I>(data);
        Node {
          data,
          state,
          start: addr,
          end: s.end_addr::<I>(data, sizes),
          is_final: false,
          num_trans: 1,
          sizes,
          final_output: O::zero(),
        }
      },
      State::AnyTrans(s) => {
        let data = &data[..addr + 1];
        let sizes = s.sizes::<I>(data);
        let num_trans = s.num_trans::<I>(data);
        Node {
          data,
          state,
          start: addr,
          end: s.end_addr::<I>(data, sizes, num_trans),
          is_final: s.is_final(),
          num_trans,
          sizes,
          final_output: s.final_output::<I, O>(data, sizes, num_trans),
        }
      },
    }
  }
  fn empty_final() -> Node<'static, O> {
    Node {
      data: &[],
      state: State::EmptyFinal,
      start: END_ADDRESS,
      end: END_ADDRESS,
      is_final: true,
      num_trans: 0,
      sizes: IOSize::new(),
      final_output: O::zero(),
    }
  }
  /// Returns a placeholder node which is not intended for use
  pub fn placeholder() -> Node<'static, O> { Self::empty_final() }
  /// Gets the ith transition for this node
  pub(crate) fn transition<I: Input>(&self, i: usize) -> Transition<I, O>
  where
    Bytes<I>: Deserialize,
    Bytes<O>: Deserialize, {
    let (input, output, addr): (I, O, usize) = match self.state {
      State::EmptyFinal => unreachable!(),
      State::OneTransNext(s) => {
        assert_eq!(i, 0);
        (s.input(self), O::zero(), s.trans_addr(self))
      },
      State::OneTrans(s) => {
        assert_eq!(i, 0);
        (
          s.input(self),
          s.output::<I, O>(self),
          s.trans_addr::<I, O>(self),
        )
      },
      State::AnyTrans(s) => (
        s.input(self, i),
        s.output::<I, O>(self, i),
        s.trans_addr::<I, O>(self, i),
      ),
    };
    Transition {
      input,
      output,
      addr,
    }
  }
  // Returns which # input/transition this byte is if it exists or none otherwise
  pub(crate) fn find_input<I: Input>(&self, b: I) -> Option<usize>
  where
    Bytes<I>: Deserialize, {
    match self.state {
      State::EmptyFinal => None,

      State::OneTransNext(s) if s.input::<I, O>(self) == b => Some(0),
      State::OneTransNext(_) => None,

      State::OneTrans(s) if s.input::<I, O>(self) == b => Some(0),
      State::OneTrans(_) => None,

      State::AnyTrans(s) => s.find_input(self, b),
    }
  }
  /// Returns the range of inputs from start to end inclusive.
  pub(crate) fn find_input_range<I: Input>(&self, range: &Range<I>) -> Range<usize>
  where
    Bytes<I>: Deserialize, {
    match self.state {
      State::EmptyFinal => 0..0,

      State::OneTransNext(s) if range.contains(&s.input::<I, O>(self)) => 0..1,
      State::OneTransNext(_) => 0..0,

      State::OneTrans(s) if range.contains(&s.input::<I, O>(self)) => 0..1,
      State::OneTrans(_) => 0..0,

      State::AnyTrans(s) => s.find_input_range(self, range),
    }
  }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum State {
  // 1 trans | next | common input
  OneTransNext(StateOneTransNext),
  // 1 trans | !next | common input
  OneTrans(StateOneTrans),
  // !1 trans | ?final | # transitions
  AnyTrans(StateAnyTrans),
  EmptyFinal,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct StateOneTransNext(u8);
#[derive(Clone, Copy, Debug)]
pub(crate) struct StateOneTrans(u8);
#[derive(Clone, Copy, Debug)]
pub(crate) struct StateAnyTrans(u8);

const TRANS_NEXT_MASK: u8 = 0b11_000000;
const MARKER_BITS: u8 = !TRANS_NEXT_MASK;
const TRANS_AND_NEXT: u8 = 0b11;
const TRANS_NOT_NEXT: u8 = 0b10;

impl State {
  fn new(data: &[u8], addr: CompiledAddr) -> Self {
    if addr == END_ADDRESS {
      return State::EmptyFinal;
    }
    let v = data[addr];
    match (v & TRANS_NEXT_MASK) >> 6 {
      TRANS_AND_NEXT => State::OneTransNext(StateOneTransNext(v)),
      TRANS_NOT_NEXT => State::OneTrans(StateOneTrans(v)),
      _ => State::AnyTrans(StateAnyTrans(v)),
    }
  }
}

impl StateOneTransNext {
  const fn new() -> Self { Self(TRANS_NEXT_MASK) }
  fn compile<W: Write, I: Input>(mut wtr: W, _: CompiledAddr, input: I) -> io::Result<()>
  where
    Bytes<I>: Serialize, {
    let mut state = StateOneTransNext::new();
    Bytes(input).write_le(&mut wtr)?;
    Bytes(state.0).write_le(&mut wtr)?;
    Ok(())
  }
  const fn input_len<I: Input>(self) -> usize { size_of::<I>() }
  const fn end_addr<I: Input>(self, data: &[u8]) -> usize { data.len() - 1 - self.input_len::<I>() }
  fn input<I: Input, O>(self, node: &Node<'_, O>) -> I
  where
    Bytes<I>: Deserialize, {
    Bytes::read_le(
      &mut &node.data[node.start - self.input_len::<I>()..],
      self.input_len::<I>() as u8,
    )
    .unwrap()
    .inner()
    // node.data[node.start - 1]
  }
  const fn trans_addr<O>(self, node: &Node<'_, O>) -> CompiledAddr { node.end - 1 }
  fn trans_iter<I: Input, O>(self, node: &Node<'_, O>) -> Once<(I, CompiledAddr)>
  where
    Bytes<I>: Deserialize, {
    once((self.input(node), node.end - 1))
  }
}

impl StateOneTrans {
  const fn new() -> Self { Self(0b10_000000) }
  fn compile<W: Write, I: Input, O: Output>(
    mut wtr: W,
    addr: CompiledAddr,
    trans: Transition<I, O>,
  ) -> io::Result<()>
  where
    Bytes<O>: Serialize,
    Bytes<I>: Serialize, {
    let out_bytes = if trans.output.is_zero() {
      0
    } else {
      Bytes(trans.output).write_le(&mut wtr)?
    };
    let trans_bytes = Bytes(delta(addr, trans.addr)).write_le(&mut wtr)?;
    let mut io_sizes = IOSize::new();
    io_sizes.set_output_bytes(out_bytes);
    io_sizes.set_transition_bytes(trans_bytes);
    Bytes(io_sizes.encode()).write_le(&mut wtr)?;
    let _in_bytes = Bytes(trans.input).write_le(&mut wtr)?;
    // TODO encode in_bytes into state?
    let mut state = Self::new();
    Bytes(state.0).write_le(&mut wtr)?;
    Ok(())
  }
  fn sizes<I: Input>(self, data: &[u8]) -> IOSize {
    let i = data.len() - 2 - self.input_len::<I>();
    IOSize::decode(data[i])
  }
  const fn input_len<I: Input>(self) -> usize { size_of::<I>() }

  /// returns index at start of node
  fn end_addr<I: Input>(self, data: &[u8], sizes: IOSize) -> usize {
    data.len() - 1
      - self.input_len::<I>()
      - 1 // IOSize byte
      - sizes.transition_bytes()
      - sizes.output_bytes()
  }
  fn input<I: Input, O>(self, node: &Node<'_, O>) -> I
  where
    Bytes<I>: Deserialize, {
    Bytes::read_le(
      &mut &node.data[node.start - self.input_len::<I>()..],
      self.input_len::<I>() as u8,
    )
    .unwrap()
    .inner()
    // node.data[node.start - 1]
  }
  fn output<I: Input, O: Output>(self, node: &Node<'_, O>) -> O
  where
    Bytes<O>: Deserialize, {
    let osize = node.sizes.output_bytes();
    if osize == 0 {
      return O::zero();
    }
    let i = node.start
      - 1 // IOSize
      - self.input_len::<I>()
      - node.sizes.transition_bytes()
      - osize;
    Bytes::<O>::read_le(&mut &node.data[i..], osize as u8)
      .unwrap()
      .inner()
  }
  /// Returns the address of the next transition for this state
  fn trans_addr<I: Input, O: Output>(self, node: &Node<'_, O>) -> CompiledAddr
  where
    Bytes<O>: Deserialize, {
    let tsize = node.sizes.transition_bytes();
    // assert_ne!(tsize, 0, "Encoded 0 size transition for one trans");
    let i = node.start
      - 1 // IOSize
      - self.input_len::<I>()
      - tsize;
    let delta = Bytes::<u64>::read_le(&mut &node.data[i..], tsize as u8)
      .unwrap()
      .inner();
    undo_delta(node.end, delta)
  }
  fn trans_iter<I: Input, O: Output>(self, node: &Node<'_, O>) -> Once<Transition<I, O>>
  where
    Bytes<I>: Deserialize,
    Bytes<O>: Deserialize, {
    let input_size = self.input_len::<I>();
    let mut i = node.start - 1 - input_size;
    let input = Bytes::<I>::read_le(&mut &node.data[i..], input_size as u8)
      .unwrap()
      .inner();
    let tsize = node.sizes.transition_bytes();
    let delta = if tsize == 0 {
      0
    } else {
      i = i - tsize;
      Bytes::<u64>::read_le(&mut &node.data[i..], tsize as u8)
        .unwrap()
        .inner()
    };
    let addr = undo_delta(node.end, delta);
    let osize = node.sizes.output_bytes();
    let output = if osize == 0 {
      O::zero()
    } else {
      let i = i - osize;
      Bytes::<O>::read_le(&mut &node.data[i..], osize as u8)
        .unwrap()
        .inner()
    };
    once(Transition {
      input,
      addr,
      output,
    })
  }
}

impl StateAnyTrans {
  const fn new() -> Self { Self(0b00_000000) }
  fn compile<W: Write, I: Input, O: Output>(
    mut wtr: W,
    addr: CompiledAddr,
    node: &BuilderNode<I, O>,
  ) -> io::Result<()>
  where
    Bytes<O>: Serialize,
    Bytes<I>: Serialize, {
    assert!(node.transitions.len() <= I::max_value().as_usize());
    let mut sink = io::sink();
    let obytes_init = Bytes(node.final_output).write_le(&mut sink).unwrap();
    let (tbytes, obytes, any_outs) = node.transitions.iter().fold(
      (0, obytes_init, !node.final_output.is_zero()),
      |(tbytes, obytes, any_outs), trans| {
        let next_tsize = Pack(delta(addr, trans.addr)).size();
        (
          tbytes.max(next_tsize),
          obytes.max(Bytes(trans.output).write_le(&mut sink).unwrap()),
          any_outs || !trans.output.is_zero(),
        )
      },
    );
    let obytes = if any_outs { obytes } else { 0 };
    let mut iosize = IOSize::new();
    iosize.set_output_bytes(obytes);
    iosize.set_transition_bytes(tbytes);
    let mut state = StateAnyTrans::new();
    if node.is_final {
      state.set_final();
    }
    state.set_state_num_trans(node.transitions.len());
    if any_outs {
      if node.is_final {
        // TODO make this packed
        Bytes(node.final_output).write_le(&mut wtr)?;
      }
      for t in node.transitions.iter().rev() {
        // TODO make this packed
        assert_eq!(obytes, Bytes(t.output).write_le(&mut wtr)?);
      }
    }
    for t in node.transitions.iter().rev() {
      let t_written = Bytes(PackTo(delta(addr, t.addr), tbytes)).write_le(&mut wtr)?;
      assert_eq!(t_written, tbytes);
    }
    for t in node.transitions.iter().rev() {
      Bytes(t.input).write_le(&mut wtr)?;
    }
    Bytes(iosize.encode()).write_le(&mut wtr)?;
    if state.state_num_trans().is_none() {
      assert_ne!(node.transitions.len(), 0, "UNREACHABLE 0 encoded in state");
      let s = I::from_usize(node.transitions.len() - 1);
      Bytes(s).write_le(&mut wtr)?;
      assert_ne!(state.num_trans_len::<I>(), 0);
    } else {
      assert_eq!(state.num_trans_len::<I>(), 0);
    }
    Bytes(state.0).write_le(&mut wtr)?;
    Ok(())
  }
  fn set_final(&mut self) { self.0 |= 0b01_000000; }
  const fn is_final(self) -> bool { self.0 & 0b01_000000 != 0 }
  fn sizes<I: Input>(self, data: &[u8]) -> IOSize {
    let i = data.len() - 1 - self.num_trans_len::<I>() - 1;
    IOSize::decode(data[i])
  }
  /// Attempts to encode number of transitions in the state
  // TODO return if successful or not
  fn set_state_num_trans(&mut self, n: usize) {
    if n >= 256 {
      return;
    }
    let n = (n as u8).saturating_add(1);
    if n <= !0b11_000000 {
      self.0 = (self.0 & 0b11_000000) | n;
    }
  }
  /// Number of transitions encoded into the state
  /// If it is 0 it implies that the number of transitions is encoded elsewhere
  fn state_num_trans(self) -> Option<u8> {
    let n = self.0 & !0b11_000000;
    Some(n).filter(|&n| n != 0).map(|n| n - 1)
  }
  // the total size of all the transitions for the given sizes
  fn total_trans_bytes<I: Input>(self, sizes: IOSize, n: usize) -> usize {
    // If I choose to add an index to self I need to add that in here
    // the n addition at the start is the size of the input bytes (1 per transition currently)
    (n * self.input_len::<I>()) + (n * sizes.transition_bytes())
  }
  // returns whether or not the ntrans byte exists
  fn num_trans_len<I: Input>(self) -> usize {
    if self.state_num_trans().is_none() {
      self.input_len::<I>()
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
    let input_len = self.input_len::<I>();
    Bytes::<I>::read_le(&mut &data[data.len() - 1 - input_len..], input_len as u8)
      .unwrap()
      .inner()
      .as_usize()
      + 1
  }
  const fn input_len<I: Input>(self) -> usize { size_of::<I>() }
  fn input<I: Input, O>(self, node: &Node<'_, O>, i: usize) -> I
  where
    Bytes<I>: Deserialize, {
    let input_len = self.input_len::<I>();
    let at = node.start
      - self.num_trans_len::<I>()
      - 1 // IOSize
      // TODO if add index need to add it in here as well
      - (i+1) * input_len;
    Bytes::read_le(&mut &node.data[at..], input_len as u8)
      .unwrap()
      .inner()
  }
  fn final_output<I: Input, O: Output>(self, data: &[u8], sizes: IOSize, num_trans: usize) -> O
  where
    Bytes<O>: Deserialize, {
    let osize = sizes.output_bytes();
    if osize == 0 || !self.is_final() {
      return O::zero();
    }
    let at = data.len()
      - 1
      - self.num_trans_len::<I>()
      - 1
      - self.total_trans_bytes::<I>(sizes, num_trans)
      - ((num_trans + 1) * osize);
    Bytes::<O>::read_le(&mut &data[at..], osize as u8)
      .unwrap()
      .inner()
  }
  fn end_addr<I: Input>(self, data: &[u8], sizes: IOSize, num_trans: usize) -> CompiledAddr {
    let osize = sizes.output_bytes();
    let final_osize = if self.is_final() { osize } else { 0 };
    data.len()
      - 1
      - self.num_trans_len::<I>()
      - 1 // IOSize
      - self.total_trans_bytes::<I>(sizes, num_trans)
      - (osize * num_trans)
      - final_osize
  }
  fn trans_addr<I: Input, O>(self, node: &Node<'_, O>, i: usize) -> CompiledAddr {
    assert!(i < node.num_trans);
    let tsize = node.sizes.transition_bytes();
    if tsize == 0 {
      return END_ADDRESS;
    };
    let at = node.start
      - self.num_trans_len::<I>()
      - 1 // iosize
      - node.num_trans * self.input_len::<I>() // inputs
      - ((i+1) * tsize);
    let delta = Bytes::<u64>::read_le(&mut &node.data[at..], tsize as u8)
      .unwrap()
      .inner();
    undo_delta(node.end, delta)
  }
  fn output<I: Input, O: Output>(self, node: &Node<'_, O>, i: usize) -> O
  where
    Bytes<O>: Deserialize, {
    let osize = node.sizes.output_bytes();
    if osize == 0 {
      return O::zero();
    };
    let at = node.start
      - self.num_trans_len::<I>()
      - 1 // iosize
      - self.total_trans_bytes::<I>(node.sizes, node.num_trans)
      - ((i+1) * osize);
    Bytes::<O>::read_le(&mut &node.data[at..], osize as u8)
      .unwrap()
      .inner()
  }
  fn find_input<O, I: Input>(self, node: &Node<'_, O>, b: I) -> Option<usize>
  where
    Bytes<I>: Deserialize, {
    let input_len = self.input_len::<I>();
    let end = node.start - self.num_trans_len::<I>() - 1;
    let start = end - node.num_trans * input_len;
    // Iterate from left to right then flip number
    node.data[start..end]
      .chunks_exact(input_len)
      .map(|mut chunk| Bytes::read_le(&mut chunk, input_len as u8).unwrap().inner())
      .position(|i| i == b)
      .map(|i| node.num_trans - i - 1)
  }
  fn find_input_range<O, I: Input>(self, node: &Node<'_, O>, r: &Range<I>) -> Range<usize>
  where
    Bytes<I>: Deserialize, {
    let input_len = self.input_len::<I>();
    let end = node.start - self.num_trans_len::<I>() - 1;
    let start = end - node.num_trans * input_len;
    let mut iter = node.data[start..end]
      .chunks_exact(input_len)
      .map(|mut chunk| Bytes::read_le(&mut chunk, input_len as u8).unwrap().inner());
    // going from greatest to least
    let end = iter
      .position(|i| r.end <= i)
      .map(|i| node.num_trans - i - 1);
    let end = if let Some(v) = end { v } else { return 0..0 };
    let start = iter
      .position(|i| r.start >= i)
      .map(|i| node.num_trans - i - 1);
    let start = if let Some(v) = start {
      v
    } else {
      return 0..end;
    };
    start..end
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
  // TODO check that this properly works
  let delta: usize = u64_to_usize(delta);
  if delta == END_ADDRESS {
    END_ADDRESS
  } else {
    node_addr - delta
  }
}
