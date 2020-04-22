use crate::{
  build::MAGIC_NUMBER,
  bytes::{Bytes, Deserialize},
  error::Result,
  input::Input,
  node::Node,
  output::Output,
};
use std::marker::PhantomData;

/// Represents a location in the FST.
pub type CompiledAddr = usize;
pub(crate) const END_ADDRESS: CompiledAddr = 0;
pub(crate) const INVALID_ADDRESS: CompiledAddr = 1;

#[derive(Debug)]
pub struct Fst<D, I, O> {
  meta: Meta,
  data: D,
  /// What is the output type of this FST?
  _output_type: PhantomData<O>,
  /// What is the input type of this FST?
  _input_type: PhantomData<I>,
}

#[derive(Debug)]
pub struct Meta {
  root_addr: CompiledAddr,
  // need to encode type here as well
  len: usize,
}

impl<D: AsRef<[u8]>, I: Input, O: Output> Fst<D, I, O> {
  pub fn new(data: D) -> Result<Fst<D, I, O>, I> {
    let bytes = data.as_ref();
    let initial_byte = Bytes::<u64>::read_le(&mut &bytes[..], 8)?.inner();
    assert_eq!(initial_byte, MAGIC_NUMBER);
    let end = bytes.len();
    let root_addr = u64_to_usize(Bytes::<u64>::read_le(&mut &bytes[end - 8..], 8)?.inner());
    let len = u64_to_usize(Bytes::<u64>::read_le(&mut &bytes[end - 16..], 8)?.inner());
    let meta = Meta { root_addr, len };
    Ok(Fst {
      meta,
      data,

      _output_type: PhantomData,
      _input_type: PhantomData,
    })
  }
  pub fn get(&self, key: &[I]) -> Option<O>
  where
    Bytes<I>: Deserialize,
    Bytes<O>: Deserialize, {
    let mut node = self.root();
    let mut out = O::zero();
    for &b in key {
      node = node.find_input(b).map(|i| {
        let t = node.transition::<I>(i);
        out = out.cat(&t.output);
        self.node(t.addr)
      })?;
    }
    if !node.is_final {
      None
    } else {
      Some(out.cat(&node.final_output))
    }
  }
  pub fn contains_key(&self, key: &[I]) -> bool
  where
    Bytes<I>: Deserialize,
    Bytes<O>: Deserialize, {
    let mut node = self.root();
    for &b in key {
      let next = node.find_input(b).map(|i| {
        let t: Transition<I, O> = node.transition::<I>(i);
        self.node(t.addr)
      });
      node = if let Some(node) = next {
        node
      } else {
        return false;
      }
    }
    node.is_final
  }
  pub(crate) fn root(&self) -> Node<'_, O>
  where
    Bytes<O>: Deserialize,
    Bytes<I>: Deserialize, {
    self.node(self.meta.root_addr)
  }
  pub(crate) fn node(&self, addr: CompiledAddr) -> Node<'_, O>
  where
    Bytes<O>: Deserialize,
    Bytes<I>: Deserialize, {
    Node::new::<I>(addr, &self.data.as_ref())
  }
  pub fn len(&self) -> usize { self.meta.len }
  pub fn is_empty(&self) -> bool { self.len() == 0 }
  pub(crate) fn nbytes(&self) -> usize { self.data.as_ref().len() }
}

impl<I, O> Fst<Vec<u8>, I, O> {
  #[inline]
  pub fn recycle(self) -> Vec<u8> { self.data }
}

#[derive(Debug, Copy, Clone, Hash, Eq, PartialEq)]
pub struct Transition<I, O> {
  /// Input value associated with this transition
  // TODO convert this into a parametrized type for larger input values
  pub input: I,
  /// Output value associated with this transition
  pub output: O,
  /// Where is the next node from this transition
  pub addr: CompiledAddr,
}

pub(crate) fn u64_to_usize(v: u64) -> usize {
  use std::convert::TryInto;
  v.try_into().unwrap()
}
