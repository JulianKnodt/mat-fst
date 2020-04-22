use crate::{
  bytes::*,
  counting_writer::CountingWriter,
  error::{Error, Result},
  fst::{CompiledAddr, Fst, Transition, END_ADDRESS, INVALID_ADDRESS},
  input::Input,
  output::{Output, Prefix},
};
use num::Zero;
use std::{
  collections::{hash_map::Entry, HashMap},
  io::{self, Write},
};

pub const MAGIC_NUMBER: u64 = 0xFD15EA5E;

#[derive(Debug)]
pub struct LastTransition<I> {
  input: I,
  num_out: u32,
}

#[derive(Debug)]
pub struct PartialNode<I> {
  node: BuilderNode<I>,
  last: Option<LastTransition<I>>,
}

impl<I: Input> PartialNode<I> {
  fn last_compiled(&mut self, addr: CompiledAddr) {
    // if there was some previous partial transition
    // we can now assign it an address
    if let Some(LastTransition { input, num_out }) = self.last.take() {
      self.node.transitions.push(Transition {
        input,
        num_out,
        addr,
      });
    }
  }
  fn add_output_prefix(&mut self, prefix: u32) {
    for t in &mut self.node.transitions {
      t.num_out = prefix.cat(&t.num_out);
    }
    if let Some(ref mut t) = self.last {
      t.num_out = prefix.cat(&t.num_out);
    }
  }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct BuilderNode<I> {
  // Map of transitions to next node address
  pub transitions: Vec<Transition<I>>,
  pub(crate) idx: usize,
}

impl<I: Input> BuilderNode<I> {
  #[inline]
  fn new(idx: usize) -> Self {
    Self {
      transitions: vec![],
      idx,
    }
  }
}

#[derive(Debug)]
pub struct PartialNodes<I>(Vec<PartialNode<I>>);

impl<I: Input> PartialNodes<I> {
  fn new() -> Self {
    let mut partials = PartialNodes(Vec::with_capacity(64));
    partials.push_empty();
    partials
  }
  fn len(&self) -> usize { self.0.len() }
  fn push_empty(&mut self) {
    self.0.push(PartialNode {
      node: BuilderNode::new(0),
      last: None,
    });
  }
  /// Pops a final element from this
  fn pop_empty(&mut self) -> BuilderNode<I> {
    let PartialNode { node, last } = self.0.pop().unwrap();
    assert!(last.is_none());
    node
  }
  fn pop_freeze(&mut self, addr: CompiledAddr) -> BuilderNode<I> {
    let mut unfinished = self.0.pop().unwrap();
    unfinished.last_compiled(addr);
    unfinished.node
  }
  fn top_last_freeze(&mut self, addr: CompiledAddr) {
    self.0.last_mut().unwrap().last_compiled(addr);
  }
  fn pop_root(&mut self) -> BuilderNode<I> {
    assert_eq!(self.0.len(), 1);
    assert!(self.0[0].last.is_none());
    self.0.pop().unwrap().node
  }
  fn find_common_prefix_set_output(&mut self, key: &[I], mut o: u32) -> (usize, u32) {
    let mut i = 0;
    while i < key.len() {
      let prefix_change = match self.0[i].last.as_mut() {
        Some(ref mut t) if t.input == key[i] => {
          i += 1;
          let common_pre = t.num_out.prefix(&o);
          let add_prefix = t.num_out.rm_pre(&common_pre);
          o = o.rm_pre(&common_pre);
          t.num_out = common_pre;
          add_prefix
        },
        _ => break,
      };
      if !prefix_change.is_zero() {
        self.0[i].add_output_prefix(prefix_change);
      }
    }
    (i, o)
  }
  fn add_suffix(&mut self, key: &[I], o: u32) {
    if key.is_empty() {
      return;
    }
    // Mark the last element on the stack transitioning to the rest of the keys
    let last = self.0.last_mut().unwrap();
    assert!(last.last.is_none());
    last.last = Some(LastTransition {
      input: key[0],
      num_out: o,
    });
    for i in 1..key.len() {
      let k = key[i];
      self.0.push(PartialNode {
        node: BuilderNode::new(i),
        last: Some(LastTransition {
          input: k,
          num_out: 0,
        }),
      })
    }
    self.push_empty();
  }
}

/// A builder for a Fixed Finite State Transducer
#[derive(Debug)]
pub struct Builder<Wtr: Write, I, O, const N: usize> {
  wtr: CountingWriter<Wtr>,
  unfinished: PartialNodes<I>,
  registry: HashMap<BuilderNode<I>, CompiledAddr>,
  outputs: Vec<O>,

  /// The last location written to for this builder
  last_addr: CompiledAddr,

  /// The last sequence added to this builder
  last: Option<Vec<I>>,

  /// How many nodes have been written to this builder
  len: usize,
}

impl<I: Input, O: Output, const N: usize> Builder<Vec<u8>, I, O, N>
where
  Bytes<I>: Serialize,
  Bytes<O>: Serialize,
{
  pub fn memory() -> Result<Self, I> { Builder::new(Vec::with_capacity(10240)) }
  pub fn from_buffer(mut v: Vec<u8>) -> Result<Self, I> {
    v.clear();
    Builder::new(v)
  }
}

impl<W: Write, I: Input, O: Output, const N: usize> Builder<W, I, O, N>
where
  Bytes<I>: Serialize,
  Bytes<O>: Serialize,
{
  pub fn new(mut w: W) -> Result<Self, I> {
    let mut wtr = CountingWriter::new(w);
    // Write a magic number to ensure that the first few bytes are not addressable by the rest
    Bytes(MAGIC_NUMBER).write_le(&mut wtr)?;

    Ok(Builder {
      wtr,
      unfinished: PartialNodes::new(),
      registry: HashMap::new(),
      outputs: vec![],
      last: None,
      last_addr: INVALID_ADDRESS,
      len: 0,
    })
  }
  pub fn into_fst(self) -> Fst<W, I, O>
  where
    W: AsRef<[u8]>,
    Bytes<O>: Deserialize, {
    self.into_inner().and_then(Fst::new).unwrap()
  }
  pub fn insert(&mut self, key: [I; N], o: O) -> Result<(), I> {
    let key = key.as_ref();
    if let Some(ref mut last) = self.last {
      if key < last.as_slice() {
        return Err(Error::OutOfOrder {
          prev: last.to_vec(),
          next: key.to_vec(),
        });
      }
    }
    let idx = self.outputs.len() as u32;
    self.outputs.push(o);
    // insert one to indicate that there is one more value being added
    self.insert_output(key, idx)?;
    // mark the last key as updated
    if let Some(ref mut last) = self.last {
      last.clear();
      last.extend_from_slice(key);
    } else {
      self.last = Some(key.to_vec());
    }
    self.len += 1;
    Ok(())
  }
  fn insert_output(&mut self, key: &[I], val: u32) -> Result<(), I> {
    assert!(!key.is_empty(), "Cannot insert empty key");
    let (prefix_len, out) = self.unfinished.find_common_prefix_set_output(key, val);
    assert_ne!(
      prefix_len,
      key.len(),
      "Multiple values inserted at same position"
    );
    self.compile_from(prefix_len)?;
    self.unfinished.add_suffix(&key[prefix_len..], out);
    Ok(())
  }
  /// Compiles all states after the given state not including that state.
  fn compile_from(&mut self, from: usize) -> Result<(), I> {
    let mut addr = INVALID_ADDRESS;
    while from + 1 < self.unfinished.len() {
      let node = if addr == INVALID_ADDRESS {
        self.unfinished.pop_empty()
      } else {
        self.unfinished.pop_freeze(addr)
      };
      addr = self.compile(&node)?;
      assert_ne!(addr, INVALID_ADDRESS, "Invalid address after compilation");
    }
    self.unfinished.top_last_freeze(addr);
    Ok(())
  }
  fn compile(&mut self, node: &BuilderNode<I>) -> Result<CompiledAddr, I> {
    if node.transitions.is_empty() {
      return Ok(END_ADDRESS);
    }
    let addr = match self.registry.entry(node.clone()) {
      Entry::Occupied(v) => *v.get(),
      Entry::Vacant(ve) => {
        let start_addr = self.wtr.count() as CompiledAddr;
        node.compile(&mut self.wtr, node.idx + 1 == N, self.last_addr, start_addr)?;
        self.last_addr = self.wtr.count() as CompiledAddr - 1;
        ve.insert(self.last_addr);
        self.last_addr
      },
    };
    Ok(addr)
  }
  fn into_inner(mut self) -> Result<W, I> {
    // Flush all nodes except for the first
    self.compile_from(0)?;
    let root = self.unfinished.pop_root();
    let root_addr = self.compile(&root)?;
    for o in self.outputs {
      Bytes(o).write_le(&mut self.wtr)?;
    }
    Bytes(self.len as u64).write_le(&mut self.wtr)?;
    Bytes(root_addr as u64).write_le(&mut self.wtr)?;
    self.wtr.flush()?;
    Ok(self.wtr.inner())
  }
  fn finish(self) -> Result<(), I> { self.into_inner().map(|_| ()) }
}

/// Testing construction of the fst but not operations that read from it
#[cfg(test)]
mod tests {
  use super::*;
  use crate::output::Unit;
  #[test]
  fn empty() {
    let builder: Builder<_, u8, u64, 2> = Builder::memory().expect("Could not create builder");
    assert!(builder.into_inner().is_ok())
  }
  #[test]
  fn add_one() {
    let mut builder: Builder<_, u8, _, 3> = Builder::memory().expect("Could not create builder");
    assert!(builder.insert([0, 1, 2], Unit).is_ok());
    assert!(builder.into_inner().is_ok())
  }
  #[test]
  fn add_unrelated() {
    let mut builder: Builder<_, u8, _, 3> = Builder::memory().expect("Could not create builder");
    assert!(builder.insert([0, 1, 2], Unit).is_ok());
    assert!(builder.insert([3, 4, 5], Unit).is_ok());
    assert!(builder.into_inner().is_ok())
  }
  #[test]
  fn add_with_prefix() {
    let mut builder: Builder<_, u8, _, 3> = Builder::memory().expect("Could not create builder");
    assert!(builder.insert([0, 1, 2], Unit).is_ok());
    assert!(builder.insert([0, 1, 3], Unit).is_ok());
    assert!(builder.into_inner().is_ok())
  }
}
