use num::Bounded;
use std::{
  fmt::{Debug, Display},
  hash::Hash,
};

/// A trait which represents a possible input language type
pub trait Input:
  Display + Debug + Hash + Bounded + Default + Sized + Clone + Copy + Eq + Ord {
  fn as_usize(self) -> usize;
  fn from_usize(i: usize) -> Self;
}
// Input types that we allow
impl Input for u8 {
  fn as_usize(self) -> usize { self as usize }
  fn from_usize(i: usize) -> Self { i as Self }
}
impl Input for u16 {
  fn as_usize(self) -> usize { self as usize }
  fn from_usize(i: usize) -> Self { i as Self }
}
impl Input for u32 {
  fn as_usize(self) -> usize { self as usize }
  fn from_usize(i: usize) -> Self { i as Self }
}
