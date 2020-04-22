use crate::bytes::*;
/// Defining associated output alphabets for the FST
use num::{Float, One, Zero};
use std::{
  cmp,
  fmt::Debug,
  hash::{Hash, Hasher},
  mem::transmute,
  ops::{Add, Mul, Sub},
};

/// Prefix represents two types which can have a common prefix between them from the same type
/// as itself
pub trait Prefix {
  /// Returns the prefix of two elements
  fn prefix(&self, o: &Self) -> Self;
}

/// Represents an associatable output alphabet for the FST
// it should be noted that zero also enforces Add between types
pub trait Output: Prefix + Zero + Sized + Clone + Copy + Debug {
  /// Appends o to self
  fn cat(&self, o: &Self) -> Self;
  /// Removes the prefix from self
  fn rm_pre(&self, prefix: &Self) -> Self;
}

/*
impl<T> Output for T
where
  T: Prefix + Zero + Sub<Self, Output = Self> + Hash + Add<Self, Output = Self> + Copy + Eq + Debug,
{
  fn cat(&self, o: &Self) -> Self { *self + *o }
  fn rm_pre(&self, o: &Self) -> Self { *self - *o }
}
*/

macro_rules! default_prefix {
  ($t: ty) => {
    impl Prefix for $t {
      fn prefix(&self, o: &Self) -> Self { cmp::min(*self, *o) }
    }
  };
}
default_prefix!(u8);
default_prefix!(u16);
default_prefix!(u32);
default_prefix!(u64);

macro_rules! default_output {
  ($t: ty) => {
    impl Output for $t {
      fn cat(&self, o: &Self) -> Self { *self + *o }
      fn rm_pre(&self, o: &Self) -> Self { *self - *o }
    }
  };
}

default_output!(u8);
default_output!(u16);
default_output!(u32);
default_output!(u64);

/// A float which implements Ord and Eq for use with FST packages
/// Assuming that the float constructed
#[derive(PartialOrd, Debug, Copy, Clone)]
pub struct FiniteFloat<T>(T);
impl<T: Float> Ord for FiniteFloat<T> {
  #[inline]
  fn cmp(&self, o: &Self) -> std::cmp::Ordering { self.partial_cmp(o).unwrap() }
}
impl<T: PartialEq> PartialEq for FiniteFloat<T> {
  #[inline]
  fn eq(&self, o: &Self) -> bool { self.0 == o.0 }
}
impl<T: Float> Eq for FiniteFloat<T> {}
impl<T: Zero + Float> Zero for FiniteFloat<T> {
  fn zero() -> Self { FiniteFloat(T::zero()) }
  fn is_zero(&self) -> bool { self.0.is_zero() }
}
impl<T: One + Float> One for FiniteFloat<T> {
  fn one() -> Self { FiniteFloat(T::one()) }
  fn is_one(&self) -> bool { self.0.is_one() }
}

impl<T: Float> Add for FiniteFloat<T> {
  type Output = Self;
  #[inline]
  fn add(self, o: Self) -> Self::Output { FiniteFloat(self.0 + o.0) }
}

impl<T: Float> Mul for FiniteFloat<T> {
  type Output = Self;
  #[inline]
  fn mul(self, o: Self) -> Self::Output { FiniteFloat::new(self.0 * o.0) }
}

impl<T: Float> Hash for FiniteFloat<T>
where
  Bytes<T>: Serialize,
{
  fn hash<H: Hasher>(&self, state: &mut H) {
    let mut buf = [0u8; 4];
    Bytes(self.0).write_le(&mut buf.as_mut()).unwrap();
    buf.hash(state)
  }
}
impl<T: Float> FiniteFloat<T> {
  pub fn new(f: T) -> Self {
    // normalize to positive 0 as well
    let f = if f.is_zero() { T::zero() } else { f };
    assert!(f.is_finite());
    FiniteFloat(f)
  }
  #[inline]
  pub fn inner(&self) -> T { self.0 }
}
impl<T: Float + Debug> Prefix for FiniteFloat<T> {
  fn prefix(&self, o: &Self) -> Self {
    // which strategy works best?
    // FiniteFloat::new(self.0.min(o.0))
    FiniteFloat::zero()
  }
}
impl<T: Float + Debug> Output for FiniteFloat<T>
where
  Bytes<T>: Serialize,
{
  fn cat(&self, o: &Self) -> Self { FiniteFloat::new(self.0 + o.0) }
  fn rm_pre(&self, o: &Self) -> Self { FiniteFloat::new(self.0 - o.0) }
}

/// f32 which is internally casted to a u32
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TransmutedF32(u32);
impl TransmutedF32 {
  pub fn new(v: f32) -> Self { Self(v.to_bits()) }
  pub fn v(&self) -> f32 { f32::from_bits(self.0) }
}
impl Zero for TransmutedF32 {
  fn zero() -> Self { Self::new(0.0) }
  fn is_zero(&self) -> bool { self.v().is_zero() }
}

impl Add for TransmutedF32 {
  type Output = Self;
  fn add(self, o: Self) -> Self { Self::new(self.v() + o.v()) }
}

impl Sub for TransmutedF32 {
  type Output = Self;
  fn sub(self, o: Self) -> Self { Self::new(self.v() - o.v()) }
}

impl Mul for TransmutedF32 {
  type Output = Self;
  fn mul(self, o: Self) -> Self { Self::new(self.v() * o.v()) }
}

impl Prefix for TransmutedF32 {
  fn prefix(&self, o: &Self) -> Self { TransmutedF32(self.0.min(o.0)) }
}

impl Output for TransmutedF32 {
  fn cat(&self, o: &Self) -> Self { TransmutedF32(self.0 + o.0) }
  fn rm_pre(&self, o: &Self) -> Self { TransmutedF32(self.0 - o.0) }
}

/// A ZST which is an FST compatible unit type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Unit;
impl Add for Unit {
  type Output = Self;
  fn add(self, _: Self) -> Self::Output { Unit }
}
impl Zero for Unit {
  #[inline]
  fn zero() -> Self { Unit }
  #[inline]
  fn is_zero(&self) -> bool { true }
}
impl Output for Unit {
  fn cat(&self, o: &Self) -> Self { Unit }
  fn rm_pre(&self, o: &Self) -> Self { Unit }
}
impl Prefix for Unit {
  fn prefix(&self, _: &Self) -> Self { Unit }
}
