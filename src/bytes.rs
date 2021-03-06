use crate::output::FiniteFloat;
use std::{
  io::{self, Read, Write},
  mem::size_of,
};

pub trait Serialize: Sized {
  /// Returns the number of bytes written on serialization
  fn write_le<W: Write>(self, dst: &mut W) -> io::Result<u8>;
  fn pack<W: Write>(self, _: &mut W) -> io::Result<u8> { todo!() }
  fn pack_to<W: Write>(self, _: u8, _: &mut W) -> io::Result<()> { todo!() }
}

pub trait Deserialize: Sized {
  /// Reads self from src for a given size
  fn read_le<R: Read>(src: &mut R, size: u8) -> io::Result<Self>;
}

/// A newtype wrapper for reading and writing bytes.
#[derive(Debug)]
pub struct Bytes<T>(pub T);

impl<T> Bytes<T> {
  pub fn inner(self) -> T { self.0 }
}

impl<T> From<T> for Bytes<T> {
  fn from(t: T) -> Self { Bytes(t) }
}

macro_rules! SerDesUnsigned {
  ($u: ty) => {
    impl Serialize for Bytes<$u> {
      fn write_le<W: Write>(self, dst: &mut W) -> io::Result<u8> {
        let bytes = self.0.to_le_bytes();
        dst.write_all(&bytes)?;
        Ok(size_of::<$u>() as u8)
      }
      fn pack_to<W: Write>(self, n: u8, dst: &mut W) -> io::Result<()> {
        let buf = self.inner().to_le_bytes();
        dst.write_all(&buf[..n as usize])?;
        Ok(())
      }
      fn pack<W: Write>(self, dst: &mut W) -> io::Result<u8> {
        let v = self.inner();
        let size = Pack(v).size();
        Bytes(v).pack_to(size, dst).map(|_| size)
      }
    }
    impl Deserialize for Bytes<$u> {
      #[inline]
      fn read_le<R: Read>(from: &mut R, n: u8) -> io::Result<Self> {
        debug_assert!(n <= (size_of::<$u>() as u8), "Got size of {:?}", n);
        let mut buf = [0; size_of::<$u>()];
        from.read_exact(&mut buf[..n as usize])?;
        Ok(<$u>::from_le_bytes(buf).into())
      }
    }
  };
}

SerDesUnsigned!(u8);
SerDesUnsigned!(u16);
SerDesUnsigned!(u32);
SerDesUnsigned!(u64);
SerDesUnsigned!(usize);

impl Serialize for Bytes<f32> {
  fn write_le<W: Write>(self, dst: &mut W) -> io::Result<u8> {
    let bytes = self.0.to_le_bytes();
    dst.write_all(&bytes)?;
    Ok(4)
  }
}

impl Serialize for Bytes<FiniteFloat<f32>> {
  fn write_le<W: Write>(self, dst: &mut W) -> io::Result<u8> { Bytes(self.0.inner()).write_le(dst) }
}

impl Deserialize for Bytes<f32> {
  fn read_le<R: Read>(from: &mut R, n: u8) -> io::Result<Self> {
    assert!(n <= 4);
    let mut buf = [0; 4];
    from.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf).into())
  }
}

impl Deserialize for Bytes<FiniteFloat<f32>> {
  fn read_le<R: Read>(from: &mut R, n: u8) -> io::Result<Self> {
    Ok(FiniteFloat::new(Bytes::<f32>::read_le(from, n)?.inner()).into())
  }
}

use crate::output::Unit;
impl Serialize for Bytes<Unit> {
  fn write_le<W: Write>(self, _: &mut W) -> io::Result<u8> { Ok(0) }
}

impl Deserialize for Bytes<Unit> {
  fn read_le<R: Read>(_: &mut R, n: u8) -> io::Result<Self> {
    assert_eq!(n, 0);
    Ok(Bytes(Unit))
  }
}

/// PackTo packs the given type to a specific size
#[derive(Debug)]
pub struct PackTo<T>(pub T, pub u8);

/// Packing is exclusively for serializing data and for deserializing the type itself should be
/// used.
#[derive(Debug, Copy, Clone)]
pub struct Pack<T>(pub T);
impl Pack<u8> {
  pub fn size(self) -> u8 {
    let n = self.0;
    if n == 0 {
      0
    } else {
      1
    }
  }
}

impl Pack<u16> {
  pub fn size(self) -> u8 {
    let n = self.0;
    if n == 0 {
      0
    } else if n < 1 << 8 {
      1
    } else {
      2
    }
  }
}

impl Pack<u32> {
  pub fn size(self) -> u8 {
    let n = self.0;
    if n == 0 {
      0
    } else if n < 1 << 8 {
      1
    } else if n < 1 << 16 {
      2
    } else if n < 1 << 24 {
      3
    } else {
      4
    }
  }
}

impl Pack<u64> {
  pub fn size(self) -> u8 {
    let n = self.0;
    if n == 0 {
      0
    } else if n < 1 << 8 {
      1
    } else if n < 1 << 16 {
      2
    } else if n < 1 << 24 {
      3
    } else if n < 1 << 32 {
      4
    } else if n < 1 << 40 {
      5
    } else if n < 1 << 48 {
      6
    } else if n < 1 << 56 {
      7
    } else {
      8
    }
  }
}

impl Pack<usize> {
  // Different implementations of size depending on pointer widths
  #[cfg(target_pointer_width = "32")]
  pub fn size(self) -> u8 { Pack(self.0 as u32).size() }
  #[cfg(not(target_pointer_width = "32"))]
  pub fn size(self) -> u8 { Pack(self.0 as u64).size() }
}

#[cfg(test)]
mod bytes_test {
  use super::*;
  quickcheck! {
    fn pack_serdes_u32(v: u32) -> bool {
      let mut buffer = vec![];
      let written = Bytes(v).pack(&mut buffer).unwrap() as u8;
      v == Bytes::<u32>::read_le(&mut buffer.as_slice(), written).unwrap().inner()
    }
  }
  quickcheck! {
    fn pack_serdes_u64(v: u64) -> bool {
      let mut buffer = vec![];
      let written = Bytes(v).pack(&mut buffer).unwrap() as u8;
      v == Bytes::<u64>::read_le(&mut buffer.as_slice(), written).unwrap().inner()
    }
  }
  quickcheck! {
    fn pack_serdes_ff_f32(v: f32) -> bool {
      let v = FiniteFloat::new(v);
      let mut buffer = vec![];
      let written = Bytes(v).write_le(&mut buffer).unwrap() as u8;
      v == Bytes::read_le(&mut buffer.as_slice(), written).unwrap().inner()
    }
  }
}
