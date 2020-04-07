#[cfg(test)]
use crate::{build::Builder, output::Unit};
#[test]
fn one_unit() {
  let mut builder = Builder::memory().unwrap();
  assert!(builder.insert([0u8, 1, 2], Unit).is_ok());
  let fst = builder.into_fst();
  assert_eq!(fst.get(&[0, 1, 2]), Some(Unit));
  assert_eq!(fst.get(&[0, 1, 3]), None);
}
#[test]
fn one_u32() {
  let mut builder = Builder::memory().unwrap();
  assert!(builder.insert([0u8, 1, 2], 3u32).is_ok());
  let fst = builder.into_fst();
  assert_eq!(fst.get(&[0, 1, 2]), Some(3));
}
#[test]
fn one_u64() {
  let mut builder = Builder::memory().unwrap();
  assert!(builder.insert([0u8, 1, 2], 3u64).is_ok());
  let fst = builder.into_fst();
  assert_eq!(fst.get(&[0, 1, 2]), Some(3));
}
#[test]
fn one_u16_u64() {
  let mut builder = Builder::memory().unwrap();
  assert!(builder.insert([0u16, 1, 2], 3u64).is_ok());
  let fst = builder.into_fst();
  assert_eq!(fst.get(&[0, 1, 2]), Some(3));
}
#[test]
fn many_unit() {
  let mut builder = Builder::memory().unwrap();
  for i in 0u8..=20 {
    for j in 0..=20 {
      for k in 0..=20 {
        assert!(builder.insert([i, j, k], Unit).is_ok());
      }
    }
  }
  let fst = builder.into_fst();
  for i in 0u8..=20 {
    for j in 0..=20 {
      for k in 0..=20 {
        assert_eq!(fst.get(&[i, j, k]), Some(Unit));
      }
    }
  }
}
#[test]
fn many_u32() {
  let mut builder = Builder::memory().unwrap();
  for i in 0u8..=20 {
    for j in 0..=20 {
      for k in 0..=20 {
        assert!(builder.insert([i, j, k], i + j + k).is_ok());
      }
    }
  }
  let fst = builder.into_fst();
  for i in 0u8..=20 {
    for j in 0..=20 {
      for k in 0..=20 {
        assert_eq!(fst.get(&[i, j, k]), Some(i + j + k));
      }
    }
  }
}
#[test]
fn many_f32() {
  use crate::output::FiniteFloat;
  let mut builder = Builder::memory().unwrap();
  for i in 0u8..=20 {
    for j in 0..=20 {
      for k in 0..=20 {
        assert!(builder
          .insert([i, j, k], FiniteFloat::new((i + j + k) as f32))
          .is_ok());
      }
    }
  }
  let fst = builder.into_fst();
  for i in 0u8..=20 {
    for j in 0..=20 {
      for k in 0..=20 {
        assert_eq!(
          fst.get(&[i, j, k]),
          Some(FiniteFloat::new((i + j + k) as f32))
        );
      }
    }
  }
}
#[test]
fn iter() {
  use crate::output::FiniteFloat;
  let mut builder = Builder::memory().unwrap();
  let n = 5u8;
  for i in 0..=n {
    for j in 0..=n {
      for k in 0..=n {
        assert!(builder
          .insert([i, j, k], FiniteFloat::new((i + j + k) as f32))
          .is_ok());
      }
    }
  }
  let fst = builder.into_fst();
  let expected = (0..=n).flat_map(|i| {
    (0..=n).flat_map(move |j| (0..=n).map(move |k| FiniteFloat::new((i + j + k) as f32)))
  });
  assert!(fst.values().eq(expected));
}
