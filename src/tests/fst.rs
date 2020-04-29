use crate::{
  build::Builder,
  output::{FiniteFloat, Unit},
};

#[test]
fn one_unit() {
  let mut builder: Builder<_, u8, Unit, 3> = Builder::memory().unwrap();
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
  let amt = 3u8;
  let mut builder = Builder::memory().unwrap();
  for i in 0u8..amt {
    for j in 0..amt {
      for k in 0..amt {
        assert!(builder.insert([i, j, k], Unit).is_ok());
      }
    }
  }
  let fst = builder.into_fst();
  for i in 0u8..amt {
    for j in 0..amt {
      for k in 0..amt {
        assert_eq!(fst.get(&[i, j, k]), Some(Unit), "Missing {:?}", [i, j, k]);
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
        assert_eq!(fst.get(&[i, j, k]), Some(i + j + k), "Missing {:?}", [
          i, j, k
        ]);
      }
    }
  }
}

#[test]
fn many_f32() {
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
fn f32_precision() {
  use crate::output::FiniteFloat;
  let mut builder = Builder::memory().unwrap();
  let scale_down = 9999999999999999999999999f32;

  for i in 1u8..=20 {
    for j in 1..=20 {
      let v = (i as f32 + j as f32) / scale_down;
      assert_ne!(0.0, v);
      assert!(builder.insert([i, j], FiniteFloat::new(v)).is_ok());
    }
  }
  let eps = 0.00000001;
  let fst = builder.into_fst();
  for i in 1u8..=20 {
    for j in 1..=20 {
      let got = fst.get(&[i, j]).expect("Unexpected missing value").inner();
      let expected = (i as f32 + j as f32) / scale_down;
      assert!((got - expected).abs() < eps, "{}", (got - expected).abs());
    }
  }
}
#[test]
fn many_u16_u64() {
  let mut builder = Builder::memory().unwrap();
  for i in 0u16..=300 {
    for j in 0..=2 {
      assert!(builder.insert([i, j], i + j).is_ok());
    }
  }
  let fst = builder.into_fst();
  for i in 0u16..=300 {
    for j in 0..=2 {
      assert_eq!(fst.get(&[i, j]), Some(i + j));
    }
  }
}
