use crate::{
  build::Builder,
  matrix::Matrix,
  output::{FiniteFloat, Unit},
};
use num::{One, Zero};

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
/*
#[test]
fn viter() {
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
*/

#[test]
fn iter() {
  use crate::output::FiniteFloat;
  let n = 5u8;
  let items = (0..n).flat_map(move |i| {
    (0..n).flat_map(move |j| (0..n).map(move |k| ([i, j, k], FiniteFloat::new((i + j + k) as f32))))
  });
  let mat = Matrix::new([n, n, n], items);
  let expected = (0..n).flat_map(|i| {
    (0..n).flat_map(move |j| (0..n).map(move |k| ([i, j, k], FiniteFloat::new((i + j + k) as f32))))
  });
  assert!(
    mat.iter().eq(expected),
    "Got {:?}",
    mat.iter().collect::<Vec<_>>()
  );
}

#[test]
fn vec_matmul() {
  let mul = FiniteFloat::new(0.3);
  let mat: Matrix<_, _, _, 2> = Matrix::eye(16u8);
  let one = vec![mul; 16];
  let out = mat.vecmul(&one[..]);
  for i in out {
    assert_eq!(mul, i)
  }
}

#[test]
fn transpose() {
  let mat: Matrix<_, _, u32, 2> = Matrix::eye(16u8);
  // just check that it doesn't throw
  let _ = mat.transpose();
}

#[test]
fn matmul() {
  let a: Matrix<_, _, u32, 2> = Matrix::eye(16u8);
  let b: Matrix<_, _, u32, 2> = Matrix::eye(16u8);
  // just check that it doesn't throw
  let c = a.matmul(&b);
  assert!(c.iter().eq(a.iter()));
}

#[test]
fn convolve() {
  let cap = 6u8;
  let a: Matrix<_, _, FiniteFloat<f32>, 2> = Matrix::eye(cap);
  let l = FiniteFloat::one();
  let o = FiniteFloat::<f32>::zero();
  let h = FiniteFloat::new(0.5);
  /*
  let output = a.convolve_2d([[l]]);
  */
  /*
  let output = a.convolve_2d([
    [o, o, o],
    [o, l, o],
    [o, o, o],
  ]);
  */
  let output = a.convolve_2d([
    [o, o, o, o, o],
    [o, o, h, o, o],
    [o, h, l, h, o],
    [o, o, h, o, o],
    [o, o, o, o, o],
  ]);
  for i in 0..cap {
    for j in 0..cap {
      if i == j || i == j + 1 || j == i + 1 {
        assert!(output[[i, j]].is_one());
      } else {
        assert!(output[[i, j]].is_zero());
      }
    }
  }
}

#[test]
#[cfg(feature = "parallel")]
fn par_vec_matmul() {
  let mul = FiniteFloat::new(0.3);
  let mat: Matrix<_, _, _, 2> = Matrix::eye(16u8);
  let one = vec![mul; 16];
  let out = mat.par_vecmul(&one[..]);
  for i in out {
    assert_eq!(mul, i)
  }
}
