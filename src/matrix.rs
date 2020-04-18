use crate::{
  build::Builder, bytes::*, coo::COO, dense::Dense, fst::Fst, input::Input, node::Node,
  output::Output, util::within,
};
use num::{One, Zero};
use std::{
  array::LengthAtMost32,
  cmp::Ordering,
  mem::replace,
  ops::{Index, Mul, RangeBounds, Sub},
};

#[derive(Debug)]
pub struct Matrix<D, I, O, const N: usize>
where
  [I; N]: LengthAtMost32, {
  // row, col, etc
  pub dims: [I; N],
  pub(crate) data: Fst<D, I, O>,
}

impl<I, O, const N: usize> Matrix<Vec<u8>, I, O, N>
where
  I: Input,
  O: Output,
  Bytes<O>: Serialize + Deserialize,
  Bytes<I>: Serialize + Deserialize,
  [I; N]: LengthAtMost32,
{
  pub fn new<Iter: Iterator<Item = ([I; N], O)>>(dims: [I; N], i: Iter) -> Self {
    let mut builder = Builder::memory().unwrap();
    for (k, v) in i {
      assert!(
        within(k, dims),
        "Point outside of given dims: {:?} < {:?}",
        dims,
        k
      );
      builder.insert(k, v).expect("Failed to insert");
    }
    let data = builder.into_fst();
    Matrix { dims, data }
  }
}

impl<D, I, O> Matrix<D, I, O, 2>
where
  D: AsRef<[u8]>,
  I: Input,
  O: Output,
  O: Mul<Output = O>,
  Bytes<O>: Deserialize,
  Bytes<I>: Deserialize,
{
  #[inline]
  fn get(&self, idxs: [I; 2]) -> O { self.data.get(&idxs[..]).unwrap_or_else(O::zero) }
  /// Performs vector multiplication of this matrix with some dense vector
  #[inline]
  pub fn vecmul(&self, vec: &[O]) -> Vec<O> {
    let mut out = vec![O::zero(); self.dims[0].as_usize()];
    self.vecmul_into(vec, &mut out);
    out
  }
  /// Performs vector multiplication and puts the result into some destination buffer
  pub fn vecmul_into(&self, vec: &[O], out: &mut [O]) {
    assert_eq!(
      self.dims[1].as_usize(),
      vec.len(),
      "Dimension mismatch, expected vector of len {}",
      self.dims[1]
    );
    let rows = self.dims[0].as_usize();
    assert!(
      rows <= out.len(),
      "Dimension mismatch, expected output of size {}",
      self.dims[0]
    );
    self.eager_iter(|[y, x], v| {
      let y = y.as_usize();
      out[y] = out[y] + v * vec[x.as_usize()];
    });
    /*
    self.iter().for_each(|([y, x], v)| {
      let y = y.as_usize();
      out[y] = out[y] + v * vec[x.as_usize()];
    });
    */
  }
}

impl<D, I, O, const N: usize> Matrix<D, I, O, N>
where
  D: AsRef<[u8]>,
  I: Input,
  O: Output + Mul<Output = O>,
  Bytes<O>: Serialize + Deserialize,
  Bytes<I>: Serialize + Deserialize,
  [I; N]: LengthAtMost32,
{
  pub fn hadamard<'a>(&'a self, o: &'a Self) -> Matrix<Vec<u8>, I, O, N>
  where
    [(O, usize, Node<'a, O>); N]: LengthAtMost32, {
    assert_eq!(self.shape(), o.shape());
    let mut a = self.iter();
    let mut b = self.iter().peekable();
    let mut builder = Builder::memory().unwrap();
    for (i_a, o_a) in a {
      loop {
        if let Some(&(i_b, o_b)) = b.peek() {
          match i_b.cmp(&i_a) {
            Ordering::Less => {
              b.next();
              builder.insert(i_b, o_b).unwrap();
            },
            Ordering::Equal => {
              builder.insert(i_a, o_b * o_a).unwrap();
              b.next();
              break;
            },
            Ordering::Greater => {
              builder.insert(i_a, o_a).unwrap();
              break;
            },
          }
        } else {
          builder.insert(i_a, o_a).unwrap();
          break;
        }
      }
    }
    // clean up remaining b values
    for (i_b, o_b) in b {
      builder.insert(i_b, o_b).unwrap();
    }
    Matrix {
      dims: self.shape(),
      data: builder.into_fst(),
    }
  }
  pub fn transpose<'a>(&'a self) -> Matrix<Vec<u8>, I, O, N>
  where
    [(O, usize, Node<'a, O>); N]: LengthAtMost32, {
    let mut inv_items = Vec::with_capacity(self.data.len());
    inv_items.extend(self.iter().map(|(mut i, o)| {
      i.reverse();
      (i, o)
    }));
    // TODO This is expensive? Work on a way to make this cheaper
    inv_items.sort_unstable_by_key(|&(i, _)| i);
    let mut out = Builder::memory().unwrap();
    for (mut i, o) in inv_items {
      out.insert(i, o).unwrap();
    }
    let mut dims = self.shape();
    dims.reverse();
    Matrix {
      dims,
      data: out.into_fst(),
    }
  }
}

impl<D, I, O> Matrix<D, I, O, 2>
where
  D: AsRef<[u8]>,
  I: Input,
  O: Output,
  O: Mul<Output = O>,
  Bytes<O>: Serialize + Deserialize,
  Bytes<I>: Serialize + Deserialize,
{
  pub fn matmul<'a>(&'a self, rhs: &'a Self) -> Matrix<Vec<u8>, I, O, 2> {
    assert_eq!(
      self.shape()[1],
      rhs.shape()[0],
      "Mismatched matmul dimensions"
    );
    if rhs.data.is_empty() || self.data.is_empty() {
      let dims = [self.shape()[0], rhs.shape()[1]];
      let mut out = Builder::memory().unwrap();
      let data = out.into_fst();
      return Matrix { dims, data };
    }
    let rhs_t = rhs.transpose();
    self.matmul_pretransposed(&rhs_t)
  }
  pub fn matmul_pretransposed<'a, D2>(
    &'a self,
    rhs_t: &'a Matrix<D2, I, O, 2>,
  ) -> Matrix<Vec<u8>, I, O, 2>
  where
    D2: AsRef<[u8]>, {
    assert_eq!(
      self.shape()[1],
      rhs_t.shape()[1],
      "Mismatched matmul dimensions"
    );
    let rows = self.shape()[0];
    let cols = rhs_t.shape()[0];
    let dims = [rows, cols];
    let mut out = Builder::memory().unwrap();
    if rhs_t.data.is_empty() || self.data.is_empty() {
      let data = out.into_fst();
      return Matrix { dims, data };
    }
    // iterating over in row order
    let mut a = self.iter().peekable();
    // need to maintain a buffer per col
    let mut row_buf = vec![O::zero(); self.shape()[1].as_usize()];
    let mut curr_row = I::zero();
    while curr_row < rows {
      for v in row_buf.iter_mut() {
        *v = O::zero();
      }
      while let Some(&([y, x], o)) = a.peek() {
        if y > curr_row {
          break;
        }
        assert_eq!(y, curr_row);
        assert!(a.next().is_some());
        row_buf[x.as_usize()] = o;
      }
      let mut b = rhs_t.pred(|p| !p.is_empty() && !row_buf[p[0].as_usize()].is_zero());
      // iterate through columns of b
      let ([mut curr_x, y], mut acc) = b.next().unwrap();
      acc = acc * row_buf[y.as_usize()];
      for ([x, y], o) in b {
        assert!(x >= curr_x);
        // println!("{:?} {}", [x, y], curr_x);
        if x > curr_x {
          if !acc.is_zero() {
            out.insert([curr_row, curr_x], acc);
          }
          acc = o * row_buf[y.as_usize()];
          curr_x = x;
        } else {
          acc = acc + o * row_buf[y.as_usize()];
        }
      }
      if !acc.is_zero() {
        out.insert([curr_row, curr_x], acc);
      }

      // go on to next row
      curr_row = curr_row + I::one();
    }
    let dims = [rows, cols];
    let data = out.into_fst();
    Matrix { dims, data }
  }
  /// Convolves this matrix row-wise by the kernel
  /// To convolve columnwise, transpose this matrix then convolve.
  pub fn convolve_1d<const K: usize>(&self, kernel: [O; K]) -> Matrix<Vec<u8>, I, O, 2>
  where
    I: Sub<Output = I>, {
    use crate::circ_buf::CircularBuffer;
    assert!(K > 0);
    let mut out = Builder::memory().unwrap();
    // circular buffer of items
    let mut buf: CircularBuffer<O, K> = CircularBuffer::new(O::zero());

    // this is the x coordinate of the last value in the circular buffer
    let mut last_x = I::zero();
    let mut curr_row = I::zero();

    for ([y, x], o) in self.iter() {
      assert!(y >= curr_row);
      if y > curr_row {
        for (i, v) in buf.shift(K, O::zero()).enumerate() {
          if let Some(i) = (last_x.as_usize() + i).checked_sub(K) {
            out.insert([curr_row, I::from_usize(i)], v);
          }
        }
        last_x = x;
        curr_row = y;
      }
      assert!(x >= last_x);
      let diff = x - last_x;
      for (i, v) in buf.shift(diff.as_usize(), O::zero()).enumerate() {
        if let Some(i) = (last_x.as_usize() + i).checked_sub(K) {
          out.insert([curr_row, I::from_usize(i)], v);
        }
      }
      last_x = x;
      for i in 0..K {
        buf.set(i, buf.get(i) + kernel[K - i - 1] * o);
      }
    }
    for (i, v) in buf.shift(K, O::zero()).enumerate() {
      if let Some(i) = (last_x.as_usize() + i).checked_sub(K) {
        out.insert([curr_row, I::from_usize(i)], v);
      }
    }
    Matrix {
      dims: self.shape(),
      data: out.into_fst(),
    }
  }

  pub fn convolve_2d<const K: usize>(&self, kernel: [[O; K]; K]) -> Dense<I, O, 2>
  where
    I: Sub<Output = I>, {
    let mut out = Dense::new(self.dims);
    self.convolve_2d_into(kernel, &mut out);
    out
  }

  /// Convolves this matrix by a 2D kernel of size K known at compile time.
  /// Creating a new output in the Coordinate Format.
  pub fn convolve_2d_into<const K: usize>(&self, kernel: [[O; K]; K], out: &mut Dense<I, O, 2>)
  where
    I: Sub<Output = I>, {
    use crate::circ_buf::CircularBuffer2D;
    assert!(K > 0);
    assert!(K % 2 == 1, "Only supports odd kernels");
    // circular buffer of items
    let mut buf: CircularBuffer2D<O, K> = CircularBuffer2D::new(O::zero());
    let mid = (K - 1) / 2;

    // central buffer coordinate
    let mut l_c = [I::zero(); 2];
    for (i, o) in self.iter() {
      debug_assert!(i >= l_c, "Invariant broken, not iterating in order");
      let [y, x] = i;
      // how much we're shifting in each direction
      let d0 = (y - l_c[0]).as_usize();
      // flush buffer in the x direction might have to flush if wrapped
      let d1 = x.as_usize().checked_sub(l_c[1].as_usize()).unwrap_or(K);
      buf.eager_shift_modify([d0, d1], |y_coord, x_coord, v| {
        let val = replace(v, O::zero());
        let dy = (l_c[0].as_usize() + mid).checked_sub(K - 1 - x_coord);
        let dy = if let Some(dy) = dy { dy } else { return };
        let dy = I::from_usize(dy);
        if dy >= self.dims[0] {
          return;
        }
        let dx = (l_c[1].as_usize() + mid).checked_sub(K - 1 - y_coord);
        let dx = if let Some(dx) = dx { dx } else { return };
        let dx = I::from_usize(dx);
        if dx >= self.dims[1] {
          return;
        }
        let output = &mut out[[dy, dx]];
        *output = *output + val;
      });
      // update lc position of buffer
      l_c = i;
      for i in 0..K {
        for j in 0..K {
          let v = kernel[K - i - 1][K - j - 1];
          let mut item = buf.entry([i, j]);
          *item = *item + v * o;
        }
      }
    }
    let [y, x] = l_c;
    buf.eager_shift_modify([K, K], |y_coord, x_coord, v| {
      let dy = (y.as_usize() + mid).checked_sub(K - 1 - y_coord);
      let dy = if let Some(dy) = dy { dy } else { return };
      let dy = I::from_usize(dy);
      if dy >= self.dims[0] {
        return;
      }
      let dx = (x.as_usize() + mid).checked_sub(K - 1 - x_coord);
      let dx = if let Some(dx) = dx { dx } else { return };
      let dx = I::from_usize(dx);
      if dx >= self.dims[1] {
        return;
      }
      let o = &mut out[[dy, dx]];
      *o = *o + *v;
    })
  }
}

impl<D, I, O, const N: usize> Matrix<D, I, O, N>
where
  D: AsRef<[u8]>,
  I: Input,
  O: Output,
  O: Mul<Output = O>,
  Bytes<O>: Deserialize,
  Bytes<I>: Deserialize,
  [I; N]: LengthAtMost32,
{
  #[inline]
  /// Returns the dimensions of this matrix
  pub fn shape(&self) -> [I; N] { self.dims }
  pub fn count_nonzero(&self) -> usize { self.data.len() }
  /// Returns the size of this matrix in bytes
  pub fn nbytes(&self) -> usize { self.data.nbytes() }
  pub fn sparsity(&self) -> f64 {
    let total = self.shape().iter().map(|l| l.as_usize()).product::<usize>();
    (self.count_nonzero() as f64) / (total as f64)
  }
}

impl<I, O, const N: usize> Matrix<Vec<u8>, I, O, N>
where
  I: Input,
  O: Output + One,
  O: Mul<Output = O>,
  Bytes<O>: Serialize + Deserialize,
  Bytes<I>: Serialize + Deserialize,
  [I; N]: LengthAtMost32,
{
  pub fn eye(n: I) -> Self {
    Matrix::new(
      [n; N],
      (0..n.as_usize()).map(|i| ([I::from_usize(i); N], O::one())),
    )
  }
}

impl<D, I, O, const N: usize> Matrix<D, I, O, N>
where
  D: AsRef<[u8]>,
  I: Input,
  O: Output,
  Bytes<O>: Deserialize,
  Bytes<I>: Deserialize,
  [I; N]: LengthAtMost32,
{
  pub fn to_coo<'a>(&'a self) -> COO<I, O, N>
  where
    [(O, usize, Node<'a, O>); N]: LengthAtMost32, {
    COO::from_iter(self.dims, self.iter())
  }
}
