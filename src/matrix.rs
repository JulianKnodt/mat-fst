use crate::{
  build::Builder,
  bytes::*,
  coo::COO,
  dense::Dense,
  fst::Fst,
  input::Input,
  node::{immediate_iter, immediate_range_iter, Node},
  output::Output,
  util::within,
};
use num::One;
use std::{
  array::LengthAtMost32,
  cmp::Ordering,
  mem::replace,
  ops::{Mul, Sub},
};

#[derive(Debug)]
pub struct Matrix<D, I, O, const N: usize>
where
  [I; N]: LengthAtMost32, {
  // row, col, etc
  pub dims: [I; N],
  pub data: Fst<D, I, O>,
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
  pub fn get(&self, idxs: [I; 2]) -> O { self.data.get(&idxs[..]).unwrap_or_else(O::zero) }
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
      out.len() >= rows,
      "Dimension mismatch, expected output of size {}",
      self.dims[0]
    );
    let data = self.data.data.as_ref();
    for t0 in immediate_iter::<I>(self.data.meta.root_addr, data) {
      let y = t0.input.as_usize();
      let start = t0.num_out;
      for (offset, x) in immediate_range_iter::<I>(t0.addr, data).enumerate() {
        assert!(self.data.outputs.len() as u32 > start + offset as u32);
        out[y] = out[y] + vec[x.as_usize()] * self.data.outputs[(start + offset as u32) as usize];
      }
    }
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
    [(u32, usize, Node<'a, I>); N]: LengthAtMost32, {
    assert_eq!(self.shape(), o.shape());
    let a = self.iter();
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
    [(u32, usize, Node<'a, I>); N]: LengthAtMost32, {
    let mut inv_items = Vec::with_capacity(self.data.len());
    inv_items.extend(self.iter().map(|(mut i, o)| {
      i.reverse();
      (i, o)
    }));
    // TODO This is expensive? Work on a way to make this cheaper
    inv_items.sort_unstable_by_key(|&(i, _)| i);
    let mut out = Builder::memory().unwrap();
    for (i, o) in inv_items {
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
      // handle simple case where
      let dims = [self.shape()[0], rhs.shape()[1]];
      let out: Builder<_, _, _, 2> = Builder::memory().unwrap();
      let data = out.into_fst();
      return Matrix { dims, data };
    }
    let rhs_t = rhs.transpose();
    let mut buffer = Builder::memory().unwrap();
    self.matmul_buf(&rhs_t, &mut buffer);
    let data = buffer.into_fst();
    let dims = [self.shape()[0], rhs.shape()[1]];
    Matrix { dims, data }
  }
  /// Writes the output of matrix multiplication into out
  pub fn matmul_buf<'a, D2, DOut>(
    &'a self,
    rhs_t: &'a Matrix<D2, I, O, 2>,
    out: &mut Builder<DOut, I, O, 2>,
  ) where
    D2: AsRef<[u8]>,
    DOut: AsRef<[u8]> + std::io::Write, {
    assert_eq!(
      self.shape()[1],
      rhs_t.shape()[1],
      "Mismatched matmul dimensions"
    );
    let rows = self.shape()[0];
    if rhs_t.data.is_empty() || self.data.is_empty() {
      return;
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
      let mut b = rhs_t.iter();
      // iterate through columns of b
      let ([mut curr_x, y], mut acc) = b.next().unwrap();
      acc = acc * row_buf[y.as_usize()];
      for ([x, y], o) in b {
        assert!(x >= curr_x);
        if x > curr_x {
          if !acc.is_zero() {
            out.insert([curr_row, curr_x], acc).unwrap();
          }
          acc = o * row_buf[y.as_usize()];
          curr_x = x;
        } else {
          acc = acc + o * row_buf[y.as_usize()];
        }
      }
      if !acc.is_zero() {
        out.insert([curr_row, curr_x], acc).unwrap();
      }

      // go on to next row
      curr_row = curr_row + I::one();
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
          let item = buf.entry([i, j]);
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
    [(u32, usize, Node<'a, I>); N]: LengthAtMost32, {
    COO::from_iter(self.dims, self.iter())
  }
}
