use crate::{bytes::*, csr::CSR, input::Input, matrix::Matrix, output::Output, util::within};
use std::{array::LengthAtMost32, collections::BTreeMap};

#[derive(Debug)]
pub struct COO<I, O, const N: usize>
where
  [I; N]: LengthAtMost32, {
  pub(crate) items: BTreeMap<[I; N], O>,
  dims: [I; N],
}

impl<I, O, const N: usize> COO<I, O, N>
where
  I: Input,
  O: Output,
  [I; N]: LengthAtMost32,
{
  pub fn new(dims: [I; N]) -> Self {
    Self {
      dims,
      items: BTreeMap::new(),
    }
  }
  /// Builds this COO from an iterator over values, skipping those outside of dims
  pub fn from_iter<Iter>(dims: [I; N], iter: Iter) -> Self
  where
    Iter: Iterator<Item = ([I; N], O)>, {
    let items: BTreeMap<[I; N], O> = iter.filter(|&(i, _)| within(i, dims)).collect();
    Self { items, dims }
  }
  pub fn get(&self, i: [I; N]) -> O { self.items.get(&i).copied().unwrap_or_else(O::zero) }
  pub fn set(&mut self, idx: [I; N], o: O) -> O {
    if !within(idx, self.dims) {
      return O::zero();
    }
    self.items.insert(idx, o).unwrap_or_else(O::zero)
  }
}

impl<I, O, const N: usize> COO<I, O, N>
where
  I: Input,
  O: Output,
  [I; N]: LengthAtMost32,
  Bytes<I>: Serialize + Deserialize,
  Bytes<O>: Serialize + Deserialize,
{
  pub fn write_fst(&self) -> Matrix<Vec<u8>, I, O, N> {
    Matrix::new(self.dims, self.items.iter().map(|(&i, &o)| (i, o)))
  }
}

impl<I, O> COO<I, O, 2>
where
  I: Input,
  O: Output,
{
  pub fn to_csr(&self) -> CSR<I, O> {
    let dims = self.dims;
    let mut row_ptrs = Vec::with_capacity(dims[0].as_usize());
    let nnz = self.items.len();
    let mut cols = Vec::with_capacity(nnz);
    let mut values = Vec::with_capacity(nnz);
    let mut last_row_start = 0;
    let mut last_row = I::zero();
    for (&i, &o) in self.items.iter() {
      while i[0] > last_row {
        row_ptrs.push(last_row_start);
        // number of non-zero
        last_row_start = values.len();
        last_row = last_row + I::one();
      }
      // always push column and value
      cols.push(i[1]);
      values.push(o);
    }
    while row_ptrs.len() < dims[0].as_usize() {
      row_ptrs.push(nnz);
    }
    CSR {
      row_ptrs,
      cols,
      values,
      dims,
    }
  }
}
