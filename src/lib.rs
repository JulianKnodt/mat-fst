#![feature(const_generics, const_generic_impls_guard, const_fn, is_sorted)]
#![allow(incomplete_features)]

extern crate num;

pub mod build;
mod bytes;
mod counting_writer;
mod error;
mod fst;
pub mod iter;
mod node;
// mod node_iter;
#[cfg(feature = "parallel")]
pub mod par_iter;

// Fst generalization traits
pub mod input;
pub mod output;

// matrix related stuff
pub mod matrix;

// utilities for sparsity
mod circ_buf;
pub mod util;

// alternate matrix formats
pub mod coo;
pub mod csr;
pub mod dense;

#[cfg(test)]
#[macro_use]
extern crate quickcheck;
#[cfg(test)]
mod tests;
