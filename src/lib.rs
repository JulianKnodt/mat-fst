#![feature(const_generics, const_generic_impls_guard, const_fn)]
#![allow(incomplete_features, unused)]

extern crate num;

mod build;
mod bytes;
mod counting_writer;
mod error;
mod fst;
mod iter;
mod node;

// Fst generalization traits
mod input;
mod output;

// matrix related stuff
pub mod matrix;

#[cfg(test)]
#[macro_use]
extern crate quickcheck;
#[cfg(test)]
mod unit_tests;
