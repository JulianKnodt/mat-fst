use std::io;

#[derive(Debug)]
pub enum Error<I> {
  IO(io::Error),
  // Duplicate key without same output value
  DuplicateKey { key: Vec<I> },
  // Got input sequence out of order
  OutOfOrder { prev: Vec<I>, next: Vec<I> },
}

impl<I> From<io::Error> for Error<I> {
  fn from(io_error: io::Error) -> Self { Error::IO(io_error) }
}

pub type Result<T, I> = std::result::Result<T, Error<I>>;
