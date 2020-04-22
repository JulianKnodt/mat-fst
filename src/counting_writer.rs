use std::io::{self, Write};

/// Writer which tracks how many bytes have been written to the underlying writter
#[derive(Debug)]
pub struct CountingWriter<W> {
  wtr: W,
  /// Bytes written to the underlying writer
  count: u64,
}

impl<W: Write> CountingWriter<W> {
  pub fn new(wtr: W) -> Self { Self { wtr, count: 0 } }
  pub fn count(&self) -> u64 { self.count }
  pub fn inner(self) -> W { self.wtr }
}

impl CountingWriter<Vec<u8>> {
  pub fn reset(&mut self) {
    self.wtr.clear();
    self.count = 0;
  }
}

impl<W: Write> Write for CountingWriter<W> {
  fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
    let n = self.wtr.write(buf)?;
    self.count += n as u64;
    Ok(n)
  }
  fn flush(&mut self) -> io::Result<()> { self.wtr.flush() }
}
