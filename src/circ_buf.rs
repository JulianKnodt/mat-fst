/// A circular buffer of fixed size
pub struct CircularBuffer<T, const L: usize> {
  items: [T; L],
  start_index: usize,
}

impl<T: Copy, const L: usize> CircularBuffer<T, L> {
  pub fn new(val: T) -> Self {
    CircularBuffer {
      items: [val; L],
      start_index: 0,
    }
  }
  #[inline]
  pub fn get(&self, i: usize) -> T {
    assert!(i < L, "Out of bounds");
    self.items[(self.start_index + i) % L]
  }
  #[inline]
  pub fn set(&mut self, i: usize, t: T) {
    assert!(i < L, "Out of bounds");
    self.items[(self.start_index + i) % L] = t;
  }
  /// Shifts this circular buffer by at most L
  /// returning elements visited along the way
  /// and setting them to a default as they're passed
  pub fn shift(&mut self, by: usize, def: T) -> impl Iterator<Item = T> + '_ {
    let by = by.min(L);
    let s0 = self.start_index;
    self.start_index = (s0 + by) % L;
    (s0..s0 + by).map(|i| i % L).map(move |i| {
      let out = self.items[i];
      self.items[i] = def;
      out
    })
  }
}

/// A 2 dimensional square circular buffer
pub struct CircularBuffer2D<T, const L: usize> {
  pub(crate) items: [[T; L]; L],
  start_coord: [usize; 2],
}

impl<T: Copy, const L: usize> CircularBuffer2D<T, L> {
  pub fn new(val: T) -> Self {
    CircularBuffer2D {
      items: [[val; L]; L],
      start_coord: [0, 0],
    }
  }
  #[inline]
  pub fn get(&self, c: [usize; 2]) -> T {
    assert!(c < [L, L]);
    let [c0, c1] = c;
    let [s0, s1] = self.start_coord;
    self.items[(c0 + s0) % L][(c1 + s1) % L]
  }
  #[inline]
  pub fn set(&mut self, c: [usize; 2], t: T) {
    assert!(c < [L, L]);
    let [c0, c1] = c;
    let [s0, s1] = self.start_coord;
    self.items[(c0 + s0) % L][(c1 + s1) % L] = t;
  }

  /// Gets item covered if a certain range of this iterator was shifted
  pub fn covered(&self, c: [usize; 2]) -> impl Iterator<Item = ([usize; 2], T)> + '_ {
    let [c0, c1] = c;
    let c0 = c0.min(L);
    let c1 = c1.min(L);
    let [s0, s1] = self.start_coord;
    (0..c0)
      .flat_map(move |y| (0..L).map(move |x| ([y, x], self.items[(s0 + y) % L][(s1 + x) % L])))
      .chain(
        (c0..L).flat_map(move |y| {
          (0..c1).map(move |x| ([y, x], self.items[(s0 + y) % L][(s1 + x) % L]))
        }),
      )
  }
  pub fn shift(&mut self, c: [usize; 2], def: T) {
    let [c0, c1] = c;
    let c0 = c0.min(L);
    let c1 = c1.min(L);
    let [s0, s1] = self.start_coord;
    self.start_coord = [(s0 + c0) % L, (s1 + c1) % L];
    for y in 0..c0 {
      for x in 0..L {
        self.items[(s0 + y) % L][(s1 + x) % L] = def;
      }
    }
    for y in c0..L {
      for x in 0..c1 {
        self.items[(s0 + y) % L][(s1 + x) % L] = def;
      }
    }
  }
}
