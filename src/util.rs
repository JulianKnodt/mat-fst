/// Computes a threshold for a given set of items which has at least the target sparsity
pub fn compute_threshold<O: Ord + Copy, I: Iterator<Item = O>>(
  items: I,
  target_sparsity: f64,
) -> O {
  assert!(
    target_sparsity <= 1.0,
    "Cannot compute sparsity greater than 1"
  );
  let mut items = items.collect::<Vec<_>>();
  let dest_index = ((items.len() - 1) as f64 * target_sparsity).ceil() as usize;
  items.sort_unstable();
  items[dest_index]
}

#[inline]
pub fn within<I: Ord, const N: usize>(pt: [I; N], bounds: [I; N]) -> bool {
  for i in 0..N {
    if bounds[i] <= pt[i] {
      return false;
    }
  }
  true
}
