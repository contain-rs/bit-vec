//! Arithmetic functions.

#[inline]
pub fn div_rem(num: usize, divisor: usize) -> (usize, usize) {
    (num / divisor, num % divisor)
}

#[inline]
pub fn round_up_to_next(unrounded: usize, target_alignment: usize) -> usize {
    assert!(target_alignment.is_power_of_two());
    (unrounded + target_alignment - 1) & !(target_alignment - 1)
}
