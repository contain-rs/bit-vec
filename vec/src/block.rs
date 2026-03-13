use crate::local_prelude::ops::*;
use crate::local_prelude::*;

/// Abstracts over a pile of bits (basically unsigned primitives)
pub trait BitBlock:
    Copy
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Shl<usize, Output = Self>
    + Shr<usize, Output = Self>
    + Not<Output = Self>
    + BitAnd<Self, Output = Self>
    + BitOr<Self, Output = Self>
    + BitXor<Self, Output = Self>
    + Rem<Self, Output = Self>
    + BitOrAssign<Self>
    + Eq
    + Ord
    + hash::Hash
{
    /// How many bits it has
    const BITS_: usize;
    /// How many bytes it has
    const BYTES_: usize = Self::BITS_ / 8;
    /// Convert a byte into this type (lowest-order bits set)
    fn from_byte(byte: u8) -> Self;
    /// Count the number of 1's in the bitwise repr
    fn count_ones(self) -> usize;
    /// Count the number of 0's in the bitwise repr
    fn count_zeros(self) -> usize {
        Self::BITS_ - self.count_ones()
    }
    /// Get `0`
    const ZERO_: Self;
    /// Get `1`
    const ONE_: Self;
}

macro_rules! bit_block_impl {
    ($(($t: ident, $size: expr)),*) => ($(
        impl BitBlock for $t {
            const BITS_: usize = $size;
            #[inline]
            fn from_byte(byte: u8) -> Self { $t::from(byte) }
            #[inline]
            fn count_ones(self) -> usize { self.count_ones() as usize }
            #[inline]
            fn count_zeros(self) -> usize { self.count_zeros() as usize }
            const ONE_: Self = 1;
            const ZERO_: Self = 0;
        }
    )*)
}

bit_block_impl! {
    (u8, 8),
    (u16, 16),
    (u32, 32),
    (u64, 64),
    (usize, usize::BITS as usize)
}
