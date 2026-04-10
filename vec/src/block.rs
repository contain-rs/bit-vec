use crate::local_prelude::ops::*;
use crate::local_prelude::*;

pub type Target<B: BitBlockOrStore> = <Block::<B> as BitBlock>::Target;

/// Abstracts over a pile of bits (basically unsigned primitives)
pub trait BitBlock where Self: Sized, Self::Target:
    Copy
    + Add<Self::Target, Output = Self::Target>
    + Sub<Self::Target, Output = Self::Target>
    + Shl<usize, Output = Self::Target>
    + Shr<usize, Output = Self::Target>
    + Not<Output = Self::Target>
    + BitAnd<Self::Target, Output = Self::Target>
    + BitOr<Self::Target, Output = Self::Target>
    + BitXor<Self::Target, Output = Self::Target>
    + Rem<Self::Target, Output = Self::Target>
    + BitOrAssign<Self::Target>
    + Eq
    + Ord
    + hash::Hash
    + Into<Self>
{
    type Target;
    /// How many bits it has
    const BITS_: usize;
    /// How many bytes it has
    const BYTES_: usize = Self::BITS_ / 8;
    /// Convert a byte into this type (lowest-order bits set)
    fn from_byte(byte: u8) -> Self::Target;
    /// Count the number of 1's in the bitwise repr
    fn count_ones(&self) -> usize;
    /// Count the number of 0's in the bitwise repr
    fn count_zeros(&self) -> usize {
        Self::BITS_ - self.count_ones()
    }
    fn from(target: Self::Target) -> Self;
    fn load(&self) -> Self::Target;
    fn get_mut(&mut self) -> &mut Self::Target;
    /// Get `0`
    const ZERO_: Self::Target;
    /// Get `1`
    const ONE_: Self::Target;
}

macro_rules! bit_block_impl {
    ($(($t: ident, $size: expr)),*) => ($(
        impl BitBlock for $t {
            type Target = Self;
            const BITS_: usize = $size;
            #[inline]
            fn from_byte(byte: u8) -> Self::Target { <$t as From<u8>>::from(byte) }
            #[inline]
            fn count_ones(&self) -> usize { self.load().count_ones() as usize }
            #[inline]
            fn count_zeros(&self) -> usize { self.load().count_zeros() as usize }
            #[inline]
            fn get_mut(&mut self) -> &mut Self {
                self
            }
            #[inline]
            fn load(&self) -> Self {
                *self
            }
            #[inline]
            fn from(val: Self) -> Self {
                val
            }
            const ONE_: Self::Target = 1;
            const ZERO_: Self::Target = 0;
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
