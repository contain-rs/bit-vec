use portable_atomic::{AtomicU8, AtomicU16, AtomicU32, AtomicU64, AtomicU128, AtomicUsize, Ordering};

use crate::local_prelude::*;

macro_rules! atomic_block_impl {
    ($(($t: ident, $u:ident, $size: expr)),*) => ($(
        impl BitBlock for $t {
            type Target = $u;
            const BITS_: usize = $size;
            #[inline]
            fn from_byte(byte: u8) -> $u { <$u as From<u8>>::from(byte) }
            #[inline]
            fn count_ones(&self) -> usize { BitBlock::load(self).count_ones() as usize }
            #[inline]
            fn count_zeros(&self) -> usize { BitBlock::load(self).count_zeros() as usize }
            #[inline]
            fn from(val: $u) -> Self { From::from(val) }
            #[inline]
            fn load(&self) -> Self::Target {
                self.load(Ordering::Relaxed)
            }
            #[inline]
            fn get_mut(&mut self) -> &mut Self::Target {
                self.get_mut()
            }
            const ONE_: Self::Target = 1;
            const ZERO_: Self::Target = 0;
        }
    )*)
}

atomic_block_impl! {
    (AtomicU8, u8, 8),
    (AtomicU16, u16, 16),
    (AtomicU32, u32, 32),
    (AtomicU64, u64, 64),
    (AtomicU128, u128, 128),
    (AtomicUsize, usize, usize::BITS as usize)
}
