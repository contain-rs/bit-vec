use portable_atomic::{AtomicU8, AtomicU16, AtomicU32, AtomicU64, AtomicU128, AtomicUsize};

use crate::local_prelude::*;

macro_rules! atomic_bit_block_or_store_impl {
    ($($t: ident),*) => ($(
        impl BitBlockOrStore for $t {
            type Store = Vec<Self>;
        }
    )*)
}

atomic_bit_block_or_store_impl! {
    AtomicU8,
    AtomicU16,
    AtomicU32,
    AtomicU64,
    AtomicU128,
    AtomicUsize
}
