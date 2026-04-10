use portable_atomic::{AtomicU32, Ordering};

use crate::BitVec;
use crate::local_prelude::*;

macro_rules! atomic (
    ($Atomic:ty) => {
        impl BitVec<Vec<$Atomic>> {
            pub fn fetch_set(&self, i: usize, elem: bool, order: Ordering) -> bool {
                self.ensure_invariant();
                assert!(
                    i < self.nbits,
                    "index out of bounds: {:?} >= {:?}",
                    i,
                    self.nbits
                );
                let bits = mem::size_of::<$Atomic>() * 8;
                let w = i / bits;
                let b = i % bits;
                let flag = <$Atomic>::ONE_ << b;
                if elem {
                    self.storage.slice()[w].fetch_or(flag, order) & flag == flag
                } else {
                    self.storage.slice()[w].fetch_and(!flag, order) & flag == flag
                }
            }
        }

        impl BitVec<$Atomic> {
            pub fn fetch_set(&self, i: usize, elem: bool, order: Ordering) -> bool {
                self.ensure_invariant();
                assert!(
                    i < self.nbits,
                    "index out of bounds: {:?} >= {:?}",
                    i,
                    self.nbits
                );
                let bits = mem::size_of::<$Atomic>() * 8;
                let w = i / bits;
                let b = i % bits;
                let flag = <$Atomic>::ONE_ << b;
                if elem {
                    self.storage.slice()[w].fetch_or(flag, order) & flag == flag
                } else {
                    self.storage.slice()[w].fetch_and(!flag, order) & flag == flag
                }
            }
        }
    }
);

atomic!(portable_atomic::AtomicU8);
atomic!(portable_atomic::AtomicU16);
atomic!(portable_atomic::AtomicU32);
atomic!(portable_atomic::AtomicU64);
atomic!(portable_atomic::AtomicU128);
atomic!(portable_atomic::AtomicUsize);

// for IDE

// impl BitVec<Vec<AtomicU32>> {
//     pub fn fetch_set(&self, i: usize, elem: bool, order: Ordering) -> bool {
//         self.ensure_invariant();
//         assert!(
//             i < self.nbits,
//             "index out of bounds: {:?} >= {:?}",
//             i,
//             self.nbits
//         );
//         let bits = mem::size_of::<AtomicU32>() * 8;
//         let w = i / bits;
//         let b = i % bits;
//         let flag = AtomicU32::ONE_ << b;
//         if elem {
//             self.storage.slice()[w].fetch_or(flag, order) & flag == flag
//         } else {
//             self.storage.slice()[w].fetch_and(!flag, order) & flag == flag
//         }
//     }
// }
