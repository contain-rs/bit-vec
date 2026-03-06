//! Implements access to a matrix's individual rows.

use core::{mem, ops};

use super::{FALSE, TRUE};
use crate::local_prelude::*;
use crate::util::div_rem;

/// A slice of bit vector's blocks.
pub struct BitSlice {
    pub(crate) slice: [Block],
}

impl BitSlice {
    /// Creates a new slice from a slice of blocks.
    #[inline]
    pub fn new(slice: &[Block]) -> &Self {
        unsafe { mem::transmute(slice) }
    }

    /// Creates a new slice from a mutable slice of blocks.
    #[inline]
    pub fn new_mut(slice: &mut [Block]) -> &mut Self {
        unsafe { mem::transmute(slice) }
    }

    /// Iterates over bits.
    #[inline]
    pub fn iter_bits(&self, len: usize) -> impl Iterator<Item = bool> + '_ {
        (0 .. len).map(|i| self[i])
    }

    /// Iterates over the slice's blocks.
    pub fn iter_blocks(&self) -> impl Iterator<Item = &Block> {
        self.slice.iter()
    }

    /// Iterates over the slice's blocks, yielding mutable references.
    pub fn iter_blocks_mut(&mut self) -> impl Iterator<Item = &mut Block> {
        self.slice.iter_mut()
    }

    /// Returns `true` if a bit is enabled in the bit vector slice, or `false` otherwise.
    #[inline]
    pub fn get(&self, bit: usize) -> bool {
        let (block, i) = div_rem(bit, BITS);
        match self.slice.get(block) {
            None => false,
            Some(b) => (b & (1 << i)) != 0,
        }
    }

    /// Returns a small integer-sized slice of the bit vector slice.
    #[inline]
    pub fn small_slice_aligned(&self, bit: usize, len: u8) -> u32 {
        let (block, i) = div_rem(bit, BITS);
        match self.slice.get(block) {
            None => 0,
            Some(&b) => {
                let len_mask = (1 << len) - 1;
                (b >> i) & len_mask
            }
        }
    }
}

/// Returns `true` if a bit is enabled in the bit vector slice,
/// or `false` otherwise.
impl ops::Index<usize> for BitSlice {
    type Output = bool;

    #[inline]
    fn index(&self, bit: usize) -> &bool {
        let (block, i) = div_rem(bit, BITS);
        match self.slice.get(block) {
            None => &FALSE,
            Some(b) => {
                if (b & (1 << i)) != 0 {
                    &TRUE
                } else {
                    &FALSE
                }
            }
        }
    }
}

impl ops::BitOrAssign for &mut BitSlice {
    fn bitor_assign(&mut self, rhs: Self) {
        debug_assert_eq!(self.slice.len(), rhs.slice.len());
        for (dst, src) in self.iter_blocks_mut().zip(rhs.iter_blocks()) {
            *dst |= src;
        }
    }
}
