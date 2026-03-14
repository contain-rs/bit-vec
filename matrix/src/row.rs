//! Implements access to a matrix's individual rows.

use core::{mem, ops};

use bit_vec::BitBlock;

use super::{FALSE, TRUE};
// use crate::local_prelude::*;
use crate::util::div_rem;

/// A slice of bit vector's blocks.
pub struct BitSlice<Block> {
    pub(crate) slice: [Block],
}

impl<Block: BitBlock> BitSlice<Block> {
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
        (0..len).map(|i| self[i])
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
        let (block, i) = div_rem(bit, Block::BITS_);
        match self.slice.get(block) {
            None => false,
            Some(&b) => (b & (Block::ONE_ << i)) != Block::ZERO_,
        }
    }

    /// Returns a small integer-sized slice of the bit vector slice.
    #[inline]
    pub fn small_slice_aligned(&self, bit: usize, len: u8) -> Block {
        let (block, i) = div_rem(bit, Block::BITS_);
        match self.slice.get(block) {
            None => Block::ZERO_,
            Some(&b) => {
                let len_mask = (Block::ONE_ << len as usize) - Block::ONE_;
                (b >> i) & len_mask
            }
        }
    }
}

/// Returns `true` if a bit is enabled in the bit vector slice,
/// or `false` otherwise.
impl<Block: BitBlock> ops::Index<usize> for BitSlice<Block> {
    type Output = bool;

    #[inline]
    fn index(&self, bit: usize) -> &bool {
        let (block, i) = div_rem(bit, Block::BITS_);
        match self.slice.get(block) {
            None => &FALSE,
            Some(&b) => {
                if (b & (Block::ONE_ << i)) != Block::ZERO_ {
                    &TRUE
                } else {
                    &FALSE
                }
            }
        }
    }
}

impl<Block: BitBlock> ops::BitOrAssign for &mut BitSlice<Block> {
    fn bitor_assign(&mut self, rhs: Self) {
        debug_assert_eq!(self.slice.len(), rhs.slice.len());
        for (dst, src) in self.iter_blocks_mut().zip(rhs.iter_blocks()) {
            *dst |= *src;
        }
    }
}
