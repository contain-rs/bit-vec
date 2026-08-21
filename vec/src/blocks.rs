use crate::local_prelude::*;
use core::slice;

impl<C: BitContainer> BitVec<C::Block, C> {
    /// Iterator over the underlying blocks of data
    #[inline]
    pub fn blocks(&self) -> Blocks<'_, C::Block> {
        // (2)
        Blocks {
            iter: self.storage.slice().iter(),
        }
    }
}

/// An iterator over the blocks of a `BitVec`.
#[derive(Clone)]
pub struct Blocks<'a, B: 'a + BitBlock> {
    iter: slice::Iter<'a, B>,
}

impl<B: BitBlock> Iterator for Blocks<'_, B> {
    type Item = B;

    #[inline]
    fn next(&mut self) -> Option<B> {
        self.iter.next().cloned()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<B: BitBlock> DoubleEndedIterator for Blocks<'_, B> {
    #[inline]
    fn next_back(&mut self) -> Option<B> {
        self.iter.next_back().cloned()
    }
}

impl<B: BitBlock> ExactSizeIterator for Blocks<'_, B> {}
