use crate::{local_prelude::*, vec::BitVec};

impl<B: BitBlockOrStore> BitVec<B> {
    /// Iterator over the underlying blocks of data
    #[inline]
    pub fn blocks(&self) -> Blocks<'_, B> {
        // (2)
        Blocks {
            iter: self.storage.slice().iter(),
        }
    }
}

/// An iterator over the blocks of a `BitVec`.
#[derive(Clone)]
pub struct Blocks<'a, B: 'a + BitBlockOrStore> {
    iter: slice::Iter<'a, Block<B>>,
}

impl<B: BitBlockOrStore> Iterator for Blocks<'_, B> {
    type Item = Block<B>;

    #[inline]
    fn next(&mut self) -> Option<Block<B>> {
        self.iter.next().cloned()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<B: BitBlockOrStore> DoubleEndedIterator for Blocks<'_, B> {
    #[inline]
    fn next_back(&mut self) -> Option<Block<B>> {
        self.iter.next_back().cloned()
    }
}

impl<B: BitBlockOrStore> ExactSizeIterator for Blocks<'_, B> {}
