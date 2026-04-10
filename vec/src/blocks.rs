use crate::{local_prelude::*, vec::BitVec};

impl<B: BitBlockOrStore> BitVec<B> {
    /// Iterator over the underlying blocks of data
    #[inline]
    pub fn blocks(&self) -> Blocks<'_, B> {
        // (2)
        Blocks {
            iter: self.storage.slice().iter().map(Block::<B>::load),
        }
    }

    #[inline]
    pub fn block_refs(&self) -> BlockRefs<'_, B> {
        self.storage.slice().iter()
    }
}

/// An iterator over the blocks of a `BitVec`.
#[derive(Clone)]
pub struct Blocks<'a, B: 'a + BitBlockOrStore> {
    iter: iter::Map<slice::Iter<'a, Block<B>>, fn(&'a Block<B>) -> Target<B>>,
}

pub type BlockRefs<'a, B: 'a + BitBlockOrStore> = slice::Iter<'a, Block<B>>;

impl<B: BitBlockOrStore> Iterator for Blocks<'_, B> {
    type Item = Block<B>;

    #[inline]
    fn next(&mut self) -> Option<Block<B>> {
        self.iter.next().map(|b| b.into())
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<B: BitBlockOrStore> DoubleEndedIterator for Blocks<'_, B> {
    #[inline]
    fn next_back(&mut self) -> Option<Block<B>> {
        self.iter.next_back().map(|b| b.into())
    }
}

impl<B: BitBlockOrStore> ExactSizeIterator for Blocks<'_, B> {}
