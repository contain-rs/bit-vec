use crate::local_prelude::*;
use core::slice;

impl<C: BitContainer> BitVec<C::Block, C> {
    /// Iterator over mutable refs to the underlying blocks of data.
    #[inline]
    pub(crate) fn blocks_mut(&mut self) -> BlocksMut<'_, C::Block> {
        // (2)
        BlocksMut(self.storage.slice_mut().iter_mut())
    }
}

pub struct BlocksMut<'a, B: BitBlock>(slice::IterMut<'a, B>);

impl<'a, B> Iterator for BlocksMut<'a, B>
where
    B: BitBlock,
{
    type Item = &'a mut B;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl<B: BitBlock> DoubleEndedIterator for BlocksMut<'_, B> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0.next_back()
    }
}

impl<B: BitBlock> ExactSizeIterator for BlocksMut<'_, B> {}
