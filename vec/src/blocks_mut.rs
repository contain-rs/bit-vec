use crate::{local_prelude::*, vec::BitVec};

impl<B: BitBlockOrStore> BitVec<B> {
    /// Iterator over mutable refs to the underlying blocks of data.
    #[inline]
    pub(crate) fn blocks_mut(&mut self) -> BlocksMut<'_, B> {
        // (2)
        self.storage.slice_mut().iter_mut()
    }
}

pub type BlocksMut<'a, B: BitBlockOrStore> = slice::IterMut<'a, Block<B>>;
