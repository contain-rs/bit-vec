use crate::{local_prelude::*, BitVec};

pub struct IntoIter<B: BitBlockOrStore = u32> {
    bit_vec: BitVec<B>,
    range: ops::Range<usize>,
}

impl<B: BitBlockOrStore> Iterator for IntoIter<B> {
    type Item = bool;

    #[inline]
    fn next(&mut self) -> Option<bool> {
        self.range.next().map(|i| self.bit_vec.get(i).unwrap())
    }
}

impl<B: BitBlockOrStore> DoubleEndedIterator for IntoIter<B> {
    #[inline]
    fn next_back(&mut self) -> Option<bool> {
        self.range.next_back().map(|i| self.bit_vec.get(i).unwrap())
    }
}

impl<B: BitBlockOrStore> ExactSizeIterator for IntoIter<B> {}

impl<B: BitBlockOrStore> IntoIterator for BitVec<B> {
    type Item = bool;
    type IntoIter = IntoIter<B>;

    #[inline]
    fn into_iter(self) -> IntoIter<B> {
        let nbits = self.nbits;
        IntoIter {
            bit_vec: self,
            range: 0..nbits,
        }
    }
}
