use crate::local_prelude::*;
use core::ops;

impl<C: BitContainer> BitVec<C::Block, C> {
    /// Returns an iterator over the elements of the vector in order.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let bv = BitVec::from_bytes(&[0b01110100, 0b10010010]);
    /// assert_eq!(bv.iter().filter(|x| *x).count(), 7);
    /// ```
    #[inline]
    pub fn iter(&self) -> Iter<'_, C::Block, C> {
        self.ensure_invariant();
        Iter {
            bit_vec: self,
            range: 0..self.nbits,
        }
    }
}

/// An iterator for `BitVec`.
#[derive(Clone)]
pub struct Iter<'a, B: BitBlock = u32, C: 'a + BitContainer<Block = B> = Vec<B>> {
    bit_vec: &'a BitVec<B, C>,
    range: ops::Range<usize>,
}

impl<C: BitContainer> Iterator for Iter<'_, C::Block, C> {
    type Item = bool;

    #[inline]
    fn next(&mut self) -> Option<bool> {
        // NB: indexing is slow for extern crates when it has to go through &TRUE or &FALSE
        // variables.  get is more direct, and unwrap is fine since we're sure of the range.
        self.range.next().map(|i| self.bit_vec.get(i).unwrap())
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        // This override is used by the compiler to optimize Iterator::skip.
        // Without this, the default implementation of Iterator::nth is used, which walks over
        // the whole iterator up to n.
        self.range.nth(n).and_then(|i| self.bit_vec.get(i))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.range.size_hint()
    }
}

impl<C: BitContainer> DoubleEndedIterator for Iter<'_, C::Block, C> {
    #[inline]
    fn next_back(&mut self) -> Option<bool> {
        self.range.next_back().map(|i| self.bit_vec.get(i).unwrap())
    }
}

impl<C: BitContainer> ExactSizeIterator for Iter<'_, C::Block, C> {}

impl<'a, C: BitContainer> IntoIterator for &'a BitVec<C::Block, C> {
    type Item = bool;
    type IntoIter = Iter<'a, C::Block, C>;

    #[inline]
    fn into_iter(self) -> Iter<'a, C::Block, C> {
        self.iter()
    }
}
