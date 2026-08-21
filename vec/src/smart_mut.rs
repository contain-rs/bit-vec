use crate::local_prelude::*;
use core::cell::RefCell;
use core::ops;

/// An iterator for mutable references to the bits in a `BitVec`.
pub struct IterMut<'a, B: 'a + BitBlock = u32, C: 'a + BitContainer<Block = B> = Vec<B>> {
    pub(crate) vec: Rc<RefCell<&'a mut BitVec<B, C>>>,
    range: ops::Range<usize>,
}

impl<'a, C: BitContainer> Iterator for IterMut<'a, C::Block, C> {
    type Item = MutBorrowedBit<'a, C::Block, C>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let index = self.range.next();
        self.get(index)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.range.size_hint()
    }
}

impl<C: BitContainer> DoubleEndedIterator for IterMut<'_, C::Block, C> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        let index = self.range.next_back();
        self.get(index)
    }
}

impl<C: BitContainer> ExactSizeIterator for IterMut<'_, C::Block, C> {}

#[derive(Debug)]
pub struct MutBorrowedBit<'a, B: 'a + BitBlock, C: 'a + BitContainer<Block = B> = Vec<B>> {
    vec: Rc<RefCell<&'a mut BitVec<B, C>>>,
    index: usize,
    #[cfg(debug_assertions)]
    old_value: bool,
    new_value: bool,
}

impl<C: BitContainer> ops::Deref for MutBorrowedBit<'_, C::Block, C> {
    type Target = bool;

    fn deref(&self) -> &Self::Target {
        &self.new_value
    }
}

impl<C: BitContainer> ops::DerefMut for MutBorrowedBit<'_, C::Block, C> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.new_value
    }
}

impl<'a, B: 'a + BitBlock, C: 'a + BitContainer<Block = B>> Drop for MutBorrowedBit<'a, B, C> {
    fn drop(&mut self) {
        let mut vec = (*self.vec).borrow_mut();
        #[cfg(debug_assertions)]
        debug_assert_eq!(
            Some(self.old_value),
            vec.get(self.index),
            "Mutably-borrowed bit was modified externally!"
        );
        vec.set(self.index, self.new_value);
    }
}

impl<'a, C: 'a + BitContainer> IterMut<'a, C::Block, C> {
    fn get(&mut self, index: Option<usize>) -> Option<MutBorrowedBit<'a, C::Block, C>> {
        let value = (*self.vec).borrow().get(index?)?;
        Some(MutBorrowedBit {
            vec: self.vec.clone(),
            index: index?,
            #[cfg(debug_assertions)]
            old_value: value,
            new_value: value,
        })
    }
}

impl<C: BitContainer> BitVec<C::Block, C> {
    /// Retrieves a smart pointer to the value at index `i`, or `None` if the index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let mut bv = BitVec::from_bytes(&[0b01100000]);
    /// *bv.get_mut(0).unwrap() = true;
    /// *bv.get_mut(1).unwrap() = false;
    /// assert!(bv.get_mut(100).is_none());
    /// assert_eq!(bv, BitVec::from_bytes(&[0b10100000]));
    /// ```
    #[inline]
    pub fn get_mut(&mut self, index: usize) -> Option<MutBorrowedBit<'_, C::Block, C>> {
        self.get(index).map(move |value| MutBorrowedBit {
            vec: Rc::new(RefCell::new(self)),
            index,
            #[cfg(debug_assertions)]
            old_value: value,
            new_value: value,
        })
    }

    /// Retrieves a smart pointer to the value at index `i`, without doing bounds checking.
    ///
    /// # Safety
    ///
    /// Calling this method with out-of-bounds `index` may cause undefined behavior even when
    /// the result is not used.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let mut bv = BitVec::from_bytes(&[0b01100000]);
    /// unsafe {
    ///     *bv.get_unchecked_mut(0) = true;
    ///     *bv.get_unchecked_mut(1) = false;
    /// }
    /// assert_eq!(bv, BitVec::from_bytes(&[0b10100000]));
    /// ```
    #[inline]
    pub unsafe fn get_unchecked_mut(&mut self, index: usize) -> MutBorrowedBit<'_, C::Block, C> {
        let value = self.get_unchecked(index);
        MutBorrowedBit {
            #[cfg(debug_assertions)]
            old_value: value,
            new_value: value,
            vec: Rc::new(RefCell::new(self)),
            index,
        }
    }

    /// Returns an iterator over mutable smart pointers to the elements of the vector in order.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let mut a = BitVec::from_elem(8, false);
    /// a.iter_mut().enumerate().for_each(|(index, mut bit)| {
    ///     *bit = if index % 2 == 1 { true } else { false };
    /// });
    /// assert!(a.eq_vec(&[
    ///    false, true, false, true, false, true, false, true
    /// ]));
    /// ```
    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'_, C::Block, C> {
        self.ensure_invariant();
        let nbits = self.nbits;
        IterMut {
            vec: Rc::new(RefCell::new(self)),
            range: 0..nbits,
        }
    }
}
