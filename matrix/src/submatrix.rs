//! Submatrix of bits.

use core::cmp;
use core::fmt;
use core::mem;
use core::ops::RangeBounds;
use core::ops::{Index, IndexMut};
use core::slice;

use crate::local_prelude::*;
use crate::util::{div_rem, round_up_to_next};

/// Immutable access to a range of matrix's rows.
pub struct BitSubMatrix<'a, B: BitBlock> {
    pub(crate) slice: &'a [B],
    pub(crate) row_bits: usize,
}

/// Mutable access to a range of matrix's rows.
pub struct BitSubMatrixMut<'a, B: BitBlock> {
    pub(crate) slice: &'a mut [B],
    pub(crate) row_bits: usize,
}

impl<'a, B: BitBlock> BitSubMatrix<'a, B> {
    // /// Returns a new BitSubMatrix.
    // pub(crate) fn new(slice: &[B], row_bits: usize) -> BitSubMatrix<'_, B> {
    //     BitSubMatrix { slice, row_bits }
    // }

    /// Forms a BitSubMatrix from a pointer and dimensions.
    ///
    /// # Safety
    ///
    /// Can construct an ill-formed value, thus the function is marked as
    /// unsafe.
    #[inline]
    pub unsafe fn from_raw_parts(ptr: *const B, rows: usize, row_bits: usize) -> Self {
        BitSubMatrix {
            slice: slice::from_raw_parts(ptr, round_up_to_next(row_bits, B::BITS) / B::BITS * rows),
            row_bits,
        }
    }

    /// Iterates over the matrix's rows in the form of immutable slices.
    pub fn iter(&self) -> impl Iterator<Item = &BitSlice<B>> {
        fn f<B: BitBlock>(arg: &[B]) -> &BitSlice<B> {
            // Safety:
            // This is currently the only way to construct a custom DST.
            // We wish the layout of DSTs were defined.
            unsafe { mem::transmute(arg) }
        }
        let row_size = round_up_to_next(self.row_bits, B::BITS) / B::BITS;
        self.slice.chunks(row_size).map(f::<B>)
    }

    fn row_size(&self) -> usize {
        round_up_to_next(self.row_bits, B::BITS) / B::BITS
    }
}

impl<'a, B: BitBlock> BitSubMatrixMut<'a, B> {
    /// Returns a new `BitSubMatrixMut`.
    pub(crate) fn new(slice: &mut [B], row_bits: usize) -> BitSubMatrixMut<'_, B> {
        BitSubMatrixMut { slice, row_bits }
    }

    /// Forms a `BitSubMatrix` from a pointer and dimensions.
    ///
    /// # Safety
    ///
    /// Can construct an ill-formed value, thus the function is unsafe.
    #[inline]
    pub unsafe fn from_raw_parts(ptr: *mut B, rows: usize, row_bits: usize) -> Self {
        BitSubMatrixMut {
            slice: slice::from_raw_parts_mut(
                ptr,
                round_up_to_next(row_bits, B::BITS) / B::BITS * rows,
            ),
            row_bits,
        }
    }

    /// Returns the number of rows.
    #[inline]
    fn num_rows(&self) -> usize {
        self.slice.len().checked_div(self.row_size()).unwrap_or(0)
    }

    /// Returns the number of columns.
    #[inline]
    pub fn num_cols(&self) -> usize {
        self.row_bits
    }

    /// Sets the value of a bit. The first argument is the row number.
    ///
    /// # Panics
    ///
    /// Panics if `(row, col)` is out of bounds.
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, enabled: bool) {
        let row_size_in_bits = round_up_to_next(self.row_bits, B::BITS);
        let bit = row * row_size_in_bits + col;
        let (block, i) = div_rem(bit, B::BITS);
        assert!(
            block < self.slice.len() && col < self.row_bits,
            "invalid index given to `BitSubMatrixMut::set`"
        );
        unsafe {
            // Safety:
            // We check for `block` being within bounds in the assert above.
            let elt = self.slice.get_unchecked_mut(block);
            if enabled {
                *elt |= B::ONE << i;
            } else {
                *elt = *elt & !(B::ONE << i);
            }
        }
    }

    /// Sets the value of a bit. The first argument is the row number.
    ///
    /// # Safety
    ///
    /// Unsafe if `(row, col)` is out of bounds.
    #[inline]
    pub unsafe fn set_unchecked(&mut self, row: usize, col: usize, enabled: bool) {
        let row_size_in_bits = round_up_to_next(self.row_bits, B::BITS);
        let bit = row * row_size_in_bits + col;
        let (block, i) = div_rem(bit, B::BITS);
        unsafe {
            // Safety:
            // Unsafe if `(row, col)` is out of bounds.
            let elt = self.slice.get_unchecked_mut(block);
            if enabled {
                *elt |= B::ONE << i;
            } else {
                *elt = *elt & !(B::ONE << i);
            }
        }
    }

    /// Returns a slice of the matrix's rows.
    pub fn sub_matrix<R: RangeBounds<usize>>(&self, range: R) -> BitSubMatrix<'_, B> {
        BitSubMatrix {
            slice: &self.slice[(
                range.start_bound().map(|&s| s * self.row_size()),
                range.end_bound().map(|&e| e * self.row_size()),
            )],
            row_bits: self.row_bits,
        }
    }

    /// Given a row's index, returns a slice of all rows above that row, a reference to said row,
    /// and a slice of all rows below.
    ///
    /// Functionally equivalent to `(self.sub_matrix(0..row), &self[row],
    /// self.sub_matrix(row..self.num_rows()))`.
    #[inline]
    pub fn split_at(&self, row: usize) -> (BitSubMatrix<'_, B>, BitSubMatrix<'_, B>) {
        (
            self.sub_matrix(0..row),
            self.sub_matrix(row..self.num_rows()),
        )
    }

    /// Given a row's index, returns a slice of all rows above that row, a reference to said row,
    /// and a slice of all rows below.
    #[inline]
    pub fn split_at_mut(&mut self, row: usize) -> (BitSubMatrixMut<'_, B>, BitSubMatrixMut<'_, B>) {
        let (first, second) = self.slice.split_at_mut(row * self.row_size());
        (
            BitSubMatrixMut::new(first, self.row_bits),
            BitSubMatrixMut::new(second, self.row_bits),
        )
    }

    /// Computes the transitive closure of the binary relation
    /// represented by this square bit matrix.
    ///
    /// Modifies this matrix in place using Warshall's algorithm.
    ///
    /// After this operation, the matrix will describe a transitive
    /// relation. This means that, for any indices `a`, `b`, `c`,
    /// if `M[(a, b)]` and `M[(b, c)]`, then `M[(a, c)]`.
    ///
    /// # Complexity
    ///
    /// The time complexity is **O(n^3)**, where `n` is the number
    /// of columns and rows.
    ///
    /// # Panics
    ///
    /// The matrix must be square for this operation to succeed.
    pub fn transitive_closure(&mut self) {
        assert!(self.is_square());
        for pos in 0..self.row_bits {
            let (mut rows0, mut rows1a) = self.split_at_mut(pos);
            let (mut row, mut rows1b) = rows1a.split_at_mut(1);
            for mut dst_row in rows0.iter_mut().chain(rows1b.iter_mut()) {
                if dst_row[pos] {
                    dst_row |= &mut row[0];
                }
            }
        }
    }

    /// Determines whether the number of rows equals the number of columns.
    ///
    /// This means the matrix is square.
    fn is_square(&self) -> bool {
        self.num_rows() == self.row_bits
    }

    /// Computes the reflexive closure of the binary relation represented by
    /// this bit matrix. The matrix can be rectangular.
    ///
    /// The reflexive closure means that for every `x`` that will be within bounds,
    /// `M[(x, x)]` is true.
    ///
    /// In other words, modifies this matrix in-place by making all
    /// bits on the diagonal set.
    pub fn reflexive_closure(&mut self) {
        for i in 0..cmp::min(self.row_bits, self.num_rows()) {
            self.set(i, i, true);
        }
    }

    /// Iterates over the matrix's rows in the form of mutable slices.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut BitSlice<B>> {
        fn f<B: BitBlock>(arg: &mut [B]) -> &mut BitSlice<B> {
            // Safety:
            // This is currently the only way to construct a custom DST.
            // We wish the layout of DSTs were defined.
            unsafe { mem::transmute(arg) }
        }
        self.slice.chunks_mut(self.row_size()).map(f::<B>)
    }

    fn row_size(&self) -> usize {
        round_up_to_next(self.row_bits, B::BITS) / B::BITS
    }
}

/// Returns the matrix's row in the form of a mutable slice.
impl<'a, B: BitBlock> Index<usize> for BitSubMatrixMut<'a, B> {
    type Output = BitSlice<B>;

    #[inline]
    fn index(&self, row: usize) -> &Self::Output {
        // Safety:
        // This is currently the only way to construct a custom DST.
        // We wish the layout of DSTs were defined.
        unsafe { mem::transmute(&self.slice[row * self.row_size()..(row + 1) * self.row_size()]) }
    }
}

/// Returns the matrix's row in the form of a mutable slice.
impl<'a, B: BitBlock> IndexMut<usize> for BitSubMatrixMut<'a, B> {
    #[inline]
    fn index_mut(&mut self, row: usize) -> &mut Self::Output {
        let row_size = self.row_size();
        // Safety:
        // This is currently the only way to construct a custom DST.
        // We wish the layout of DSTs were defined.
        unsafe { mem::transmute(&mut self.slice[row * row_size..(row + 1) * row_size]) }
    }
}

/// Returns the matrix's row in the form of a mutable slice.
impl<'a, B: BitBlock> Index<usize> for BitSubMatrix<'a, B> {
    type Output = BitSlice<B>;

    #[inline]
    fn index(&self, row: usize) -> &Self::Output {
        let row_size = self.row_size();
        // Safety:
        // This is currently the only way to construct a custom DST.
        // We wish the layout of DSTs were defined.
        unsafe { mem::transmute(&self.slice[row * row_size..(row + 1) * row_size]) }
    }
}

impl<'a, B: BitBlock> fmt::Debug for BitSubMatrix<'a, B> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        for row in self.iter() {
            for bit in row.iter_bits(self.row_bits) {
                write!(fmt, "{}", if bit { 1 } else { 0 })?;
            }
            writeln!(fmt)?;
        }
        Ok(())
    }
}
