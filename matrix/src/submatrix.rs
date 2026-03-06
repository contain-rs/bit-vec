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
pub struct BitSubMatrix<'a> {
    pub(crate) slice: &'a [Block],
    pub(crate) row_bits: usize,
}

/// Mutable access to a range of matrix's rows.
pub struct BitSubMatrixMut<'a> {
    pub(crate) slice: &'a mut [Block],
    pub(crate) row_bits: usize,
}

impl<'a> BitSubMatrix<'a> {
    /// Returns a new BitSubMatrix.
    pub fn new(slice: &[Block], row_bits: usize) -> BitSubMatrix<'_> {
        BitSubMatrix {
            slice,
            row_bits,
        }
    }

    /// Forms a BitSubMatrix from a pointer and dimensions.
    /// 
    /// # Safety
    /// 
    /// Can construct an ill-formed value, thus the function is marked as
    /// unsafe.
    #[inline]
    pub unsafe fn from_raw_parts(ptr: *const Block, rows: usize, row_bits: usize) -> Self {
        BitSubMatrix {
            slice: slice::from_raw_parts(ptr, round_up_to_next(row_bits, BITS) / BITS * rows),
            row_bits,
        }
    }

    /// Iterates over the matrix's rows in the form of immutable slices.
    pub fn iter(&self) -> impl Iterator<Item = &BitSlice> {
        fn f(arg: &[Block]) -> &BitSlice {
            unsafe { mem::transmute(arg) }
        }
        let row_size = round_up_to_next(self.row_bits, BITS) / BITS;
        self.slice.chunks(row_size).map(f)
    }
}

impl<'a> BitSubMatrixMut<'a> {
    /// Returns a new `BitSubMatrixMut`.
    pub fn new(slice: &mut [Block], row_bits: usize) -> BitSubMatrixMut<'_> {
        BitSubMatrixMut {
            slice,
            row_bits,
        }
    }

    /// Forms a `BitSubMatrix` from a pointer and dimensions.
    /// 
    /// # Safety
    /// 
    /// Can construct an ill-formed value, thus the function is unsafe.
    #[inline]
    pub unsafe fn from_raw_parts(ptr: *mut Block, rows: usize, row_bits: usize) -> Self {
        BitSubMatrixMut {
            slice: slice::from_raw_parts_mut(ptr, round_up_to_next(row_bits, BITS) / BITS * rows),
            row_bits,
        }
    }

    /// Returns the number of rows.
    #[inline]
    fn num_rows(&self) -> usize {
        let row_size = round_up_to_next(self.row_bits, BITS) / BITS;
        self.slice.len().checked_div(row_size).unwrap_or(0)
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
        let row_size_in_bits = round_up_to_next(self.row_bits, BITS);
        let bit = row * row_size_in_bits + col;
        let (block, i) = div_rem(bit, BITS);
        assert!(block < self.slice.len() && col < self.row_bits);
        unsafe {
            let elt = self.slice.get_unchecked_mut(block);
            if enabled {
                *elt |= 1 << i;
            } else {
                *elt &= !(1 << i);
            }
        }
    }

    /// Returns a slice of the matrix's rows.
    pub fn sub_matrix<R: RangeBounds<usize>>(&self, range: R) -> BitSubMatrix<'_> {
        let row_size = round_up_to_next(self.row_bits, BITS) / BITS;
        BitSubMatrix {
            slice: &self.slice[(
                range.start_bound().map(|&s| s * row_size),
                range.end_bound().map(|&e| e * row_size),
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
    pub fn split_at(&self, row: usize) -> (BitSubMatrix<'_>, BitSubMatrix<'_>) {
        (
            self.sub_matrix(0..row),
            self.sub_matrix(row..self.num_rows()),
        )
    }

    /// Given a row's index, returns a slice of all rows above that row, a reference to said row,
    /// and a slice of all rows below.
    #[inline]
    pub fn split_at_mut(&mut self, row: usize) -> (BitSubMatrixMut<'_>, BitSubMatrixMut<'_>) {
        let row_size = round_up_to_next(self.row_bits, BITS) / BITS;
        let (first, second) = self.slice.split_at_mut(row * row_size);
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
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut BitSlice> {
        fn f(arg: &mut [Block]) -> &mut BitSlice {
            unsafe { mem::transmute(arg) }
        }
        let row_size = round_up_to_next(self.row_bits, BITS) / BITS;
        self.slice.chunks_mut(row_size).map(f)
    }
}

/// Returns the matrix's row in the form of a mutable slice.
impl<'a> Index<usize> for BitSubMatrixMut<'a> {
    type Output = BitSlice;

    #[inline]
    fn index(&self, row: usize) -> &BitSlice {
        let row_size = round_up_to_next(self.row_bits, BITS) / BITS;
        unsafe { mem::transmute(&self.slice[row * row_size..(row + 1) * row_size]) }
    }
}

/// Returns the matrix's row in the form of a mutable slice.
impl<'a> IndexMut<usize> for BitSubMatrixMut<'a> {
    #[inline]
    fn index_mut(&mut self, row: usize) -> &mut BitSlice {
        let row_size = round_up_to_next(self.row_bits, BITS) / BITS;
        unsafe { mem::transmute(&mut self.slice[row * row_size..(row + 1) * row_size]) }
    }
}

/// Returns the matrix's row in the form of a mutable slice.
impl<'a> Index<usize> for BitSubMatrix<'a> {
    type Output = BitSlice;

    #[inline]
    fn index(&self, row: usize) -> &BitSlice {
        let row_size = round_up_to_next(self.row_bits, BITS) / BITS;
        unsafe { mem::transmute(&self.slice[row * row_size..(row + 1) * row_size]) }
    }
}

impl<'a> fmt::Debug for BitSubMatrix<'a> {
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
