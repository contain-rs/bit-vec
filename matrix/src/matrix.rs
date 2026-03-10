//! Matrix of bits.
//!
//! # Examples
//!
//! Gets a mutable reference to the square bit matrix within this
//! rectangular matrix, then performs a transitive closure.
//!
//! ```rust
//! use bit_matrix::BitMatrix;
//!
//! let mut matrix = BitMatrix::new(7, 5);
//! matrix.set(1, 2, true);
//! matrix.set(2, 3, true);
//! matrix.set(3, 4, true);
//!
//! {
//!     let mut sub_matrix = matrix.sub_matrix_mut(1 .. 6);
//!     sub_matrix.transitive_closure();
//! }
//! assert!(matrix[(1, 4)]);
//!
//! matrix.reflexive_closure();
//! assert!(matrix[(0, 0)]);
//! assert!(matrix[(1, 1)]);
//! assert!(matrix[(2, 2)]);
//! assert!(matrix[(3, 3)]);
//! ```

use core::cmp;
use core::ops::{Index, IndexMut, RangeBounds};

use bit_vec::BitVec;

use super::{FALSE, TRUE};
use crate::local_prelude::*;
use crate::util::round_up_to_next;

/// A matrix of bits.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(
    feature = "miniserde",
    derive(miniserde::Serialize, miniserde::Deserialize)
)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BitMatrix {
    bit_vec: BitVec,
    row_bits: usize,
}

// Matrix

impl BitMatrix {
    /// Create a new BitMatrix with specific numbers of bits in columns and rows.
    pub fn new(rows: usize, row_bits: usize) -> Self {
        BitMatrix {
            bit_vec: BitVec::from_elem(round_up_to_next(row_bits, BITS) * rows, false),
            row_bits,
        }
    }

    /// Returns the number of rows.
    #[inline]
    fn num_rows(&self) -> usize {
        if self.row_bits == 0 {
            0
        } else {
            let row_blocks = round_up_to_next(self.row_bits, BITS) / BITS;
            self.bit_vec.storage().len() / row_blocks
        }
    }

    /// Returns the number of columns.
    #[inline]
    pub fn num_cols(&self) -> usize {
        self.row_bits
    }

    /// Returns the matrix's size as `(rows, columns)`.
    pub fn size(&self) -> (usize, usize) {
        (self.num_rows(), self.row_bits)
    }

    /// Sets the value of a bit.
    ///
    /// # Panics
    ///
    /// Panics if `(row, col)` is out of bounds.
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, enabled: bool) {
        let row_size_in_bits = round_up_to_next(self.row_bits, BITS);
        self.bit_vec.set(row * row_size_in_bits + col, enabled);
    }

    /// Sets the value of all bits.
    #[inline]
    pub fn set_all(&mut self, enabled: bool) {
        self.bit_vec.fill(enabled);
    }

    /// Grows the matrix in-place, adding `num_rows` rows filled with `value`.
    pub fn grow(&mut self, num_rows: usize, value: bool) {
        self.bit_vec
            .grow(round_up_to_next(self.row_bits, BITS) * num_rows, value);
    }

    /// Truncates the matrix.
    pub fn truncate(&mut self, num_rows: usize) {
        self.bit_vec
            .truncate(round_up_to_next(self.row_bits, BITS) * num_rows);
    }

    /// Returns a slice of the matrix's rows.
    #[inline]
    pub fn sub_matrix<R: RangeBounds<usize>>(&self, range: R) -> BitSubMatrix<'_> {
        let row_size = round_up_to_next(self.row_bits, BITS) / BITS;
        BitSubMatrix {
            slice: &self.bit_vec.storage()[(
                range.start_bound().map(|&s| s * row_size),
                range.end_bound().map(|&e| e * row_size),
            )],
            row_bits: self.row_bits,
        }
    }

    /// Returns a slice of the matrix's rows.
    #[inline]
    pub fn sub_matrix_mut<R: RangeBounds<usize>>(&mut self, range: R) -> BitSubMatrixMut<'_> {
        let row_size = self.row_size();
        // Safety:
        //
        unsafe {
            BitSubMatrixMut {
                slice: &mut self.bit_vec.storage_mut()[(
                    range.start_bound().map(|&s| s * row_size),
                    range.end_bound().map(|&e| e * row_size),
                )],
                row_bits: self.row_bits,
            }
        }
    }

    fn row_size(&self) -> usize {
        round_up_to_next(self.row_bits, BITS) / BITS
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
        let (first, second) = unsafe { self.bit_vec.storage_mut().split_at_mut(row * row_size) };
        (
            BitSubMatrixMut::new(first, self.row_bits),
            BitSubMatrixMut::new(second, self.row_bits),
        )
    }

    /// Iterate over bits in the specified row.
    pub fn iter_row(&self, row: usize) -> impl Iterator<Item = bool> + '_ {
        BitSlice::new(&self[row].slice).iter_bits(self.row_bits)
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
        Into::<BitSubMatrixMut>::into(self).transitive_closure();
    }

    /// Determines whether the number of rows equals the number of columns.
    ///
    /// This means the matrix is square.
    pub fn is_square(&self) -> bool {
        self.num_rows() == self.row_bits
    }

    /// Determines whether the matrix is empty.
    pub fn is_empty(&self) -> bool {
        self.size() == (0, 0)
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
}

/// Gains immutable access to the matrix's row in the form of a `BitSlice`.
impl Index<usize> for BitMatrix {
    type Output = BitSlice;

    #[inline]
    fn index(&self, row: usize) -> &BitSlice {
        let row_size = round_up_to_next(self.row_bits, BITS) / BITS;
        BitSlice::new(&self.bit_vec.storage()[row * row_size..(row + 1) * row_size])
    }
}

/// Gains mutable access to the matrix's row in the form of a `BitSlice`.
impl IndexMut<usize> for BitMatrix {
    #[inline]
    fn index_mut(&mut self, row: usize) -> &mut BitSlice {
        let row_size = round_up_to_next(self.row_bits, BITS) / BITS;
        unsafe {
            BitSlice::new_mut(&mut self.bit_vec.storage_mut()[row * row_size..(row + 1) * row_size])
        }
    }
}

/// Returns `true` if a bit is enabled in the matrix, or `false` otherwise.
///
/// The first index in the tuple is row number, and the second is column
/// number.
impl Index<(usize, usize)> for BitMatrix {
    type Output = bool;

    #[inline]
    fn index(&self, (row, col): (usize, usize)) -> &bool {
        let row_size_in_bits = round_up_to_next(self.row_bits, BITS);
        if self.bit_vec.get(row * row_size_in_bits + col).unwrap() {
            &TRUE
        } else {
            &FALSE
        }
    }
}

impl<'a> From<&'a mut BitMatrix> for BitSubMatrixMut<'a> {
    fn from(value: &'a mut BitMatrix) -> Self {
        unsafe { BitSubMatrixMut::new(value.bit_vec.storage_mut(), value.row_bits) }
    }
}

// Tests

#[test]
fn test_empty() {
    let mut matrix = BitMatrix::new(0, 0);
    for _ in 0..3 {
        assert_eq!(matrix.num_rows(), 0);
        assert_eq!(matrix.size(), (0, 0));
        assert!(matrix.is_square());
        assert!(matrix.is_empty());
        matrix.transitive_closure();
    }
}
