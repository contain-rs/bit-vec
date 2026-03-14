//! Implements bit matrices.
//!
//! # Examples
//!
//! Gets a mutable reference to the square bit matrix within this
//! rectangular matrix, then performs a transitive closure.
//!
//! ```rust
//! use bit_matrix::BitMatrix;
//!
//! let mut matrix = <BitMatrix>::new(7, 5);
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
//!
//! This simple example calculates the transitive closure of 4x4 bit matrix.
//!
//! ```rust
//! use bit_matrix::BitMatrix;
//!
//! let mut matrix = <BitMatrix>::new(4, 4);
//! let points = &[
//!     (0, 0),
//!     (0, 1),
//!     (0, 3),
//!     (1, 0),
//!     (1, 2),
//!     (2, 0),
//!     (2, 1),
//!     (3, 1),
//!     (3, 3),
//! ];
//! for &(i, j) in points {
//!     matrix.set(i, j, true);
//! }
//! matrix.transitive_closure();
//!
//! let mut expected_matrix = BitMatrix::new(4, 4);
//! for i in 0..4 {
//!     for j in 0..4 {
//!         expected_matrix.set(i, j, true);
//!     }
//! }
//!
//! assert_eq!(matrix, expected_matrix);
//! ```

#![deny(
    missing_docs,
    missing_copy_implementations,
    trivial_casts,
    trivial_numeric_casts,
    unused_import_braces,
    unused_qualifications
)]
#![cfg_attr(test, deny(warnings))]
#![no_std]

mod block;
mod matrix;
mod row;
mod submatrix;
mod util;

pub use matrix::BitMatrix;

/// A value for borrowing through the `Index` trait.
pub static TRUE: bool = true;
/// A value for borrowing through the `Index` trait.
pub static FALSE: bool = false;

pub(crate) mod local_prelude {
    pub(crate) use crate::block::Block;
    pub use crate::row::BitSlice;
    pub use crate::submatrix::{BitSubMatrix, BitSubMatrixMut};
}
