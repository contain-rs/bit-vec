//! Implements bit matrices.

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

pub mod block;
pub mod matrix;
pub mod row;
pub mod submatrix;
mod util;

pub use matrix::BitMatrix;

/// A value for borrowing through the `Index` trait.
pub static TRUE: bool = true;
/// A value for borrowing through the `Index` trait.
pub static FALSE: bool = false;

pub(crate) mod local_prelude {
    pub use crate::block::{Block, BITS};
    // pub use crate::matrix::BitMatrix;
    pub use crate::row::BitSlice;
    pub use crate::submatrix::{BitSubMatrix, BitSubMatrixMut};
}
