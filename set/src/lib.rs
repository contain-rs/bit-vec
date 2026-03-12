// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Description
//!
//! An implementation of a set using a bit vector as an underlying
//! representation for holding unsigned numerical elements.
//!
//! It should also be noted that the amount of storage necessary for holding a
//! set of objects is proportional to the maximum of the objects when viewed
//! as a `usize`.
//!
//! # Examples
//!
//! ```
//! use bit_set::BitSet;
//!
//! // It's a regular set
//! let mut s = BitSet::new();
//! s.insert(0);
//! s.insert(3);
//! s.insert(7);
//!
//! s.remove(7);
//!
//! if !s.contains(7) {
//!     println!("There is no 7");
//! }
//!
//! // Can initialize from a `BitVec`
//! let other = BitSet::from_bytes(&[0b11010000]);
//!
//! s.union_with(&other);
//!
//! // Print 0, 1, 3 in some order
//! for x in s.iter() {
//!     println!("{}", x);
//! }
//!
//! // Can convert back to a `BitVec`
//! let bv = s.into_bit_vec();
//! assert!(bv[3]);
//! ```
#![doc(html_root_url = "https://docs.rs/bit-set/0.8.0")]
#![deny(clippy::shadow_reuse)]
#![deny(clippy::shadow_same)]
#![deny(clippy::shadow_unrelated)]
#![no_std]

#[cfg(any(test, feature = "std"))]
extern crate std;

#[cfg(feature = "nanoserde")]
extern crate alloc;

mod set;
mod util;
mod iter;

pub mod local_prelude {
    pub use bit_vec::{BitBlock, BitBlockOrStore, BitStore, BitVec, Blocks};
    pub use core::cmp::Ordering;
    pub use core::{hash, fmt, cmp};
    pub use core::iter::{self, Chain, Enumerate, FromIterator, Repeat, Skip, Take};
}

pub use set::BitSet;
pub use bit_vec::{BitStore, BitBlockOrStore};
