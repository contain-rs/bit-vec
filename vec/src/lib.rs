// Copyright 2012-2023 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// FIXME(Gankro): BitVec and BitSet are very tightly coupled. Ideally (for
// maintenance), they should be in separate files/modules, with BitSet only
// using BitVec's public API. This will be hard for performance though, because
// `BitVec` will not want to leak its internal representation while its internal
// representation as `u32`s must be assumed for best performance.

// (1) Be careful, most things can overflow here because the amount of bits in
//     memory can overflow `usize`.
// (2) Make sure that the underlying vector has no excess length:
//     E. g. `nbits == 16`, `storage.len() == 2` would be excess length,
//     because the last word isn't used at all. This is important because some
//     methods rely on it (for *CORRECTNESS*).
// (3) Make sure that the unused bits in the last word are zeroed out, again
//     other methods rely on it for *CORRECTNESS*.
// (4) `BitSet` is tightly coupled with `BitVec`, so any changes you make in
// `BitVec` will need to be reflected in `BitSet`.

//! # Description
//!
//! Dynamic collections implemented with compact bit vectors.
//!
//! # Examples
//!
//! This is a simple example of the [Sieve of Eratosthenes][sieve]
//! which calculates prime numbers up to a given limit.
//!
//! [sieve]: http://en.wikipedia.org/wiki/Sieve_of_Eratosthenes
//!
//! ```
//! use bit_vec::BitVec;
//!
//! let max_prime = 10000;
//!
//! // Store the primes as a BitVec
//! let primes = {
//!     // Assume all numbers are prime to begin, and then we
//!     // cross off non-primes progressively
//!     let mut bv = BitVec::from_elem(max_prime, true);
//!
//!     // Neither 0 nor 1 are prime
//!     bv.set(0, false);
//!     bv.set(1, false);
//!
//!     for i in 2.. 1 + (max_prime as f64).sqrt() as usize {
//!         // if i is a prime
//!         if bv[i] {
//!             // Mark all multiples of i as non-prime (any multiples below i * i
//!             // will have been marked as non-prime previously)
//!             for j in i.. {
//!                 if i * j >= max_prime {
//!                     break;
//!                 }
//!                 bv.set(i * j, false)
//!             }
//!         }
//!     }
//!     bv
//! };
//!
//! // Simple primality tests below our max bound
//! let print_primes = 20;
//! print!("The primes below {} are: ", print_primes);
//! for x in 0..print_primes {
//!     if primes.get(x).unwrap_or(false) {
//!         print!("{} ", x);
//!     }
//! }
//! println!();
//!
//! let num_primes = primes.iter().filter(|x| *x).count();
//! println!("There are {} primes below {}", num_primes, max_prime);
//! assert_eq!(num_primes, 1_229);
//! ```

#![doc(html_root_url = "https://docs.rs/bit-vec/0.9.0/bit_vec/")]
#![no_std]
#![deny(clippy::shadow_reuse)]
#![deny(clippy::shadow_same)]
#![deny(clippy::shadow_unrelated)]
#![warn(clippy::multiple_inherent_impl)]
#![warn(clippy::multiple_crate_versions)]
#![warn(clippy::single_match)]
#![warn(clippy::missing_safety_doc)]
#![allow(type_alias_bounds)]
#![cfg_attr(feature = "allocator_api", feature(allocator_api))]

#[cfg(any(test, feature = "std"))]
#[macro_use]
extern crate std;

#[cfg(feature = "borsh")]
extern crate borsh;
#[cfg(feature = "miniserde")]
extern crate miniserde;
#[cfg(feature = "nanoserde")]
extern crate nanoserde;
#[cfg(feature = "serde")]
extern crate serde;

#[cfg(not(feature = "std"))]
extern crate alloc;

mod block;
mod block_or_store;
mod blocks;
mod blocks_mut;
mod into_iter;
mod iter;
mod smart_mut;
mod store;
mod util;
mod vec;

pub use block::BitBlock;
pub use block_or_store::BitBlockOrStore;
pub use blocks::Blocks;
pub use blocks_mut::BlocksMut;
pub use into_iter::IntoIter;
pub use iter::Iter;
pub use smart_mut::{IterMut, MutBorrowedBit};
pub use store::BitStore;
pub use vec::BitVec;

mod local_prelude {
    #[cfg(not(feature = "std"))]
    pub use alloc::rc::Rc;
    #[cfg(not(feature = "std"))]
    pub use alloc::string::String;
    #[cfg(not(feature = "std"))]
    pub use alloc::vec::Vec;
    #[cfg(not(feature = "std"))]
    pub use alloc::boxed::Box;

    #[cfg(feature = "std")]
    pub use std::rc::Rc;
    #[cfg(feature = "std")]
    pub use std::string::String;
    #[cfg(feature = "std")]
    pub use std::vec::Vec;
    #[cfg(feature = "std")]
    pub use std::boxed::Box;

    pub use core::cell::RefCell;
    pub use core::cmp::Ordering;
    pub use core::fmt::Write;
    pub use core::iter::FromIterator;
    pub use core::{cmp, fmt, hash, iter, mem, ops, slice};

    #[cfg(feature = "nanoserde")]
    pub use nanoserde::{DeBin, DeJson, DeRon, SerBin, SerJson, SerRon};

    pub use crate::block::BitBlock;
    pub use crate::block_or_store::BitBlockOrStore;
    pub use crate::store::BitStore;
    pub(crate) use crate::util::Block;
}
