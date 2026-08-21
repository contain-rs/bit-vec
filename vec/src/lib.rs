// Copyright 2012-2026 The Rust Project Developers. See the COPYRIGHT
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

#![doc(html_root_url = "https://docs.rs/bit-vec/0.11.0/bit_vec/")]
#![no_std]
#![deny(clippy::shadow_reuse)]
#![deny(clippy::shadow_same)]
#![deny(clippy::shadow_unrelated)]
#![allow(clippy::multiple_inherent_impl)]
#![warn(clippy::multiple_crate_versions)]
#![warn(clippy::single_match)]
#![warn(clippy::missing_safety_doc)]
// FIXME https://github.com/near/borsh/issues/159
#![allow(clippy::multiple_crate_versions)]

#[cfg(any(test, feature = "std"))]
#[macro_use]
extern crate std;

#[cfg(not(feature = "std"))]
extern crate alloc;

mod block;
mod blocks;
mod blocks_mut;
mod container;
mod iter;
#[cfg(any(feature = "miniserde", feature = "serde", feature = "borsh"))]
mod serde;
mod smart_mut;
mod util;
mod vec;

mod local_prelude {
    pub use crate::block::BitBlock;
    pub use crate::container::BitContainer;
    pub use crate::vec::BitVec;

    #[cfg(all(not(feature = "std"), feature = "borsh"))]
    pub use alloc::borrow::ToOwned;
    #[cfg(all(not(feature = "std"), feature = "miniserde"))]
    pub use alloc::boxed::Box;
    #[cfg(not(feature = "std"))]
    pub use alloc::rc::Rc;
    #[cfg(not(feature = "std"))]
    pub use alloc::string::String;
    #[cfg(not(feature = "std"))]
    pub use alloc::vec::Vec;

    #[cfg(all(feature = "std", feature = "borsh"))]
    pub use std::borrow::ToOwned;
    #[cfg(all(feature = "std", feature = "miniserde"))]
    pub use std::boxed::Box;
    #[cfg(feature = "std")]
    pub use std::rc::Rc;
    #[cfg(feature = "std")]
    pub use std::string::String;
    #[cfg(feature = "std")]
    pub use std::vec::Vec;
}

pub use block::BitBlock;
pub use blocks::Blocks;
pub use container::BitContainer;
pub use iter::Iter;
pub use smart_mut::IterMut;
pub use vec::BitVec;
