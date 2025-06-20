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

#![doc(html_root_url = "https://docs.rs/bit-vec/0.8.0")]
#![no_std]
#![deny(clippy::shadow_reuse)]
#![deny(clippy::shadow_same)]
#![deny(clippy::shadow_unrelated)]
#![warn(clippy::multiple_inherent_impl)]
#![warn(clippy::multiple_crate_versions)]
#![warn(clippy::single_match)]
#![warn(clippy::missing_safety_doc)]

#[cfg(any(test, feature = "std"))]
#[macro_use]
extern crate std;
#[cfg(feature = "std")]
use std::rc::Rc;
#[cfg(feature = "std")]
use std::string::String;
#[cfg(feature = "std")]
use std::vec::Vec;

#[cfg(feature = "serde")]
extern crate serde;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "borsh")]
extern crate borsh;
#[cfg(feature = "miniserde")]
extern crate miniserde;
#[cfg(feature = "nanoserde")]
extern crate nanoserde;
#[cfg(feature = "nanoserde")]
use nanoserde::{DeBin, DeJson, DeRon, SerBin, SerJson, SerRon};

#[cfg(not(feature = "std"))]
#[macro_use]
extern crate alloc;
#[cfg(not(feature = "std"))]
use alloc::rc::Rc;
#[cfg(not(feature = "std"))]
use alloc::string::String;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use core::cell::RefCell;
use core::cmp;
use core::cmp::Ordering;
use core::fmt::{self, Write};
use core::hash;
use core::iter::repeat;
use core::iter::FromIterator;
use core::mem;
use core::ops::*;
use core::slice;

type MutBlocks<'a, B> = slice::IterMut<'a, B>;

/// Abstracts over a pile of bits (basically unsigned primitives)
pub trait BitBlock:
    Copy
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Shl<usize, Output = Self>
    + Shr<usize, Output = Self>
    + Not<Output = Self>
    + BitAnd<Self, Output = Self>
    + BitOr<Self, Output = Self>
    + BitXor<Self, Output = Self>
    + Rem<Self, Output = Self>
    + Eq
    + Ord
    + hash::Hash
{
    /// How many bits it has
    fn bits() -> usize;
    /// How many bytes it has
    #[inline]
    fn bytes() -> usize {
        Self::bits() / 8
    }
    /// Convert a byte into this type (lowest-order bits set)
    fn from_byte(byte: u8) -> Self;
    /// Count the number of 1's in the bitwise repr
    fn count_ones(self) -> usize;
    /// Count the number of 0's in the bitwise repr
    fn count_zeros(self) -> usize {
        Self::bits() - self.count_ones()
    }
    /// Get `0`
    fn zero() -> Self;
    /// Get `1`
    fn one() -> Self;
}

macro_rules! bit_block_impl {
    ($(($t: ident, $size: expr)),*) => ($(
        impl BitBlock for $t {
            #[inline]
            fn bits() -> usize { $size }
            #[inline]
            fn from_byte(byte: u8) -> Self { $t::from(byte) }
            #[inline]
            fn count_ones(self) -> usize { self.count_ones() as usize }
            #[inline]
            fn count_zeros(self) -> usize { self.count_zeros() as usize }
            #[inline]
            fn one() -> Self { 1 }
            #[inline]
            fn zero() -> Self { 0 }
        }
    )*)
}

bit_block_impl! {
    (u8, 8),
    (u16, 16),
    (u32, 32),
    (u64, 64),
    (usize, core::mem::size_of::<usize>() * 8)
}

fn reverse_bits(byte: u8) -> u8 {
    let mut result = 0;
    for i in 0..u8::bits() {
        result |= ((byte >> i) & 1) << (u8::bits() - 1 - i);
    }
    result
}

static TRUE: bool = true;
static FALSE: bool = false;

/// The bitvector type.
///
/// # Examples
///
/// ```
/// use bit_vec::BitVec;
///
/// let mut bv = BitVec::from_elem(10, false);
///
/// // insert all primes less than 10
/// bv.set(2, true);
/// bv.set(3, true);
/// bv.set(5, true);
/// bv.set(7, true);
/// println!("{:?}", bv);
/// println!("total bits set to true: {}", bv.iter().filter(|x| *x).count());
///
/// // flip all values in bitvector, producing non-primes less than 10
/// bv.negate();
/// println!("{:?}", bv);
/// println!("total bits set to true: {}", bv.iter().filter(|x| *x).count());
///
/// // reset bitvector to empty
/// bv.clear();
/// println!("{:?}", bv);
/// println!("total bits set to true: {}", bv.iter().filter(|x| *x).count());
/// ```
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "borsh",
    derive(borsh::BorshDeserialize, borsh::BorshSerialize)
)]
#[cfg_attr(
    feature = "miniserde",
    derive(miniserde::Deserialize, miniserde::Serialize)
)]
#[cfg_attr(
    feature = "nanoserde",
    derive(DeBin, DeJson, DeRon, SerBin, SerJson, SerRon)
)]
pub struct BitVec<B = u32> {
    /// Internal representation of the bit vector
    storage: Vec<B>,
    /// The number of valid bits in the internal representation
    nbits: usize,
}

// FIXME(Gankro): NopeNopeNopeNopeNope (wait for IndexGet to be a thing)
impl<B: BitBlock> Index<usize> for BitVec<B> {
    type Output = bool;

    #[inline]
    fn index(&self, i: usize) -> &bool {
        if self.get(i).expect("index out of bounds") {
            &TRUE
        } else {
            &FALSE
        }
    }
}

/// Computes how many blocks are needed to store that many bits
fn blocks_for_bits<B: BitBlock>(bits: usize) -> usize {
    // If we want 17 bits, dividing by 32 will produce 0. So we add 1 to make sure we
    // reserve enough. But if we want exactly a multiple of 32, this will actually allocate
    // one too many. So we need to check if that's the case. We can do that by computing if
    // bitwise AND by `32 - 1` is 0. But LLVM should be able to optimize the semantically
    // superior modulo operator on a power of two to this.
    //
    // Note that we can technically avoid this branch with the expression
    // `(nbits + U32_BITS - 1) / 32::BITS`, but if nbits is almost usize::MAX this will overflow.
    if bits % B::bits() == 0 {
        bits / B::bits()
    } else {
        bits / B::bits() + 1
    }
}

/// Computes the bitmask for the final word of the vector
fn mask_for_bits<B: BitBlock>(bits: usize) -> B {
    // Note especially that a perfect multiple of U32_BITS should mask all 1s.
    (!B::zero()) >> ((B::bits() - bits % B::bits()) % B::bits())
}

type B = u32;

impl BitVec<u32> {
    /// Creates an empty `BitVec`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    /// let mut bv = BitVec::new();
    /// ```
    #[inline]
    pub fn new() -> Self {
        Default::default()
    }
}

impl BitVec<u32> {
    /// Creates a `BitVec` that holds `nbits` elements, setting each element
    /// to `bit`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let mut bv = BitVec::from_elem(10, false);
    /// assert_eq!(bv.len(), 10);
    /// for x in bv.iter() {
    ///     assert_eq!(x, false);
    /// }
    /// ```
    #[inline]
    pub fn from_elem(len: usize, bit: bool) -> Self {
        let nblocks = blocks_for_bits::<B>(len);
        let mut bit_vec = BitVec {
            storage: vec![if bit { !B::zero() } else { B::zero() }; nblocks],
            nbits: len,
        };
        bit_vec.fix_last_block();
        bit_vec
    }

    /// Constructs a new, empty `BitVec` with the specified capacity.
    ///
    /// The bitvector will be able to hold at least `capacity` bits without
    /// reallocating. If `capacity` is 0, it will not allocate.
    ///
    /// It is important to note that this function does not specify the
    /// *length* of the returned bitvector, but only the *capacity*.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        BitVec {
            storage: Vec::with_capacity(blocks_for_bits::<B>(capacity)),
            nbits: 0,
        }
    }

    /// Transforms a byte-vector into a `BitVec`. Each byte becomes eight bits,
    /// with the most significant bits of each byte coming first. Each
    /// bit becomes `true` if equal to 1 or `false` if equal to 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let bv = BitVec::from_bytes(&[0b10100000, 0b00010010]);
    /// assert!(bv.eq_vec(&[true, false, true, false,
    ///                     false, false, false, false,
    ///                     false, false, false, true,
    ///                     false, false, true, false]));
    /// ```
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let len = bytes
            .len()
            .checked_mul(u8::bits())
            .expect("capacity overflow");
        let mut bit_vec = BitVec::with_capacity(len);
        let complete_words = bytes.len() / B::bytes();
        let extra_bytes = bytes.len() % B::bytes();

        bit_vec.nbits = len;

        for i in 0..complete_words {
            let mut accumulator = B::zero();
            for idx in 0..B::bytes() {
                accumulator |= B::from_byte(reverse_bits(bytes[i * B::bytes() + idx])) << (idx * 8)
            }
            bit_vec.storage.push(accumulator);
        }

        if extra_bytes > 0 {
            let mut last_word = B::zero();
            for (i, &byte) in bytes[complete_words * B::bytes()..].iter().enumerate() {
                last_word |= B::from_byte(reverse_bits(byte)) << (i * 8);
            }
            bit_vec.storage.push(last_word);
        }

        bit_vec
    }

    /// Creates a `BitVec` of the specified length where the value at each index
    /// is `f(index)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let bv = BitVec::from_fn(5, |i| { i % 2 == 0 });
    /// assert!(bv.eq_vec(&[true, false, true, false, true]));
    /// ```
    #[inline]
    pub fn from_fn<F>(len: usize, mut f: F) -> Self
    where
        F: FnMut(usize) -> bool,
    {
        let mut bit_vec = BitVec::from_elem(len, false);
        for i in 0..len {
            bit_vec.set(i, f(i));
        }
        bit_vec
    }
}

impl<B: BitBlock> BitVec<B> {
    /// Applies the given operation to the blocks of self and other, and sets
    /// self to be the result. This relies on the caller not to corrupt the
    /// last word.
    #[inline]
    fn process<F>(&mut self, other: &BitVec<B>, mut op: F) -> bool
    where
        F: FnMut(B, B) -> B,
    {
        assert_eq!(self.len(), other.len());
        debug_assert_eq!(self.storage.len(), other.storage.len());
        let mut changed_bits = B::zero();
        for (a, b) in self.blocks_mut().zip(other.blocks()) {
            let w = op(*a, b);
            changed_bits = changed_bits | (*a ^ w);
            *a = w;
        }
        changed_bits != B::zero()
    }

    /// Iterator over mutable refs to the underlying blocks of data.
    #[inline]
    fn blocks_mut(&mut self) -> MutBlocks<B> {
        // (2)
        self.storage.iter_mut()
    }

    /// Iterator over the underlying blocks of data
    #[inline]
    pub fn blocks(&self) -> Blocks<B> {
        // (2)
        Blocks {
            iter: self.storage.iter(),
        }
    }

    /// Exposes the raw block storage of this `BitVec`.
    ///
    /// Only really intended for `BitSet`.
    #[inline]
    pub fn storage(&self) -> &[B] {
        &self.storage
    }

    /// Exposes the raw block storage of this `BitVec`.
    ///
    /// # Safety
    ///
    /// Can probably cause unsafety. Only really intended for `BitSet`.
    #[inline]
    pub unsafe fn storage_mut(&mut self) -> &mut Vec<B> {
        &mut self.storage
    }

    /// Helper for procedures involving spare space in the last block.
    #[inline]
    fn last_block_with_mask(&self) -> Option<(B, B)> {
        let extra_bits = self.len() % B::bits();
        if extra_bits > 0 {
            let mask = (B::one() << extra_bits) - B::one();
            let storage_len = self.storage.len();
            Some((self.storage[storage_len - 1], mask))
        } else {
            None
        }
    }

    /// Helper for procedures involving spare space in the last block.
    #[inline]
    fn last_block_mut_with_mask(&mut self) -> Option<(&mut B, B)> {
        let extra_bits = self.len() % B::bits();
        if extra_bits > 0 {
            let mask = (B::one() << extra_bits) - B::one();
            let storage_len = self.storage.len();
            Some((&mut self.storage[storage_len - 1], mask))
        } else {
            None
        }
    }

    /// An operation might screw up the unused bits in the last block of the
    /// `BitVec`. As per (3), it's assumed to be all 0s. This method fixes it up.
    fn fix_last_block(&mut self) {
        if let Some((last_block, used_bits)) = self.last_block_mut_with_mask() {
            *last_block = *last_block & used_bits;
        }
    }

    /// Operations such as change detection for xnor, nor and nand are easiest
    /// to implement when unused bits are all set to 1s.
    fn fix_last_block_with_ones(&mut self) {
        if let Some((last_block, used_bits)) = self.last_block_mut_with_mask() {
            *last_block = *last_block | !used_bits;
        }
    }

    /// Check whether last block's invariant is fine.
    fn is_last_block_fixed(&self) -> bool {
        if let Some((last_block, used_bits)) = self.last_block_with_mask() {
            last_block & !used_bits == B::zero()
        } else {
            true
        }
    }

    /// Ensure the invariant for the last block.
    ///
    /// An operation might screw up the unused bits in the last block of the
    /// `BitVec`.
    ///
    /// This method fails in case the last block is not fixed. The check
    /// is skipped outside testing.
    #[inline]
    fn ensure_invariant(&self) {
        if cfg!(test) {
            debug_assert!(self.is_last_block_fixed());
        }
    }

    /// Retrieves the value at index `i`, or `None` if the index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let bv = BitVec::from_bytes(&[0b01100000]);
    /// assert_eq!(bv.get(0), Some(false));
    /// assert_eq!(bv.get(1), Some(true));
    /// assert_eq!(bv.get(100), None);
    ///
    /// // Can also use array indexing
    /// assert_eq!(bv[1], true);
    /// ```
    #[inline]
    pub fn get(&self, i: usize) -> Option<bool> {
        self.ensure_invariant();
        if i >= self.nbits {
            return None;
        }
        let w = i / B::bits();
        let b = i % B::bits();
        self.storage
            .get(w)
            .map(|&block| (block & (B::one() << b)) != B::zero())
    }

    /// Retrieves the value at index `i`, without doing bounds checking.
    ///
    /// For a safe alternative, see `get`.
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is undefined behavior
    /// even if the resulting reference is not used.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let bv = BitVec::from_bytes(&[0b01100000]);
    /// unsafe {
    ///     assert_eq!(bv.get_unchecked(0), false);
    ///     assert_eq!(bv.get_unchecked(1), true);
    /// }
    /// ```
    #[inline]
    pub unsafe fn get_unchecked(&self, i: usize) -> bool {
        self.ensure_invariant();
        let w = i / B::bits();
        let b = i % B::bits();
        let block = *self.storage.get_unchecked(w);
        block & (B::one() << b) != B::zero()
    }

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
    pub fn get_mut(&mut self, index: usize) -> Option<MutBorrowedBit<B>> {
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
    pub unsafe fn get_unchecked_mut(&mut self, index: usize) -> MutBorrowedBit<B> {
        let value = self.get_unchecked(index);
        MutBorrowedBit {
            #[cfg(debug_assertions)]
            old_value: value,
            new_value: value,
            vec: Rc::new(RefCell::new(self)),
            index,
        }
    }

    /// Sets the value of a bit at an index `i`.
    ///
    /// # Panics
    ///
    /// Panics if `i` is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let mut bv = BitVec::from_elem(5, false);
    /// bv.set(3, true);
    /// assert_eq!(bv[3], true);
    /// ```
    #[inline]
    pub fn set(&mut self, i: usize, x: bool) {
        self.ensure_invariant();
        assert!(
            i < self.nbits,
            "index out of bounds: {:?} >= {:?}",
            i,
            self.nbits
        );
        let w = i / B::bits();
        let b = i % B::bits();
        let flag = B::one() << b;
        let val = if x {
            self.storage[w] | flag
        } else {
            self.storage[w] & !flag
        };
        self.storage[w] = val;
    }

    /// Sets all bits to 1.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let before = 0b01100000;
    /// let after  = 0b11111111;
    ///
    /// let mut bv = BitVec::from_bytes(&[before]);
    /// bv.set_all();
    /// assert_eq!(bv, BitVec::from_bytes(&[after]));
    /// ```
    #[inline]
    pub fn set_all(&mut self) {
        self.ensure_invariant();
        for w in &mut self.storage {
            *w = !B::zero();
        }
        self.fix_last_block();
    }

    /// Flips all bits.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let before = 0b01100000;
    /// let after  = 0b10011111;
    ///
    /// let mut bv = BitVec::from_bytes(&[before]);
    /// bv.negate();
    /// assert_eq!(bv, BitVec::from_bytes(&[after]));
    /// ```
    #[inline]
    pub fn negate(&mut self) {
        self.ensure_invariant();
        for w in &mut self.storage {
            *w = !*w;
        }
        self.fix_last_block();
    }

    /// Calculates the union of two bitvectors. This acts like the bitwise `or`
    /// function.
    ///
    /// Sets `self` to the union of `self` and `other`. Both bitvectors must be
    /// the same length. Returns `true` if `self` changed.
    ///
    /// # Panics
    ///
    /// Panics if the bitvectors are of different lengths.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let a   = 0b01100100;
    /// let b   = 0b01011010;
    /// let res = 0b01111110;
    ///
    /// let mut a = BitVec::from_bytes(&[a]);
    /// let b = BitVec::from_bytes(&[b]);
    ///
    /// assert!(a.union(&b));
    /// assert_eq!(a, BitVec::from_bytes(&[res]));
    /// ```
    #[deprecated(since = "0.7.0", note = "Please use the 'or' function instead")]
    #[inline]
    pub fn union(&mut self, other: &Self) -> bool {
        self.or(other)
    }

    /// Calculates the intersection of two bitvectors. This acts like the
    /// bitwise `and` function.
    ///
    /// Sets `self` to the intersection of `self` and `other`. Both bitvectors
    /// must be the same length. Returns `true` if `self` changed.
    ///
    /// # Panics
    ///
    /// Panics if the bitvectors are of different lengths.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let a   = 0b01100100;
    /// let b   = 0b01011010;
    /// let res = 0b01000000;
    ///
    /// let mut a = BitVec::from_bytes(&[a]);
    /// let b = BitVec::from_bytes(&[b]);
    ///
    /// assert!(a.intersect(&b));
    /// assert_eq!(a, BitVec::from_bytes(&[res]));
    /// ```
    #[deprecated(since = "0.7.0", note = "Please use the 'and' function instead")]
    #[inline]
    pub fn intersect(&mut self, other: &Self) -> bool {
        self.and(other)
    }

    /// Calculates the bitwise `or` of two bitvectors.
    ///
    /// Sets `self` to the union of `self` and `other`. Both bitvectors must be
    /// the same length. Returns `true` if `self` changed.
    ///
    /// # Panics
    ///
    /// Panics if the bitvectors are of different lengths.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let a   = 0b01100100;
    /// let b   = 0b01011010;
    /// let res = 0b01111110;
    ///
    /// let mut a = BitVec::from_bytes(&[a]);
    /// let b = BitVec::from_bytes(&[b]);
    ///
    /// assert!(a.or(&b));
    /// assert_eq!(a, BitVec::from_bytes(&[res]));
    /// ```
    #[inline]
    pub fn or(&mut self, other: &Self) -> bool {
        self.ensure_invariant();
        debug_assert!(other.is_last_block_fixed());
        self.process(other, |w1, w2| (w1 | w2))
    }

    /// Calculates the bitwise `and` of two bitvectors.
    ///
    /// Sets `self` to the intersection of `self` and `other`. Both bitvectors
    /// must be the same length. Returns `true` if `self` changed.
    ///
    /// # Panics
    ///
    /// Panics if the bitvectors are of different lengths.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let a   = 0b01100100;
    /// let b   = 0b01011010;
    /// let res = 0b01000000;
    ///
    /// let mut a = BitVec::from_bytes(&[a]);
    /// let b = BitVec::from_bytes(&[b]);
    ///
    /// assert!(a.and(&b));
    /// assert_eq!(a, BitVec::from_bytes(&[res]));
    /// ```
    #[inline]
    pub fn and(&mut self, other: &Self) -> bool {
        self.ensure_invariant();
        debug_assert!(other.is_last_block_fixed());
        self.process(other, |w1, w2| (w1 & w2))
    }

    /// Calculates the difference between two bitvectors.
    ///
    /// Sets each element of `self` to the value of that element minus the
    /// element of `other` at the same index. Both bitvectors must be the same
    /// length. Returns `true` if `self` changed.
    ///
    /// # Panics
    ///
    /// Panics if the bitvectors are of different length.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let a   = 0b01100100;
    /// let b   = 0b01011010;
    /// let a_b = 0b00100100; // a - b
    /// let b_a = 0b00011010; // b - a
    ///
    /// let mut bva = BitVec::from_bytes(&[a]);
    /// let bvb = BitVec::from_bytes(&[b]);
    ///
    /// assert!(bva.difference(&bvb));
    /// assert_eq!(bva, BitVec::from_bytes(&[a_b]));
    ///
    /// let bva = BitVec::from_bytes(&[a]);
    /// let mut bvb = BitVec::from_bytes(&[b]);
    ///
    /// assert!(bvb.difference(&bva));
    /// assert_eq!(bvb, BitVec::from_bytes(&[b_a]));
    /// ```
    #[inline]
    pub fn difference(&mut self, other: &Self) -> bool {
        self.ensure_invariant();
        debug_assert!(other.is_last_block_fixed());
        self.process(other, |w1, w2| (w1 & !w2))
    }

    /// Calculates the xor of two bitvectors.
    ///
    /// Sets `self` to the xor of `self` and `other`. Both bitvectors must be
    /// the same length. Returns `true` if `self` changed.
    ///
    /// # Panics
    ///
    /// Panics if the bitvectors are of different length.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let a   = 0b01100110;
    /// let b   = 0b01010100;
    /// let res = 0b00110010;
    ///
    /// let mut a = BitVec::from_bytes(&[a]);
    /// let b = BitVec::from_bytes(&[b]);
    ///
    /// assert!(a.xor(&b));
    /// assert_eq!(a, BitVec::from_bytes(&[res]));
    /// ```
    #[inline]
    pub fn xor(&mut self, other: &Self) -> bool {
        self.ensure_invariant();
        debug_assert!(other.is_last_block_fixed());
        self.process(other, |w1, w2| (w1 ^ w2))
    }

    /// Calculates the nand of two bitvectors.
    ///
    /// Sets `self` to the nand of `self` and `other`. Both bitvectors must be
    /// the same length. Returns `true` if `self` changed.
    ///
    /// # Panics
    ///
    /// Panics if the bitvectors are of different length.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let a   = 0b01100110;
    /// let b   = 0b01010100;
    /// let res = 0b10111011;
    ///
    /// let mut a = BitVec::from_bytes(&[a]);
    /// let b = BitVec::from_bytes(&[b]);
    ///
    /// assert!(a.nand(&b));
    /// assert_eq!(a, BitVec::from_bytes(&[res]));
    /// ```
    #[inline]
    pub fn nand(&mut self, other: &Self) -> bool {
        self.ensure_invariant();
        debug_assert!(other.is_last_block_fixed());
        self.fix_last_block_with_ones();
        let result = self.process(other, |w1, w2| !(w1 & w2));
        self.fix_last_block();
        result
    }

    /// Calculates the nor of two bitvectors.
    ///
    /// Sets `self` to the nor of `self` and `other`. Both bitvectors must be
    /// the same length. Returns `true` if `self` changed.
    ///
    /// # Panics
    ///
    /// Panics if the bitvectors are of different length.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let a   = 0b01100110;
    /// let b   = 0b01010100;
    /// let res = 0b10001001;
    ///
    /// let mut a = BitVec::from_bytes(&[a]);
    /// let b = BitVec::from_bytes(&[b]);
    ///
    /// assert!(a.nor(&b));
    /// assert_eq!(a, BitVec::from_bytes(&[res]));
    /// ```
    #[inline]
    pub fn nor(&mut self, other: &Self) -> bool {
        self.ensure_invariant();
        debug_assert!(other.is_last_block_fixed());
        self.fix_last_block_with_ones();
        let result = self.process(other, |w1, w2| !(w1 | w2));
        self.fix_last_block();
        result
    }

    /// Calculates the xnor of two bitvectors.
    ///
    /// Sets `self` to the xnor of `self` and `other`. Both bitvectors must be
    /// the same length. Returns `true` if `self` changed.
    ///
    /// # Panics
    ///
    /// Panics if the bitvectors are of different length.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let a   = 0b01100110;
    /// let b   = 0b01010100;
    /// let res = 0b11001101;
    ///
    /// let mut a = BitVec::from_bytes(&[a]);
    /// let b = BitVec::from_bytes(&[b]);
    ///
    /// assert!(a.xnor(&b));
    /// assert_eq!(a, BitVec::from_bytes(&[res]));
    /// ```
    #[inline]
    pub fn xnor(&mut self, other: &Self) -> bool {
        self.ensure_invariant();
        debug_assert!(other.is_last_block_fixed());
        self.fix_last_block_with_ones();
        let result = self.process(other, |w1, w2| !(w1 ^ w2));
        self.fix_last_block();
        result
    }

    /// Returns `true` if all bits are 1.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let mut bv = BitVec::from_elem(5, true);
    /// assert_eq!(bv.all(), true);
    ///
    /// bv.set(1, false);
    /// assert_eq!(bv.all(), false);
    /// ```
    #[inline]
    pub fn all(&self) -> bool {
        self.ensure_invariant();
        let mut last_word = !B::zero();
        // Check that every block but the last is all-ones...
        self.blocks().all(|elem| {
            let tmp = last_word;
            last_word = elem;
            tmp == !B::zero()
            // and then check the last one has enough ones
        }) && (last_word == mask_for_bits(self.nbits))
    }

    /// Returns the number of ones in the binary representation.
    ///
    /// Also known as the
    /// [Hamming weight](https://en.wikipedia.org/wiki/Hamming_weight).
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let mut bv = BitVec::from_elem(100, true);
    /// assert_eq!(bv.count_ones(), 100);
    ///
    /// bv.set(50, false);
    /// assert_eq!(bv.count_ones(), 99);
    /// ```
    #[inline]
    pub fn count_ones(&self) -> u64 {
        self.ensure_invariant();
        // Add the number of ones of each block.
        self.blocks().map(|elem| elem.count_ones() as u64).sum()
    }

    /// Returns the number of zeros in the binary representation.
    ///
    /// Also known as the opposite of
    /// [Hamming weight](https://en.wikipedia.org/wiki/Hamming_weight).
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let mut bv = BitVec::from_elem(100, false);
    /// assert_eq!(bv.count_zeros(), 100);
    ///
    /// bv.set(50, true);
    /// assert_eq!(bv.count_zeros(), 99);
    /// ```
    #[inline]
    pub fn count_zeros(&self) -> u64 {
        self.ensure_invariant();
        // Add the number of zeros of each block.
        let extra_zeros = (B::bits() - (self.len() % B::bits())) % B::bits();
        self.blocks()
            .map(|elem| elem.count_zeros() as u64)
            .sum::<u64>()
            - extra_zeros as u64
    }

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
    pub fn iter(&self) -> Iter<B> {
        self.ensure_invariant();
        Iter {
            bit_vec: self,
            range: 0..self.nbits,
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
    pub fn iter_mut(&mut self) -> IterMut<B> {
        self.ensure_invariant();
        let nbits = self.nbits;
        IterMut {
            vec: Rc::new(RefCell::new(self)),
            range: 0..nbits,
        }
    }

    /// Moves all bits from `other` into `Self`, leaving `other` empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let mut a = BitVec::from_bytes(&[0b10000000]);
    /// let mut b = BitVec::from_bytes(&[0b01100001]);
    ///
    /// a.append(&mut b);
    ///
    /// assert_eq!(a.len(), 16);
    /// assert_eq!(b.len(), 0);
    /// assert!(a.eq_vec(&[true, false, false, false, false, false, false, false,
    ///                    false, true, true, false, false, false, false, true]));
    /// ```
    pub fn append(&mut self, other: &mut Self) {
        self.ensure_invariant();
        debug_assert!(other.is_last_block_fixed());

        let b = self.len() % B::bits();
        let o = other.len() % B::bits();
        let will_overflow = (b + o > B::bits()) || (o == 0 && b != 0);

        self.nbits += other.len();
        other.nbits = 0;

        if b == 0 {
            self.storage.append(&mut other.storage);
        } else {
            self.storage.reserve(other.storage.len());

            for block in other.storage.drain(..) {
                {
                    let last = self.storage.last_mut().unwrap();
                    *last = *last | (block << b);
                }
                self.storage.push(block >> (B::bits() - b));
            }

            // Remove additional block if the last shift did not overflow
            if !will_overflow {
                self.storage.pop();
            }
        }
    }

    /// Splits the `BitVec` into two at the given bit,
    /// retaining the first half in-place and returning the second one.
    ///
    /// # Panics
    ///
    /// Panics if `at` is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    /// let mut a = BitVec::new();
    /// a.push(true);
    /// a.push(false);
    /// a.push(false);
    /// a.push(true);
    ///
    /// let b = a.split_off(2);
    ///
    /// assert_eq!(a.len(), 2);
    /// assert_eq!(b.len(), 2);
    /// assert!(a.eq_vec(&[true, false]));
    /// assert!(b.eq_vec(&[false, true]));
    /// ```
    pub fn split_off(&mut self, at: usize) -> Self {
        self.ensure_invariant();
        assert!(at <= self.len(), "`at` out of bounds");

        let mut other = BitVec::<B>::default();

        if at == 0 {
            mem::swap(self, &mut other);
            return other;
        } else if at == self.len() {
            return other;
        }

        let w = at / B::bits();
        let b = at % B::bits();
        other.nbits = self.nbits - at;
        self.nbits = at;
        if b == 0 {
            // Split at block boundary
            other.storage = self.storage.split_off(w);
        } else {
            other.storage.reserve(self.storage.len() - w);

            {
                let mut iter = self.storage[w..].iter();
                let mut last = *iter.next().unwrap();
                for &cur in iter {
                    other.storage.push((last >> b) | (cur << (B::bits() - b)));
                    last = cur;
                }
                other.storage.push(last >> b);
            }

            self.storage.truncate(w + 1);
            self.fix_last_block();
        }

        other
    }

    /// Returns `true` if all bits are 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let mut bv = BitVec::from_elem(10, false);
    /// assert_eq!(bv.none(), true);
    ///
    /// bv.set(3, true);
    /// assert_eq!(bv.none(), false);
    /// ```
    #[inline]
    pub fn none(&self) -> bool {
        self.blocks().all(|w| w == B::zero())
    }

    /// Returns `true` if any bit is 1.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let mut bv = BitVec::from_elem(10, false);
    /// assert_eq!(bv.any(), false);
    ///
    /// bv.set(3, true);
    /// assert_eq!(bv.any(), true);
    /// ```
    #[inline]
    pub fn any(&self) -> bool {
        !self.none()
    }

    /// Organises the bits into bytes, such that the first bit in the
    /// `BitVec` becomes the high-order bit of the first byte. If the
    /// size of the `BitVec` is not a multiple of eight then trailing bits
    /// will be filled-in with `false`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let mut bv = BitVec::from_elem(3, true);
    /// bv.set(1, false);
    ///
    /// assert_eq!(bv.to_bytes(), [0b10100000]);
    ///
    /// let mut bv = BitVec::from_elem(9, false);
    /// bv.set(2, true);
    /// bv.set(8, true);
    ///
    /// assert_eq!(bv.to_bytes(), [0b00100000, 0b10000000]);
    /// ```
    pub fn to_bytes(&self) -> Vec<u8> {
        self.ensure_invariant();
        // Oh lord, we're mapping this to bytes bit-by-bit!
        fn bit<B: BitBlock>(bit_vec: &BitVec<B>, byte: usize, bit: usize) -> u8 {
            let offset = byte * 8 + bit;
            if offset >= bit_vec.nbits {
                0
            } else {
                (bit_vec[offset] as u8) << (7 - bit)
            }
        }

        let len = self.nbits / 8 + if self.nbits % 8 == 0 { 0 } else { 1 };
        (0..len)
            .map(|i| {
                bit(self, i, 0)
                    | bit(self, i, 1)
                    | bit(self, i, 2)
                    | bit(self, i, 3)
                    | bit(self, i, 4)
                    | bit(self, i, 5)
                    | bit(self, i, 6)
                    | bit(self, i, 7)
            })
            .collect()
    }

    /// Compares a `BitVec` to a slice of `bool`s.
    /// Both the `BitVec` and slice must have the same length.
    ///
    /// # Panics
    ///
    /// Panics if the `BitVec` and slice are of different length.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let bv = BitVec::from_bytes(&[0b10100000]);
    ///
    /// assert!(bv.eq_vec(&[true, false, true, false,
    ///                     false, false, false, false]));
    /// ```
    #[inline]
    pub fn eq_vec(&self, v: &[bool]) -> bool {
        assert_eq!(self.nbits, v.len());
        self.iter().zip(v.iter().cloned()).all(|(b1, b2)| b1 == b2)
    }

    /// Shortens a `BitVec`, dropping excess elements.
    ///
    /// If `len` is greater than the vector's current length, this has no
    /// effect.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let mut bv = BitVec::from_bytes(&[0b01001011]);
    /// bv.truncate(2);
    /// assert!(bv.eq_vec(&[false, true]));
    /// ```
    #[inline]
    pub fn truncate(&mut self, len: usize) {
        self.ensure_invariant();
        if len < self.len() {
            self.nbits = len;
            // This fixes (2).
            self.storage.truncate(blocks_for_bits::<B>(len));
            self.fix_last_block();
        }
    }

    /// Reserves capacity for at least `additional` more bits to be inserted in the given
    /// `BitVec`. The collection may reserve more space to avoid frequent reallocations.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity overflows `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let mut bv = BitVec::from_elem(3, false);
    /// bv.reserve(10);
    /// assert_eq!(bv.len(), 3);
    /// assert!(bv.capacity() >= 13);
    /// ```
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        let desired_cap = self
            .len()
            .checked_add(additional)
            .expect("capacity overflow");
        let storage_len = self.storage.len();
        if desired_cap > self.capacity() {
            self.storage
                .reserve(blocks_for_bits::<B>(desired_cap) - storage_len);
        }
    }

    /// Reserves the minimum capacity for exactly `additional` more bits to be inserted in the
    /// given `BitVec`. Does nothing if the capacity is already sufficient.
    ///
    /// Note that the allocator may give the collection more space than it requests. Therefore
    /// capacity can not be relied upon to be precisely minimal. Prefer `reserve` if future
    /// insertions are expected.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity overflows `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let mut bv = BitVec::from_elem(3, false);
    /// bv.reserve(10);
    /// assert_eq!(bv.len(), 3);
    /// assert!(bv.capacity() >= 13);
    /// ```
    #[inline]
    pub fn reserve_exact(&mut self, additional: usize) {
        let desired_cap = self
            .len()
            .checked_add(additional)
            .expect("capacity overflow");
        let storage_len = self.storage.len();
        if desired_cap > self.capacity() {
            self.storage
                .reserve_exact(blocks_for_bits::<B>(desired_cap) - storage_len);
        }
    }

    /// Returns the capacity in bits for this bit vector. Inserting any
    /// element less than this amount will not trigger a resizing.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let mut bv = BitVec::new();
    /// bv.reserve(10);
    /// assert!(bv.capacity() >= 10);
    /// ```
    #[inline]
    pub fn capacity(&self) -> usize {
        self.storage.capacity().saturating_mul(B::bits())
    }

    /// Grows the `BitVec` in-place, adding `n` copies of `value` to the `BitVec`.
    ///
    /// # Panics
    ///
    /// Panics if the new len overflows a `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let mut bv = BitVec::from_bytes(&[0b01001011]);
    /// bv.grow(2, true);
    /// assert_eq!(bv.len(), 10);
    /// assert_eq!(bv.to_bytes(), [0b01001011, 0b11000000]);
    /// ```
    pub fn grow(&mut self, n: usize, value: bool) {
        self.ensure_invariant();

        // Note: we just bulk set all the bits in the last word in this fn in multiple places
        // which is technically wrong if not all of these bits are to be used. However, at the end
        // of this fn we call `fix_last_block` at the end of this fn, which should fix this.

        let new_nbits = self.nbits.checked_add(n).expect("capacity overflow");
        let new_nblocks = blocks_for_bits::<B>(new_nbits);
        let full_value = if value { !B::zero() } else { B::zero() };

        // Correct the old tail word, setting or clearing formerly unused bits
        let num_cur_blocks = blocks_for_bits::<B>(self.nbits);
        if self.nbits % B::bits() > 0 {
            let mask = mask_for_bits::<B>(self.nbits);
            if value {
                let block = &mut self.storage[num_cur_blocks - 1];
                *block = *block | !mask;
            } else {
                // Extra bits are already zero by invariant.
            }
        }

        // Fill in words after the old tail word
        let stop_idx = cmp::min(self.storage.len(), new_nblocks);
        for idx in num_cur_blocks..stop_idx {
            self.storage[idx] = full_value;
        }

        // Allocate new words, if needed
        if new_nblocks > self.storage.len() {
            let to_add = new_nblocks - self.storage.len();
            self.storage.extend(repeat(full_value).take(to_add));
        }

        // Adjust internal bit count
        self.nbits = new_nbits;

        self.fix_last_block();
    }

    /// Removes the last bit from the `BitVec`, and returns it. Returns `None` if the `BitVec` is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let mut bv = BitVec::from_bytes(&[0b01001001]);
    /// assert_eq!(bv.pop(), Some(true));
    /// assert_eq!(bv.pop(), Some(false));
    /// assert_eq!(bv.len(), 6);
    /// ```
    #[inline]
    pub fn pop(&mut self) -> Option<bool> {
        self.ensure_invariant();

        if self.is_empty() {
            None
        } else {
            let i = self.nbits - 1;
            let ret = self[i];
            // (3)
            self.set(i, false);
            self.nbits = i;
            if self.nbits % B::bits() == 0 {
                // (2)
                self.storage.pop();
            }
            Some(ret)
        }
    }

    /// Pushes a `bool` onto the end.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let mut bv = BitVec::new();
    /// bv.push(true);
    /// bv.push(false);
    /// assert!(bv.eq_vec(&[true, false]));
    /// ```
    #[inline]
    pub fn push(&mut self, elem: bool) {
        if self.nbits % B::bits() == 0 {
            self.storage.push(B::zero());
        }
        let insert_pos = self.nbits;
        self.nbits = self.nbits.checked_add(1).expect("Capacity overflow");
        self.set(insert_pos, elem);
    }

    /// Returns the total number of bits in this vector
    #[inline]
    pub fn len(&self) -> usize {
        self.nbits
    }

    /// Sets the number of bits that this `BitVec` considers initialized.
    ///
    /// # Safety
    ///
    /// Almost certainly can cause bad stuff. Only really intended for `BitSet`.
    #[inline]
    pub unsafe fn set_len(&mut self, len: usize) {
        self.nbits = len;
    }

    /// Returns true if there are no bits in this vector
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clears all bits in this vector.
    #[inline]
    pub fn clear(&mut self) {
        self.ensure_invariant();
        for w in &mut self.storage {
            *w = B::zero();
        }
    }

    /// Shrinks the capacity of the underlying storage as much as
    /// possible.
    ///
    /// It will drop down as close as possible to the length but the
    /// allocator may still inform the underlying storage that there
    /// is space for a few more elements/bits.
    pub fn shrink_to_fit(&mut self) {
        self.storage.shrink_to_fit();
    }

    /// Inserts a given bit at index `at`, shifting all bits after by one
    ///
    /// # Panics
    /// Panics if `at` is out of bounds for `BitVec`'s length (that is, if `at > BitVec::len()`)
    ///
    /// # Examples
    ///```
    /// use bit_vec::BitVec;
    ///
    /// let mut b = BitVec::new();
    ///
    /// b.push(true);
    /// b.push(true);
    /// b.insert(1, false);
    ///
    /// assert!(b.eq_vec(&[true, false, true]));
    ///```
    ///
    /// # Time complexity
    /// Takes O([`len`]) time. All items after the insertion index must be
    /// shifted to the right. In the worst case, all elements are shifted when
    /// the insertion index is 0.
    ///
    /// [`len`]: Self::len
    pub fn insert(&mut self, at: usize, bit: bool) {
        assert!(
            at <= self.nbits,
            "insertion index (is {at}) should be <= len (is {nbits})",
            nbits = self.nbits
        );

        let last_block_bits = self.nbits % B::bits();
        let block_at = at / B::bits(); // needed block
        let bit_at = at % B::bits(); // index within the block

        if last_block_bits == 0 {
            self.storage.push(B::zero());
        }

        self.nbits += 1;

        let mut carry = self.storage[block_at] >> (B::bits() - 1);
        let lsbits_mask = (B::one() << bit_at) - B::one();
        let set_bit = if bit { B::one() } else { B::zero() } << bit_at;
        self.storage[block_at] = (self.storage[block_at] & lsbits_mask)
            | ((self.storage[block_at] & !lsbits_mask) << 1)
            | set_bit;

        for block_ref in &mut self.storage[block_at + 1..] {
            let curr_carry = *block_ref >> (B::bits() - 1);
            *block_ref = *block_ref << 1 | carry;
            carry = curr_carry;
        }
    }

    /// Appends an element if there is sufficient spare capacity, otherwise an error is returned
    /// with the element.
    ///
    /// Unlike [`push`] this method will not reallocate when there's insufficient capacity.
    /// The caller should use [`reserve`] to ensure that there is enough capacity.
    ///
    /// [`push`]: Self::push
    /// [`reserve`]: Self::reserve
    ///
    /// # Examples
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let initial_capacity = 64;
    /// let mut bitvec = BitVec::with_capacity(64);
    ///
    /// for _ in 0..initial_capacity - 1 {
    ///     bitvec.push(false);
    /// }
    ///
    /// assert_eq!(bitvec.len(), initial_capacity - 1); // there is space for only 1 bit
    ///
    /// assert_eq!(bitvec.push_within_capacity(true), Ok(())); // Successfully push a bit
    /// assert_eq!(bitvec.len(), initial_capacity); // So we can't push within capacity anymore
    ///
    /// assert_eq!(bitvec.push_within_capacity(true), Err(true));
    /// assert_eq!(bitvec.len(), initial_capacity);
    /// assert_eq!(bitvec.capacity(), initial_capacity);
    /// ```
    ///
    /// # Time Complexity
    /// Takes *O(1)* time.
    pub fn push_within_capacity(&mut self, bit: bool) -> Result<(), bool> {
        let len = self.len();

        if len == self.capacity() {
            return Err(bit);
        }

        let bits = B::bits();

        if len % bits == 0 {
            self.storage.push(B::zero());
        }

        let block_at = len / bits;
        let bit_at = len % bits;
        let flag = if bit { B::one() << bit_at } else { B::zero() };

        self.ensure_invariant();

        self.nbits += 1;

        self.storage[block_at] = self.storage[block_at] | flag; // set the bit

        Ok(())
    }
}

impl<B: BitBlock> Default for BitVec<B> {
    #[inline]
    fn default() -> Self {
        BitVec {
            storage: Vec::new(),
            nbits: 0,
        }
    }
}

impl<B: BitBlock> FromIterator<bool> for BitVec<B> {
    #[inline]
    fn from_iter<I: IntoIterator<Item = bool>>(iter: I) -> Self {
        let mut ret: Self = Default::default();
        ret.extend(iter);
        ret
    }
}

impl<B: BitBlock> Extend<bool> for BitVec<B> {
    #[inline]
    fn extend<I: IntoIterator<Item = bool>>(&mut self, iterable: I) {
        self.ensure_invariant();
        let iterator = iterable.into_iter();
        let (min, _) = iterator.size_hint();
        self.reserve(min);
        for element in iterator {
            self.push(element)
        }
    }
}

impl<B: BitBlock> Clone for BitVec<B> {
    #[inline]
    fn clone(&self) -> Self {
        self.ensure_invariant();
        BitVec {
            storage: self.storage.clone(),
            nbits: self.nbits,
        }
    }

    #[inline]
    fn clone_from(&mut self, source: &Self) {
        debug_assert!(source.is_last_block_fixed());
        self.nbits = source.nbits;
        self.storage.clone_from(&source.storage);
    }
}

impl<B: BitBlock> PartialOrd for BitVec<B> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<B: BitBlock> Ord for BitVec<B> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.ensure_invariant();
        debug_assert!(other.is_last_block_fixed());
        let mut a = self.iter();
        let mut b = other.iter();
        loop {
            match (a.next(), b.next()) {
                (Some(x), Some(y)) => match x.cmp(&y) {
                    Ordering::Equal => {}
                    otherwise => return otherwise,
                },
                (None, None) => return Ordering::Equal,
                (None, _) => return Ordering::Less,
                (_, None) => return Ordering::Greater,
            }
        }
    }
}

impl<B: BitBlock> fmt::Display for BitVec<B> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        self.ensure_invariant();
        for bit in self {
            fmt.write_char(if bit { '1' } else { '0' })?;
        }
        Ok(())
    }
}

impl<B: BitBlock> fmt::Debug for BitVec<B> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        self.ensure_invariant();
        let mut storage = String::with_capacity(self.len() + self.len() / B::bits());
        for (i, bit) in self.iter().enumerate() {
            if i != 0 && i % B::bits() == 0 {
                storage.push(' ');
            }
            storage.push(if bit { '1' } else { '0' });
        }
        fmt.debug_struct("BitVec")
            .field("storage", &storage)
            .field("nbits", &self.nbits)
            .finish()
    }
}

impl<B: BitBlock> hash::Hash for BitVec<B> {
    #[inline]
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.ensure_invariant();
        self.nbits.hash(state);
        for elem in self.blocks() {
            elem.hash(state);
        }
    }
}

impl<B: BitBlock> cmp::PartialEq for BitVec<B> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        if self.nbits != other.nbits {
            self.ensure_invariant();
            other.ensure_invariant();
            return false;
        }
        self.blocks().zip(other.blocks()).all(|(w1, w2)| w1 == w2)
    }
}

impl<B: BitBlock> cmp::Eq for BitVec<B> {}

/// An iterator for `BitVec`.
#[derive(Clone)]
pub struct Iter<'a, B: 'a = u32> {
    bit_vec: &'a BitVec<B>,
    range: Range<usize>,
}

#[derive(Debug)]
pub struct MutBorrowedBit<'a, B: 'a + BitBlock> {
    vec: Rc<RefCell<&'a mut BitVec<B>>>,
    index: usize,
    #[cfg(debug_assertions)]
    old_value: bool,
    new_value: bool,
}

/// An iterator for mutable references to the bits in a `BitVec`.
pub struct IterMut<'a, B: 'a + BitBlock = u32> {
    vec: Rc<RefCell<&'a mut BitVec<B>>>,
    range: Range<usize>,
}

impl<'a, B: 'a + BitBlock> IterMut<'a, B> {
    fn get(&mut self, index: Option<usize>) -> Option<MutBorrowedBit<'a, B>> {
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

impl<B: BitBlock> Deref for MutBorrowedBit<'_, B> {
    type Target = bool;

    fn deref(&self) -> &Self::Target {
        &self.new_value
    }
}

impl<B: BitBlock> DerefMut for MutBorrowedBit<'_, B> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.new_value
    }
}

impl<B: BitBlock> Drop for MutBorrowedBit<'_, B> {
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

impl<B: BitBlock> Iterator for Iter<'_, B> {
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

impl<'a, B: BitBlock> Iterator for IterMut<'a, B> {
    type Item = MutBorrowedBit<'a, B>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let index = self.range.next();
        self.get(index)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.range.size_hint()
    }
}

impl<B: BitBlock> DoubleEndedIterator for Iter<'_, B> {
    #[inline]
    fn next_back(&mut self) -> Option<bool> {
        self.range.next_back().map(|i| self.bit_vec.get(i).unwrap())
    }
}

impl<B: BitBlock> DoubleEndedIterator for IterMut<'_, B> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        let index = self.range.next_back();
        self.get(index)
    }
}

impl<B: BitBlock> ExactSizeIterator for Iter<'_, B> {}

impl<B: BitBlock> ExactSizeIterator for IterMut<'_, B> {}

impl<'a, B: BitBlock> IntoIterator for &'a BitVec<B> {
    type Item = bool;
    type IntoIter = Iter<'a, B>;

    #[inline]
    fn into_iter(self) -> Iter<'a, B> {
        self.iter()
    }
}

pub struct IntoIter<B = u32> {
    bit_vec: BitVec<B>,
    range: Range<usize>,
}

impl<B: BitBlock> Iterator for IntoIter<B> {
    type Item = bool;

    #[inline]
    fn next(&mut self) -> Option<bool> {
        self.range.next().map(|i| self.bit_vec.get(i).unwrap())
    }
}

impl<B: BitBlock> DoubleEndedIterator for IntoIter<B> {
    #[inline]
    fn next_back(&mut self) -> Option<bool> {
        self.range.next_back().map(|i| self.bit_vec.get(i).unwrap())
    }
}

impl<B: BitBlock> ExactSizeIterator for IntoIter<B> {}

impl<B: BitBlock> IntoIterator for BitVec<B> {
    type Item = bool;
    type IntoIter = IntoIter<B>;

    #[inline]
    fn into_iter(self) -> IntoIter<B> {
        let nbits = self.nbits;
        IntoIter {
            bit_vec: self,
            range: 0..nbits,
        }
    }
}

/// An iterator over the blocks of a `BitVec`.
#[derive(Clone)]
pub struct Blocks<'a, B: 'a> {
    iter: slice::Iter<'a, B>,
}

impl<B: BitBlock> Iterator for Blocks<'_, B> {
    type Item = B;

    #[inline]
    fn next(&mut self) -> Option<B> {
        self.iter.next().cloned()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<B: BitBlock> DoubleEndedIterator for Blocks<'_, B> {
    #[inline]
    fn next_back(&mut self) -> Option<B> {
        self.iter.next_back().cloned()
    }
}

impl<B: BitBlock> ExactSizeIterator for Blocks<'_, B> {}

#[cfg(test)]
mod tests {
    #![allow(clippy::shadow_reuse)]
    #![allow(clippy::shadow_same)]
    #![allow(clippy::shadow_unrelated)]

    use super::{BitVec, Iter, Vec};

    // This is stupid, but I want to differentiate from a "random" 32
    const U32_BITS: usize = 32;

    #[test]
    fn test_display_output() {
        assert_eq!(format!("{}", BitVec::new()), "");
        assert_eq!(format!("{}", BitVec::from_elem(1, true)), "1");
        assert_eq!(format!("{}", BitVec::from_elem(8, false)), "00000000")
    }

    #[test]
    fn test_debug_output() {
        assert_eq!(
            format!("{:?}", BitVec::new()),
            "BitVec { storage: \"\", nbits: 0 }"
        );
        assert_eq!(
            format!("{:?}", BitVec::from_elem(1, true)),
            "BitVec { storage: \"1\", nbits: 1 }"
        );
        assert_eq!(
            format!("{:?}", BitVec::from_elem(8, false)),
            "BitVec { storage: \"00000000\", nbits: 8 }"
        );
        assert_eq!(
            format!("{:?}", BitVec::from_elem(33, true)),
            "BitVec { storage: \"11111111111111111111111111111111 1\", nbits: 33 }"
        );
        assert_eq!(
            format!(
                "{:?}",
                BitVec::from_bytes(&[0b111, 0b000, 0b1110, 0b0001, 0b11111111, 0b00000000])
            ),
            "BitVec { storage: \"00000111000000000000111000000001 1111111100000000\", nbits: 48 }"
        )
    }

    #[test]
    fn test_0_elements() {
        let act = BitVec::new();
        let exp = Vec::new();
        assert!(act.eq_vec(&exp));
        assert!(act.none() && act.all());
    }

    #[test]
    fn test_1_element() {
        let mut act = BitVec::from_elem(1, false);
        assert!(act.eq_vec(&[false]));
        assert!(act.none() && !act.all());
        act = BitVec::from_elem(1, true);
        assert!(act.eq_vec(&[true]));
        assert!(!act.none() && act.all());
    }

    #[test]
    fn test_2_elements() {
        let mut b = BitVec::from_elem(2, false);
        b.set(0, true);
        b.set(1, false);
        assert_eq!(format!("{}", b), "10");
        assert!(!b.none() && !b.all());
    }

    #[test]
    fn test_10_elements() {
        // all 0

        let mut act = BitVec::from_elem(10, false);
        assert!(
            (act.eq_vec(&[false, false, false, false, false, false, false, false, false, false]))
        );
        assert!(act.none() && !act.all());
        // all 1

        act = BitVec::from_elem(10, true);
        assert!((act.eq_vec(&[true, true, true, true, true, true, true, true, true, true])));
        assert!(!act.none() && act.all());
        // mixed

        act = BitVec::from_elem(10, false);
        act.set(0, true);
        act.set(1, true);
        act.set(2, true);
        act.set(3, true);
        act.set(4, true);
        assert!((act.eq_vec(&[true, true, true, true, true, false, false, false, false, false])));
        assert!(!act.none() && !act.all());
        // mixed

        act = BitVec::from_elem(10, false);
        act.set(5, true);
        act.set(6, true);
        act.set(7, true);
        act.set(8, true);
        act.set(9, true);
        assert!((act.eq_vec(&[false, false, false, false, false, true, true, true, true, true])));
        assert!(!act.none() && !act.all());
        // mixed

        act = BitVec::from_elem(10, false);
        act.set(0, true);
        act.set(3, true);
        act.set(6, true);
        act.set(9, true);
        assert!((act.eq_vec(&[true, false, false, true, false, false, true, false, false, true])));
        assert!(!act.none() && !act.all());
    }

    #[test]
    fn test_31_elements() {
        // all 0

        let mut act = BitVec::from_elem(31, false);
        assert!(act.eq_vec(&[
            false, false, false, false, false, false, false, false, false, false, false, false,
            false, false, false, false, false, false, false, false, false, false, false, false,
            false, false, false, false, false, false, false
        ]));
        assert!(act.none() && !act.all());
        // all 1

        act = BitVec::from_elem(31, true);
        assert!(act.eq_vec(&[
            true, true, true, true, true, true, true, true, true, true, true, true, true, true,
            true, true, true, true, true, true, true, true, true, true, true, true, true, true,
            true, true, true
        ]));
        assert!(!act.none() && act.all());
        // mixed

        act = BitVec::from_elem(31, false);
        act.set(0, true);
        act.set(1, true);
        act.set(2, true);
        act.set(3, true);
        act.set(4, true);
        act.set(5, true);
        act.set(6, true);
        act.set(7, true);
        assert!(act.eq_vec(&[
            true, true, true, true, true, true, true, true, false, false, false, false, false,
            false, false, false, false, false, false, false, false, false, false, false, false,
            false, false, false, false, false, false
        ]));
        assert!(!act.none() && !act.all());
        // mixed

        act = BitVec::from_elem(31, false);
        act.set(16, true);
        act.set(17, true);
        act.set(18, true);
        act.set(19, true);
        act.set(20, true);
        act.set(21, true);
        act.set(22, true);
        act.set(23, true);
        assert!(act.eq_vec(&[
            false, false, false, false, false, false, false, false, false, false, false, false,
            false, false, false, false, true, true, true, true, true, true, true, true, false,
            false, false, false, false, false, false
        ]));
        assert!(!act.none() && !act.all());
        // mixed

        act = BitVec::from_elem(31, false);
        act.set(24, true);
        act.set(25, true);
        act.set(26, true);
        act.set(27, true);
        act.set(28, true);
        act.set(29, true);
        act.set(30, true);
        assert!(act.eq_vec(&[
            false, false, false, false, false, false, false, false, false, false, false, false,
            false, false, false, false, false, false, false, false, false, false, false, false,
            true, true, true, true, true, true, true
        ]));
        assert!(!act.none() && !act.all());
        // mixed

        act = BitVec::from_elem(31, false);
        act.set(3, true);
        act.set(17, true);
        act.set(30, true);
        assert!(act.eq_vec(&[
            false, false, false, true, false, false, false, false, false, false, false, false,
            false, false, false, false, false, true, false, false, false, false, false, false,
            false, false, false, false, false, false, true
        ]));
        assert!(!act.none() && !act.all());
    }

    #[test]
    fn test_32_elements() {
        // all 0

        let mut act = BitVec::from_elem(32, false);
        assert!(act.eq_vec(&[
            false, false, false, false, false, false, false, false, false, false, false, false,
            false, false, false, false, false, false, false, false, false, false, false, false,
            false, false, false, false, false, false, false, false
        ]));
        assert!(act.none() && !act.all());
        // all 1

        act = BitVec::from_elem(32, true);
        assert!(act.eq_vec(&[
            true, true, true, true, true, true, true, true, true, true, true, true, true, true,
            true, true, true, true, true, true, true, true, true, true, true, true, true, true,
            true, true, true, true
        ]));
        assert!(!act.none() && act.all());
        // mixed

        act = BitVec::from_elem(32, false);
        act.set(0, true);
        act.set(1, true);
        act.set(2, true);
        act.set(3, true);
        act.set(4, true);
        act.set(5, true);
        act.set(6, true);
        act.set(7, true);
        assert!(act.eq_vec(&[
            true, true, true, true, true, true, true, true, false, false, false, false, false,
            false, false, false, false, false, false, false, false, false, false, false, false,
            false, false, false, false, false, false, false
        ]));
        assert!(!act.none() && !act.all());
        // mixed

        act = BitVec::from_elem(32, false);
        act.set(16, true);
        act.set(17, true);
        act.set(18, true);
        act.set(19, true);
        act.set(20, true);
        act.set(21, true);
        act.set(22, true);
        act.set(23, true);
        assert!(act.eq_vec(&[
            false, false, false, false, false, false, false, false, false, false, false, false,
            false, false, false, false, true, true, true, true, true, true, true, true, false,
            false, false, false, false, false, false, false
        ]));
        assert!(!act.none() && !act.all());
        // mixed

        act = BitVec::from_elem(32, false);
        act.set(24, true);
        act.set(25, true);
        act.set(26, true);
        act.set(27, true);
        act.set(28, true);
        act.set(29, true);
        act.set(30, true);
        act.set(31, true);
        assert!(act.eq_vec(&[
            false, false, false, false, false, false, false, false, false, false, false, false,
            false, false, false, false, false, false, false, false, false, false, false, false,
            true, true, true, true, true, true, true, true
        ]));
        assert!(!act.none() && !act.all());
        // mixed

        act = BitVec::from_elem(32, false);
        act.set(3, true);
        act.set(17, true);
        act.set(30, true);
        act.set(31, true);
        assert!(act.eq_vec(&[
            false, false, false, true, false, false, false, false, false, false, false, false,
            false, false, false, false, false, true, false, false, false, false, false, false,
            false, false, false, false, false, false, true, true
        ]));
        assert!(!act.none() && !act.all());
    }

    #[test]
    fn test_33_elements() {
        // all 0

        let mut act = BitVec::from_elem(33, false);
        assert!(act.eq_vec(&[
            false, false, false, false, false, false, false, false, false, false, false, false,
            false, false, false, false, false, false, false, false, false, false, false, false,
            false, false, false, false, false, false, false, false, false
        ]));
        assert!(act.none() && !act.all());
        // all 1

        act = BitVec::from_elem(33, true);
        assert!(act.eq_vec(&[
            true, true, true, true, true, true, true, true, true, true, true, true, true, true,
            true, true, true, true, true, true, true, true, true, true, true, true, true, true,
            true, true, true, true, true
        ]));
        assert!(!act.none() && act.all());
        // mixed

        act = BitVec::from_elem(33, false);
        act.set(0, true);
        act.set(1, true);
        act.set(2, true);
        act.set(3, true);
        act.set(4, true);
        act.set(5, true);
        act.set(6, true);
        act.set(7, true);
        assert!(act.eq_vec(&[
            true, true, true, true, true, true, true, true, false, false, false, false, false,
            false, false, false, false, false, false, false, false, false, false, false, false,
            false, false, false, false, false, false, false, false
        ]));
        assert!(!act.none() && !act.all());
        // mixed

        act = BitVec::from_elem(33, false);
        act.set(16, true);
        act.set(17, true);
        act.set(18, true);
        act.set(19, true);
        act.set(20, true);
        act.set(21, true);
        act.set(22, true);
        act.set(23, true);
        assert!(act.eq_vec(&[
            false, false, false, false, false, false, false, false, false, false, false, false,
            false, false, false, false, true, true, true, true, true, true, true, true, false,
            false, false, false, false, false, false, false, false
        ]));
        assert!(!act.none() && !act.all());
        // mixed

        act = BitVec::from_elem(33, false);
        act.set(24, true);
        act.set(25, true);
        act.set(26, true);
        act.set(27, true);
        act.set(28, true);
        act.set(29, true);
        act.set(30, true);
        act.set(31, true);
        assert!(act.eq_vec(&[
            false, false, false, false, false, false, false, false, false, false, false, false,
            false, false, false, false, false, false, false, false, false, false, false, false,
            true, true, true, true, true, true, true, true, false
        ]));
        assert!(!act.none() && !act.all());
        // mixed

        act = BitVec::from_elem(33, false);
        act.set(3, true);
        act.set(17, true);
        act.set(30, true);
        act.set(31, true);
        act.set(32, true);
        assert!(act.eq_vec(&[
            false, false, false, true, false, false, false, false, false, false, false, false,
            false, false, false, false, false, true, false, false, false, false, false, false,
            false, false, false, false, false, false, true, true, true
        ]));
        assert!(!act.none() && !act.all());
    }

    #[test]
    fn test_equal_differing_sizes() {
        let v0 = BitVec::from_elem(10, false);
        let v1 = BitVec::from_elem(11, false);
        assert_ne!(v0, v1);
    }

    #[test]
    fn test_equal_greatly_differing_sizes() {
        let v0 = BitVec::from_elem(10, false);
        let v1 = BitVec::from_elem(110, false);
        assert_ne!(v0, v1);
    }

    #[test]
    fn test_equal_sneaky_small() {
        let mut a = BitVec::from_elem(1, false);
        a.set(0, true);

        let mut b = BitVec::from_elem(1, true);
        b.set(0, true);

        assert_eq!(a, b);
    }

    #[test]
    fn test_equal_sneaky_big() {
        let mut a = BitVec::from_elem(100, false);
        for i in 0..100 {
            a.set(i, true);
        }

        let mut b = BitVec::from_elem(100, true);
        for i in 0..100 {
            b.set(i, true);
        }

        assert_eq!(a, b);
    }

    #[test]
    fn test_from_bytes() {
        let bit_vec = BitVec::from_bytes(&[0b10110110, 0b00000000, 0b11111111]);
        let str = concat!("10110110", "00000000", "11111111");
        assert_eq!(format!("{}", bit_vec), str);
    }

    #[test]
    fn test_to_bytes() {
        let mut bv = BitVec::from_elem(3, true);
        bv.set(1, false);
        assert_eq!(bv.to_bytes(), [0b10100000]);

        let mut bv = BitVec::from_elem(9, false);
        bv.set(2, true);
        bv.set(8, true);
        assert_eq!(bv.to_bytes(), [0b00100000, 0b10000000]);
    }

    #[test]
    fn test_from_bools() {
        let bools = [true, false, true, true];
        let bit_vec: BitVec = bools.iter().copied().collect();
        assert_eq!(format!("{}", bit_vec), "1011");
    }

    #[test]
    fn test_to_bools() {
        let bools = vec![false, false, true, false, false, true, true, false];
        assert_eq!(
            BitVec::from_bytes(&[0b00100110])
                .iter()
                .collect::<Vec<bool>>(),
            bools
        );
    }

    #[test]
    fn test_bit_vec_iterator() {
        let bools = vec![true, false, true, true];
        let bit_vec: BitVec = bools.iter().copied().collect();

        assert_eq!(bit_vec.iter().collect::<Vec<bool>>(), bools);

        let long: Vec<_> = (0..10000).map(|i| i % 2 == 0).collect();
        let bit_vec: BitVec = long.iter().copied().collect();
        assert_eq!(bit_vec.iter().collect::<Vec<bool>>(), long)
    }

    #[test]
    fn test_small_difference() {
        let mut b1 = BitVec::from_elem(3, false);
        let mut b2 = BitVec::from_elem(3, false);
        b1.set(0, true);
        b1.set(1, true);
        b2.set(1, true);
        b2.set(2, true);
        assert!(b1.difference(&b2));
        assert!(b1[0]);
        assert!(!b1[1]);
        assert!(!b1[2]);
    }

    #[test]
    fn test_big_difference() {
        let mut b1 = BitVec::from_elem(100, false);
        let mut b2 = BitVec::from_elem(100, false);
        b1.set(0, true);
        b1.set(40, true);
        b2.set(40, true);
        b2.set(80, true);
        assert!(b1.difference(&b2));
        assert!(b1[0]);
        assert!(!b1[40]);
        assert!(!b1[80]);
    }

    #[test]
    fn test_small_xor() {
        let mut a = BitVec::from_bytes(&[0b0011]);
        let b = BitVec::from_bytes(&[0b0101]);
        let c = BitVec::from_bytes(&[0b0110]);
        assert!(a.xor(&b));
        assert_eq!(a, c);
    }

    #[test]
    fn test_small_xnor() {
        let mut a = BitVec::from_bytes(&[0b0011]);
        let b = BitVec::from_bytes(&[0b1111_0101]);
        let c = BitVec::from_bytes(&[0b1001]);
        assert!(a.xnor(&b));
        assert_eq!(a, c);
    }

    #[test]
    fn test_small_nand() {
        let mut a = BitVec::from_bytes(&[0b1111_0011]);
        let b = BitVec::from_bytes(&[0b1111_0101]);
        let c = BitVec::from_bytes(&[0b1110]);
        assert!(a.nand(&b));
        assert_eq!(a, c);
    }

    #[test]
    fn test_small_nor() {
        let mut a = BitVec::from_bytes(&[0b0011]);
        let b = BitVec::from_bytes(&[0b1111_0101]);
        let c = BitVec::from_bytes(&[0b1000]);
        assert!(a.nor(&b));
        assert_eq!(a, c);
    }

    #[test]
    fn test_big_xor() {
        let mut a = BitVec::from_bytes(&[
            // 88 bits
            0, 0, 0b00010100, 0, 0, 0, 0, 0b00110100, 0, 0, 0,
        ]);
        let b = BitVec::from_bytes(&[
            // 88 bits
            0, 0, 0b00010100, 0, 0, 0, 0, 0, 0, 0, 0b00110100,
        ]);
        let c = BitVec::from_bytes(&[
            // 88 bits
            0, 0, 0, 0, 0, 0, 0, 0b00110100, 0, 0, 0b00110100,
        ]);
        assert!(a.xor(&b));
        assert_eq!(a, c);
    }

    #[test]
    fn test_big_xnor() {
        let mut a = BitVec::from_bytes(&[
            // 88 bits
            0, 0, 0b00010100, 0, 0, 0, 0, 0b00110100, 0, 0, 0,
        ]);
        let b = BitVec::from_bytes(&[
            // 88 bits
            0, 0, 0b00010100, 0, 0, 0, 0, 0, 0, 0, 0b00110100,
        ]);
        let c = BitVec::from_bytes(&[
            // 88 bits
            !0,
            !0,
            !0,
            !0,
            !0,
            !0,
            !0,
            !0b00110100,
            !0,
            !0,
            !0b00110100,
        ]);
        assert!(a.xnor(&b));
        assert_eq!(a, c);
    }

    #[test]
    fn test_small_clear() {
        let mut b = BitVec::from_elem(14, true);
        assert!(!b.none() && b.all());
        b.clear();
        assert!(b.none() && !b.all());
    }

    #[test]
    fn test_big_clear() {
        let mut b = BitVec::from_elem(140, true);
        assert!(!b.none() && b.all());
        b.clear();
        assert!(b.none() && !b.all());
    }

    #[test]
    fn test_bit_vec_lt() {
        let mut a = BitVec::from_elem(5, false);
        let mut b = BitVec::from_elem(5, false);

        assert!(a >= b && b >= a);
        b.set(2, true);
        assert!(a < b);
        a.set(3, true);
        assert!(a < b);
        a.set(2, true);
        assert!(a >= b && b < a);
        b.set(0, true);
        assert!(a < b);
    }

    #[test]
    fn test_ord() {
        let mut a = BitVec::from_elem(5, false);
        let mut b = BitVec::from_elem(5, false);

        assert!(a == b);
        a.set(1, true);
        assert!(a > b && a >= b);
        assert!(b < a && b <= a);
        b.set(1, true);
        b.set(2, true);
        assert!(b > a && b >= a);
        assert!(a < b && a <= b);
    }

    #[test]
    fn test_small_bit_vec_tests() {
        let v = BitVec::from_bytes(&[0]);
        assert!(!v.all());
        assert!(!v.any());
        assert!(v.none());

        let v = BitVec::from_bytes(&[0b00010100]);
        assert!(!v.all());
        assert!(v.any());
        assert!(!v.none());

        let v = BitVec::from_bytes(&[0xFF]);
        assert!(v.all());
        assert!(v.any());
        assert!(!v.none());
    }

    #[test]
    fn test_big_bit_vec_tests() {
        let v = BitVec::from_bytes(&[
            // 88 bits
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]);
        assert!(!v.all());
        assert!(!v.any());
        assert!(v.none());

        let v = BitVec::from_bytes(&[
            // 88 bits
            0, 0, 0b00010100, 0, 0, 0, 0, 0b00110100, 0, 0, 0,
        ]);
        assert!(!v.all());
        assert!(v.any());
        assert!(!v.none());

        let v = BitVec::from_bytes(&[
            // 88 bits
            0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        ]);
        assert!(v.all());
        assert!(v.any());
        assert!(!v.none());
    }

    #[test]
    fn test_bit_vec_push_pop() {
        let mut s = BitVec::from_elem(5 * U32_BITS - 2, false);
        assert_eq!(s.len(), 5 * U32_BITS - 2);
        assert!(!s[5 * U32_BITS - 3]);
        s.push(true);
        s.push(true);
        assert!(s[5 * U32_BITS - 2]);
        assert!(s[5 * U32_BITS - 1]);
        // Here the internal vector will need to be extended
        s.push(false);
        assert!(!s[5 * U32_BITS]);
        s.push(false);
        assert!(!s[5 * U32_BITS + 1]);
        assert_eq!(s.len(), 5 * U32_BITS + 2);
        // Pop it all off
        assert_eq!(s.pop(), Some(false));
        assert_eq!(s.pop(), Some(false));
        assert_eq!(s.pop(), Some(true));
        assert_eq!(s.pop(), Some(true));
        assert_eq!(s.len(), 5 * U32_BITS - 2);
    }

    #[test]
    fn test_bit_vec_truncate() {
        let mut s = BitVec::from_elem(5 * U32_BITS, true);

        assert_eq!(s, BitVec::from_elem(5 * U32_BITS, true));
        assert_eq!(s.len(), 5 * U32_BITS);
        s.truncate(4 * U32_BITS);
        assert_eq!(s, BitVec::from_elem(4 * U32_BITS, true));
        assert_eq!(s.len(), 4 * U32_BITS);
        // Truncating to a size > s.len() should be a noop
        s.truncate(5 * U32_BITS);
        assert_eq!(s, BitVec::from_elem(4 * U32_BITS, true));
        assert_eq!(s.len(), 4 * U32_BITS);
        s.truncate(3 * U32_BITS - 10);
        assert_eq!(s, BitVec::from_elem(3 * U32_BITS - 10, true));
        assert_eq!(s.len(), 3 * U32_BITS - 10);
        s.truncate(0);
        assert_eq!(s, BitVec::from_elem(0, true));
        assert_eq!(s.len(), 0);
    }

    #[test]
    fn test_bit_vec_reserve() {
        let mut s = BitVec::from_elem(5 * U32_BITS, true);
        // Check capacity
        assert!(s.capacity() >= 5 * U32_BITS);
        s.reserve(2 * U32_BITS);
        assert!(s.capacity() >= 7 * U32_BITS);
        s.reserve(7 * U32_BITS);
        assert!(s.capacity() >= 12 * U32_BITS);
        s.reserve_exact(7 * U32_BITS);
        assert!(s.capacity() >= 12 * U32_BITS);
        s.reserve(7 * U32_BITS + 1);
        assert!(s.capacity() > 12 * U32_BITS);
        // Check that length hasn't changed
        assert_eq!(s.len(), 5 * U32_BITS);
        s.push(true);
        s.push(false);
        s.push(true);
        assert!(s[5 * U32_BITS - 1]);
        assert!(s[5 * U32_BITS]);
        assert!(!s[5 * U32_BITS + 1]);
        assert!(s[5 * U32_BITS + 2]);
    }

    #[test]
    fn test_bit_vec_grow() {
        let mut bit_vec = BitVec::from_bytes(&[0b10110110, 0b00000000, 0b10101010]);
        bit_vec.grow(32, true);
        assert_eq!(
            bit_vec,
            BitVec::from_bytes(&[0b10110110, 0b00000000, 0b10101010, 0xFF, 0xFF, 0xFF, 0xFF])
        );
        bit_vec.grow(64, false);
        assert_eq!(
            bit_vec,
            BitVec::from_bytes(&[
                0b10110110, 0b00000000, 0b10101010, 0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0, 0, 0, 0, 0
            ])
        );
        bit_vec.grow(16, true);
        assert_eq!(
            bit_vec,
            BitVec::from_bytes(&[
                0b10110110, 0b00000000, 0b10101010, 0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0, 0, 0, 0, 0,
                0xFF, 0xFF
            ])
        );
    }

    #[test]
    fn test_bit_vec_extend() {
        let mut bit_vec = BitVec::from_bytes(&[0b10110110, 0b00000000, 0b11111111]);
        let ext = BitVec::from_bytes(&[0b01001001, 0b10010010, 0b10111101]);
        bit_vec.extend(ext.iter());
        assert_eq!(
            bit_vec,
            BitVec::from_bytes(&[
                0b10110110, 0b00000000, 0b11111111, 0b01001001, 0b10010010, 0b10111101
            ])
        );
    }

    #[test]
    fn test_bit_vec_append() {
        // Append to BitVec that holds a multiple of U32_BITS bits
        let mut a = BitVec::from_bytes(&[0b10100000, 0b00010010, 0b10010010, 0b00110011]);
        let mut b = BitVec::new();
        b.push(false);
        b.push(true);
        b.push(true);

        a.append(&mut b);

        assert_eq!(a.len(), 35);
        assert_eq!(b.len(), 0);
        assert!(b.capacity() >= 3);

        assert!(a.eq_vec(&[
            true, false, true, false, false, false, false, false, false, false, false, true, false,
            false, true, false, true, false, false, true, false, false, true, false, false, false,
            true, true, false, false, true, true, false, true, true
        ]));

        // Append to arbitrary BitVec
        let mut a = BitVec::new();
        a.push(true);
        a.push(false);

        let mut b =
            BitVec::from_bytes(&[0b10100000, 0b00010010, 0b10010010, 0b00110011, 0b10010101]);

        a.append(&mut b);

        assert_eq!(a.len(), 42);
        assert_eq!(b.len(), 0);
        assert!(b.capacity() >= 40);

        assert!(a.eq_vec(&[
            true, false, true, false, true, false, false, false, false, false, false, false, false,
            true, false, false, true, false, true, false, false, true, false, false, true, false,
            false, false, true, true, false, false, true, true, true, false, false, true, false,
            true, false, true
        ]));

        // Append to empty BitVec
        let mut a = BitVec::new();
        let mut b =
            BitVec::from_bytes(&[0b10100000, 0b00010010, 0b10010010, 0b00110011, 0b10010101]);

        a.append(&mut b);

        assert_eq!(a.len(), 40);
        assert_eq!(b.len(), 0);
        assert!(b.capacity() >= 40);

        assert!(a.eq_vec(&[
            true, false, true, false, false, false, false, false, false, false, false, true, false,
            false, true, false, true, false, false, true, false, false, true, false, false, false,
            true, true, false, false, true, true, true, false, false, true, false, true, false,
            true
        ]));

        // Append empty BitVec
        let mut a =
            BitVec::from_bytes(&[0b10100000, 0b00010010, 0b10010010, 0b00110011, 0b10010101]);
        let mut b = BitVec::new();

        a.append(&mut b);

        assert_eq!(a.len(), 40);
        assert_eq!(b.len(), 0);

        assert!(a.eq_vec(&[
            true, false, true, false, false, false, false, false, false, false, false, true, false,
            false, true, false, true, false, false, true, false, false, true, false, false, false,
            true, true, false, false, true, true, true, false, false, true, false, true, false,
            true
        ]));
    }

    #[test]
    fn test_bit_vec_split_off() {
        // Split at 0
        let mut a = BitVec::new();
        a.push(true);
        a.push(false);
        a.push(false);
        a.push(true);

        let b = a.split_off(0);

        assert_eq!(a.len(), 0);
        assert_eq!(b.len(), 4);

        assert!(b.eq_vec(&[true, false, false, true]));

        // Split at last bit
        a.truncate(0);
        a.push(true);
        a.push(false);
        a.push(false);
        a.push(true);

        let b = a.split_off(4);

        assert_eq!(a.len(), 4);
        assert_eq!(b.len(), 0);

        assert!(a.eq_vec(&[true, false, false, true]));

        // Split at block boundary
        let mut a =
            BitVec::from_bytes(&[0b10100000, 0b00010010, 0b10010010, 0b00110011, 0b11110011]);

        let b = a.split_off(32);

        assert_eq!(a.len(), 32);
        assert_eq!(b.len(), 8);

        assert!(a.eq_vec(&[
            true, false, true, false, false, false, false, false, false, false, false, true, false,
            false, true, false, true, false, false, true, false, false, true, false, false, false,
            true, true, false, false, true, true
        ]));
        assert!(b.eq_vec(&[true, true, true, true, false, false, true, true]));

        // Don't split at block boundary
        let mut a = BitVec::from_bytes(&[
            0b10100000, 0b00010010, 0b10010010, 0b00110011, 0b01101011, 0b10101101,
        ]);

        let b = a.split_off(13);

        assert_eq!(a.len(), 13);
        assert_eq!(b.len(), 35);

        assert!(a.eq_vec(&[
            true, false, true, false, false, false, false, false, false, false, false, true, false
        ]));
        assert!(b.eq_vec(&[
            false, true, false, true, false, false, true, false, false, true, false, false, false,
            true, true, false, false, true, true, false, true, true, false, true, false, true,
            true, true, false, true, false, true, true, false, true
        ]));
    }

    #[test]
    fn test_into_iter() {
        let bools = [true, false, true, true];
        let bit_vec: BitVec = bools.iter().copied().collect();
        let mut iter = bit_vec.into_iter();
        assert_eq!(Some(true), iter.next());
        assert_eq!(Some(false), iter.next());
        assert_eq!(Some(true), iter.next());
        assert_eq!(Some(true), iter.next());
        assert_eq!(None, iter.next());
        assert_eq!(None, iter.next());

        let bit_vec: BitVec = bools.iter().copied().collect();
        let mut iter = bit_vec.into_iter();
        assert_eq!(Some(true), iter.next_back());
        assert_eq!(Some(true), iter.next_back());
        assert_eq!(Some(false), iter.next_back());
        assert_eq!(Some(true), iter.next_back());
        assert_eq!(None, iter.next_back());
        assert_eq!(None, iter.next_back());

        let bit_vec: BitVec = bools.iter().copied().collect();
        let mut iter = bit_vec.into_iter();
        assert_eq!(Some(true), iter.next_back());
        assert_eq!(Some(true), iter.next());
        assert_eq!(Some(false), iter.next());
        assert_eq!(Some(true), iter.next_back());
        assert_eq!(None, iter.next());
        assert_eq!(None, iter.next_back());
    }

    #[test]
    fn iter() {
        let b = BitVec::with_capacity(10);
        let _a: Iter = b.iter();
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_serialization() {
        let bit_vec: BitVec = BitVec::new();
        let serialized = serde_json::to_string(&bit_vec).unwrap();
        let unserialized: BitVec = serde_json::from_str(&serialized).unwrap();
        assert_eq!(bit_vec, unserialized);

        let bools = vec![true, false, true, true];
        let bit_vec: BitVec = bools.iter().map(|n| *n).collect();
        let serialized = serde_json::to_string(&bit_vec).unwrap();
        let unserialized = serde_json::from_str(&serialized).unwrap();
        assert_eq!(bit_vec, unserialized);
    }

    #[cfg(feature = "miniserde")]
    #[test]
    fn test_miniserde_serialization() {
        let bit_vec: BitVec = BitVec::new();
        let serialized = miniserde::json::to_string(&bit_vec);
        let unserialized: BitVec = miniserde::json::from_str(&serialized[..]).unwrap();
        assert_eq!(bit_vec, unserialized);

        let bools = vec![true, false, true, true];
        let bit_vec: BitVec = bools.iter().map(|n| *n).collect();
        let serialized = miniserde::json::to_string(&bit_vec);
        let unserialized = miniserde::json::from_str(&serialized[..]).unwrap();
        assert_eq!(bit_vec, unserialized);
    }

    #[cfg(feature = "nanoserde")]
    #[test]
    fn test_nanoserde_json_serialization() {
        use nanoserde::{DeJson, SerJson};

        let bit_vec: BitVec = BitVec::new();
        let serialized = bit_vec.serialize_json();
        let unserialized: BitVec = BitVec::deserialize_json(&serialized[..]).unwrap();
        assert_eq!(bit_vec, unserialized);

        let bools = vec![true, false, true, true];
        let bit_vec: BitVec = bools.iter().map(|n| *n).collect();
        let serialized = bit_vec.serialize_json();
        let unserialized = BitVec::deserialize_json(&serialized[..]).unwrap();
        assert_eq!(bit_vec, unserialized);
    }

    #[cfg(feature = "borsh")]
    #[test]
    fn test_borsh_serialization() {
        let bit_vec: BitVec = BitVec::new();
        let serialized = borsh::to_vec(&bit_vec).unwrap();
        let unserialized: BitVec = borsh::from_slice(&serialized[..]).unwrap();
        assert_eq!(bit_vec, unserialized);

        let bools = vec![true, false, true, true];
        let bit_vec: BitVec = bools.iter().map(|n| *n).collect();
        let serialized = borsh::to_vec(&bit_vec).unwrap();
        let unserialized = borsh::from_slice(&serialized[..]).unwrap();
        assert_eq!(bit_vec, unserialized);
    }

    #[test]
    fn test_bit_vec_unaligned_small_append() {
        let mut a = BitVec::from_elem(8, false);
        a.set(7, true);

        let mut b = BitVec::from_elem(16, false);
        b.set(14, true);

        let mut c = BitVec::from_elem(8, false);
        c.set(6, true);
        c.set(7, true);

        a.append(&mut b);
        a.append(&mut c);

        assert_eq!(&[1, 0, 2, 3][..], &*a.to_bytes());
    }

    #[test]
    fn test_bit_vec_unaligned_large_append() {
        let mut a = BitVec::from_elem(48, false);
        a.set(47, true);

        let mut b = BitVec::from_elem(48, false);
        b.set(46, true);

        let mut c = BitVec::from_elem(48, false);
        c.set(46, true);
        c.set(47, true);

        a.append(&mut b);
        a.append(&mut c);

        assert_eq!(
            &[
                0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x03
            ][..],
            &*a.to_bytes()
        );
    }

    #[test]
    fn test_bit_vec_append_aligned_to_unaligned() {
        let mut a = BitVec::from_elem(2, true);
        let mut b = BitVec::from_elem(32, false);
        let mut c = BitVec::from_elem(8, true);
        a.append(&mut b);
        a.append(&mut c);
        assert_eq!(&[0xc0, 0x00, 0x00, 0x00, 0x3f, 0xc0][..], &*a.to_bytes());
    }

    #[test]
    fn test_count_ones() {
        for i in 0..1000 {
            let mut t = BitVec::from_elem(i, true);
            let mut f = BitVec::from_elem(i, false);
            assert_eq!(i as u64, t.count_ones());
            assert_eq!(0_u64, f.count_ones());
            if i > 20 {
                t.set(10, false);
                t.set(i - 10, false);
                assert_eq!(i - 2, t.count_ones() as usize);
                f.set(10, true);
                f.set(i - 10, true);
                assert_eq!(2, f.count_ones());
            }
        }
    }

    #[test]
    fn test_count_zeros() {
        for i in 0..1000 {
            let mut tbits = BitVec::from_elem(i, true);
            let mut fbits = BitVec::from_elem(i, false);
            assert_eq!(i as u64, fbits.count_zeros());
            assert_eq!(0_u64, tbits.count_zeros());
            if i > 20 {
                fbits.set(10, true);
                fbits.set(i - 10, true);
                assert_eq!(i - 2, fbits.count_zeros() as usize);
                tbits.set(10, false);
                tbits.set(i - 10, false);
                assert_eq!(2, tbits.count_zeros());
            }
        }
    }

    #[test]
    fn test_get_mut() {
        let mut a = BitVec::from_elem(3, false);
        let mut a_bit_1 = a.get_mut(1).unwrap();
        assert!(!*a_bit_1);
        *a_bit_1 = true;
        drop(a_bit_1);
        assert!(a.eq_vec(&[false, true, false]));
    }
    #[test]
    fn test_iter_mut() {
        let mut a = BitVec::from_elem(8, false);
        a.iter_mut().enumerate().for_each(|(index, mut bit)| {
            *bit = index % 2 == 1;
        });
        assert!(a.eq_vec(&[false, true, false, true, false, true, false, true]));
    }

    #[test]
    fn test_insert_at_zero() {
        let mut v = BitVec::new();

        v.insert(0, false);
        v.insert(0, true);
        v.insert(0, false);
        v.insert(0, true);
        v.insert(0, false);
        v.insert(0, true);

        assert_eq!(v.len(), 6);
        assert_eq!(v.storage().len(), 1);
        assert!(v.eq_vec(&[true, false, true, false, true, false]));
    }

    #[test]
    fn test_insert_at_end() {
        let mut v = BitVec::new();

        v.insert(v.len(), true);
        v.insert(v.len(), false);
        v.insert(v.len(), true);
        v.insert(v.len(), false);
        v.insert(v.len(), true);
        v.insert(v.len(), false);

        assert_eq!(v.storage().len(), 1);
        assert_eq!(v.len(), 6);
        assert!(v.eq_vec(&[true, false, true, false, true, false]));
    }

    #[test]
    fn test_insert_at_block_boundaries() {
        let mut v = BitVec::from_elem(32, false);

        assert_eq!(v.storage().len(), 1);

        v.insert(31, true);

        assert_eq!(v.len(), 33);

        assert!(matches!(v.get(31), Some(true)));
        assert!(v.eq_vec(&[
            false, false, false, false, false, false, false, false, false, false, false, false,
            false, false, false, false, false, false, false, false, false, false, false, false,
            false, false, false, false, false, false, false, true, false
        ]));

        assert_eq!(v.storage().len(), 2);
    }

    #[test]
    fn test_insert_at_block_boundaries_1() {
        let mut v = BitVec::from_elem(64, false);

        assert_eq!(v.storage().len(), 2);

        v.insert(63, true);

        assert_eq!(v.len(), 65);

        assert!(matches!(v.get(63), Some(true)));
        assert!(v.eq_vec(&[
            false, false, false, false, false, false, false, false, false, false, false, false,
            false, false, false, false, false, false, false, false, false, false, false, false,
            false, false, false, false, false, false, false, false, false, false, false, false,
            false, false, false, false, false, false, false, false, false, false, false, false,
            false, false, false, false, false, false, false, false, false, false, false, false,
            false, false, false, true, false
        ]));

        assert_eq!(v.storage().len(), 3);
    }

    #[test]
    fn test_push_within_capacity_with_suffice_cap() {
        let mut v = BitVec::from_elem(16, true);

        assert!(v.push_within_capacity(false).is_ok());

        for i in 0..16 {
            assert_eq!(v.get(i), Some(true));
        }

        assert_eq!(v.get(16), Some(false));
        assert_eq!(v.len(), 17);
    }

    #[test]
    fn test_push_within_capacity_at_brink() {
        let mut v = BitVec::from_elem(31, true);

        assert!(v.push_within_capacity(false).is_ok());

        assert_eq!(v.get(31), Some(false));
        assert_eq!(v.len(), v.capacity());
        assert_eq!(v.len(), 32);

        assert_eq!(v.push_within_capacity(false), Err(false));
        assert_eq!(v.capacity(), 32);

        for i in 0..31 {
            assert_eq!(v.get(i), Some(true));
        }
        assert_eq!(v.get(31), Some(false));
    }

    #[test]
    fn test_push_within_capacity_at_brink_with_mul_blocks() {
        let mut v = BitVec::from_elem(95, true);

        assert!(v.push_within_capacity(false).is_ok());

        assert_eq!(v.get(95), Some(false));
        assert_eq!(v.len(), v.capacity());
        assert_eq!(v.len(), 96);

        assert_eq!(v.push_within_capacity(false), Err(false));
        assert_eq!(v.capacity(), 96);

        for i in 0..95 {
            assert_eq!(v.get(i), Some(true));
        }
        assert_eq!(v.get(95), Some(false));
    }

    #[test]
    fn test_push_within_capacity_storage_push() {
        let mut v = BitVec::with_capacity(64);

        for _ in 0..32 {
            v.push(true);
        }

        assert_eq!(v.len(), 32);

        assert!(v.push_within_capacity(false).is_ok());

        assert_eq!(v.len(), 33);

        for i in 0..32 {
            assert_eq!(v.get(i), Some(true));
        }
        assert_eq!(v.get(32), Some(false));
    }
}
