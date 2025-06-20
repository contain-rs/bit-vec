// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(stable_features)]
#![feature(test)]
#![feature(hint_assert_unchecked)]

extern crate bit_vec;
extern crate rand;
extern crate rand_xorshift;
extern crate test;

use bit_vec::BitVec;
use rand::{Rng, RngCore, SeedableRng};
use rand_xorshift::XorShiftRng;
use test::{black_box, Bencher};

const HUGE_BENCH_BITS: usize = 1 << 20;
const BENCH_BITS: usize = 1 << 14;
const U32_BITS: usize = 32;

fn small_rng() -> XorShiftRng {
    XorShiftRng::from_os_rng()
}

#[bench]
fn bench_usize_small(b: &mut Bencher) {
    let mut r = small_rng();
    let mut bit_vec = 0_usize;
    b.iter(|| {
        for _ in 0..100 {
            bit_vec |= 1 << ((r.next_u32() as usize) % U32_BITS);
        }
        black_box(&bit_vec);
    });
}

#[bench]
fn bench_bit_set_big_fixed(b: &mut Bencher) {
    let mut r = small_rng();
    let mut bit_vec = BitVec::from_elem(BENCH_BITS, false);
    b.iter(|| {
        for _ in 0..100 {
            bit_vec.set((r.next_u32() as usize) % BENCH_BITS, true);
        }
        black_box(&bit_vec);
    });
}

#[bench]
fn bench_bit_set_big_variable(b: &mut Bencher) {
    let mut r = small_rng();
    let mut bit_vec = BitVec::from_elem(BENCH_BITS, false);
    b.iter(|| {
        for _ in 0..100 {
            bit_vec.set((r.next_u32() as usize) % BENCH_BITS, r.random());
        }
        black_box(&bit_vec);
    });
}

#[bench]
fn bench_bit_set_small(b: &mut Bencher) {
    let mut r = small_rng();
    let mut bit_vec = BitVec::from_elem(U32_BITS, false);
    b.iter(|| {
        for _ in 0..100 {
            bit_vec.set((r.next_u32() as usize) % U32_BITS, true);
        }
        black_box(&bit_vec);
    });
}

#[bench]
fn bench_bit_get_checked_small(b: &mut Bencher) {
    let mut r = small_rng();
    let size = 200;
    let mut bit_vec = BitVec::from_elem(size, false);
    for _ in 0..20 {
        bit_vec.set((r.next_u32() as usize) % size, true);
    }
    let bit_vec = black_box(bit_vec);
    b.iter(|| {
        for _ in 0..100 {
            black_box(bit_vec.get((r.next_u32() as usize) % size));
        }
    });
}

#[bench]
fn bench_bit_get_unchecked_small(b: &mut Bencher) {
    let mut r = small_rng();
    let size = 200;
    let mut bit_vec = BitVec::from_elem(size, false);
    for _ in 0..20 {
        bit_vec.set((r.next_u32() as usize) % size, true);
    }
    let bit_vec = black_box(bit_vec);
    b.iter(|| {
        for _ in 0..100 {
            unsafe {
                black_box(bit_vec.get_unchecked((r.next_u32() as usize) % size));
            }
        }
    });
}

#[bench]
fn bench_bit_get_unchecked_small_assume(b: &mut Bencher) {
    let mut r = small_rng();
    let size = 200;
    let mut bit_vec = BitVec::from_elem(size, false);
    for _ in 0..20 {
        bit_vec.set((r.next_u32() as usize) % size, true);
    }
    let bit_vec = black_box(bit_vec);
    b.iter(|| {
        for _ in 0..100 {
            unsafe {
                let idx = (r.next_u32() as usize) % size;
                ::std::hint::assert_unchecked(!(idx >= bit_vec.len()));
                black_box(bit_vec.get(idx));
            }
        }
    });
}

#[bench]
fn bench_bit_vec_big_or(b: &mut Bencher) {
    let mut b1 = BitVec::from_elem(BENCH_BITS, false);
    let b2 = BitVec::from_elem(BENCH_BITS, false);
    b.iter(|| b1.or(&b2))
}

#[bench]
fn bench_bit_vec_big_xnor(b: &mut Bencher) {
    let mut b1 = BitVec::from_elem(BENCH_BITS, false);
    let b2 = BitVec::from_elem(BENCH_BITS, false);
    b.iter(|| b1.xnor(&b2))
}

#[bench]
fn bench_bit_vec_big_negate_xor(b: &mut Bencher) {
    let mut b1 = BitVec::from_elem(BENCH_BITS, false);
    let b2 = BitVec::from_elem(BENCH_BITS, false);
    b.iter(|| {
        let res = b1.xor(&b2);
        b1.negate();
        res
    })
}

#[bench]
fn bench_bit_vec_huge_xnor(b: &mut Bencher) {
    let mut b1 = BitVec::from_elem(HUGE_BENCH_BITS, false);
    let b2 = BitVec::from_elem(HUGE_BENCH_BITS, false);
    b.iter(|| b1.xnor(&b2))
}

#[bench]
fn bench_bit_vec_huge_negate_xor(b: &mut Bencher) {
    let mut b1 = BitVec::from_elem(HUGE_BENCH_BITS, false);
    let b2 = BitVec::from_elem(HUGE_BENCH_BITS, false);
    b.iter(|| {
        let res = b1.xor(&b2);
        b1.negate();
        res
    })
}

#[bench]
fn bench_bit_vec_small_iter(b: &mut Bencher) {
    let bit_vec = BitVec::from_elem(U32_BITS, false);
    b.iter(|| {
        let mut sum = 0;
        for _ in 0..10 {
            for pres in &bit_vec {
                sum += pres as usize;
            }
        }
        sum
    })
}

#[bench]
fn bench_bit_vec_big_iter(b: &mut Bencher) {
    let bit_vec = BitVec::from_elem(BENCH_BITS, false);
    b.iter(|| {
        let mut sum = 0;
        for pres in &bit_vec {
            sum += pres as usize;
        }
        sum
    })
}

#[bench]
fn bench_from_elem(b: &mut Bencher) {
    let cap = black_box(BENCH_BITS);
    let bit = black_box(true);
    b.iter(|| {
        // create a BitVec and popcount it
        BitVec::from_elem(cap, bit)
            .blocks()
            .fold(0, |acc, b| acc + b.count_ones())
    });
    b.bytes = cap as u64 / 8;
}

#[bench]
fn bench_erathostenes(b: &mut test::Bencher) {
    let mut primes = vec![];
    b.iter(|| {
        primes.clear();
        let mut sieve = BitVec::from_elem(1 << 16, true);
        black_box(&mut sieve);
        let mut i = 2;
        while i < sieve.len() {
            if sieve[i] {
                primes.push(i);
            }
            let mut j = i;
            while j < sieve.len() {
                sieve.set(j, false);
                j += i;
            }
            i += 1;
        }
        black_box(&mut sieve);
    });
}

#[bench]
fn bench_erathostenes_set_all(b: &mut test::Bencher) {
    let mut primes = vec![];
    let mut sieve = BitVec::from_elem(1 << 16, true);
    b.iter(|| {
        primes.clear();
        black_box(&mut sieve);
        sieve.set_all();
        black_box(&mut sieve);
        let mut i = 2;
        while i < sieve.len() {
            if sieve[i] {
                primes.push(i);
            }
            let mut j = i;
            while j < sieve.len() {
                sieve.set(j, false);
                j += i;
            }
            i += 1;
        }
        black_box(&mut sieve);
    });
}

#[bench]
fn bench_iter_skip(b: &mut test::Bencher) {
    let start = 3 << 20;
    let p = 16411;
    let g = 9749; // 9749 is a primitive root modulo 16411, so we can generate numbers mod p in a seemingly random order
    let end = start + p;
    let mut tbl = BitVec::from_elem(end, false);
    let mut r = g;
    for i in start..end {
        tbl.set(i, r & 1 != 0);
        r = r * g % p;
    }
    b.iter(|| {
        black_box(&mut tbl);
        // start is large relative to end-start, so before Iterator::nth was implemented for bitvec this would
        // have been much slower
        black_box(tbl.iter().skip(start).filter(|&v| v).count());
    });
}
