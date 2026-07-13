// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::hint::black_box;

use bit_vec::BitVec;
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use rand::{RngExt, SeedableRng};
use rand_xorshift::XorShiftRng;

const HUGE_BENCH_BITS: usize = 1 << 20;
const BENCH_BITS: usize = 1 << 14;

fn small_rng() -> XorShiftRng {
    XorShiftRng::from_rng(&mut rand::rng())
}

fn usize_small(c: &mut Criterion) {
    let mut r = small_rng();
    let mut bit_vec = 0_usize;

    c.bench_function("usize_small", |b| {
        b.iter(|| {
            for _ in 0..100 {
                bit_vec |= 1 << (r.random::<u32>() as usize % u32::BITS as usize);
            }

            black_box(&bit_vec);
        });
    });
}

fn to_bytes(c: &mut Criterion) {
    let mut bit_vec = BitVec::from_elem(BENCH_BITS, false);
    let mut r = small_rng();

    for _ in 0..BENCH_BITS / 10 {
        bit_vec.set((r.random::<u32>() as usize) % BENCH_BITS, true);
    }

    c.bench_function("to_bytes", |b| {
        b.iter(|| {
            black_box(bit_vec.to_bytes());
        });
    });
}

fn bit_set_big_fixed(c: &mut Criterion) {
    let mut r = small_rng();
    let mut bit_vec = BitVec::from_elem(BENCH_BITS, false);

    c.bench_function("bit_set_big_fixed", |b| {
        b.iter(|| {
            for _ in 0..100 {
                bit_vec.set((r.random::<u32>() as usize) % BENCH_BITS, true);
            }

            black_box(&bit_vec);
        });
    });
}

fn bit_set_big_variable(c: &mut Criterion) {
    let mut r = small_rng();
    let mut bit_vec = BitVec::from_elem(BENCH_BITS, false);

    c.bench_function("bit_set_big_variable", |b| {
        b.iter(|| {
            for _ in 0..100 {
                bit_vec.set((r.random::<u32>() as usize) % BENCH_BITS, r.random());
            }

            black_box(&bit_vec);
        });
    });
}

fn bit_set_small(c: &mut Criterion) {
    let mut r = small_rng();
    let mut bit_vec = BitVec::from_elem(u32::BITS as usize, false);

    c.bench_function("bit_set_small", |b| {
        b.iter(|| {
            for _ in 0..100 {
                bit_vec.set((r.random::<u32>() as usize) % u32::BITS as usize, true);
            }

            black_box(&bit_vec);
        });
    });
}

fn bit_get_checked_small(c: &mut Criterion) {
    let mut r = small_rng();
    let size = 200;
    let mut bit_vec = BitVec::from_elem(size, false);

    for _ in 0..20 {
        bit_vec.set((r.random::<u32>() as usize) % size, true);
    }

    let bit_vec = black_box(bit_vec);
    c.bench_function("bit_get_checked_small", |b| {
        b.iter(|| {
            for _ in 0..100 {
                black_box(bit_vec.get((r.random::<u32>() as usize) % size));
            }
        });
    });
}

fn bit_get_unchecked_small(c: &mut Criterion) {
    let mut r = small_rng();
    let size = 200;
    let mut bit_vec = BitVec::from_elem(size, false);

    for _ in 0..20 {
        bit_vec.set((r.random::<u32>() as usize) % size, true);
    }

    let bit_vec = black_box(bit_vec);
    c.bench_function("bit_get_unchecked_small", |b| {
        b.iter(|| {
            for _ in 0..100 {
                // Safety: This is just a benchmark of an unsafe fn.
                unsafe {
                    black_box(bit_vec.get_unchecked((r.random::<u32>() as usize) % size));
                }
            }
        });
    });
}

fn bit_get_unchecked_small_assume(c: &mut Criterion) {
    let mut r = small_rng();
    let size = 200;
    let mut bit_vec = BitVec::from_elem(size, false);

    for _ in 0..20 {
        bit_vec.set((r.random::<u32>() as usize) % size, true);
    }

    let bit_vec = black_box(bit_vec);
    c.bench_function("bit_get_unchecked_small_assume", |b| {
        b.iter(|| {
            for _ in 0..100 {
                // Safety: This is just a benchmark with an unsafe fn call.
                unsafe {
                    let idx = (r.random::<u32>() as usize) % size;

                    std::hint::assert_unchecked(idx < bit_vec.len());
                    black_box(bit_vec.get(idx));
                }
            }
        });
    });
}

fn bit_vec_big_or(c: &mut Criterion) {
    let mut b1 = BitVec::from_elem(BENCH_BITS, false);
    let b2 = BitVec::from_elem(BENCH_BITS, false);

    c.bench_function("bit_vec_big_or", |b| b.iter(|| b1.or(&b2)));
}

fn bit_vec_big_xnor(c: &mut Criterion) {
    let mut b1 = BitVec::from_elem(BENCH_BITS, false);
    let b2 = BitVec::from_elem(BENCH_BITS, false);

    c.bench_function("bit_vec_big_xnor", |b| b.iter(|| b1.xnor(&b2)));
}

fn bit_vec_big_negate_xor(c: &mut Criterion) {
    let mut b1 = BitVec::from_elem(BENCH_BITS, false);
    let b2 = BitVec::from_elem(BENCH_BITS, false);

    c.bench_function("bit_vec_big_negate_xor", |b| {
        b.iter(|| {
            let res = b1.xor(&b2);
            b1.negate();
            res
        })
    });
}

fn bit_vec_huge_xnor(c: &mut Criterion) {
    let mut b1 = BitVec::from_elem(HUGE_BENCH_BITS, false);
    let b2 = BitVec::from_elem(HUGE_BENCH_BITS, false);

    c.bench_function("bit_vec_huge_xnor", |b| b.iter(|| b1.xnor(&b2)));
}

fn bit_vec_huge_negate_xor(c: &mut Criterion) {
    let mut b1 = BitVec::from_elem(HUGE_BENCH_BITS, false);
    let b2 = BitVec::from_elem(HUGE_BENCH_BITS, false);

    c.bench_function("bit_vec_huge_negate_xor", |b| {
        b.iter(|| {
            let res = b1.xor(&b2);
            b1.negate();
            res
        })
    });
}

fn bit_vec_small_iter(c: &mut Criterion) {
    let bit_vec = BitVec::from_elem(u32::BITS as usize, false);

    c.bench_function("bit_vec_small_iter", |b| {
        b.iter(|| {
            let mut sum = 0;
            for _ in 0..10 {
                for pres in &bit_vec {
                    sum += pres as usize;
                }
            }
            sum
        })
    });
}

fn bit_vec_big_iter(c: &mut Criterion) {
    let bit_vec = BitVec::from_elem(BENCH_BITS, false);

    c.bench_function("bit_vec_big_iter", |b| {
        b.iter(|| {
            let mut sum = 0;
            for pres in &bit_vec {
                sum += pres as usize;
            }
            sum
        })
    });
}

fn from_elem(c: &mut Criterion) {
    let cap = black_box(BENCH_BITS);
    let bit = black_box(true);
    let mut group = c.benchmark_group("from_elem");

    group.throughput(Throughput::Bytes(cap as u64 / 8));
    group.bench_function("from_elem", |b| {
        b.iter(|| {
            // create a BitVec and popcount it
            BitVec::from_elem(cap, bit)
                .blocks()
                .fold(0, |acc, b| acc + b.count_ones())
        });
    });
    group.finish();
}

fn eratosthenes(c: &mut Criterion) {
    let mut primes = vec![];

    c.bench_function("eratosthenes", |b| {
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
    });
}

fn eratosthenes_set_all(c: &mut Criterion) {
    let mut primes = vec![];
    let mut sieve = BitVec::from_elem(1 << 16, true);

    c.bench_function("eratosthenes_set_all", |b| {
        b.iter(|| {
            primes.clear();

            black_box(&mut sieve);
            sieve.fill(true);
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
    });
}

fn iter_skip(c: &mut Criterion) {
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

    c.bench_function("iter_skip", |b| {
        b.iter(|| {
            black_box(&mut tbl);
            // start is large relative to end-start, so before Iterator::nth was
            // implemented for bitvec this would have been much slower
            black_box(tbl.iter().skip(start).filter(|&v| v).count());
        });
    });
}

criterion_group!(
    benches,
    usize_small,
    to_bytes,
    bit_set_big_fixed,
    bit_set_big_variable,
    bit_set_small,
    bit_get_checked_small,
    bit_get_unchecked_small,
    bit_get_unchecked_small_assume,
    bit_vec_big_or,
    bit_vec_big_xnor,
    bit_vec_big_negate_xor,
    bit_vec_huge_xnor,
    bit_vec_huge_negate_xor,
    bit_vec_small_iter,
    bit_vec_big_iter,
    from_elem,
    eratosthenes,
    eratosthenes_set_all,
    iter_skip,
);
criterion_main!(benches);
