#![feature(test)]

extern crate test;

use bit_vec_extra::prelude::*;
use test::Bencher;

const LEN: usize = 1 << 10;

#[bench]
fn iter_proxy(bench: &mut Bencher) {
	let a = bits![1; LEN];
	bench.iter(|| a.iter().all(|b| *b));
}

#[bench]
fn iter_ref(bench: &mut Bencher) {
	let a = bits![1; LEN];
	bench.iter(|| a.iter().by_refs().all(|b| *b));
}

#[bench]
fn iter_val(bench: &mut Bencher) {
	let a = bits![1; LEN];
	bench.iter(|| a.iter().by_vals().all(|b| b));
}

#[bench]
fn iter_bools(bench: &mut Bencher) {
	let a = [true; LEN];
	bench.iter(|| a.iter().copied().all(|b| b));
}
