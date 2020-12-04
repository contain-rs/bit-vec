<div align="center">
  <h1>bit-vec</h1>
  <p>
    <strong>A vector of bits.</strong>
  </p>
  <p>

[![crates.io](https://img.shields.io/crates/v/bit-vec?label=latest)](https://crates.io/crates/bit-vec)
[![Documentation](https://docs.rs/bit-vec/badge.svg?version=0.6.3)](https://docs.rs/bit-vec/0.6.3/bit_vec/)
[![Build Status](https://travis-ci.org/contain-rs/bit-vec.svg?branch=master)](https://travis-ci.org/contain-rs/bit-vec)
[![rustc 1.0+]][Rust 1.0]
[![serde_derive: rustc 1.31+]][Rust 1.31]
<br />
<br />
[![Dependency Status](https://deps.rs/crate/bit-vec/0.6.3/status.svg)](https://deps.rs/crate/bit-vec/0.6.3)
[![Download Status](https://img.shields.io/crates/d/bit-vec.svg)](https://crates.io/crates/bit-vec)

  </p>
</div>

[rustc 1.0+]: https://img.shields.io/badge/rustc-1.0%2B-blue.svg
[serde_derive: rustc 1.31+]: https://img.shields.io/badge/serde_derive-rustc_1.31+-lightgray.svg
[Rust 1.0]: https://blog.rust-lang.org/2015/05/15/Rust-1.0.html
[Rust 1.31]: https://blog.rust-lang.org/2018/12/06/Rust-1.31-and-rust-2018.html

## Usage

Add this to your Cargo.toml:

```toml
[dependencies]
bit-vec = "0.6"
```

and this to your crate root:

```rust
extern crate bit_vec;
```

If you want [serde](https://github.com/serde-rs/serde) support, include the feature like this:

```toml
[dependencies]
bit-vec = { version = "0.6", features = ["serde"] }
```

If you want to use bit-vec in a program that has `#![no_std]`, just drop default features:

```toml
[dependencies]
bit-vec = { version = "0.6", default-features = false }
```
