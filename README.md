<div align="center">
  <h1>bit-vec</h1>
  <p>
    <strong>A compact vector of bits.</strong>
  </p>
  <p>

[![crates.io][crates.io shield]][crates.io link]
[![Documentation][docs.rs badge]][docs.rs link]
![Rust CI][github ci badge]
[![rustc 1.0+]][Rust 1.0]
[![serde_derive: rustc 1.31+]][Rust 1.31]
<br />
<br />
[![Dependency Status][deps.rs status]][deps.rs link]
[![Download Status][download count shield]][crates.io link]

  </p>
</div>

[crates.io shield]: https://img.shields.io/crates/v/bit-vec?label=latest
[crates.io link]: https://crates.io/crates/bit-vec
[docs.rs badge]: https://docs.rs/bit-vec/badge.svg?version=0.6.3
[docs.rs link]: https://docs.rs/bit-vec/0.6.3/bit_vec/
[github ci badge]: https://github.com/contain-rs/linked-hash-map/workflows/Rust/badge.svg?branch=master
[rustc 1.0+]: https://img.shields.io/badge/rustc-1.0%2B-blue.svg
[serde_derive: rustc 1.31+]: https://img.shields.io/badge/serde_derive-rustc_1.31+-lightgray.svg
[Rust 1.0]: https://blog.rust-lang.org/2015/05/15/Rust-1.0.html
[Rust 1.31]: https://blog.rust-lang.org/2018/12/06/Rust-1.31-and-rust-2018.html
[deps.rs status]: https://deps.rs/crate/bit-vec/0.6.3/status.svg
[deps.rs link]: https://deps.rs/crate/bit-vec/0.6.3
[shields.io download count]: https://img.shields.io/crates/d/bit-vec.svg

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

<!-- cargo-rdme -->
