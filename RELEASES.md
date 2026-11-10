Version 0.11.0 (TO BE RELEASED)
==========================

<a id="v0.11.0"></a>


Version 0.10.2
==========================

<a id="v0.10.2"></a>

- fixed `split_off` bug meaning it returned an invalid value
- fixed `xnor` and `nor` bug meaning it returned an invalid value

Version 0.10.1
==========================

<a id="v0.10.1"></a>

- indexing into a `BitVec` has simpler codegen (https://github.com/contain-rs/bit-vec/pull/153)

Version 0.10.0 (VULNERABILITY FIX)
==========================

<a id="v0.10.0"></a>

- fixed a soundness issue where `deserialize`ing from an untrusted source
  could make `fn get_unchecked` perform undefined behavior, and could make
  other functions return incorrect results
- removed nanoserde support
- moved all the crates into a workspace

Version 0.9.1
==========================

<a id="v0.9.1"></a>

- Fixed issue with the serde feature

Version 0.9.0
==========================

<a id="v0.9.0"></a>

- Minimal Supported Rust Version is now 1.82
- `fn remove` is implemented
- `fn fill` is implemented
- `fn remove_all` is implemented
- `fn clear` is **deprecated**. Please use `.fill(false)` instead!
- `fn push_within_capacity` is implemented
- bug fix: `.skip(n)` on our iterators is now O(1) instead of O(n) time
- `fn to_bytes` is optimized with a lookup table
- some clippy lints are enabled
- nanoserde version is now 0.2

Version 0.8.0 (2024-07-16)
==========================

<a id="v0.8.0"></a>

- `fn insert` is implemented
- `impl Display` is implemented
- `impl Debug` has different output
