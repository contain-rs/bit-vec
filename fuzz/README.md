# fuzzer for bit-vec, bit-set and bit-matrix

Based on fuzzing in `smallvec`.

# fuzzing

```sh
cargo afl build --release --bin bit_ops --features afl && cargo afl fuzz -i in -o out target/release/bit_ops
```
