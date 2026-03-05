# fuzzer for bit-vec

Based on fuzzing in `smallvec`.

# fuzzing

```sh
cargo afl build --release --bin bitvec_ops --features afl && cargo afl fuzz -i in -o out target/release/bitvec_ops
```
