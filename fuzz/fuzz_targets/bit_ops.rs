//! Simple fuzzer testing all available `BitVec`, `BitSet` and `BitMatrix` operations
use bit_vec::{BitBlockOrStore, BitVec};
use bit_set::BitSet;
#[cfg(not(feature = "nanoserde"))]
use smallvec::SmallVec;

// There's no point growing too much, so try not to grow
// over this size.
const CAP_GROWTH: usize = 256;

macro_rules! next_usize {
    ($b:ident) => {
        $b.next().unwrap_or(0) as usize
    };
}

macro_rules! next_u8 {
    ($b:ident) => {
        $b.next().unwrap_or(0)
    };
}

fn black_box_bit_vec<T: BitBlockOrStore>(s: &BitVec<T>) {
    // print to work as a black_box
    print!("{}", s);
}

fn black_box_bit_set<T: BitBlockOrStore>(s: &BitSet<T>) {
    // print to work as a black_box
    print!("{}", s);
}

fn do_test<T: BitBlockOrStore>(data: &[u8]) -> BitVec<T> {
    let mut v = BitVec::<T>::new_general();

    let mut bytes = data.iter().copied();

    while let Some(op) = bytes.next() {
        match op % 23 {
            0 => {
                v = BitVec::new_general();
            }
            1 => {
                v = BitVec::with_capacity_general(next_usize!(bytes));
            }
            2 => {
                v = BitVec::from_bytes_general(&v.to_bytes()[..]);
            }
            3 => {}
            4 => {
                if v.len() < CAP_GROWTH {
                    v.push(next_u8!(bytes) < 128)
                }
            }
            5 => {
                v.pop();
            }
            6 => v.grow(next_usize!(bytes) + v.len(), next_u8!(bytes) < 128),
            7 => {
                if v.len() < CAP_GROWTH {
                    v.reserve(next_usize!(bytes))
                }
            }
            8 => {
                if v.len() < CAP_GROWTH {
                    v.reserve_exact(next_usize!(bytes))
                }
            }
            9 => v.shrink_to_fit(),
            10 => v.truncate(next_usize!(bytes)),
            11 => black_box_bit_vec(&v),
            12 => {
                if !v.is_empty() {
                    v.remove(next_usize!(bytes) % v.len());
                }
            }
            13 => {
                v.fill(false);
            }
            14 => {
                if !v.is_empty() {
                    v.remove(next_usize!(bytes) % v.len());
                }
            }
            15 => {
                let insert_pos = next_usize!(bytes) % (v.len() + 1);
                v.insert(insert_pos, next_u8!(bytes) < 128);
            }

            16 => {
                v = BitVec::from_bytes_general(&v.to_bytes()[..]);
            }

            17 => {
                v = BitVec::from_bytes_general(data);
            }

            18 => {
                if v.len() < CAP_GROWTH {
                    let mut v2 = BitVec::<T>::from_bytes_general(data);
                    v.append(&mut v2);
                }
            }

            19 => {
                if v.len() < CAP_GROWTH {
                    v.reserve(next_usize!(bytes));
                }
            }

            20 => {
                if v.len() < CAP_GROWTH {
                    v.reserve_exact(next_usize!(bytes));
                }
            }
            21 => {
                let slice = vec![next_u8!(bytes); next_usize!(bytes)];
                v = BitVec::<T>::from_bytes_general(&slice[..]);
            }
            22 => {
                v.fill(true);
            }
            _ => panic!("booo"),
        }
    }
    v
}

fn do_test_set<T: BitBlockOrStore>(data: &[u8]) -> BitSet<T> {
    let mut v = BitSet::<T>::new_general();

    let mut bytes = data.iter().copied();

    while let Some(op) = bytes.next() {
        match op % 16 {
            0 => {
                v = BitSet::new_general();
            }
            1 => {
                v = BitSet::with_capacity_general(next_usize!(bytes));
            }
            2 => {
                v = BitSet::from_bytes_general(&v.get_ref().to_bytes()[..]);
            }
            3 => {
                if v.get_ref().len() < CAP_GROWTH {
                    v.reserve_len(next_usize!(bytes))
                }
            }
            4 => {
                if v.get_ref().len() < CAP_GROWTH {
                    v.reserve_len_exact(next_usize!(bytes))
                }
            }
            5 => v.shrink_to_fit(),
            6 => v.truncate(next_usize!(bytes)),
            7 => black_box_bit_set(&v),
            8 => {
                if !v.is_empty() {
                    v.remove(next_usize!(bytes) % v.get_ref().len());
                }
            }
            9 => {
                v.reset();
            }
            10 => {
                let insert_pos = next_usize!(bytes) % (v.get_ref().len() + 1);
                v.insert(insert_pos);
            }

            11 => {
                v = BitSet::from_bytes_general(&v.get_ref().to_bytes()[..]);
            }

            12 => {
                v = BitSet::from_bytes_general(data);
            }

            13 => {
                if v.get_ref().len() < CAP_GROWTH {
                    v.reserve_len(next_usize!(bytes));
                }
            }

            14 => {
                if v.get_ref().len() < CAP_GROWTH {
                    v.reserve_len_exact(next_usize!(bytes));
                }
            }
            15 => {
                let slice = vec![next_u8!(bytes); next_usize!(bytes)];
                v = BitSet::<T>::from_bytes_general(&slice[..]);
            }
            _ => panic!("booo"),
        }
    }
    v
}

fn do_test_all(data: &[u8]) {
    do_test::<u32>(data);
    do_test::<u8>(data);
    do_test::<u16>(data);
    do_test::<u64>(data);
    #[cfg(not(feature = "nanoserde"))]
    do_test::<SmallVec<[u32; 8]>>(data);
    do_test::<Vec<u16>>(data);

    do_test_set::<u32>(data);
    do_test_set::<u8>(data);
    do_test_set::<u16>(data);
    do_test_set::<u64>(data);
    #[cfg(not(feature = "nanoserde"))]
    do_test_set::<SmallVec<[u32; 8]>>(data);
    do_test_set::<Vec<u16>>(data);
}

#[cfg(feature = "afl")]
fn main() {
    afl::fuzz!(|data| {
        // Remove the panic hook so we can actually catch panic
        // See https://github.com/rust-fuzz/afl.rs/issues/150
        std::panic::set_hook(Box::new(|_| {}));
        do_test_all(data);
    });
}

#[cfg(feature = "honggfuzz")]
fn main() {
    loop {
        honggfuzz::fuzz!(|data| {
            // Remove the panic hook so we can actually catch panic
            // See https://github.com/rust-fuzz/afl.rs/issues/150
            std::panic::set_hook(Box::new(|_| {}));
            do_test_all(data);
        });
    }
}

#[cfg(test)]
mod tests {
    fn extend_vec_from_hex(hex: &str, out: &mut Vec<u8>) {
        let mut b = 0;
        for (idx, c) in hex.as_bytes().iter().enumerate() {
            b <<= 4;
            match *c {
                b'A'..=b'F' => b |= c - b'A' + 10,
                b'a'..=b'f' => b |= c - b'a' + 10,
                b'0'..=b'9' => b |= c - b'0',
                b'\n' => {}
                b' ' => {}
                _ => panic!("Bad hex"),
            }
            if (idx & 1) == 1 {
                out.push(b);
                b = 0;
            }
        }
    }

    #[test]
    fn duplicate_crash() {
        let mut a = Vec::new();
        // paste the output of `xxd -p <crash_dump>` here and run `cargo test`
        extend_vec_from_hex(
            r#"
            787c4a1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d4a1d1d1d1d1d1d1d
            1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d1d27271d1d1d1d1d1d2727fffe
            270a610a
            "#,
            &mut a,
        );
        super::do_test_all(&a);
    }
}
