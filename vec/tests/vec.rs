#[cfg(test)]
#[generic_tests::define]
mod tests {
    #![allow(clippy::shadow_reuse)]
    #![allow(clippy::shadow_same)]
    #![allow(clippy::shadow_unrelated)]
    #![allow(clippy::extra_unused_type_parameters)]

    use bit_vec::{BitBlockOrStore, BitVec, Iter};

    // This is stupid, but I want to differentiate from a "random" 32
    const U32_BITS: usize = 32;

    #[test]
    fn test_display_output<S: BitBlockOrStore>() {
        assert_eq!(format!("{}", BitVec::<S>::new_general()), "");
        assert_eq!(format!("{}", BitVec::<S>::from_elem_general(1, true)), "1");
        assert_eq!(
            format!("{}", BitVec::<S>::from_elem_general(8, false)),
            "00000000"
        )
    }

    #[test]
    fn test_debug_output<S: BitBlockOrStore>() {
        assert_eq!(
            format!("{:?}", BitVec::<S>::new_general()),
            "BitVec { storage: \"\", nbits: 0 }"
        );
        assert_eq!(
            format!("{:?}", BitVec::<S>::from_elem_general(1, true)),
            "BitVec { storage: \"1\", nbits: 1 }"
        );
        assert_eq!(
            format!("{:?}", BitVec::<S>::from_elem_general(8, false)),
            "BitVec { storage: \"00000000\", nbits: 8 }"
        );
        assert_eq!(
            format!("{:?}", BitVec::<S>::from_elem_general(33, true)).replace(" ", ""),
            "BitVec{storage:\"111111111111111111111111111111111\",nbits:33}"
        );
        assert_eq!(
            format!(
                "{:?}",
                BitVec::<S>::from_bytes_general(&[
                    0b111, 0b000, 0b1110, 0b0001, 0b11111111, 0b00000000
                ])
            )
            .replace(" ", ""),
            "BitVec{storage:\"000001110000000000001110000000011111111100000000\",nbits:48}"
        )
    }

    #[test]
    fn test_0_elements<S: BitBlockOrStore>() {
        let act = BitVec::<S>::new_general();
        let exp = Vec::new();
        assert!(act.eq_vec(&exp));
        assert!(act.none() && act.all());
    }

    #[test]
    fn test_1_element<S: BitBlockOrStore>() {
        let mut act = BitVec::<S>::from_elem_general(1, false);
        assert!(act.eq_vec(&[false]));
        assert!(act.none() && !act.all());
        act = BitVec::<S>::from_elem_general(1, true);
        assert!(act.eq_vec(&[true]));
        assert!(!act.none() && act.all());
    }

    #[test]
    fn test_2_elements<S: BitBlockOrStore>() {
        let mut b = BitVec::<S>::from_elem_general(2, false);
        b.set(0, true);
        b.set(1, false);
        assert_eq!(format!("{}", b), "10");
        assert!(!b.none() && !b.all());
    }

    #[test]
    fn test_10_elements<S: BitBlockOrStore>() {
        // all 0

        let mut act = BitVec::<S>::from_elem_general(10, false);
        assert!(
            (act.eq_vec(&[false, false, false, false, false, false, false, false, false, false]))
        );
        assert!(act.none() && !act.all());
        // all 1

        act = BitVec::<S>::from_elem_general(10, true);
        assert!((act.eq_vec(&[true, true, true, true, true, true, true, true, true, true])));
        assert!(!act.none() && act.all());
        // mixed

        act = BitVec::<S>::from_elem_general(10, false);
        act.set(0, true);
        act.set(1, true);
        act.set(2, true);
        act.set(3, true);
        act.set(4, true);
        assert!((act.eq_vec(&[true, true, true, true, true, false, false, false, false, false])));
        assert!(!act.none() && !act.all());
        // mixed

        act = BitVec::<S>::from_elem_general(10, false);
        act.set(5, true);
        act.set(6, true);
        act.set(7, true);
        act.set(8, true);
        act.set(9, true);
        assert!((act.eq_vec(&[false, false, false, false, false, true, true, true, true, true])));
        assert!(!act.none() && !act.all());
        // mixed

        act = BitVec::<S>::from_elem_general(10, false);
        act.set(0, true);
        act.set(3, true);
        act.set(6, true);
        act.set(9, true);
        assert!((act.eq_vec(&[true, false, false, true, false, false, true, false, false, true])));
        assert!(!act.none() && !act.all());
    }

    #[test]
    fn test_31_elements<S: BitBlockOrStore>() {
        // all 0

        let mut act = BitVec::<S>::from_elem_general(31, false);
        assert!(act.eq_vec(&[
            false, false, false, false, false, false, false, false, false, false, false, false,
            false, false, false, false, false, false, false, false, false, false, false, false,
            false, false, false, false, false, false, false
        ]));
        assert!(act.none() && !act.all());
        // all 1

        act = BitVec::<S>::from_elem_general(31, true);
        assert!(act.eq_vec(&[
            true, true, true, true, true, true, true, true, true, true, true, true, true, true,
            true, true, true, true, true, true, true, true, true, true, true, true, true, true,
            true, true, true
        ]));
        assert!(!act.none() && act.all());
        // mixed

        act = BitVec::<S>::from_elem_general(31, false);
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

        act = BitVec::<S>::from_elem_general(31, false);
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

        act = BitVec::<S>::from_elem_general(31, false);
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

        act = BitVec::<S>::from_elem_general(31, false);
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
    fn test_32_elements<S: BitBlockOrStore>() {
        // all 0

        let mut act = BitVec::<S>::from_elem_general(32, false);
        assert!(act.eq_vec(&[
            false, false, false, false, false, false, false, false, false, false, false, false,
            false, false, false, false, false, false, false, false, false, false, false, false,
            false, false, false, false, false, false, false, false
        ]));
        assert!(act.none() && !act.all());
        // all 1

        act = BitVec::<S>::from_elem_general(32, true);
        assert!(act.eq_vec(&[
            true, true, true, true, true, true, true, true, true, true, true, true, true, true,
            true, true, true, true, true, true, true, true, true, true, true, true, true, true,
            true, true, true, true
        ]));
        assert!(!act.none() && act.all());
        // mixed

        act = BitVec::<S>::from_elem_general(32, false);
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

        act = BitVec::<S>::from_elem_general(32, false);
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

        act = BitVec::<S>::from_elem_general(32, false);
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

        act = BitVec::<S>::from_elem_general(32, false);
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
    fn test_33_elements<S: BitBlockOrStore>() {
        // all 0

        let mut act = BitVec::<S>::from_elem_general(33, false);
        assert!(act.eq_vec(&[
            false, false, false, false, false, false, false, false, false, false, false, false,
            false, false, false, false, false, false, false, false, false, false, false, false,
            false, false, false, false, false, false, false, false, false
        ]));
        assert!(act.none() && !act.all());
        // all 1

        act = BitVec::<S>::from_elem_general(33, true);
        assert!(act.eq_vec(&[
            true, true, true, true, true, true, true, true, true, true, true, true, true, true,
            true, true, true, true, true, true, true, true, true, true, true, true, true, true,
            true, true, true, true, true
        ]));
        assert!(!act.none() && act.all());
        // mixed

        act = BitVec::<S>::from_elem_general(33, false);
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

        act = BitVec::<S>::from_elem_general(33, false);
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

        act = BitVec::<S>::from_elem_general(33, false);
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

        act = BitVec::<S>::from_elem_general(33, false);
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
    fn test_equal_differing_sizes<S: BitBlockOrStore>() {
        let v0 = BitVec::<S>::from_elem_general(10, false);
        let v1 = BitVec::<S>::from_elem_general(11, false);
        assert_ne!(v0, v1);
    }

    #[test]
    fn test_equal_greatly_differing_sizes<S: BitBlockOrStore>() {
        let v0 = BitVec::<S>::from_elem_general(10, false);
        let v1 = BitVec::<S>::from_elem_general(110, false);
        assert_ne!(v0, v1);
    }

    #[test]
    fn test_equal_sneaky_small<S: BitBlockOrStore>() {
        let mut a = BitVec::<S>::from_elem_general(1, false);
        a.set(0, true);

        let mut b = BitVec::<S>::from_elem_general(1, true);
        b.set(0, true);

        assert_eq!(a, b);
    }

    #[test]
    fn test_equal_sneaky_big<S: BitBlockOrStore>() {
        let mut a = BitVec::<S>::from_elem_general(100, false);
        for i in 0..100 {
            a.set(i, true);
        }

        let mut b = BitVec::<S>::from_elem_general(100, true);
        for i in 0..100 {
            b.set(i, true);
        }

        assert_eq!(a, b);
    }

    #[test]
    fn test_from_bytes<S: BitBlockOrStore>() {
        let bit_vec = BitVec::<S>::from_bytes_general(&[0b10110110, 0b00000000, 0b11111111]);
        let str = concat!("10110110", "00000000", "11111111");
        assert_eq!(format!("{}", bit_vec), str);
    }

    #[test]
    fn test_to_bytes<S: BitBlockOrStore>() {
        let mut bv = BitVec::<S>::from_elem_general(3, true);
        bv.set(1, false);
        assert_eq!(bv.to_bytes(), [0b10100000]);

        let mut bv = BitVec::<S>::from_elem_general(9, false);
        bv.set(2, true);
        bv.set(8, true);
        assert_eq!(bv.to_bytes(), [0b00100000, 0b10000000]);
    }

    #[test]
    fn test_from_bools<S: BitBlockOrStore>() {
        let bools = [true, false, true, true];
        let bit_vec: BitVec = bools.iter().copied().collect();
        assert_eq!(format!("{}", bit_vec), "1011");
    }

    #[test]
    fn test_to_bools<S: BitBlockOrStore>() {
        let bools = vec![false, false, true, false, false, true, true, false];
        assert_eq!(
            BitVec::<S>::from_bytes_general(&[0b00100110])
                .iter()
                .collect::<Vec<bool>>(),
            bools
        );
    }

    #[test]
    fn test_bit_vec_iterator<S: BitBlockOrStore>() {
        let bools = vec![true, false, true, true];
        let bit_vec: BitVec = bools.iter().copied().collect();

        assert_eq!(bit_vec.iter().collect::<Vec<bool>>(), bools);

        let long: Vec<_> = (0..10000).map(|i| i % 2 == 0).collect();
        let bit_vec: BitVec = long.iter().copied().collect();
        assert_eq!(bit_vec.iter().collect::<Vec<bool>>(), long)
    }

    #[test]
    fn test_small_difference<S: BitBlockOrStore>() {
        let mut b1 = BitVec::<S>::from_elem_general(3, false);
        let mut b2 = BitVec::<S>::from_elem_general(3, false);
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
    fn test_big_difference<S: BitBlockOrStore>() {
        let mut b1 = BitVec::<S>::from_elem_general(100, false);
        let mut b2 = BitVec::<S>::from_elem_general(100, false);
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
    fn test_small_xor<S: BitBlockOrStore>() {
        let mut a = BitVec::<S>::from_bytes_general(&[0b0011]);
        let b = BitVec::<S>::from_bytes_general(&[0b0101]);
        let c = BitVec::<S>::from_bytes_general(&[0b0110]);
        assert!(a.xor(&b));
        assert_eq!(a, c);
    }

    #[test]
    fn test_small_xnor<S: BitBlockOrStore>() {
        let mut a = BitVec::<S>::from_bytes_general(&[0b0011]);
        let b = BitVec::<S>::from_bytes_general(&[0b1111_0101]);
        let c = BitVec::<S>::from_bytes_general(&[0b1001]);
        assert!(a.xnor(&b));
        assert_eq!(a, c);
    }

    #[test]
    fn test_small_nand<S: BitBlockOrStore>() {
        let mut a = BitVec::<S>::from_bytes_general(&[0b1111_0011]);
        let b = BitVec::<S>::from_bytes_general(&[0b1111_0101]);
        let c = BitVec::<S>::from_bytes_general(&[0b1110]);
        assert!(a.nand(&b));
        assert_eq!(a, c);
    }

    #[test]
    fn test_small_nor<S: BitBlockOrStore>() {
        let mut a = BitVec::<S>::from_bytes_general(&[0b0011]);
        let b = BitVec::<S>::from_bytes_general(&[0b1111_0101]);
        let c = BitVec::<S>::from_bytes_general(&[0b1000]);
        assert!(a.nor(&b));
        assert_eq!(a, c);
    }

    #[test]
    fn test_big_xor<S: BitBlockOrStore>() {
        let mut a = BitVec::<S>::from_bytes_general(&[
            // 88 bits
            0, 0, 0b00010100, 0, 0, 0, 0, 0b00110100, 0, 0, 0,
        ]);
        let b = BitVec::<S>::from_bytes_general(&[
            // 88 bits
            0, 0, 0b00010100, 0, 0, 0, 0, 0, 0, 0, 0b00110100,
        ]);
        let c = BitVec::<S>::from_bytes_general(&[
            // 88 bits
            0, 0, 0, 0, 0, 0, 0, 0b00110100, 0, 0, 0b00110100,
        ]);
        assert!(a.xor(&b));
        assert_eq!(a, c);
    }

    #[test]
    fn test_big_xnor<S: BitBlockOrStore>() {
        let mut a = BitVec::<S>::from_bytes_general(&[
            // 88 bits
            0, 0, 0b00010100, 0, 0, 0, 0, 0b00110100, 0, 0, 0,
        ]);
        let b = BitVec::<S>::from_bytes_general(&[
            // 88 bits
            0, 0, 0b00010100, 0, 0, 0, 0, 0, 0, 0, 0b00110100,
        ]);
        let c = BitVec::<S>::from_bytes_general(&[
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
    fn test_small_fill<S: BitBlockOrStore>() {
        let mut b = BitVec::<S>::from_elem_general(14, true);
        assert!(!b.none() && b.all());
        b.fill(false);
        assert!(b.none() && !b.all());
        b.fill(true);
        assert!(!b.none() && b.all());
    }

    #[test]
    fn test_big_fill<S: BitBlockOrStore>() {
        let mut b = BitVec::<S>::from_elem_general(140, true);
        assert!(!b.none() && b.all());
        b.fill(false);
        assert!(b.none() && !b.all());
        b.fill(true);
        assert!(!b.none() && b.all());
    }

    #[test]
    fn test_bit_vec_lt<S: BitBlockOrStore>() {
        let mut a = BitVec::<S>::from_elem_general(5, false);
        let mut b = BitVec::<S>::from_elem_general(5, false);

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
    fn test_ord<S: BitBlockOrStore>() {
        let mut a = BitVec::<S>::from_elem_general(5, false);
        let mut b = BitVec::<S>::from_elem_general(5, false);

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
    fn test_small_bit_vec_tests<S: BitBlockOrStore>() {
        let v = BitVec::<S>::from_bytes_general(&[0]);
        assert!(!v.all());
        assert!(!v.any());
        assert!(v.none());

        let v = BitVec::<S>::from_bytes_general(&[0b00010100]);
        assert!(!v.all());
        assert!(v.any());
        assert!(!v.none());

        let v = BitVec::<S>::from_bytes_general(&[0xFF]);
        assert!(v.all());
        assert!(v.any());
        assert!(!v.none());
    }

    #[test]
    fn test_big_bit_vec_tests<S: BitBlockOrStore>() {
        let v = BitVec::<S>::from_bytes_general(&[
            // 88 bits
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]);
        assert!(!v.all());
        assert!(!v.any());
        assert!(v.none());

        let v = BitVec::<S>::from_bytes_general(&[
            // 88 bits
            0, 0, 0b00010100, 0, 0, 0, 0, 0b00110100, 0, 0, 0,
        ]);
        assert!(!v.all());
        assert!(v.any());
        assert!(!v.none());

        let v = BitVec::<S>::from_bytes_general(&[
            // 88 bits
            0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        ]);
        assert!(v.all());
        assert!(v.any());
        assert!(!v.none());
    }

    #[test]
    fn test_bit_vec_push_pop<S: BitBlockOrStore>() {
        let mut s = BitVec::<S>::from_elem_general(5 * U32_BITS - 2, false);
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
    fn test_bit_vec_truncate<S: BitBlockOrStore>() {
        let mut s = BitVec::<S>::from_elem_general(5 * U32_BITS, true);

        assert_eq!(s, BitVec::<S>::from_elem_general(5 * U32_BITS, true));
        assert_eq!(s.len(), 5 * U32_BITS);
        s.truncate(4 * U32_BITS);
        assert_eq!(s, BitVec::<S>::from_elem_general(4 * U32_BITS, true));
        assert_eq!(s.len(), 4 * U32_BITS);
        // Truncating to a size > s.len() should be a noop
        s.truncate(5 * U32_BITS);
        assert_eq!(s, BitVec::<S>::from_elem_general(4 * U32_BITS, true));
        assert_eq!(s.len(), 4 * U32_BITS);
        s.truncate(3 * U32_BITS - 10);
        assert_eq!(s, BitVec::<S>::from_elem_general(3 * U32_BITS - 10, true));
        assert_eq!(s.len(), 3 * U32_BITS - 10);
        s.truncate(0);
        assert_eq!(s, BitVec::<S>::from_elem_general(0, true));
        assert_eq!(s.len(), 0);
    }

    #[test]
    fn test_bit_vec_reserve<S: BitBlockOrStore>() {
        let mut s = BitVec::<S>::from_elem_general(5 * U32_BITS, true);
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
    fn test_bit_vec_grow<S: BitBlockOrStore>() {
        let mut bit_vec = BitVec::<S>::from_bytes_general(&[0b10110110, 0b00000000, 0b10101010]);
        bit_vec.grow(32, true);
        assert_eq!(
            bit_vec,
            BitVec::<S>::from_bytes_general(&[
                0b10110110, 0b00000000, 0b10101010, 0xFF, 0xFF, 0xFF, 0xFF
            ])
        );
        bit_vec.grow(64, false);
        assert_eq!(
            bit_vec,
            BitVec::<S>::from_bytes_general(&[
                0b10110110, 0b00000000, 0b10101010, 0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0, 0, 0, 0, 0
            ])
        );
        bit_vec.grow(16, true);
        assert_eq!(
            bit_vec,
            BitVec::<S>::from_bytes_general(&[
                0b10110110, 0b00000000, 0b10101010, 0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0, 0, 0, 0, 0,
                0xFF, 0xFF
            ])
        );
    }

    #[test]
    fn test_bit_vec_extend<S: BitBlockOrStore>() {
        let mut bit_vec = BitVec::<S>::from_bytes_general(&[0b10110110, 0b00000000, 0b11111111]);
        let ext = BitVec::<S>::from_bytes_general(&[0b01001001, 0b10010010, 0b10111101]);
        bit_vec.extend(ext.iter());
        assert_eq!(
            bit_vec,
            BitVec::<S>::from_bytes_general(&[
                0b10110110, 0b00000000, 0b11111111, 0b01001001, 0b10010010, 0b10111101
            ])
        );
    }

    #[test]
    fn test_bit_vec_append<S: BitBlockOrStore>() {
        // Append to BitVec that holds a multiple of U32_BITS bits
        let mut a =
            BitVec::<S>::from_bytes_general(&[0b10100000, 0b00010010, 0b10010010, 0b00110011]);
        let mut b = BitVec::<S>::new_general();
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
        let mut a = BitVec::<S>::new_general();
        a.push(true);
        a.push(false);

        let mut b = BitVec::<S>::from_bytes_general(&[
            0b10100000, 0b00010010, 0b10010010, 0b00110011, 0b10010101,
        ]);

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
        let mut a = BitVec::<S>::new_general();
        let mut b = BitVec::<S>::from_bytes_general(&[
            0b10100000, 0b00010010, 0b10010010, 0b00110011, 0b10010101,
        ]);

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
        let mut a = BitVec::<S>::from_bytes_general(&[
            0b10100000, 0b00010010, 0b10010010, 0b00110011, 0b10010101,
        ]);
        let mut b = BitVec::<S>::new_general();

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
    fn test_bit_vec_split_off<S: BitBlockOrStore>() {
        // Split at 0
        let mut a = BitVec::<S>::new_general();
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
        let mut a = BitVec::<S>::from_bytes_general(&[
            0b10100000, 0b00010010, 0b10010010, 0b00110011, 0b11110011,
        ]);

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
        let mut a = BitVec::<S>::from_bytes_general(&[
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
    fn test_into_iter<S: BitBlockOrStore>() {
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
    fn test_iter<S: BitBlockOrStore>() {
        let b = BitVec::<S>::with_capacity_general(10);
        let _a: Iter<S> = b.iter();
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_serialization<S: BitBlockOrStore>()
    where
        S::Store: serde::Serialize + for<'a> serde::Deserialize<'a>,
    {
        let bit_vec: BitVec<S> = BitVec::<S>::new_general();
        let serialized = serde_json::to_string(&bit_vec).unwrap();
        let unserialized: BitVec<S> = serde_json::from_str(&serialized[..]).unwrap();
        assert_eq!(bit_vec, unserialized);

        let bools = vec![true, false, true, true];
        let bit_vec: BitVec = bools.iter().map(|n| *n).collect();
        let serialized = serde_json::to_string(&bit_vec).unwrap();
        let unserialized = serde_json::from_str(&serialized).unwrap();
        assert_eq!(bit_vec, unserialized);
    }

    #[cfg(feature = "miniserde")]
    #[test]
    fn test_miniserde_serialization<
        S: BitBlockOrStore + miniserde::Serialize + miniserde::Deserialize,
    >() {
        let bit_vec = BitVec::<S>::new_general();
        let serialized = miniserde::json::to_string(&bit_vec);
        let unserialized: BitVec<S> = miniserde::json::from_str(&serialized[..]).unwrap();
        assert_eq!(bit_vec, unserialized);

        let bools = vec![true, false, true, true];
        let bit_vec: BitVec = bools.iter().map(|n| *n).collect();
        let serialized = miniserde::json::to_string(&bit_vec);
        let unserialized = miniserde::json::from_str(&serialized[..]).unwrap();
        assert_eq!(bit_vec, unserialized);
    }

    #[cfg(feature = "nanoserde")]
    #[test]
    fn test_nanoserde_json_serialization<
        S: BitBlockOrStore
            + nanoserde::DeBin
            + nanoserde::DeJson
            + nanoserde::DeRon
            + nanoserde::SerBin
            + nanoserde::SerJson
            + nanoserde::SerRon,
    >() {
        use nanoserde::{DeJson, SerJson};

        let bit_vec = BitVec::<S>::new_general();
        let serialized = bit_vec.serialize_json();
        let unserialized = BitVec::<S>::deserialize_json(&serialized[..]).unwrap();
        assert_eq!(bit_vec, unserialized);

        let bools = vec![true, false, true, true];
        let bit_vec: BitVec<S> = bools.iter().map(|n| *n).collect();
        let serialized = bit_vec.serialize_json();
        let unserialized = BitVec::<S>::deserialize_json(&serialized[..]).unwrap();
        assert_eq!(bit_vec, unserialized);
    }

    #[cfg(feature = "borsh")]
    #[test]
    fn test_borsh_serialization<S: BitBlockOrStore>() {
        let bit_vec = BitVec::<S>::new_general();
        let serialized = borsh::to_vec(&bit_vec).unwrap();
        let unserialized: BitVec<S> = borsh::from_slice(&serialized[..]).unwrap();
        assert_eq!(bit_vec, unserialized);

        let bools = vec![true, false, true, true];
        let bit_vec: BitVec = bools.iter().map(|n| *n).collect();
        let serialized = borsh::to_vec(&bit_vec).unwrap();
        let unserialized = borsh::from_slice(&serialized[..]).unwrap();
        assert_eq!(bit_vec, unserialized);
    }

    #[test]
    fn test_bit_vec_unaligned_small_append<S: BitBlockOrStore>() {
        let mut a = BitVec::<S>::from_elem_general(8, false);
        a.set(7, true);

        let mut b = BitVec::<S>::from_elem_general(16, false);
        b.set(14, true);

        let mut c = BitVec::<S>::from_elem_general(8, false);
        c.set(6, true);
        c.set(7, true);

        a.append(&mut b);
        a.append(&mut c);

        assert_eq!(&[1, 0, 2, 3][..], &*a.to_bytes());
    }

    #[test]
    fn test_bit_vec_unaligned_large_append<S: BitBlockOrStore>() {
        let mut a = BitVec::<S>::from_elem_general(48, false);
        a.set(47, true);

        let mut b = BitVec::<S>::from_elem_general(48, false);
        b.set(46, true);

        let mut c = BitVec::<S>::from_elem_general(48, false);
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
    fn test_bit_vec_append_aligned_to_unaligned<S: BitBlockOrStore>() {
        let mut a = BitVec::<S>::from_elem_general(2, true);
        let mut b = BitVec::<S>::from_elem_general(32, false);
        let mut c = BitVec::<S>::from_elem_general(8, true);
        a.append(&mut b);
        a.append(&mut c);
        assert_eq!(&[0xc0, 0x00, 0x00, 0x00, 0x3f, 0xc0][..], &*a.to_bytes());
    }

    #[test]
    fn test_count_ones<S: BitBlockOrStore>() {
        for i in 0..1000 {
            let mut t = BitVec::<S>::from_elem_general(i, true);
            let mut f = BitVec::<S>::from_elem_general(i, false);
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
    fn test_count_zeros<S: BitBlockOrStore>() {
        for i in 0..1000 {
            let mut tbits = BitVec::<S>::from_elem_general(i, true);
            let mut fbits = BitVec::<S>::from_elem_general(i, false);
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
    fn test_get_mut<S: BitBlockOrStore>() {
        let mut a = BitVec::<S>::from_elem_general(3, false);
        let mut a_bit_1 = a.get_mut(1).unwrap();
        assert!(!*a_bit_1);
        *a_bit_1 = true;
        drop(a_bit_1);
        assert!(a.eq_vec(&[false, true, false]));
    }

    #[test]
    fn test_iter_mut<S: BitBlockOrStore>() {
        let mut a = BitVec::<S>::from_elem_general(8, false);
        a.iter_mut().enumerate().for_each(|(index, mut bit)| {
            *bit = index % 2 == 1;
        });
        assert!(a.eq_vec(&[false, true, false, true, false, true, false, true]));
    }

    #[test]
    fn test_insert_at_zero<S: BitBlockOrStore>() {
        let mut v = BitVec::<S>::new_general();

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
    fn test_insert_at_end<S: BitBlockOrStore>() {
        let mut v = BitVec::<S>::new_general();

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
    fn test_insert_at_block_boundaries<S: BitBlockOrStore>() {
        let mut v = BitVec::<S>::from_elem_general(32, false);

        assert_eq!(v.storage().len(), (4 / S::BYTES).max(1));

        v.insert(31, true);

        assert_eq!(v.len(), 33);

        assert!(matches!(v.get(31), Some(true)));
        assert!(v.eq_vec(&[
            false, false, false, false, false, false, false, false, false, false, false, false,
            false, false, false, false, false, false, false, false, false, false, false, false,
            false, false, false, false, false, false, false, true, false
        ]));

        assert_eq!(v.storage().len(), 1 + 4 / S::BYTES);
    }

    #[test]
    fn test_insert_at_block_boundaries_1<S: BitBlockOrStore>() {
        let mut v = BitVec::<S>::from_elem_general(64, false);

        assert_eq!(v.storage().len(), 8 / S::BYTES);

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

        assert_eq!(v.storage().len(), 1 + 8 / S::BYTES);
    }

    #[test]
    fn test_push_within_capacity_with_suffice_cap<S: BitBlockOrStore>() {
        let mut v = BitVec::<S>::from_elem_general(16, true);

        if S::BYTES > 2 {
            assert!(v.push_within_capacity(false).is_ok());
        }

        for i in 0..16 {
            assert_eq!(v.get(i), Some(true));
        }

        if S::BYTES > 2 {
            assert_eq!(v.get(16), Some(false));
            assert_eq!(v.len(), 17);
        }
    }

    #[test]
    fn test_push_within_capacity_at_brink<S: BitBlockOrStore>() {
        let mut v = BitVec::<S>::from_elem_general(31, true);

        assert!(v.push_within_capacity(false).is_ok());

        assert_eq!(v.get(31), Some(false));
        if v.capacity() < 256 {
            assert_eq!(if S::BYTES == 8 { 64 } else { v.len() }, v.capacity());
        }
        assert_eq!(v.len(), 32);

        if v.capacity() < 256 {
            assert_eq!(
                v.push_within_capacity(false),
                if S::BYTES == 8 { Ok(()) } else { Err(false) }
            );
            assert_eq!(v.capacity(), if S::BYTES == 8 { 64 } else { 32 });
        }

        for i in 0..31 {
            assert_eq!(v.get(i), Some(true));
        }
        assert_eq!(v.get(31), Some(false));
    }

    #[test]
    fn test_push_within_capacity_at_brink_with_mul_blocks<S: BitBlockOrStore>() {
        let mut v = BitVec::<S>::from_elem_general(95, true);

        assert!(v.push_within_capacity(false).is_ok());

        assert_eq!(v.get(95), Some(false));
        if S::BYTES <= 4 && v.capacity() < 256 {
            assert_eq!(v.len(), v.capacity());
        }
        assert_eq!(v.len(), 96);

        if S::BYTES == 8 {
            assert_eq!(v.push_within_capacity(false), Ok(()));
            if v.capacity() < 256 {
                assert_eq!(v.capacity(), 128);
            }
        } else if v.capacity() < 256 {
            assert_eq!(v.push_within_capacity(false), Err(false));
            assert_eq!(v.capacity(), 96);
        }

        for i in 0..95 {
            assert_eq!(v.get(i), Some(true));
        }
        assert_eq!(v.get(95), Some(false));
    }

    #[test]
    fn test_push_within_capacity_storage_push<S: BitBlockOrStore>() {
        let mut v = BitVec::<S>::with_capacity_general(64);

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

    #[test]
    fn test_insert_remove<S: BitBlockOrStore>() {
        // two primes for no common divisors with 32
        let mut v = BitVec::<S>::from_fn_general(1024, |i| i % 11 < 7);
        for i in 0..1024 {
            let result = v.remove(i);
            v.insert(i, result);
            assert_eq!(result, i % 11 < 7);
        }

        for i in 0..1024 {
            v.insert(i, false);
            v.remove(i);
        }

        for i in 0..1024 {
            v.insert(i, true);
            v.remove(i);
        }

        for (i, result) in v.into_iter().enumerate() {
            assert_eq!(result, i % 11 < 7);
        }
    }

    #[test]
    fn test_remove_last<S: BitBlockOrStore>() {
        let mut v = BitVec::<S>::from_fn_general(1025, |i| i % 11 < 7);
        assert_eq!(v.len(), 1025);
        assert_eq!(v.remove(1024), 1024 % 11 < 7);
        assert_eq!(v.len(), 1024);
        assert_eq!(v.storage().len(), 1024 / S::BITS);
    }

    #[test]
    fn test_remove_all<S: BitBlockOrStore>() {
        let v = BitVec::<S>::from_elem_general(1024, false);
        for _ in 0..1024 {
            let mut v2 = v.clone();
            v2.remove_all();
            assert_eq!(v2.len(), 0);
            assert_eq!(v2.get(0), None);
            assert_eq!(v2, BitVec::new_general());
        }
    }

    #[instantiate_tests(<Vec<u32>>)]
    mod vec32 {}

    #[cfg(all(feature = "smallvec", not(feature = "nanoserde")))]
    #[instantiate_tests(<smallvec::SmallVec<[u32; 8]>>)]
    mod smallvec32x8 {}

    #[cfg(all(feature = "smallvec", not(feature = "nanoserde")))]
    #[instantiate_tests(<smallvec::SmallVec<[u64; 8]>>)]
    mod smallvec64x8 {}

    #[instantiate_tests(<u32>)]
    mod integer32 {}

    #[instantiate_tests(<usize>)]
    mod native {}

    #[instantiate_tests(<u16>)]
    mod integer16 {}

    #[instantiate_tests(<u8>)]
    mod integer8 {}
}

#[cfg(test)]
#[cfg(feature = "allocator_api")]
mod alloc_tests {
    use std::alloc::Global;
    use std::vec::Vec;

    use crate::BitVec;

    #[test]
    fn test_new_in() {
        let alloc = Global;
        let mut v: BitVec<Vec<u16, Global>> = BitVec::new_general_in(alloc);
        v.push(true);
        v.push(false);
        assert_eq!(v.len(), 2);
        assert_eq!(v.pop(), Some(false));
        assert_eq!(v.pop(), Some(true));
    }
}
