use bit_matrix::BitMatrix;

#[test]
fn test_submatrix() {
    let mut matrix = BitMatrix::new(5, 4);
    let points = &[
        (0, 0),
        (0, 1),
        (0, 3),
        (1, 0),
        (1, 2),
        (2, 0),
        (2, 1),
        (3, 1),
        (3, 3),
        (4, 3),
    ];
    for &(i, j) in points {
        matrix.set(i, j, true);
    }
    let submatrix = matrix.sub_matrix(1..=3);
    let mut iter = submatrix.iter();
    assert!(iter.next().unwrap().get(0));
    assert!(!iter.next().unwrap().get(2));
    assert_eq!(iter.next().unwrap().small_slice_aligned(1, 3), 0b101);
}
