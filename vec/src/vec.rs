use crate::local_prelude::*;
use crate::util::{self, FALSE, TRUE};

/// The bitvector type.
///
/// # Examples
///
/// ```
/// use bit_vec::BitVec;
///
/// let mut bv = BitVec::from_elem(10, false);
///
/// // insert all primes less than 10
/// bv.set(2, true);
/// bv.set(3, true);
/// bv.set(5, true);
/// bv.set(7, true);
/// println!("{:?}", bv);
/// println!("total bits set to true: {}", bv.iter().filter(|x| *x).count());
///
/// // flip all values in bitvector, producing non-primes less than 10
/// bv.negate();
/// println!("{:?}", bv);
/// println!("total bits set to true: {}", bv.iter().filter(|x| *x).count());
///
/// // reset bitvector to empty
/// bv.fill(false);
/// println!("{:?}", bv);
/// println!("total bits set to true: {}", bv.iter().filter(|x| *x).count());
/// ```
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[cfg_attr(
    feature = "borsh",
    derive(borsh::BorshDeserialize, borsh::BorshSerialize)
)]
#[cfg_attr(
    feature = "miniserde",
    derive(miniserde::Deserialize, miniserde::Serialize)
)]
#[cfg_attr(
    feature = "nanoserde",
    derive(DeBin, DeJson, DeRon, SerBin, SerJson, SerRon)
)]
pub struct BitVec<B: BitBlockOrStore = u32> {
    /// Internal representation of the bit vector
    pub(crate) storage: B::Store,
    /// The number of valid bits in the internal representation
    pub(crate) nbits: usize,
}

// FIXME(Gankro): NopeNopeNopeNopeNope (wait for IndexGet to be a thing)
impl<B: BitBlockOrStore> ops::Index<usize> for BitVec<B> {
    type Output = bool;

    #[inline]
    fn index(&self, i: usize) -> &bool {
        if self.get(i).expect("index out of bounds") {
            &TRUE
        } else {
            &FALSE
        }
    }
}

/// Computes how many blocks are needed to store that many bits
fn blocks_for_bits<B: BitBlockOrStore>(bits: usize) -> usize {
    // If we want 17 bits, dividing by 32 will produce 0. So we add 1 to make sure we
    // reserve enough. But if we want exactly a multiple of 32, this will actually allocate
    // one too many. So we need to check if that's the case. We can do that by computing if
    // bitwise AND by `32 - 1` is 0. But LLVM should be able to optimize the semantically
    // superior modulo operator on a power of two to this.
    //
    // Note that we can technically avoid this branch with the expression
    // `(nbits + U32_BITS - 1) / 32::BITS`, but if nbits is almost usize::MAX this will overflow.
    if bits % B::BITS == 0 {
        bits / B::BITS
    } else {
        bits / B::BITS + 1
    }
}

/// Computes the bitmask for the final word of the vector
fn mask_for_bits<B: BitBlockOrStore>(bits: usize) -> Block<B> {
    // Note especially that a perfect multiple of U32_BITS should mask all 1s.
    (!B::ZERO) >> ((B::BITS - bits % B::BITS) % B::BITS)
}

impl BitVec<u32> {
    /// Creates an empty `BitVec`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    /// let mut bv = BitVec::new();
    /// ```
    #[inline]
    pub fn new() -> Self {
        Default::default()
    }

    /// Creates a `BitVec` that holds `nbits` elements, setting each element
    /// to `bit`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let mut bv = BitVec::from_elem(10, false);
    /// assert_eq!(bv.len(), 10);
    /// for x in bv.iter() {
    ///     assert_eq!(x, false);
    /// }
    /// ```
    #[inline]
    pub fn from_elem(len: usize, bit: bool) -> Self {
        BitVec::<u32>::from_elem_general(len, bit)
    }

    /// Constructs a new, empty `BitVec` with the specified capacity.
    ///
    /// The bitvector will be able to hold at least `capacity` bits without
    /// reallocating. If `capacity` is 0, it will not allocate.
    ///
    /// It is important to note that this function does not specify the
    /// *length* of the returned bitvector, but only the *capacity*.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        BitVec::<u32>::with_capacity_general(capacity)
    }

    /// Transforms a byte-vector into a `BitVec`. Each byte becomes eight bits,
    /// with the most significant bits of each byte coming first. Each
    /// bit becomes `true` if equal to 1 or `false` if equal to 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let bv = BitVec::from_bytes(&[0b10100000, 0b00010010]);
    /// assert!(bv.eq_vec(&[true, false, true, false,
    ///                     false, false, false, false,
    ///                     false, false, false, true,
    ///                     false, false, true, false]));
    /// ```
    pub fn from_bytes(bytes: &[u8]) -> Self {
        BitVec::<u32>::from_bytes_general(bytes)
    }

    /// Creates a `BitVec` of the specified length where the value at each index
    /// is `f(index)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let bv = BitVec::from_fn(5, |i| { i % 2 == 0 });
    /// assert!(bv.eq_vec(&[true, false, true, false, true]));
    /// ```
    #[inline]
    pub fn from_fn<F>(len: usize, f: F) -> Self
    where
        F: FnMut(usize) -> bool,
    {
        BitVec::<u32>::from_fn_general(len, f)
    }
}

impl<B: BitBlockOrStore> BitVec<B> {
    /// Creates an empty `BitVec`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    /// let mut bv = BitVec::<usize>::new_general();
    /// ```
    #[inline]
    pub fn new_general() -> Self {
        Default::default()
    }

    /// Creates an empty `BitVec` using the provided allocator.
    #[inline]
    pub fn new_general_in(alloc: <B::Store as BitStore>::Alloc) -> Self {
        Self::with_capacity_general_in(0, alloc)
    }

    /// Creates a `BitVec` that holds `nbits` elements, setting each element
    /// to `bit`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let mut bv = BitVec::<usize>::from_elem_general(10, false);
    /// assert_eq!(bv.len(), 10);
    /// for x in bv.iter() {
    ///     assert_eq!(x, false);
    /// }
    /// ```
    #[inline]
    pub fn from_elem_general(len: usize, bit: bool) -> Self {
        let nblocks = blocks_for_bits::<B>(len);
        let mut storage: B::Store = B::Store::with_capacity(nblocks);
        storage.extend(iter::repeat_n(
            if bit { !B::ZERO } else { B::ZERO },
            nblocks,
        ));
        let mut bit_vec = BitVec {
            storage,
            nbits: len,
        };
        bit_vec.fix_last_block();
        bit_vec
    }

    /// Constructs a new, empty `BitVec` with the specified capacity.
    ///
    /// The bitvector will be able to hold at least `capacity` bits without
    /// reallocating. If `capacity` is 0, it will not allocate.
    ///
    /// It is important to note that this function does not specify the
    /// *length* of the returned bitvector, but only the *capacity*.
    #[inline]
    pub fn with_capacity_general(capacity: usize) -> Self {
        BitVec {
            storage: B::Store::with_capacity(blocks_for_bits::<B>(capacity)),
            nbits: 0,
        }
    }

    /// Constructs a new, empty `BitVec` with the specified capacity.
    ///
    /// The bitvector will be able to hold at least `capacity` bits without
    /// reallocating. If `capacity` is 0, it will not allocate.
    ///
    /// It is important to note that this function does not specify the
    /// *length* of the returned bitvector, but only the *capacity*.
    #[inline]
    pub fn with_capacity_general_in(capacity: usize, alloc: <B::Store as BitStore>::Alloc) -> Self {
        BitVec {
            storage: B::Store::with_capacity_in(blocks_for_bits::<B>(capacity), alloc),
            nbits: 0,
        }
    }

    /// Transforms a byte-vector into a `BitVec`. Each byte becomes eight bits,
    /// with the most significant bits of each byte coming first. Each
    /// bit becomes `true` if equal to 1 or `false` if equal to 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let bv = BitVec::<usize>::from_bytes_general(&[0b10100000, 0b00010010]);
    /// assert!(bv.eq_vec(&[true, false, true, false,
    ///                     false, false, false, false,
    ///                     false, false, false, true,
    ///                     false, false, true, false]));
    /// ```
    pub fn from_bytes_general(bytes: &[u8]) -> Self {
        let len = bytes
            .len()
            .checked_mul(u8::BITS as usize)
            .expect("capacity overflow");
        let mut bit_vec = BitVec::<B>::with_capacity_general(len);
        let complete_words = bytes.len() / B::BYTES;
        let extra_bytes = bytes.len() % B::BYTES;

        bit_vec.nbits = len;

        for i in 0..complete_words {
            let mut accumulator = B::ZERO;
            for idx in 0..B::BYTES {
                accumulator |= <B::Store as BitStore>::Block::from_byte(util::reverse_bits(
                    bytes[i * B::BYTES + idx],
                )) << (idx * 8)
            }
            bit_vec.storage.push(accumulator);
        }

        if extra_bytes > 0 {
            let mut last_word = B::ZERO;
            for (i, &byte) in bytes[complete_words * B::BYTES..].iter().enumerate() {
                last_word |=
                    <B::Store as BitStore>::Block::from_byte(util::reverse_bits(byte)) << (i * 8);
            }
            bit_vec.storage.push(last_word);
        }

        bit_vec
    }

    /// Creates a `BitVec` of the specified length where the value at each index
    /// is `f(index)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let bv = BitVec::<usize>::from_fn_general(5, |i| { i % 2 == 0 });
    /// assert!(bv.eq_vec(&[true, false, true, false, true]));
    /// ```
    #[inline]
    pub fn from_fn_general<F>(len: usize, mut f: F) -> Self
    where
        F: FnMut(usize) -> bool,
    {
        let mut bit_vec = BitVec::from_elem_general(len, false);
        for i in 0..len {
            bit_vec.set(i, f(i));
        }
        bit_vec
    }

    /// Applies the given operation to the blocks of self and other, and sets
    /// self to be the result. This relies on the caller not to corrupt the
    /// last word.
    #[inline]
    fn process<F>(&mut self, other: &BitVec<B>, mut op: F) -> bool
    where
        F: FnMut(Block<B>, Block<B>) -> Block<B>,
    {
        assert_eq!(self.len(), other.len());
        debug_assert_eq!(self.storage.len(), other.storage.len());
        let mut changed_bits = B::ZERO;
        for (a, b) in self.blocks_mut().zip(other.blocks()) {
            let w = op(*a, b);
            changed_bits |= *a ^ w;
            *a = w;
        }
        changed_bits != B::ZERO
    }

    /// Exposes the raw block storage of this `BitVec`.
    ///
    /// Only really intended for `BitSet`.
    #[inline]
    pub fn storage(&self) -> &[Block<B>] {
        self.storage.slice()
    }

    /// Exposes the raw block storage of this `BitVec`.
    ///
    /// # Safety
    ///
    /// Can probably cause unsafety. Only really intended for `BitSet`.
    #[inline]
    pub unsafe fn storage_mut(&mut self) -> &mut B::Store {
        &mut self.storage
    }

    /// Helper for procedures involving spare space in the last block.
    #[inline]
    fn last_block_with_mask(&self) -> Option<(Block<B>, Block<B>)> {
        let extra_bits = self.len() % B::BITS;
        if extra_bits > 0 {
            let mask = (B::ONE << extra_bits) - B::ONE;
            let storage_len = self.storage.len();
            Some((self.storage.slice()[storage_len - 1], mask))
        } else {
            None
        }
    }

    /// Helper for procedures involving spare space in the last block.
    #[inline]
    fn last_block_mut_with_mask(&mut self) -> Option<(&mut Block<B>, Block<B>)> {
        let extra_bits = self.len() % B::BITS;
        if extra_bits > 0 {
            let mask = (B::ONE << extra_bits) - B::ONE;
            let storage_len = self.storage.len();
            Some((&mut self.storage.slice_mut()[storage_len - 1], mask))
        } else {
            None
        }
    }

    /// An operation might screw up the unused bits in the last block of the
    /// `BitVec`. As per (3), it's assumed to be all 0s. This method fixes it up.
    fn fix_last_block(&mut self) {
        if let Some((last_block, used_bits)) = self.last_block_mut_with_mask() {
            *last_block = *last_block & used_bits;
        }
    }

    /// Operations such as change detection for xnor, nor and nand are easiest
    /// to implement when unused bits are all set to 1s.
    fn fix_last_block_with_ones(&mut self) {
        if let Some((last_block, used_bits)) = self.last_block_mut_with_mask() {
            *last_block |= !used_bits;
        }
    }

    /// Check whether last block's invariant is fine.
    fn is_last_block_fixed(&self) -> bool {
        if let Some((last_block, used_bits)) = self.last_block_with_mask() {
            last_block & !used_bits == B::ZERO
        } else {
            true
        }
    }

    /// Ensure the invariant for the last block.
    ///
    /// An operation might screw up the unused bits in the last block of the
    /// `BitVec`.
    ///
    /// This method fails in case the last block is not fixed. The check
    /// is skipped outside testing.
    #[inline]
    pub(crate) fn ensure_invariant(&self) {
        if cfg!(test) {
            debug_assert!(self.is_last_block_fixed());
        }
    }

    /// Retrieves the value at index `i`, or `None` if the index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let bv = BitVec::from_bytes(&[0b01100000]);
    /// assert_eq!(bv.get(0), Some(false));
    /// assert_eq!(bv.get(1), Some(true));
    /// assert_eq!(bv.get(100), None);
    ///
    /// // Can also use array indexing
    /// assert_eq!(bv[1], true);
    /// ```
    #[inline]
    pub fn get(&self, i: usize) -> Option<bool> {
        self.ensure_invariant();
        if i >= self.nbits {
            return None;
        }
        let w = i / B::BITS;
        let b = i % B::BITS;
        self.storage
            .slice()
            .get(w)
            .map(|&block| (block & (B::ONE << b)) != B::ZERO)
    }

    /// Retrieves the value at index `i`, without doing bounds checking.
    ///
    /// For a safe alternative, see `get`.
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is undefined behavior
    /// even if the resulting reference is not used.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let bv = BitVec::from_bytes(&[0b01100000]);
    /// unsafe {
    ///     assert_eq!(bv.get_unchecked(0), false);
    ///     assert_eq!(bv.get_unchecked(1), true);
    /// }
    /// ```
    #[inline]
    pub unsafe fn get_unchecked(&self, i: usize) -> bool {
        self.ensure_invariant();
        let w = i / B::BITS;
        let b = i % B::BITS;
        let block = *self.storage.slice().get_unchecked(w);
        block & (B::ONE << b) != B::ZERO
    }

    /// Sets the value of a bit at an index `i`.
    ///
    /// # Panics
    ///
    /// Panics if `i` is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let mut bv = BitVec::from_elem(5, false);
    /// bv.set(3, true);
    /// assert_eq!(bv[3], true);
    /// ```
    #[inline]
    pub fn set(&mut self, i: usize, x: bool) {
        self.ensure_invariant();
        assert!(
            i < self.nbits,
            "index out of bounds: {:?} >= {:?}",
            i,
            self.nbits
        );
        let w = i / B::BITS;
        let b = i % B::BITS;
        let flag = B::ONE << b;
        let val = if x {
            self.storage.slice()[w] | flag
        } else {
            self.storage.slice()[w] & !flag
        };
        self.storage.slice_mut()[w] = val;
    }

    /// Sets all bits to 1.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let before = 0b01100000;
    /// let after  = 0b11111111;
    ///
    /// let mut bv = BitVec::from_bytes(&[before]);
    /// bv.set_all();
    /// assert_eq!(bv, BitVec::from_bytes(&[after]));
    /// ```
    #[inline]
    #[deprecated(since = "0.9.0", note = "please use `.fill(true)` instead")]
    pub fn set_all(&mut self) {
        self.ensure_invariant();
        for w in self.storage.slice_mut() {
            *w = !B::ZERO;
        }
        self.fix_last_block();
    }

    /// Flips all bits.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let before = 0b01100000;
    /// let after  = 0b10011111;
    ///
    /// let mut bv = BitVec::from_bytes(&[before]);
    /// bv.negate();
    /// assert_eq!(bv, BitVec::from_bytes(&[after]));
    /// ```
    #[inline]
    pub fn negate(&mut self) {
        self.ensure_invariant();
        for w in self.storage.slice_mut() {
            *w = !*w;
        }
        self.fix_last_block();
    }

    /// Calculates the union of two bitvectors. This acts like the bitwise `or`
    /// function.
    ///
    /// Sets `self` to the union of `self` and `other`. Both bitvectors must be
    /// the same length. Returns `true` if `self` changed.
    ///
    /// # Panics
    ///
    /// Panics if the bitvectors are of different lengths.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let a   = 0b01100100;
    /// let b   = 0b01011010;
    /// let res = 0b01111110;
    ///
    /// let mut a = BitVec::from_bytes(&[a]);
    /// let b = BitVec::from_bytes(&[b]);
    ///
    /// assert!(a.union(&b));
    /// assert_eq!(a, BitVec::from_bytes(&[res]));
    /// ```
    #[deprecated(since = "0.7.0", note = "Please use the 'or' function instead")]
    #[inline]
    pub fn union(&mut self, other: &Self) -> bool {
        self.or(other)
    }

    /// Calculates the intersection of two bitvectors. This acts like the
    /// bitwise `and` function.
    ///
    /// Sets `self` to the intersection of `self` and `other`. Both bitvectors
    /// must be the same length. Returns `true` if `self` changed.
    ///
    /// # Panics
    ///
    /// Panics if the bitvectors are of different lengths.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let a   = 0b01100100;
    /// let b   = 0b01011010;
    /// let res = 0b01000000;
    ///
    /// let mut a = BitVec::from_bytes(&[a]);
    /// let b = BitVec::from_bytes(&[b]);
    ///
    /// assert!(a.intersect(&b));
    /// assert_eq!(a, BitVec::from_bytes(&[res]));
    /// ```
    #[deprecated(since = "0.7.0", note = "Please use the 'and' function instead")]
    #[inline]
    pub fn intersect(&mut self, other: &Self) -> bool {
        self.and(other)
    }

    /// Calculates the bitwise `or` of two bitvectors.
    ///
    /// Sets `self` to the union of `self` and `other`. Both bitvectors must be
    /// the same length. Returns `true` if `self` changed.
    ///
    /// # Panics
    ///
    /// Panics if the bitvectors are of different lengths.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let a   = 0b01100100;
    /// let b   = 0b01011010;
    /// let res = 0b01111110;
    ///
    /// let mut a = BitVec::from_bytes(&[a]);
    /// let b = BitVec::from_bytes(&[b]);
    ///
    /// assert!(a.or(&b));
    /// assert_eq!(a, BitVec::from_bytes(&[res]));
    /// ```
    #[inline]
    pub fn or(&mut self, other: &Self) -> bool {
        self.ensure_invariant();
        debug_assert!(other.is_last_block_fixed());
        self.process(other, |w1, w2| w1 | w2)
    }

    /// Calculates the bitwise `and` of two bitvectors.
    ///
    /// Sets `self` to the intersection of `self` and `other`. Both bitvectors
    /// must be the same length. Returns `true` if `self` changed.
    ///
    /// # Panics
    ///
    /// Panics if the bitvectors are of different lengths.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let a   = 0b01100100;
    /// let b   = 0b01011010;
    /// let res = 0b01000000;
    ///
    /// let mut a = BitVec::from_bytes(&[a]);
    /// let b = BitVec::from_bytes(&[b]);
    ///
    /// assert!(a.and(&b));
    /// assert_eq!(a, BitVec::from_bytes(&[res]));
    /// ```
    #[inline]
    pub fn and(&mut self, other: &Self) -> bool {
        self.ensure_invariant();
        debug_assert!(other.is_last_block_fixed());
        self.process(other, |w1, w2| w1 & w2)
    }

    /// Calculates the difference between two bitvectors.
    ///
    /// Sets each element of `self` to the value of that element minus the
    /// element of `other` at the same index. Both bitvectors must be the same
    /// length. Returns `true` if `self` changed.
    ///
    /// # Panics
    ///
    /// Panics if the bitvectors are of different length.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let a   = 0b01100100;
    /// let b   = 0b01011010;
    /// let a_b = 0b00100100; // a - b
    /// let b_a = 0b00011010; // b - a
    ///
    /// let mut bva = BitVec::from_bytes(&[a]);
    /// let bvb = BitVec::from_bytes(&[b]);
    ///
    /// assert!(bva.difference(&bvb));
    /// assert_eq!(bva, BitVec::from_bytes(&[a_b]));
    ///
    /// let bva = BitVec::from_bytes(&[a]);
    /// let mut bvb = BitVec::from_bytes(&[b]);
    ///
    /// assert!(bvb.difference(&bva));
    /// assert_eq!(bvb, BitVec::from_bytes(&[b_a]));
    /// ```
    #[inline]
    pub fn difference(&mut self, other: &Self) -> bool {
        self.ensure_invariant();
        debug_assert!(other.is_last_block_fixed());
        self.process(other, |w1, w2| w1 & !w2)
    }

    /// Calculates the xor of two bitvectors.
    ///
    /// Sets `self` to the xor of `self` and `other`. Both bitvectors must be
    /// the same length. Returns `true` if `self` changed.
    ///
    /// # Panics
    ///
    /// Panics if the bitvectors are of different length.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let a   = 0b01100110;
    /// let b   = 0b01010100;
    /// let res = 0b00110010;
    ///
    /// let mut a = BitVec::from_bytes(&[a]);
    /// let b = BitVec::from_bytes(&[b]);
    ///
    /// assert!(a.xor(&b));
    /// assert_eq!(a, BitVec::from_bytes(&[res]));
    /// ```
    #[inline]
    pub fn xor(&mut self, other: &Self) -> bool {
        self.ensure_invariant();
        debug_assert!(other.is_last_block_fixed());
        self.process(other, |w1, w2| w1 ^ w2)
    }

    /// Calculates the nand of two bitvectors.
    ///
    /// Sets `self` to the nand of `self` and `other`. Both bitvectors must be
    /// the same length. Returns `true` if `self` changed.
    ///
    /// # Panics
    ///
    /// Panics if the bitvectors are of different length.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let a   = 0b01100110;
    /// let b   = 0b01010100;
    /// let res = 0b10111011;
    ///
    /// let mut a = BitVec::from_bytes(&[a]);
    /// let b = BitVec::from_bytes(&[b]);
    ///
    /// assert!(a.nand(&b));
    /// assert_eq!(a, BitVec::from_bytes(&[res]));
    /// ```
    #[inline]
    pub fn nand(&mut self, other: &Self) -> bool {
        self.ensure_invariant();
        debug_assert!(other.is_last_block_fixed());
        self.fix_last_block_with_ones();
        let result = self.process(other, |w1, w2| !(w1 & w2));
        self.fix_last_block();
        result
    }

    /// Calculates the nor of two bitvectors.
    ///
    /// Sets `self` to the nor of `self` and `other`. Both bitvectors must be
    /// the same length. Returns `true` if `self` changed.
    ///
    /// # Panics
    ///
    /// Panics if the bitvectors are of different length.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let a   = 0b01100110;
    /// let b   = 0b01010100;
    /// let res = 0b10001001;
    ///
    /// let mut a = BitVec::from_bytes(&[a]);
    /// let b = BitVec::from_bytes(&[b]);
    ///
    /// assert!(a.nor(&b));
    /// assert_eq!(a, BitVec::from_bytes(&[res]));
    /// ```
    #[inline]
    pub fn nor(&mut self, other: &Self) -> bool {
        self.ensure_invariant();
        debug_assert!(other.is_last_block_fixed());
        self.fix_last_block_with_ones();
        let result = self.process(other, |w1, w2| !(w1 | w2));
        self.fix_last_block();
        result
    }

    /// Calculates the xnor of two bitvectors.
    ///
    /// Sets `self` to the xnor of `self` and `other`. Both bitvectors must be
    /// the same length. Returns `true` if `self` changed.
    ///
    /// # Panics
    ///
    /// Panics if the bitvectors are of different length.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let a   = 0b01100110;
    /// let b   = 0b01010100;
    /// let res = 0b11001101;
    ///
    /// let mut a = BitVec::from_bytes(&[a]);
    /// let b = BitVec::from_bytes(&[b]);
    ///
    /// assert!(a.xnor(&b));
    /// assert_eq!(a, BitVec::from_bytes(&[res]));
    /// ```
    #[inline]
    pub fn xnor(&mut self, other: &Self) -> bool {
        self.ensure_invariant();
        debug_assert!(other.is_last_block_fixed());
        self.fix_last_block_with_ones();
        let result = self.process(other, |w1, w2| !(w1 ^ w2));
        self.fix_last_block();
        result
    }

    /// Returns `true` if all bits are 1.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let mut bv = BitVec::from_elem(5, true);
    /// assert_eq!(bv.all(), true);
    ///
    /// bv.set(1, false);
    /// assert_eq!(bv.all(), false);
    /// ```
    #[inline]
    pub fn all(&self) -> bool {
        self.ensure_invariant();
        let mut last_word = !B::ZERO;
        // Check that every block but the last is all-ones...
        self.blocks().all(|elem| {
            let tmp = last_word;
            last_word = elem;
            tmp == !B::ZERO
            // and then check the last one has enough ones
        }) && (last_word == mask_for_bits::<B>(self.nbits))
    }

    /// Returns the number of ones in the binary representation.
    ///
    /// Also known as the
    /// [Hamming weight](https://en.wikipedia.org/wiki/Hamming_weight).
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let mut bv = BitVec::from_elem(100, true);
    /// assert_eq!(bv.count_ones(), 100);
    ///
    /// bv.set(50, false);
    /// assert_eq!(bv.count_ones(), 99);
    /// ```
    #[inline]
    pub fn count_ones(&self) -> u64 {
        self.ensure_invariant();
        // Add the number of ones of each block.
        self.blocks().map(|elem| elem.count_ones() as u64).sum()
    }

    /// Returns the number of zeros in the binary representation.
    ///
    /// Also known as the opposite of
    /// [Hamming weight](https://en.wikipedia.org/wiki/Hamming_weight).
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let mut bv = BitVec::from_elem(100, false);
    /// assert_eq!(bv.count_zeros(), 100);
    ///
    /// bv.set(50, true);
    /// assert_eq!(bv.count_zeros(), 99);
    /// ```
    #[inline]
    pub fn count_zeros(&self) -> u64 {
        self.ensure_invariant();
        // Add the number of zeros of each block.
        let extra_zeros = (B::BITS - (self.len() % B::BITS)) % B::BITS;
        self.blocks()
            .map(|elem| elem.count_zeros() as u64)
            .sum::<u64>()
            - extra_zeros as u64
    }

    /// Moves all bits from `other` into `Self`, leaving `other` empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let mut a = BitVec::from_bytes(&[0b10000000]);
    /// let mut b = BitVec::from_bytes(&[0b01100001]);
    ///
    /// a.append(&mut b);
    ///
    /// assert_eq!(a.len(), 16);
    /// assert_eq!(b.len(), 0);
    /// assert!(a.eq_vec(&[true, false, false, false, false, false, false, false,
    ///                    false, true, true, false, false, false, false, true]));
    /// ```
    pub fn append(&mut self, other: &mut Self) {
        self.ensure_invariant();
        debug_assert!(other.is_last_block_fixed());

        let b = self.len() % B::BITS;
        let o = other.len() % B::BITS;
        let will_overflow = (b + o > B::BITS) || (o == 0 && b != 0);

        self.nbits += other.len();
        other.nbits = 0;

        if b == 0 {
            self.storage.append(&mut other.storage);
        } else {
            self.storage.reserve(other.storage.len());

            for block in other.storage.drain(..) {
                {
                    let last = self.storage.slice_mut().last_mut().unwrap();
                    *last |= block << b;
                }
                self.storage.push(block >> (B::BITS - b));
            }

            // Remove additional block if the last shift did not overflow
            if !will_overflow {
                self.storage.pop();
            }
        }
    }

    /// Splits the `BitVec` into two at the given bit,
    /// retaining the first half in-place and returning the second one.
    ///
    /// # Panics
    ///
    /// Panics if `at` is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    /// let mut a = BitVec::new();
    /// a.push(true);
    /// a.push(false);
    /// a.push(false);
    /// a.push(true);
    ///
    /// let b = a.split_off(2);
    ///
    /// assert_eq!(a.len(), 2);
    /// assert_eq!(b.len(), 2);
    /// assert!(a.eq_vec(&[true, false]));
    /// assert!(b.eq_vec(&[false, true]));
    /// ```
    pub fn split_off(&mut self, at: usize) -> Self {
        self.ensure_invariant();
        assert!(at <= self.len(), "`at` out of bounds");

        let mut other = BitVec::<B>::new_general();

        if at == 0 {
            mem::swap(self, &mut other);
            return other;
        } else if at == self.len() {
            return other;
        }

        let w = at / B::BITS;
        let b = at % B::BITS;
        other.nbits = self.nbits - at;
        self.nbits = at;
        if b == 0 {
            // Split at block boundary
            other.storage = self.storage.split_off(w);
        } else {
            other.storage.reserve(self.storage.len() - w);

            {
                let mut iter = self.storage.slice()[w..].iter();
                let mut last = *iter.next().unwrap();
                for &cur in iter {
                    other.storage.push((last >> b) | (cur << (B::BITS - b)));
                    last = cur;
                }
                other.storage.push(last >> b);
            }

            self.storage.truncate(w + 1);
            self.fix_last_block();
        }

        other
    }

    /// Returns `true` if all bits are 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let mut bv = BitVec::from_elem(10, false);
    /// assert_eq!(bv.none(), true);
    ///
    /// bv.set(3, true);
    /// assert_eq!(bv.none(), false);
    /// ```
    #[inline]
    pub fn none(&self) -> bool {
        self.blocks().all(|w| w == B::ZERO)
    }

    /// Returns `true` if any bit is 1.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let mut bv = BitVec::from_elem(10, false);
    /// assert_eq!(bv.any(), false);
    ///
    /// bv.set(3, true);
    /// assert_eq!(bv.any(), true);
    /// ```
    #[inline]
    pub fn any(&self) -> bool {
        !self.none()
    }

    /// Organises the bits into bytes, such that the first bit in the
    /// `BitVec` becomes the high-order bit of the first byte. If the
    /// size of the `BitVec` is not a multiple of eight then trailing bits
    /// will be filled-in with `false`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let mut bv = BitVec::from_elem(3, true);
    /// bv.set(1, false);
    ///
    /// assert_eq!(bv.to_bytes(), [0b10100000]);
    ///
    /// let mut bv = BitVec::from_elem(9, false);
    /// bv.set(2, true);
    /// bv.set(8, true);
    ///
    /// assert_eq!(bv.to_bytes(), [0b00100000, 0b10000000]);
    /// ```
    pub fn to_bytes(&self) -> Vec<u8> {
        static REVERSE_TABLE: [u8; 256] = {
            let mut tbl = [0u8; 256];
            let mut i: u8 = 0;
            loop {
                tbl[i as usize] = i.reverse_bits();
                if i == 255 {
                    break;
                }
                i += 1;
            }
            tbl
        };
        self.ensure_invariant();

        let len = self.nbits / 8 + if self.nbits % 8 == 0 { 0 } else { 1 };
        let mut result = Vec::with_capacity(len);

        for byte_idx in 0..len {
            let mut byte = 0u8;
            for bit_idx in 0..8 {
                let offset = byte_idx * 8 + bit_idx;
                if offset < self.nbits && self[offset] {
                    byte |= 1 << bit_idx;
                }
            }
            result.push(REVERSE_TABLE[byte as usize]);
        }

        result
    }

    /// Compares a `BitVec` to a slice of `bool`s.
    /// Both the `BitVec` and slice must have the same length.
    ///
    /// # Panics
    ///
    /// Panics if the `BitVec` and slice are of different length.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let bv = BitVec::from_bytes(&[0b10100000]);
    ///
    /// assert!(bv.eq_vec(&[true, false, true, false,
    ///                     false, false, false, false]));
    /// ```
    #[inline]
    pub fn eq_vec(&self, v: &[bool]) -> bool {
        assert_eq!(self.nbits, v.len());
        self.iter().zip(v.iter().cloned()).all(|(b1, b2)| b1 == b2)
    }

    /// Shortens a `BitVec`, dropping excess elements.
    ///
    /// If `len` is greater than the vector's current length, this has no
    /// effect.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let mut bv = BitVec::from_bytes(&[0b01001011]);
    /// bv.truncate(2);
    /// assert!(bv.eq_vec(&[false, true]));
    /// ```
    #[inline]
    pub fn truncate(&mut self, len: usize) {
        self.ensure_invariant();
        if len < self.len() {
            self.nbits = len;
            // This fixes (2).
            self.storage.truncate(blocks_for_bits::<B>(len));
            self.fix_last_block();
        }
    }

    /// Reserves capacity for at least `additional` more bits to be inserted in the given
    /// `BitVec`. The collection may reserve more space to avoid frequent reallocations.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity overflows `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let mut bv = BitVec::from_elem(3, false);
    /// bv.reserve(10);
    /// assert_eq!(bv.len(), 3);
    /// assert!(bv.capacity() >= 13);
    /// ```
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        let desired_cap = self
            .len()
            .checked_add(additional)
            .expect("capacity overflow");
        let storage_len = self.storage.len();
        if desired_cap > self.capacity() {
            self.storage
                .reserve(blocks_for_bits::<B>(desired_cap) - storage_len);
        }
    }

    /// Reserves the minimum capacity for exactly `additional` more bits to be inserted in the
    /// given `BitVec`. Does nothing if the capacity is already sufficient.
    ///
    /// Note that the allocator may give the collection more space than it requests. Therefore
    /// capacity can not be relied upon to be precisely minimal. Prefer `reserve` if future
    /// insertions are expected.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity overflows `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let mut bv = BitVec::from_elem(3, false);
    /// bv.reserve(10);
    /// assert_eq!(bv.len(), 3);
    /// assert!(bv.capacity() >= 13);
    /// ```
    #[inline]
    pub fn reserve_exact(&mut self, additional: usize) {
        let desired_cap = self
            .len()
            .checked_add(additional)
            .expect("capacity overflow");
        let storage_len = self.storage.len();
        if desired_cap > self.capacity() {
            self.storage
                .reserve_exact(blocks_for_bits::<B>(desired_cap) - storage_len);
        }
    }

    /// Returns the capacity in bits for this bit vector. Inserting any
    /// element less than this amount will not trigger a resizing.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let mut bv = BitVec::new();
    /// bv.reserve(10);
    /// assert!(bv.capacity() >= 10);
    /// ```
    #[inline]
    pub fn capacity(&self) -> usize {
        self.storage.capacity().saturating_mul(B::BITS)
    }

    /// Grows the `BitVec` in-place, adding `n` copies of `value` to the `BitVec`.
    ///
    /// # Panics
    ///
    /// Panics if the new len overflows a `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let mut bv = BitVec::from_bytes(&[0b01001011]);
    /// bv.grow(2, true);
    /// assert_eq!(bv.len(), 10);
    /// assert_eq!(bv.to_bytes(), [0b01001011, 0b11000000]);
    /// ```
    pub fn grow(&mut self, n: usize, value: bool) {
        self.ensure_invariant();

        // Note: we just bulk set all the bits in the last word in this fn in multiple places
        // which is technically wrong if not all of these bits are to be used. However, at the end
        // of this fn we call `fix_last_block` at the end of this fn, which should fix this.

        let new_nbits = self.nbits.checked_add(n).expect("capacity overflow");
        let new_nblocks = blocks_for_bits::<B>(new_nbits);
        let full_value = if value { !B::ZERO } else { B::ZERO };

        // Correct the old tail word, setting or clearing formerly unused bits
        let num_cur_blocks = blocks_for_bits::<B>(self.nbits);
        if self.nbits % B::BITS > 0 {
            let mask = mask_for_bits::<B>(self.nbits);
            if value {
                let block = &mut self.storage.slice_mut()[num_cur_blocks - 1];
                *block |= !mask;
            } else {
                // Extra bits are already zero by invariant.
            }
        }

        // Fill in words after the old tail word
        let stop_idx = cmp::min(self.storage.len(), new_nblocks);
        for idx in num_cur_blocks..stop_idx {
            self.storage.slice_mut()[idx] = full_value;
        }

        // Allocate new words, if needed
        if new_nblocks > self.storage.len() {
            let to_add = new_nblocks - self.storage.len();
            self.storage.extend(iter::repeat_n(full_value, to_add));
        }

        // Adjust internal bit count
        self.nbits = new_nbits;

        self.fix_last_block();
    }

    /// Removes the last bit from the `BitVec`, and returns it. Returns `None` if the `BitVec` is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let mut bv = BitVec::from_bytes(&[0b01001001]);
    /// assert_eq!(bv.pop(), Some(true));
    /// assert_eq!(bv.pop(), Some(false));
    /// assert_eq!(bv.len(), 6);
    /// ```
    #[inline]
    pub fn pop(&mut self) -> Option<bool> {
        self.ensure_invariant();

        if self.is_empty() {
            None
        } else {
            let i = self.nbits - 1;
            let ret = self[i];
            // (3)
            self.set(i, false);
            self.nbits = i;
            if self.nbits % B::BITS == 0 {
                // (2)
                self.storage.pop();
            }
            Some(ret)
        }
    }

    /// Pushes a `bool` onto the end.
    ///
    /// # Examples
    ///
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let mut bv = BitVec::new();
    /// bv.push(true);
    /// bv.push(false);
    /// assert!(bv.eq_vec(&[true, false]));
    /// ```
    #[inline]
    pub fn push(&mut self, elem: bool) {
        if self.nbits % B::BITS == 0 {
            self.storage.push(B::ZERO);
        }
        let insert_pos = self.nbits;
        self.nbits = self.nbits.checked_add(1).expect("Capacity overflow");
        self.set(insert_pos, elem);
    }

    /// Returns the total number of bits in this vector
    #[inline]
    pub fn len(&self) -> usize {
        self.nbits
    }

    /// Sets the number of bits that this `BitVec` considers initialized.
    ///
    /// # Safety
    ///
    /// Almost certainly can cause bad stuff. Only really intended for `BitSet`.
    #[inline]
    pub unsafe fn set_len(&mut self, len: usize) {
        self.nbits = len;
    }

    /// Returns true if there are no bits in this vector
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clears all bits in this vector.
    #[inline]
    #[deprecated(since = "0.9.0", note = "please use `.fill(false)` instead")]
    pub fn clear(&mut self) {
        self.ensure_invariant();
        for w in self.storage.slice_mut() {
            *w = B::ZERO;
        }
    }

    /// Assigns all bits in this vector to the given boolean value.
    ///
    /// # Invariants
    ///
    /// - After a call to `.fill(true)`, the result of [`all`] is `true`.
    /// - After a call to `.fill(false)`, the result of [`none`] is `true`.
    ///
    /// [`all`]: Self::all
    /// [`none`]: Self::none
    #[inline]
    pub fn fill(&mut self, bit: bool) {
        self.ensure_invariant();
        let block = if bit { !B::ZERO } else { B::ZERO };
        for w in self.storage.slice_mut() {
            *w = block;
        }
        if bit {
            self.fix_last_block();
        }
    }

    /// Shrinks the capacity of the underlying storage as much as
    /// possible.
    ///
    /// It will drop down as close as possible to the length but the
    /// allocator may still inform the underlying storage that there
    /// is space for a few more elements/bits.
    pub fn shrink_to_fit(&mut self) {
        self.storage.shrink_to_fit();
    }

    /// Inserts a given bit at index `at`, shifting all bits after by one
    ///
    /// # Panics
    /// Panics if `at` is out of bounds for `BitVec`'s length (that is, if `at > BitVec::len()`)
    ///
    /// # Examples
    ///```
    /// use bit_vec::BitVec;
    ///
    /// let mut b = BitVec::new();
    ///
    /// b.push(true);
    /// b.push(true);
    /// b.insert(1, false);
    ///
    /// assert!(b.eq_vec(&[true, false, true]));
    ///```
    ///
    /// # Time complexity
    /// Takes O([`len`]) time. All items after the insertion index must be
    /// shifted to the right. In the worst case, all elements are shifted when
    /// the insertion index is 0.
    ///
    /// [`len`]: Self::len
    pub fn insert(&mut self, at: usize, bit: bool) {
        assert!(
            at <= self.nbits,
            "insertion index (is {at}) should be <= len (is {nbits})",
            nbits = self.nbits
        );
        self.ensure_invariant();

        let last_block_bits = self.nbits % B::BITS;
        let block_at = at / B::BITS; // needed block
        let bit_at = at % B::BITS; // index within the block

        if last_block_bits == 0 {
            self.storage.push(B::ZERO);
        }

        self.nbits += 1;

        let mut carry = self.storage.slice()[block_at] >> (B::BITS - 1);
        let lsbits_mask = (B::ONE << bit_at) - B::ONE;
        let set_bit = if bit { B::ONE } else { B::ZERO } << bit_at;
        self.storage.slice_mut()[block_at] = (self.storage.slice()[block_at] & lsbits_mask)
            | ((self.storage.slice()[block_at] & !lsbits_mask) << 1)
            | set_bit;

        for block_ref in &mut self.storage.slice_mut()[block_at + 1..] {
            let curr_carry = *block_ref >> (B::BITS - 1);
            *block_ref = *block_ref << 1 | carry;
            carry = curr_carry;
        }
    }

    /// Remove a bit at index `at`, shifting all bits after by one.
    ///
    /// # Panics
    /// Panics if `at` is out of bounds for `BitVec`'s length (that is, if `at >= BitVec::len()`)
    ///
    /// # Examples
    ///```
    /// use bit_vec::BitVec;
    ///
    /// let mut b = BitVec::new();
    ///
    /// b.push(true);
    /// b.push(false);
    /// b.push(false);
    /// b.push(true);
    /// assert!(!b.remove(1));
    ///
    /// assert!(b.eq_vec(&[true, false, true]));
    ///```
    ///
    /// # Time complexity
    /// Takes O([`len`]) time. All items after the removal index must be
    /// shifted to the left. In the worst case, all elements are shifted when
    /// the removal index is 0.
    ///
    /// [`len`]: Self::len
    pub fn remove(&mut self, at: usize) -> bool {
        assert!(
            at < self.nbits,
            "removal index (is {at}) should be < len (is {nbits})",
            nbits = self.nbits
        );
        self.ensure_invariant();

        self.nbits -= 1;

        let last_block_bits = self.nbits % B::BITS;
        let block_at = at / B::BITS; // needed block
        let bit_at = at % B::BITS; // index within the block

        let lsbits_mask = (B::ONE << bit_at) - B::ONE;

        let mut carry = B::ZERO;

        for block_ref in self.storage.slice_mut()[block_at + 1..].iter_mut().rev() {
            let curr_carry = *block_ref & B::ONE;
            *block_ref = *block_ref >> 1 | (carry << (B::BITS - 1));
            carry = curr_carry;
        }

        // Note: this is equivalent to `.get_unchecked(at)`, but we do
        // not want to introduce unsafe code here.
        let result = (self.storage.slice()[block_at] >> bit_at) & B::ONE == B::ONE;

        self.storage.slice_mut()[block_at] = (self.storage.slice()[block_at] & lsbits_mask)
            | ((self.storage.slice()[block_at] & (!lsbits_mask << 1)) >> 1)
            | carry << (B::BITS - 1);

        if last_block_bits == 0 {
            self.storage.pop();
        }

        result
    }

    /// Removes all bits in this vector.
    ///
    /// Note: this method is not named [`clear`] to avoid confusion whenever [`.fill(false)`]
    /// is needed.
    ///
    /// [`clear`]: Self::clear
    /// [`.fill(false)`]: Self::fill
    pub fn remove_all(&mut self) {
        self.storage.clear();
        self.nbits = 0;
    }

    /// Appends an element if there is sufficient spare capacity, otherwise an error is returned
    /// with the element.
    ///
    /// Unlike [`push`] this method will not reallocate when there's insufficient capacity.
    /// The caller should use [`reserve`] to ensure that there is enough capacity.
    ///
    /// [`push`]: Self::push
    /// [`reserve`]: Self::reserve
    ///
    /// # Examples
    /// ```
    /// use bit_vec::BitVec;
    ///
    /// let initial_capacity = 64;
    /// let mut bitvec = BitVec::with_capacity(64);
    ///
    /// for _ in 0..initial_capacity - 1 {
    ///     bitvec.push(false);
    /// }
    ///
    /// assert_eq!(bitvec.len(), initial_capacity - 1); // there is space for only 1 bit
    ///
    /// assert_eq!(bitvec.push_within_capacity(true), Ok(())); // Successfully push a bit
    /// assert_eq!(bitvec.len(), initial_capacity); // So we can't push within capacity anymore
    ///
    /// assert_eq!(bitvec.push_within_capacity(true), Err(true));
    /// assert_eq!(bitvec.len(), initial_capacity);
    /// assert_eq!(bitvec.capacity(), initial_capacity);
    /// ```
    ///
    /// # Time Complexity
    /// Takes *O(1)* time.
    pub fn push_within_capacity(&mut self, bit: bool) -> Result<(), bool> {
        let len = self.len();

        if len == self.capacity() {
            return Err(bit);
        }

        let bits = B::BITS;

        if len % bits == 0 {
            self.storage.push(B::ZERO);
        }

        let block_at = len / bits;
        let bit_at = len % bits;
        let flag = if bit { B::ONE << bit_at } else { B::ZERO };

        self.ensure_invariant();

        self.nbits += 1;

        self.storage.slice_mut()[block_at] = self.storage.slice()[block_at] | flag; // set the bit

        Ok(())
    }
}

impl<B: BitBlockOrStore> Default for BitVec<B> {
    #[inline]
    fn default() -> Self {
        BitVec {
            storage: B::Store::new_in(Default::default()),
            nbits: 0,
        }
    }
}

impl<B: BitBlockOrStore> FromIterator<bool> for BitVec<B> {
    #[inline]
    fn from_iter<I: IntoIterator<Item = bool>>(iter: I) -> Self {
        let mut ret: Self = Default::default();
        ret.extend(iter);
        ret
    }
}

impl<B: BitBlockOrStore> Extend<bool> for BitVec<B> {
    #[inline]
    fn extend<I: IntoIterator<Item = bool>>(&mut self, iterable: I) {
        self.ensure_invariant();
        let iterator = iterable.into_iter();
        let (min, _) = iterator.size_hint();
        self.reserve(min);
        for element in iterator {
            self.push(element)
        }
    }
}

impl<B: BitBlockOrStore> Clone for BitVec<B> {
    #[inline]
    fn clone(&self) -> Self {
        self.ensure_invariant();
        BitVec {
            storage: self.storage.clone(),
            nbits: self.nbits,
        }
    }

    #[inline]
    fn clone_from(&mut self, source: &Self) {
        debug_assert!(source.is_last_block_fixed());
        self.nbits = source.nbits;
        self.storage.clone_from(&source.storage);
    }
}

impl<B: BitBlockOrStore> PartialOrd for BitVec<B> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<B: BitBlockOrStore> Ord for BitVec<B> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.ensure_invariant();
        debug_assert!(other.is_last_block_fixed());
        let mut a = self.iter();
        let mut b = other.iter();
        loop {
            match (a.next(), b.next()) {
                (Some(x), Some(y)) => match x.cmp(&y) {
                    Ordering::Equal => {}
                    otherwise => return otherwise,
                },
                (None, None) => return Ordering::Equal,
                (None, _) => return Ordering::Less,
                (_, None) => return Ordering::Greater,
            }
        }
    }
}

impl<B: BitBlockOrStore> fmt::Display for BitVec<B> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        self.ensure_invariant();
        for bit in self {
            fmt.write_char(if bit { '1' } else { '0' })?;
        }
        Ok(())
    }
}

impl<B: BitBlockOrStore> fmt::Debug for BitVec<B> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        self.ensure_invariant();
        let mut storage = String::with_capacity(self.len() + self.len() / B::BITS);
        for (i, bit) in self.iter().enumerate() {
            if i != 0 && i % B::BITS == 0 {
                storage.push(' ');
            }
            storage.push(if bit { '1' } else { '0' });
        }
        fmt.debug_struct("BitVec")
            .field("storage", &storage)
            .field("nbits", &self.nbits)
            .finish()
    }
}

impl<B: BitBlockOrStore> hash::Hash for BitVec<B> {
    #[inline]
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.ensure_invariant();
        self.nbits.hash(state);
        for elem in self.blocks() {
            elem.hash(state);
        }
    }
}

impl<B: BitBlockOrStore> cmp::PartialEq for BitVec<B> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        if self.nbits != other.nbits {
            self.ensure_invariant();
            other.ensure_invariant();
            return false;
        }
        self.blocks().zip(other.blocks()).all(|(w1, w2)| w1 == w2)
    }
}

impl<B: BitBlockOrStore> cmp::Eq for BitVec<B> {}
