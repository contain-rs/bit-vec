use crate::{BitBlock, BitVec};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use bit_vec_extra::store::BitStore;
#[cfg(feature = "std")]
use std::vec::Vec;

use bit_vec_extra::order::Lsb0;
pub use bit_vec_extra::prelude;
use bit_vec_extra::prelude::*;
pub use bit_vec_extra::vec::BitVecLike;

impl<B> BitVecLike for BitVec<B> where B: BitStore + BitBlock {
	type Store = B;
	type Order = Lsb0;

	const EMPTY: Self = Self {
		storage:  Vec::new(),
		nbits: 0,
	};

	fn push(&mut self, value: bool) {
		self.push(value);
	}

	fn from_bitslice(slice: &BitSlice<Self::Store, Self::Order>) -> Self {
		Self::from_iter(slice.iter().by_vals())
	}

	fn as_bitslice(&self) -> &BitSlice<B, Lsb0> {
		&BitSlice::from_slice(&self.storage[..])[..self.nbits]
	}

	fn as_mut_bitslice(&mut self) -> &mut BitSlice<B, Lsb0> {
		&mut BitSlice::from_slice_mut(&mut self.storage[..])[..self.nbits]
	}
}

#[cfg(test)]
mod tests {
    use super::prelude;
    use crate::BitVec;
    use crate::extra::BitVecLike;

    #[test]
    fn test_as_extra() {
        let bv = BitVec::from_elem(32 + 21, true);
        let conv = bv.as_bitslice();
        assert_eq!(conv.len(), 32 + 21);
        for i in 0..32 + 21 {
            assert!(conv[i]);
        }
        for i in 32 + 21..64 {
            assert!(conv.get(i).is_none());
        }
    }
}
