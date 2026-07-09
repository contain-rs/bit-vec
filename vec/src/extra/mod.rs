use crate::{BitBlock, BitVec};

use bit_vec_extra::order::Lsb0;
pub use bit_vec_extra::prelude;

impl<B: BitBlock> BitVec<B> {
    pub fn as_extra(&self) -> &prelude::BitSlice<B, Lsb0>
    where
        B: prelude::BitStore,
    {
        &prelude::BitSlice::from_slice(&self.storage[..])[0..self.nbits]
    }

    pub fn as_mut_extra(&mut self) -> &mut prelude::BitSlice<B, Lsb0>
    where
        B: prelude::BitStore,
    {
        &mut prelude::BitSlice::from_slice_mut(&mut self.storage[..])[0..self.nbits]
    }

    // fn into_extra(self) -> prelude::BitVec<B, Lsb0> {
    // }
}

#[cfg(test)]
mod tests {
    use super::prelude;
    use crate::BitVec;

    #[test]
    fn test_as_extra() {
        let bv = BitVec::from_elem(32 + 21, true);
        let conv = bv.as_extra();
        assert_eq!(conv.len(), 32 + 21);
        for i in 0..32 + 21 {
            assert!(conv[i]);
        }
        for i in 32 + 21..64 {
            assert!(conv.get(i).is_none());
        }
    }
}
