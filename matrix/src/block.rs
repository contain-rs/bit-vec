use bit_vec::{BitBlockOrStore, BitStore};

#[allow(type_alias_bounds)]
pub(crate) type Block<B: BitBlockOrStore> = <B::Store as BitStore>::Block;
