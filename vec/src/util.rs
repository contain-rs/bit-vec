use crate::local_prelude::*;

pub(crate) type Block<B: BitBlockOrStore> = <B::Store as BitStore>::Block;

pub static TRUE: bool = true;
pub static FALSE: bool = false;

pub fn reverse_bits(byte: u8) -> u8 {
    let mut result = 0;
    for i in 0..u8::BITS {
        result |= ((byte >> i) & 1) << (u8::BITS - 1 - i);
    }
    result
}
