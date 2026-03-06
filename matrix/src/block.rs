//! Defines a building block for the bit slice.
//! 
//! Bits are stored in blocks. When the last block
//! is not full with bits, we waste some space.

/// The number of bits in a block.
pub const BITS: usize = 32;
/// The type used as storage for bits.
pub type Block = u32;
