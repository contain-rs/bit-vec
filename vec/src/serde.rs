use crate::local_prelude::*;

#[cfg(any(feature = "serde", feature = "borsh"))]
use core::{error, fmt};

#[cfg(any(feature = "serde", feature = "borsh", feature = "miniserde"))]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
#[cfg_attr(feature = "borsh", derive(borsh::BorshDeserialize))]
#[cfg(any(feature = "serde", feature = "borsh"))]
pub struct UncheckedBitVec<B: BitBlock = u32, C: BitContainer<Block = B> = Vec<B>> {
    /// Internal representation of the bit vector
    pub(crate) storage: C,
    /// The number of valid bits in the internal representation
    pub(crate) nbits: usize,
}

#[cfg(feature = "serde")]
impl<'de, C: BitContainer> serde::Deserialize<'de> for BitVec<C::Block, C>
where
    C: serde::Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::Error;
        UncheckedBitVec::<C::Block, C>::deserialize(deserializer).and_then(|unchecked| {
            let result = BitVec {
                storage: unchecked.storage,
                nbits: unchecked.nbits,
            };
            if !result.storage_len_matches_nbits() {
                Err(D::Error::custom(DeserializeError::StorageLenMismatch))
            } else if !result.is_last_block_fixed() {
                Err(D::Error::custom(DeserializeError::TrailingBits))
            } else {
                Ok(result)
            }
        })
    }
}

#[cfg(feature = "borsh")]
impl<C: BitContainer> borsh::BorshDeserialize for BitVec<C::Block, C>
where
    C: borsh::BorshDeserialize,
{
    fn deserialize_reader<R: borsh::io::Read>(reader: &mut R) -> borsh::io::Result<Self> {
        UncheckedBitVec::<C::Block, C>::deserialize_reader(reader).and_then(|unchecked| {
            let result = BitVec {
                storage: unchecked.storage,
                nbits: unchecked.nbits,
            };
            if !result.storage_len_matches_nbits() {
                Err(borsh::io::Error::other(
                    DeserializeError::StorageLenMismatch,
                ))
            } else if !result.is_last_block_fixed() {
                Err(borsh::io::Error::other(DeserializeError::TrailingBits))
            } else {
                Ok(result)
            }
        })
    }
}

#[cfg(feature = "miniserde")]
miniserde::make_place!(Place);

#[cfg(feature = "miniserde")]
struct BitVecBuilder<'a, C: BitContainer> {
    storage: Option<C>,
    nbits: Option<usize>,
    out: &'a mut Option<BitVec<C::Block, C>>,
}

#[cfg(feature = "miniserde")]
impl<C: BitContainer> miniserde::de::Visitor for Place<BitVec<C::Block, C>>
where
    C: miniserde::Deserialize,
    C::Block: miniserde::Deserialize,
{
    fn map(&mut self) -> miniserde::Result<Box<dyn miniserde::de::Map + '_>> {
        Ok(Box::new(BitVecBuilder {
            storage: None,
            nbits: None,
            out: &mut self.out,
        }))
    }
}

#[cfg(feature = "miniserde")]
impl<C: BitContainer> miniserde::de::Map for BitVecBuilder<'_, C>
where
    C: miniserde::Deserialize,
{
    fn key(&mut self, k: &str) -> miniserde::Result<&mut dyn miniserde::de::Visitor> {
        match k {
            "storage" => Ok(miniserde::Deserialize::begin(&mut self.storage)),
            "nbits" => Ok(miniserde::Deserialize::begin(&mut self.nbits)),
            _ => Ok(<dyn miniserde::de::Visitor>::ignore()),
        }
    }

    fn finish(&mut self) -> miniserde::Result<()> {
        let storage = self.storage.take().ok_or(miniserde::Error)?;
        let nbits = self.nbits.take().ok_or(miniserde::Error)?;
        let result = BitVec { storage, nbits };
        if !result.storage_len_matches_nbits() || !result.is_last_block_fixed() {
            Err(miniserde::Error)
        } else {
            *self.out = Some(result);
            Ok(())
        }
    }
}

#[cfg(feature = "miniserde")]
impl<C: BitContainer> miniserde::Deserialize for BitVec<C::Block, C>
where
    C: miniserde::Deserialize,
    C::Block: miniserde::Deserialize,
{
    fn begin(out: &mut Option<Self>) -> &mut dyn miniserde::de::Visitor {
        Place::new(out)
    }
}

#[cfg(any(feature = "serde", feature = "borsh"))]
#[derive(Debug)]
pub enum DeserializeError {
    StorageLenMismatch,
    TrailingBits,
}

#[cfg(any(feature = "serde", feature = "borsh"))]
impl fmt::Display for DeserializeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.simple_description())
    }
}

#[cfg(feature = "borsh")]
impl From<DeserializeError> for String {
    fn from(value: DeserializeError) -> Self {
        value.simple_description().to_owned()
    }
}

#[cfg(any(feature = "serde", feature = "borsh"))]
impl error::Error for DeserializeError {}

#[cfg(any(feature = "serde", feature = "borsh"))]
impl DeserializeError {
    fn simple_description(&self) -> &str {
        match self {
            DeserializeError::TrailingBits => "some out of bounds trailing bits are set",
            DeserializeError::StorageLenMismatch => "nbits and storage length isnt same",
        }
    }
}
