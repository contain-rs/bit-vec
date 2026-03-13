use crate::local_prelude::*;

macro_rules! bound_combination {
    (
        type $T:ident: [$($B:tt)*];
        $cfg0:tt => [$($Bounds0:tt)*];
        $(
            $cfg:tt => [$($Bounds:tt)*];
        )*
    ) => {
        #[cfg(not(feature = $cfg0))]
        bound_combination!(
            type $T: [$($B)*];
            $(
                $cfg => [$($Bounds)*];
            )*
        );
        #[cfg(feature = $cfg0)]
        bound_combination!(
            type $T: [$($B)* + $($Bounds0)*];
            $(
                $cfg => [$($Bounds)*];
            )*
        );
    };
    (
        type $T:ident: [$($B:tt)*];
    ) => {
        type $T: $($B)*;
    }
}

pub trait BitBlockOrStore {
    bound_combination!(
        type Store: [BitStore];
        "nanoserde" => [DeBin + DeJson + DeRon + SerBin + SerJson + SerRon];
        "serde" => [serde::Serialize + for<'a> serde::Deserialize<'a>];
        "miniserde" => [miniserde::Deserialize + miniserde::Serialize];
        "borsh" => [borsh::BorshDeserialize + borsh::BorshSerialize];
    );

    const BITS: usize = <Self::Store as BitStore>::Block::BITS_;
    const BYTES: usize = <Self::Store as BitStore>::Block::BYTES_;
    const ONE: <Self::Store as BitStore>::Block = <Self::Store as BitStore>::Block::ONE_;
    const ZERO: <Self::Store as BitStore>::Block = <Self::Store as BitStore>::Block::ZERO_;
}

macro_rules! impl_combination {
    (
        type $T:ty: [$($B:tt)*];
        $cfg0:tt => [$($Bounds0:tt)*];
        $(
            $cfg:tt => [$($Bounds:tt)*];
        )*
    ) => {
        #[cfg(not(feature = $cfg0))]
        impl_combination!(
            type $T: [$($B)*];
            $(
                $cfg => [$($Bounds)*];
            )*
        );
        #[cfg(feature = $cfg0)]
        impl_combination!(
            type $T: [$($B)* + $($Bounds0)*];
            $(
                $cfg => [$($Bounds)*];
            )*
        );
    };
    (
        type $T:ty: [$B0:tt + $($B:tt)*];
    ) => {
        impl<T: $B0 + $($B)*> BitBlockOrStore for Vec<T> {
            type Store = Self;
        }

        impl<T: $B0 + $($B)*> BitBlockOrStore for Box<Vec<T>> where Self: $($B)* {
            type Store = Self;
        }
    }
}

impl_combination!(
    type Vec<T>: [BitBlock];
    "nanoserde" => [DeBin + DeJson + DeRon + SerBin + SerJson + SerRon];
    "serde" => [serde::Serialize + for<'a> serde::Deserialize<'a>];
    "miniserde" => [miniserde::Deserialize + miniserde::Serialize];
    "borsh" => [borsh::BorshDeserialize + borsh::BorshSerialize];
);

#[cfg(all(feature = "smallvec", not(feature = "nanoserde")))]
impl<A: smallvec::Array> BitBlockOrStore for smallvec::SmallVec<A>
where
    A::Item: BitBlock,
{
    type Store = Self;
}

macro_rules! bit_block_or_store_impl {
    ($($t: ident),*) => ($(
        impl BitBlockOrStore for $t {
            type Store = Vec<Self>;
        }
    )*)
}

bit_block_or_store_impl! {
    u8,
    u16,
    u32,
    u64,
    usize
}
