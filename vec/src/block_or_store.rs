use crate::local_prelude::*;

// macro_rules! bound_combination {
//     (
//         type $T:ident: [$($B:tt)*];
//         $cfg0:tt => [$($Bounds0:tt)*];
//         $(
//             $cfg:tt => [$($Bounds:tt)*];
//         )*
//     ) => {
//         #[cfg(not(feature = $cfg0))]
//         bound_combination!(
//             type $T: [$($B)*];
//             $(
//                 $cfg => [$($Bounds)*];
//             )*
//         );
//         #[cfg(feature = $cfg0)]
//         bound_combination!(
//             type $T: [$($B)* + $($Bounds0)*];
//             $(
//                 $cfg => [$($Bounds)*];
//             )*
//         );
//     };
//     (
//         type $T:ident: [$($B:tt)*];
//     ) => {
//         type $T: $($B)*;
//     }
// }

pub trait BitBlockOrStore {
    // bound_combination!(
    //     type Store: [BitStore];
    //     "nanoserde" => [DeBin + DeJson + DeRon + SerBin + SerJson + SerRon];
    //     "serde" => [serde::Serialize + for<'a> serde::Deserialize<'a>];
    //     "miniserde" => [miniserde::Deserialize + miniserde::Serialize];
    //     "borsh" => [borsh::BorshDeserialize + borsh::BorshSerialize];
    // );
    type Store: BitStore;

    const BITS: usize = <Self::Store as BitStore>::Block::BITS_;
    const BYTES: usize = <Self::Store as BitStore>::Block::BYTES_;
    const ONE: <<Self::Store as BitStore>::Block as BitBlock>::Target = <Self::Store as BitStore>::Block::ONE_;
    const ZERO: <<Self::Store as BitStore>::Block as BitBlock>::Target = <Self::Store as BitStore>::Block::ZERO_;
}

pub trait CloneableBitBlockOrStore: BitBlockOrStore<Store: Clone> {}

#[cfg(feature = "serde")]
pub trait SerdeBitBlockOrStore<'de>: BitBlockOrStore<Store: serde::Serialize + serde::Deserialize<'de>> {}

// macro_rules! impl_combination {
//     (
//         type $T:ty: $B0:tt;
//         $cfg0:tt => $([$($Bounds0:tt)*])*;
//         $(
//             $cfg:tt => $([$($Bounds:tt)+])*;
//         )*
//     ) => {
//         #[cfg(not(feature = $cfg0))]
//         impl_combination!(
//             type $T: $B0;
//             $(
//                 $cfg => $([$($Bounds)+])*;
//             )*
//         );
//         #[cfg(feature = $cfg0)]
//         impl_combination!(
//             type $T: $B0 $(+ [$($Bounds0)*])*;
//             $(
//                 $cfg => $([$($Bounds)+])*;
//             )*
//         );
//     };
//     (
//         type $T:ty: $B0:tt $(+ [$($B:tt)+])*;
//         $cfg0:tt => $([$($Bounds0:tt)+])*;
//         $(
//             $cfg:tt => $([$($Bounds:tt)+])*;
//         )*
//     ) => {
//         #[cfg(not(feature = $cfg0))]
//         impl_combination!(
//             type $T: $B0 $(+ [$($B)+])*;
//             $(
//                 $cfg => $([$($Bounds)+])*;
//             )*
//         );
//         #[cfg(feature = $cfg0)]
//         impl_combination!(
//             type $T: $B0 $(+ [$($B)+])* $(+ [$($Bounds0)+])*;
//             $(
//                 $cfg => $([$($Bounds)+])*;
//             )*
//         );
//     };
//     (
//         type $T:ty: $B0:tt $(+ [$($B:tt)+])*;
//     ) => {
//         impl<T: $B0 $(+ $($B)+)*> BitBlockOrStore for Vec<T> {
//             type Store = Self;
//         }

//         // Note: The extra `Sized` is there just to fill the place before the `+`.
//         impl<T: $B0 $(+ $($B)+)*> BitBlockOrStore for Box<Vec<T>> where Self: Sized $(+ $($B)+)* {
//             type Store = Self;
//         }

//         impl<T: $B0 $(+ $($B)+)* + Clone> CloneableBitBlockOrStore for Vec<T> {}
//     }
// }

// impl_combination!(
//     type Vec<T>: BitBlock;
//     "nanoserde" => [DeBin] [DeJson] [DeRon] [SerBin] [SerJson] [SerRon];
//     "serde" => [serde::Serialize] [for<'a> serde::Deserialize<'a>];
//     "miniserde" => [miniserde::Deserialize] [miniserde::Serialize];
//     "borsh" => [borsh::BorshDeserialize] [borsh::BorshSerialize];
// );

impl<T: BitBlock> BitBlockOrStore for Vec<T> {
    type Store = Self;
}

impl<T: BitBlock> BitBlockOrStore for Box<Vec<T>> {
    type Store = Self;
}

impl<T: BitBlock + Clone> CloneableBitBlockOrStore for Vec<T> {}

impl<T: BitBlock + Clone> CloneableBitBlockOrStore for Box<Vec<T>> {}

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

        impl CloneableBitBlockOrStore for $t {}

        #[cfg(feature = "serde")]
        impl<'de> SerdeBitBlockOrStore<'de> for $t {}
        #[cfg(feature = "serde")]
        impl<'de> SerdeBitBlockOrStore<'de> for Vec<$t> {}
    )*)
}

bit_block_or_store_impl! {
    u8,
    u16,
    u32,
    u64,
    usize
}
