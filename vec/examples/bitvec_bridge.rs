use bit_vec::BitVec;
use bitvec::{
    field::BitField,
    order::Lsb0,
    view::BitView,
};

fn main() {
    let mut bv = BitVec::<u32>::from_elem(80, false);
    bv.set(4, true);

    let bits = bv.storage().view_bits::<Lsb0>();
    let val: u16 = bits[4..20].load_le();
    println!("len={}, bits[4]={}, loaded u16={}", bv.len(), bits[4], val);
}
