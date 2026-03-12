use crate::local_prelude::*;

#[allow(clippy::len_without_is_empty)]
pub trait BitStore: Clone {
    type Block: BitBlock;
    type Alloc: Default;
    fn new_in(alloc: Self::Alloc) -> Self;
    fn slice(&self) -> &[Self::Block];
    fn slice_mut(&mut self) -> &mut [Self::Block];
    fn len(&self) -> usize {
        self.slice().len()
    }
    fn pop(&mut self) -> Option<Self::Block>;
    fn drain<R: ops::RangeBounds<usize>>(&mut self, range: R) -> impl Iterator<Item = Self::Block>;
    fn capacity(&self) -> usize;
    fn append(&mut self, other: &mut Self);
    fn reserve(&mut self, additional: usize);
    fn push(&mut self, value: Self::Block);
    fn split_off(&mut self, at: usize) -> Self;
    fn truncate(&mut self, len: usize);
    fn reserve_exact(&mut self, len: usize);
    fn shrink_to_fit(&mut self);
    fn extend<T>(&mut self, iter: T)
    where
        T: IntoIterator<Item = Self::Block>;
    fn with_capacity(capacity: usize) -> Self;
    fn clear(&mut self);
    fn with_capacity_in(capacity: usize, alloc: Self::Alloc) -> Self;
}

#[cfg(not(feature = "allocator_api"))]
impl<T: BitBlock> BitStore for Vec<T> {
    type Block = T;
    type Alloc = ();

    fn new_in(_alloc: Self::Alloc) -> Self {
        Vec::new()
    }

    fn slice(&self) -> &[Self::Block] {
        &self[..]
    }

    fn slice_mut(&mut self) -> &mut [Self::Block] {
        &mut self[..]
    }

    fn pop(&mut self) -> Option<Self::Block> {
        Vec::pop(self)
    }

    fn drain<R: ops::RangeBounds<usize>>(&mut self, range: R) -> impl Iterator<Item = Self::Block> {
        Vec::drain(self, range)
    }

    fn capacity(&self) -> usize {
        Vec::capacity(self)
    }

    fn append(&mut self, other: &mut Self) {
        Vec::append(self, other);
    }

    fn reserve(&mut self, additional: usize) {
        Vec::reserve(self, additional);
    }

    fn push(&mut self, value: Self::Block) {
        Vec::push(self, value);
    }

    fn split_off(&mut self, at: usize) -> Self {
        Vec::split_off(self, at)
    }

    fn truncate(&mut self, len: usize) {
        Vec::truncate(self, len);
    }

    fn reserve_exact(&mut self, len: usize) {
        Vec::reserve_exact(self, len);
    }

    fn shrink_to_fit(&mut self) {
        Vec::shrink_to_fit(self);
    }

    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = Self::Block>,
    {
        Extend::extend(self, iter);
    }

    fn with_capacity_in(capacity: usize, _alloc: Self::Alloc) -> Self {
        Vec::with_capacity(capacity)
    }

    fn with_capacity(capacity: usize) -> Self {
        Vec::with_capacity(capacity)
    }

    fn clear(&mut self) {
        Vec::clear(self)
    }
}

#[cfg(feature = "allocator_api")]
impl<T: BitBlock, A> BitStore for Vec<T, A>
where
    A: core::alloc::Allocator + Clone + Default,
{
    type Block = T;
    type Alloc = A;

    fn new_in(alloc: Self::Alloc) -> Self {
        Vec::new_in(alloc)
    }

    fn slice(&self) -> &[Self::Block] {
        &self[..]
    }

    fn slice_mut(&mut self) -> &mut [Self::Block] {
        &mut self[..]
    }

    fn pop(&mut self) -> Option<Self::Block> {
        Vec::pop(self)
    }

    fn drain<R: RangeBounds<usize>>(&mut self, range: R) -> impl Iterator<Item = Self::Block> {
        Vec::drain(self, range)
    }

    fn capacity(&self) -> usize {
        Vec::capacity(self)
    }

    fn append(&mut self, other: &mut Self) {
        Vec::append(self, other);
    }

    fn reserve(&mut self, additional: usize) {
        Vec::reserve(self, additional);
    }

    fn push(&mut self, value: Self::Block) {
        Vec::push(self, value);
    }

    fn split_off(&mut self, at: usize) -> Self {
        Vec::split_off(self, at)
    }

    fn truncate(&mut self, len: usize) {
        Vec::truncate(self, len);
    }

    fn reserve_exact(&mut self, len: usize) {
        Vec::reserve_exact(self, len);
    }

    fn shrink_to_fit(&mut self) {
        Vec::shrink_to_fit(self);
    }

    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = Self::Block>,
    {
        Extend::extend(self, iter);
    }

    fn with_capacity_in(capacity: usize, alloc: A) -> Self {
        Vec::with_capacity_in(capacity, alloc)
    }

    fn with_capacity(capacity: usize) -> Self {
        Vec::with_capacity_in(capacity, A::default())
    }
}

#[cfg(feature = "smallvec")]
impl<A: smallvec::Array> BitStore for smallvec::SmallVec<A>
where
    A::Item: BitBlock,
{
    type Block = A::Item;
    type Alloc = ();

    fn slice(&self) -> &[Self::Block] {
        &self[..]
    }

    fn slice_mut(&mut self) -> &mut [Self::Block] {
        &mut self[..]
    }

    fn pop(&mut self) -> Option<Self::Block> {
        self.pop()
    }

    fn drain<R: ops::RangeBounds<usize>>(&mut self, range: R) -> impl Iterator<Item = Self::Block> {
        self.drain(range)
    }

    fn capacity(&self) -> usize {
        self.capacity()
    }

    fn append(&mut self, other: &mut Self) {
        self.append(other);
    }

    fn reserve(&mut self, additional: usize) {
        self.reserve(additional);
    }

    fn push(&mut self, value: Self::Block) {
        self.push(value);
    }

    fn split_off(&mut self, at: usize) -> Self {
        // TODO
        self.to_vec().split_off(at).into()
    }

    fn truncate(&mut self, len: usize) {
        self.truncate(len);
    }

    fn reserve_exact(&mut self, len: usize) {
        self.reserve_exact(len);
    }

    fn shrink_to_fit(&mut self) {
        self.shrink_to_fit();
    }

    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = Self::Block>,
    {
        iter::Extend::extend(self, iter);
    }

    fn with_capacity(capacity: usize) -> Self {
        smallvec::SmallVec::with_capacity(capacity)
    }

    fn clear(&mut self) {
        self.clear();
    }

    fn new_in(_alloc: ()) -> Self {
        smallvec::SmallVec::new()
    }

    fn with_capacity_in(capacity: usize, _alloc: ()) -> Self {
        smallvec::SmallVec::with_capacity(capacity)
    }
}
