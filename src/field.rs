// Code adapted from https://github.com/serai-dex/serai/blob/develop/crypto/dalek-ff-group/src/field.rs which is MIT licensed

use core::ops::{Add, AddAssign, Sub, SubAssign, Neg, Mul, MulAssign};
use core::iter::{Sum, Product};
use std::borrow::Borrow;
use subtle::{Choice, CtOption, ConstantTimeEq, ConstantTimeLess, ConditionallySelectable};
use ff::{PrimeField, Field, PrimeFieldBits, FieldBits};
use crypto_bigint::{Encoding, Integer, U256, U512};
use crypto_bigint::rand_core::RngCore;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Fe25519(pub(crate) U256);

const MODULUS: U256 =
    U256::from_be_hex("7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffed");

const WIDE_MODULUS: U512 = U512::from_be_hex(concat!(
    "0000000000000000000000000000000000000000000000000000000000000000",
    "7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffed"
));

const SQRT_MINUS_ONE: Fe25519 =
    Fe25519(U256::from_be_hex
    ("2b8324804fc1df0b2b4d00993dfbd7a72f431806ad2fe478c4ee1b274a0ea0b0"));

pub const FIELD_MODULUS: Fe25519 = Fe25519(MODULUS);

fn reduce(x: U512) -> U256 {
    U256::from_le_slice(&x.checked_rem(&WIDE_MODULUS).unwrap().to_le_bytes()[.. 32])
}

impl Add<Fe25519> for Fe25519 {
    type Output = Fe25519;

    fn add(self, rhs: Self) -> Self::Output {
        Self(U256::add_mod(&self.0, &rhs.0, &MODULUS))
    }
}
impl AddAssign<Fe25519> for Fe25519 {
    fn add_assign(&mut self, rhs: Self) {
        self.0 = U256::add_mod(&self.0, &rhs.0, &MODULUS);
    }
}

impl<'a> Add<&'a Fe25519> for Fe25519 {
    type Output = Fe25519;

    fn add(self, rhs: &'a Fe25519) -> Self::Output {
        Self(U256::add_mod(&self.0, &rhs.0, &MODULUS))
    }
}

impl<'a> AddAssign<&'a Fe25519> for Fe25519 {
    fn add_assign(&mut self, rhs: &'a Fe25519) {
        self.0 = U256::add_mod(&self.0, &rhs.0, &MODULUS);
    }
}

impl Sub<Fe25519> for Fe25519 {
    type Output = Fe25519;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(U256::sub_mod(&self.0, &rhs.0, &MODULUS))
    }
}
impl SubAssign<Fe25519> for Fe25519 {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 = U256::sub_mod(&self.0, &rhs.0, &MODULUS);
    }
}

impl<'a> Sub<&'a Fe25519> for Fe25519 {
    type Output = Fe25519;

    fn sub(self, rhs: &'a Fe25519) -> Self::Output {
        Self(U256::sub_mod(&self.0, &rhs.0, &MODULUS))
    }
}

impl<'a> SubAssign<&'a Fe25519> for Fe25519 {
    fn sub_assign(&mut self, rhs: &'a Fe25519) {
        self.0 = U256::sub_mod(&self.0, &rhs.0, &MODULUS);
    }
}

impl Mul<Fe25519> for Fe25519 {
    type Output = Fe25519;

    fn mul(self, rhs: Self) -> Self::Output {
        let wide = U256::mul_wide(&self.0, &rhs.0);
        Self(reduce(U512::from((wide.0, wide.1))))
    }
}
impl MulAssign<Fe25519> for Fe25519 {
    fn mul_assign(&mut self, rhs: Self) {
        let wide = U256::mul_wide(&self.0, &rhs.0);
        self.0 = reduce(U512::from((wide.0, wide.1)));
    }
}

impl<'a> Mul<&'a Fe25519> for Fe25519 {
    type Output = Fe25519;

    fn mul(self, rhs: &'a Fe25519) -> Self::Output {
        let wide = U256::mul_wide(&self.0, &rhs.0);
        Self(reduce(U512::from((wide.0, wide.1))))
    }
}

impl<'a> MulAssign<&'a Fe25519> for Fe25519 {
    fn mul_assign(&mut self, rhs: &'a Fe25519) {
        let wide = U256::mul_wide(&self.0, &rhs.0);
        self.0 = reduce(U512::from((wide.0, wide.1)));
    }
}

impl Neg for Fe25519 {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self(self.0.neg_mod(&MODULUS))
    }
}
  
impl<'a> Neg for &'a Fe25519 {
    type Output = Fe25519;
    fn neg(self) -> Self::Output {
        (*self).neg()
    }
}

impl ConstantTimeEq for Fe25519 {
    fn ct_eq(&self, other: &Self) -> Choice {
        self.0.ct_eq(&other.0)
    }
}

impl ConditionallySelectable for Fe25519 {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
      Fe25519(U256::conditional_select(&a.0, &b.0, choice))
    }
}

impl<T> Sum<T> for Fe25519
where
    T: Borrow<Fe25519>
{
    fn sum<I: Iterator<Item = T>>(iter: I) -> Self {
        iter.fold(Fe25519::ZERO, |acc, item| acc + item.borrow())
    }
}

impl<T> Product<T> for Fe25519
where
    T: Borrow<Fe25519>
{
    fn product<I: Iterator<Item = T>>(iter: I) -> Self {
        iter.fold(Fe25519::ONE, |acc, item| acc * item.borrow())
    }
}

impl From<u64> for Fe25519 {
    fn from(value: u64) -> Self {
        Self(U256::from_u64(value))
    }
}

impl AsRef<[u64]> for Fe25519 {
    fn as_ref(&self) -> &[u64] {
        self.0.as_words()
    }
}

impl Field for Fe25519 {

    const ZERO: Self = Self(U256::from_u8(0));
    const ONE: Self = Self(U256::from_u8(1));


    fn random(mut rng: impl RngCore) -> Self {
        let mut bytes = [0; 32];
        rng.fill_bytes(&mut bytes);
        Self(U256::from_le_bytes(bytes).checked_rem(&MODULUS).unwrap())
    }

    fn square(&self) -> Self {
        Self(reduce(self.0.square()))
    }

    fn double(&self) -> Self {
        Self((self.0 << 1).checked_rem(&MODULUS).unwrap())
    }

    fn invert(&self) -> CtOption<Self> {
        const NEG_2: Fe25519 = Self(MODULUS.saturating_sub(&U256::from_u8(2)));
        CtOption::new(self.pow_vartime(NEG_2), !self.is_zero())
    }

    fn sqrt_ratio(num: &Self, div: &Self) -> (Choice, Self) {
        ff::helpers::sqrt_ratio_generic(num, div)
    }

    // https://www.rfc-editor.org/rfc/rfc8032#section-5.1.3
    fn sqrt(&self) -> CtOption<Self> {
        let three = Fe25519::from(3u64);
        let one_by_eight = Fe25519::from(8u64).invert().unwrap();
        let exponent = (FIELD_MODULUS + three)*one_by_eight;
        let beta = self.pow_vartime(exponent); // candidate square root
        
        let beta_sq = beta.square();
        let is_sq_root = (beta_sq - self).is_zero() | (beta_sq + self).is_zero();

        let neg_not_required = (beta_sq - self).is_zero();
        let sq_root = if bool::from(neg_not_required) {
            beta
        }
        else {
            beta*SQRT_MINUS_ONE
        };
        CtOption::new(sq_root, is_sq_root)
    }

}

impl PrimeField for Fe25519 {
    type Repr = [u8; 32];

    const MODULUS: &'static str = "0x7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffed";
    const NUM_BITS: u32 = 255;
    const CAPACITY: u32 = 254;
    const TWO_INV: Self = Self(U256::from_be_hex("3ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7"));
    const MULTIPLICATIVE_GENERATOR: Self = Self(U256::from_u8(2));
    const S: u32 = 2;
    const ROOT_OF_UNITY: Self = Self(U256::from_be_hex("2b8324804fc1df0b2b4d00993dfbd7a72f431806ad2fe478c4ee1b274a0ea0b0"));
    const ROOT_OF_UNITY_INV: Self = Self(U256::from_be_hex("547cdb7fb03e20f4d4b2ff66c2042858d0bce7f952d01b873b11e4d8b5f15f3d"));
    const DELTA: Self = Self(U256::from_u8(16));

    fn from_repr(bytes: [u8; 32]) -> CtOption<Self> {
        let res = Self(U256::from_le_bytes(bytes));
        CtOption::new(res, res.0.ct_lt(&MODULUS))
    }

    fn to_repr(&self) -> [u8; 32] {
        self.0.to_le_bytes()
    }

    fn is_odd(&self) -> Choice {
        self.0.is_odd()
    }

    // fn multiplicative_generator() -> Self {
    //     Self(U256::from_u8(2))
    // }

    // fn root_of_unity() -> Self {
    //     Self(U256::from_be_hex("2b8324804fc1df0b2b4d00993dfbd7a72f431806ad2fe478c4ee1b274a0ea0b0"))
    // }
}

impl PrimeFieldBits for Fe25519 {
    type ReprBits = [u8; 32];
  
    fn to_le_bits(&self) -> FieldBits<Self::ReprBits> {
        self.to_repr().into()
    }
  
    fn char_le_bits() -> FieldBits<Self::ReprBits> {
        MODULUS.to_le_bytes().into()
    }
}

impl Fe25519 {
    pub fn get_value(&self) -> U256 {
        self.0
    }
}
  

#[cfg(test)]
mod tests {
    use super::*;
    // const TWO_INV: Fe25519 =
    //     Fe25519(U256::from_be_hex("3ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7"));

    // const ROOT_OF_UNITY_INV: Fe25519 =
    //     Fe25519(U256::from_be_hex("547cdb7fb03e20f4d4b2ff66c2042858d0bce7f952d01b873b11e4d8b5f15f3d"));

    // const DELTA: Fe25519 = Fe25519(U256::from_u8(16));


    #[test]
    fn sanity_checks() {
        let two = Fe25519::from(2u64);
        let four = two.mul(two);
        let expected_four = Fe25519::from(4u64);

        assert_eq!(four, expected_four);
        assert_eq!(Fe25519::TWO_INV*two, Fe25519::ONE);
        assert_eq!(two.pow_vartime(two), four);
        assert_eq!(two.square(), four);

        let two_inv = two.invert().unwrap();
        assert_eq!(two_inv, Fe25519::TWO_INV);
    }

    #[test]
    fn check_params() {
        assert_eq!(Fe25519::ROOT_OF_UNITY*Fe25519::ROOT_OF_UNITY_INV, Fe25519::ONE);
        assert_eq!(Fe25519::MULTIPLICATIVE_GENERATOR.pow_vartime(Fe25519(MODULUS)-Fe25519::ONE), Fe25519::ONE);
        
        let two = Fe25519::from(2u64);
        let two_pow_s = two.pow_vartime(Fe25519::from(Fe25519::S as u64));
        assert_eq!(Fe25519::ROOT_OF_UNITY.pow_vartime(two_pow_s), Fe25519::ONE);

        let t = (Fe25519(MODULUS)-Fe25519::ONE) * (two_pow_s.invert().unwrap());
        assert_eq!(Fe25519::DELTA.pow_vartime(t), Fe25519::ONE);
    }

    #[test]
    fn check_add_sub() {
        let mut rng = rand::thread_rng();
        let mut x = Fe25519::random(&mut rng);
        let y = Fe25519::random(&mut rng);
        let neg_y = -y;
        assert_eq!(y+neg_y, Fe25519::ZERO);
        assert_eq!(x+neg_y, x-y);

        let old_x = x;
        x -= y;
        assert_eq!(x + y, old_x);
        x += y;
        assert_eq!(x, old_x);
    
        let a = [x, Fe25519::from(1u64), Fe25519::from(2u64), y];
        assert_eq!(Fe25519::sum(a.iter()), x+y+Fe25519::from(3u64));
    
        let y_ref = &y;
        assert_eq!(x+y, x+y_ref);
        x += y_ref;
        assert_eq!(x-y, x-y_ref);
        x -= y_ref;
        assert_eq!(x+y, x+y_ref);
    }

    #[test]
    fn check_mul() {
        let mut rng = rand::thread_rng();
        let mut x = Fe25519::random(&mut rng);
        let y = Fe25519::random(&mut rng);
        assert_eq!(x.invert().unwrap()*x, Fe25519::ONE);

        let old_x = x;
        x *= y;
        assert_eq!(x*y.invert().unwrap(), old_x);

        let a = [x, Fe25519::from(2u64), Fe25519::from(3u64), y];
        assert_eq!(Fe25519::product(a.iter()), x*y*Fe25519::from(6u64));

        let y_ref = &y;
        assert_eq!(x*y, x*y_ref);
        x *= y_ref;
        assert_eq!(x*y.invert().unwrap(), x*y_ref.invert().unwrap());
    }

    #[test]
    fn check_repr() {
        let mut rng = rand::thread_rng();
        let x = Fe25519::random(&mut rng);
        let repr = x.to_repr();
        let roundtrip_x = Fe25519::from_repr(repr).unwrap();

        assert_eq!(x, roundtrip_x);
    }
    
    #[test]
    fn check_square_double() {
        let mut rng = rand::thread_rng();
        let x = Fe25519::random(&mut rng);
        assert_eq!(x.square(), x*x);
        assert_eq!(x.double(), x+x);
        let two = Fe25519::from(2u64);
        assert_eq!(x.double(), x*two);
    }

    #[test]
    fn check_cteq() {
        let two = Fe25519::from(2u64);
        let two_alt = Fe25519::ONE + Fe25519::ONE;
        assert!(bool::from(two.ct_eq(&two_alt)));
    }

    #[test]
    fn check_conditonal_select() {
        let one = Fe25519::ONE;
        let two = Fe25519::from(2u64);
        let x = Fe25519::conditional_select(&one, &two, Choice::from(0u8));
        let y = Fe25519::conditional_select(&one, &two, Choice::from(1u8));
        assert_eq!(x, one);
        assert_eq!(y, two);
    }

    #[test]
    fn check_square_root() {
        let two = Fe25519::from(2u64);
        assert!(bool::from(two.sqrt().is_none()));

        let four = two.square();
        let sq_root = Fe25519::sqrt(&four);
        assert!(bool::from(sq_root.is_some()));
        assert!(sq_root.unwrap() == two || sq_root.unwrap() == -two);

        let mut rng = rand::thread_rng();
        let x = Fe25519::random(&mut rng);
        let x_sq = x.square();
        let sq_root = Fe25519::sqrt(&x_sq);
        assert!(bool::from(sq_root.is_some()));
        assert!(sq_root.unwrap() == x || sq_root.unwrap() == -x);
        assert_eq!(sq_root.unwrap().square(), x_sq);
    }

}