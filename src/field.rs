// Code adapted from https://github.com/serai-dex/serai/blob/develop/crypto/dalek-ff-group/src/field.rs which is MIT licensed

use core::ops::{Add, AddAssign, Sub, SubAssign, Neg, Mul, MulAssign};
use core::iter::{Sum, Product};
use std::borrow::Borrow;
use subtle::{Choice, CtOption, ConstantTimeEq, ConstantTimeLess, ConditionallySelectable};
use ff::{PrimeField, Field, PrimeFieldBits, FieldBits};
use crypto_bigint::{Encoding, Integer, U256, U512};
use crypto_bigint::rand_core::RngCore;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Fp(pub U256);

const MODULUS: U256 =
    U256::from_be_hex("7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffed");

const WIDE_MODULUS: U512 = U512::from_be_hex(concat!(
    "0000000000000000000000000000000000000000000000000000000000000000",
    "7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffed"
));

const SQRT_MINUS_ONE: Fp =
    Fp(U256::from_be_hex
    ("2b8324804fc1df0b2b4d00993dfbd7a72f431806ad2fe478c4ee1b274a0ea0b0"));

pub const FIELD_MODULUS: Fp = Fp(MODULUS);

fn reduce(x: U512) -> U256 {
    U256::from_le_slice(&x.checked_rem(&WIDE_MODULUS).unwrap().to_le_bytes()[.. 32])
}

impl Add<Fp> for Fp {
    type Output = Fp;

    fn add(self, rhs: Self) -> Self::Output {
        Self(U256::add_mod(&self.0, &rhs.0, &MODULUS))
    }
}
impl AddAssign<Fp> for Fp {
    fn add_assign(&mut self, rhs: Self) {
        self.0 = U256::add_mod(&self.0, &rhs.0, &MODULUS);
    }
}

impl<'a> Add<&'a Fp> for Fp {
    type Output = Fp;

    fn add(self, rhs: &'a Fp) -> Self::Output {
        Self(U256::add_mod(&self.0, &rhs.0, &MODULUS))
    }
}

impl<'a> AddAssign<&'a Fp> for Fp {
    fn add_assign(&mut self, rhs: &'a Fp) {
        self.0 = U256::add_mod(&self.0, &rhs.0, &MODULUS);
    }
}

impl Sub<Fp> for Fp {
    type Output = Fp;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(U256::sub_mod(&self.0, &rhs.0, &MODULUS))
    }
}
impl SubAssign<Fp> for Fp {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 = U256::sub_mod(&self.0, &rhs.0, &MODULUS);
    }
}

impl<'a> Sub<&'a Fp> for Fp {
    type Output = Fp;

    fn sub(self, rhs: &'a Fp) -> Self::Output {
        Self(U256::sub_mod(&self.0, &rhs.0, &MODULUS))
    }
}

impl<'a> SubAssign<&'a Fp> for Fp {
    fn sub_assign(&mut self, rhs: &'a Fp) {
        self.0 = U256::sub_mod(&self.0, &rhs.0, &MODULUS);
    }
}

impl Mul<Fp> for Fp {
    type Output = Fp;

    fn mul(self, rhs: Self) -> Self::Output {
        let wide = U256::mul_wide(&self.0, &rhs.0);
        Self(reduce(U512::from((wide.0, wide.1))))
    }
}
impl MulAssign<Fp> for Fp {
    fn mul_assign(&mut self, rhs: Self) {
        let wide = U256::mul_wide(&self.0, &rhs.0);
        self.0 = reduce(U512::from((wide.0, wide.1)));
    }
}

impl<'a> Mul<&'a Fp> for Fp {
    type Output = Fp;

    fn mul(self, rhs: &'a Fp) -> Self::Output {
        let wide = U256::mul_wide(&self.0, &rhs.0);
        Self(reduce(U512::from((wide.0, wide.1))))
    }
}

impl<'a> MulAssign<&'a Fp> for Fp {
    fn mul_assign(&mut self, rhs: &'a Fp) {
        let wide = U256::mul_wide(&self.0, &rhs.0);
        self.0 = reduce(U512::from((wide.0, wide.1)));
    }
}

impl Neg for Fp {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self(self.0.neg_mod(&MODULUS))
    }
}
  
impl<'a> Neg for &'a Fp {
    type Output = Fp;
    fn neg(self) -> Self::Output {
        (*self).neg()
    }
}

impl ConstantTimeEq for Fp {
    fn ct_eq(&self, other: &Self) -> Choice {
        self.0.ct_eq(&other.0)
    }
}

impl ConditionallySelectable for Fp {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
      Fp(U256::conditional_select(&a.0, &b.0, choice))
    }
}

impl<T> Sum<T> for Fp
where
    T: Borrow<Fp>
{
    fn sum<I: Iterator<Item = T>>(iter: I) -> Self {
        iter.fold(Fp::ZERO, |acc, item| acc + item.borrow())
    }
}

impl<T> Product<T> for Fp
where
    T: Borrow<Fp>
{
    fn product<I: Iterator<Item = T>>(iter: I) -> Self {
        iter.fold(Fp::ONE, |acc, item| acc * item.borrow())
    }
}

impl From<u64> for Fp {
    fn from(value: u64) -> Self {
        Self(U256::from_u64(value))
    }
}

impl AsRef<[u64]> for Fp {
    fn as_ref(&self) -> &[u64] {
        self.0.as_words()
    }
}

impl Field for Fp {
    const ZERO: Self = Self(U256::ZERO);

    const ONE: Self = Self(U256::ONE);

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
        const NEG_2: Fp = Self(MODULUS.saturating_sub(&U256::from_u8(2)));
        CtOption::new(self.pow(NEG_2), !self.is_zero())
    }

    // https://www.rfc-editor.org/rfc/rfc8032#section-5.1.3
    fn sqrt_ratio(u: &Self, v: &Self) -> (Choice, Self) {
        let v_sq = v.square();
        let uv = (*u)*(*v);
        let uv3 = uv * v_sq;

        let v_pow4 = v_sq.square();
        let v_pow6 = v_pow4 * v_sq;
        let uv7 = uv * v_pow6;

        let five = Fp::from(5u64);
        let one_by_eight = Fp::from(8u64).invert().unwrap();
        let exponent = (FIELD_MODULUS - five)*one_by_eight;
        let beta = uv3 * uv7.pow(exponent); // candidate square root
        
        let beta_sq = beta.square();
        let is_sq_root = (*v * beta_sq - *u).is_zero() | (*v * beta_sq + *u).is_zero();

        let neg_not_required = (*v * beta_sq - *u).is_zero();
        let sq_root = if bool::from(neg_not_required) {
            beta
        }
        else {
            beta*SQRT_MINUS_ONE
        };
        (is_sq_root, sq_root)
    }
}

impl PrimeField for Fp {
    type Repr = [u8; 32];

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

    const MODULUS: &'static str = "7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffed";

    const NUM_BITS: u32 = 255;

    const CAPACITY: u32 = 254;

    const TWO_INV: Self =
        Self(U256::from_be_hex("3ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7"));

    const MULTIPLICATIVE_GENERATOR: Self = Self(U256::from_u8(2));

    const S: u32 = 2;

    const ROOT_OF_UNITY: Self =
        Self(U256::from_be_hex("2b8324804fc1df0b2b4d00993dfbd7a72f431806ad2fe478c4ee1b274a0ea0b0"));

    const ROOT_OF_UNITY_INV: Self =
        Self(U256::from_be_hex("547cdb7fb03e20f4d4b2ff66c2042858d0bce7f952d01b873b11e4d8b5f15f3d"));

    const DELTA: Self = Self(U256::from_u8(16));

}

impl PrimeFieldBits for Fp {
    type ReprBits = [u8; 32];
  
    fn to_le_bits(&self) -> FieldBits<Self::ReprBits> {
        self.to_repr().into()
    }
  
    fn char_le_bits() -> FieldBits<Self::ReprBits> {
        MODULUS.to_le_bytes().into()
    }
}
  

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanity_checks() {
        let two = Fp::from(2u64);
        let four = two.mul(two);
        let expected_four = Fp::from(4u64);

        assert_eq!(four, expected_four);
        assert_eq!(Fp::TWO_INV*two, Fp::ONE);
        assert_eq!(two.pow(two), four);
        assert_eq!(two.square(), four);

        let two_inv = two.invert().unwrap();
        assert_eq!(two_inv, Fp::TWO_INV);
    }

    #[test]
    fn check_params() {
        assert_eq!(Fp::ROOT_OF_UNITY*Fp::ROOT_OF_UNITY_INV, Fp::ONE);
        assert_eq!(Fp::MULTIPLICATIVE_GENERATOR.pow(Fp(MODULUS)-Fp::ONE), Fp::ONE);
        
        let two = Fp::from(2u64);
        let two_pow_s = two.pow(Fp::from(Fp::S as u64));
        assert_eq!(Fp::ROOT_OF_UNITY.pow(two_pow_s), Fp::ONE);

        let t = (Fp(MODULUS)-Fp::ONE) * (two_pow_s.invert().unwrap());
        assert_eq!(Fp::DELTA.pow(t), Fp::ONE);
    }

    #[test]
    fn check_add_sub() {
        let mut rng = rand::thread_rng();
        let mut x = Fp::random(&mut rng);
        let y = Fp::random(&mut rng);
        let neg_y = -y;
        assert_eq!(y+neg_y, Fp::ZERO);
        assert_eq!(x+neg_y, x-y);

        let old_x = x;
        x -= y;
        assert_eq!(x + y, old_x);
        x += y;
        assert_eq!(x, old_x);
    
        let a = [x, Fp::from(1u64), Fp::from(2u64), y];
        assert_eq!(Fp::sum(a.iter()), x+y+Fp::from(3u64));
    
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
        let mut x = Fp::random(&mut rng);
        let y = Fp::random(&mut rng);
        assert_eq!(x.invert().unwrap()*x, Fp::ONE);

        let old_x = x;
        x *= y;
        assert_eq!(x*y.invert().unwrap(), old_x);

        let a = [x, Fp::from(2u64), Fp::from(3u64), y];
        assert_eq!(Fp::product(a.iter()), x*y*Fp::from(6u64));

        let y_ref = &y;
        assert_eq!(x*y, x*y_ref);
        x *= y_ref;
        assert_eq!(x*y.invert().unwrap(), x*y_ref.invert().unwrap());
    }

    #[test]
    fn check_repr() {
        let mut rng = rand::thread_rng();
        let x = Fp::random(&mut rng);
        let repr = x.to_repr();
        let roundtrip_x = Fp::from_repr(repr).unwrap();

        assert_eq!(x, roundtrip_x);
    }
    
    #[test]
    fn check_square_double() {
        let mut rng = rand::thread_rng();
        let x = Fp::random(&mut rng);
        assert_eq!(x.square(), x*x);
        assert_eq!(x.double(), x+x);
        let two = Fp::from(2u64);
        assert_eq!(x.double(), x*two);
    }

    #[test]
    fn check_cteq() {
        let two = Fp::from(2u64);
        let two_alt = Fp::ONE + Fp::ONE;
        assert!(bool::from(two.ct_eq(&two_alt)));
    }

    #[test]
    fn check_conditonal_select() {
        let one = Fp::ONE;
        let two = Fp::from(2u64);
        let x = Fp::conditional_select(&one, &two, Choice::from(0u8));
        let y = Fp::conditional_select(&one, &two, Choice::from(1u8));
        assert_eq!(x, one);
        assert_eq!(y, two);
    }

    #[test]
    fn check_square_root() {
        let two = Fp::from(2u64);
        assert!(bool::from(two.sqrt().is_none()));

        let four = two.square();
        let (is_sq_root, sq_root) = Fp::sqrt_ratio(&four, &Fp::ONE);
        assert!(bool::from(is_sq_root));
        assert!(sq_root == two || sq_root == -two);

        let mut rng = rand::thread_rng();
        let x = Fp::random(&mut rng);
        let x_sq = x.square();
        let (is_sq_root, sq_root) = Fp::sqrt_ratio(&x_sq, &Fp::ONE);
        assert!(bool::from(is_sq_root));
        assert!(sq_root == x || sq_root == -x);
        assert_eq!(sq_root.square(), x_sq);
    }

}