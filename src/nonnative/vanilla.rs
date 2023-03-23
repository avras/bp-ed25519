use std::ops::{Add, Sub, Rem};

use ff::{PrimeField, PrimeFieldBits, Field};
use num_bigint::BigUint;
use num_traits::{Zero, One};
use crate::{field::Fe25519, curve::{AffinePoint, D}};



#[derive(Debug, Clone)]
pub struct LimbedInt<F: PrimeField + PrimeFieldBits> {
    limbs: Vec<F>,
    // limb_width: u32,
}

impl<F> Default for LimbedInt<F>
where
    F: PrimeField + PrimeFieldBits
{
    fn default() -> Self {
        // Self { limbs: vec![F::ZERO; 4], limb_width: 64 }
        Self { limbs: vec![F::ZERO; 4] }
    }
}

impl<F> From<&BigUint> for LimbedInt<F>
where
    F: PrimeField + PrimeFieldBits
{
    fn from(value: &BigUint) -> Self {
        Self {
            limbs: value
                    .to_u64_digits()
                    .into_iter()
                    .map(|d| F::from(d))
                    .collect()
        }
    }
}

impl<F> From<&LimbedInt<F>> for BigUint
where
    F: PrimeField + PrimeFieldBits
{
    fn from(value: &LimbedInt<F>) -> Self {
        let mut res: BigUint = Zero::zero();
        let one: &BigUint = &One::one();
        let mut base: BigUint = one.clone();
        for i in 0..value.len() {
            res += base.clone() * BigUint::from_bytes_le(value.limbs[i].to_repr().as_ref());
            base = base * (one << 64)
        }
        res
    }

}

impl<F> From<&Fe25519> for LimbedInt<F>
where
    F: PrimeField + PrimeFieldBits
{
    fn from(value: &Fe25519) -> Self {
        let bytes: [u8; 32] = value.to_repr().try_into().unwrap();
        let i = BigUint::from_bytes_le(&bytes);
        LimbedInt::<F>::from(&i)
    }
}

impl<F> Add<LimbedInt<F>> for LimbedInt<F>
where
    F: PrimeField + PrimeFieldBits 
 {
    type Output = Self;

    fn add(self, rhs: LimbedInt<F>) -> Self::Output {
        let sum_len = self.len().max(rhs.len());
        let mut sum = Self { limbs: vec![F::ZERO; sum_len]};
        for i in 0..sum_len {
            if i < self.len() {
                sum.limbs[i] += self.limbs[i];
            }
            if i < rhs.len() {
                sum.limbs[i] += rhs.limbs[i];
            }
        }
        sum
    }
}

impl<F> Sub<LimbedInt<F>> for LimbedInt<F>
where
    F: PrimeField + PrimeFieldBits 
 {
    type Output = Self;

    fn sub(self, rhs: LimbedInt<F>) -> Self::Output {
        let diff_len = self.len().max(rhs.len());
        let mut diff = Self { limbs: vec![F::ZERO; diff_len]};
        for i in 0..diff_len {
            if i < self.len() {
                diff.limbs[i] += self.limbs[i];
            }
            if i < rhs.len() {
                diff.limbs[i] -= rhs.limbs[i];
            }
        }
        diff
    }
}

impl<F> LimbedInt<F>
where
    F: PrimeField + PrimeFieldBits
{
    // fn new(limbs: Vec<F>) -> Self {
    //     Self { limbs }
    // }
    
    fn len(&self) -> usize {
        self.limbs.len()
    }

    fn pad_limbs(&mut self, padded_length: usize) {
        assert!(self.len() <= padded_length);
        for _ in self.len()..padded_length {
            self.limbs.push(F::ZERO);
        }
    } 

    fn calc_cubic_limbs(a: &LimbedInt<F>, b: &LimbedInt<F>, c: &LimbedInt<F>) -> Self {
        assert_eq!(a.len(), b.len());
        assert_eq!(b.len(), c.len());
        let num_limbs = a.len();
        let num_limbs_in_cubic = 3*(num_limbs-1)+1;

        let mut prod: LimbedInt<F> = Self { limbs: vec![F::ZERO; num_limbs_in_cubic] };
        for i in 0..num_limbs {
            for j in 0..num_limbs {
                for k in 0..num_limbs {
                    prod.limbs[i+j+k] += a.limbs[i] * b.limbs[j] * c.limbs[k];
                }
            }
        }
        prod
    }

    fn fold_cubic_limbs(g: &LimbedInt<F>) -> Self {
        let mut h: LimbedInt<F> = LimbedInt::default();
        assert_eq!(h.len(), 4);
        assert_eq!(g.len(), 10);
        let c = F::from(38u64);
        let c2 = F::from(1444);
        h.limbs[0] = g.limbs[0] + c*g.limbs[4] + c2*g.limbs[8];
        h.limbs[1] = g.limbs[1] + c*g.limbs[5] + c2*g.limbs[9];
        h.limbs[2] = g.limbs[2] + c*g.limbs[6];
        h.limbs[3] = g.limbs[3] + c*g.limbs[7];

        h
    }

    fn calc_quadratic_limbs(a: &LimbedInt<F>, b: &LimbedInt<F>) -> Self {
        let num_limbs_in_quadratic = a.len() + b.len() - 1;

        let mut prod: LimbedInt<F> = Self { limbs: vec![F::ZERO; num_limbs_in_quadratic] };
        for i in 0..a.len() {
            for j in 0..b.len() {
                prod.limbs[i+j] += a.limbs[i] * b.limbs[j];
            }
        }
        prod
    }

    fn fold_quadratic_limbs(f: &LimbedInt<F>) -> Self {
        let mut h: LimbedInt<F> = LimbedInt::default();
        assert_eq!(h.len(), 4);
        assert_eq!(f.len(), 7);
        let c = F::from(38u64);
        h.limbs[0] = f.limbs[0] + c*f.limbs[4];
        h.limbs[1] = f.limbs[1] + c*f.limbs[5];
        h.limbs[2] = f.limbs[2] + c*f.limbs[6];
        h.limbs[3] = f.limbs[3];

        h
    }

    fn check_difference_is_zero(a: LimbedInt<F>, b: LimbedInt<F>) -> bool {
        let diff = a - b;
        let mut carries: Vec<F> = vec![F::ZERO; diff.len()-1];
        let exp64 = BigUint::from(64u64);
        let base = F::from(2u64).pow(exp64.to_u64_digits());

        for i in 0..diff.len()-1 {
            if i == 0 {
                let limb_bits = diff.limbs[0].to_le_bits();
                let mut coeff = F::ONE;
                // Calculating carries[0] as diff.limbs[0] shifted to the right 64 times (discard the 64 LSBs)
                for (j, bit) in limb_bits.into_iter().enumerate() {
                    if  j >= 64 {
                        if bit {
                            carries[0] += coeff;
                        }
                        coeff += coeff;
                    }
                }
                assert_eq!(diff.limbs[0], carries[0]*base);
            }
            else {
                let limb_bits = (carries[i-1] + diff.limbs[i]).to_le_bits();
                let mut coeff = F::ONE;
                // Calculating carries[i] as diff.limbs[i] + carries[i-1] shifted to the right 64 times (discard the 64 LSBs)
                for (j, bit) in limb_bits.into_iter().enumerate() {
                    if  j >= 64 {
                        if bit {
                            carries[i] += coeff;
                        }
                        coeff += coeff;
                    }
                }
                assert_eq!(diff.limbs[i] + carries[i-1], carries[i]*base);
            }
        }
        diff.limbs[diff.len()-1] + carries[diff.len()-2] == F::ZERO
    }

    fn verify_cubic_product(
        a: &LimbedInt<F>,
        b: &LimbedInt<F>,
        c: &LimbedInt<F>,
        prod: &LimbedInt<F>,
    ) -> bool {
        let cubic_limbed_int = Self::calc_cubic_limbs(a, b, c);
        let h_l = Self::fold_cubic_limbs(&cubic_limbed_int);

        let one = BigUint::from(1u64);
        let q: BigUint = (one.clone() << 255) - BigUint::from(19u64);
        let h = BigUint::from(&h_l);
        let r = h.clone().rem(&q);
        assert_eq!(BigUint::from(&cubic_limbed_int).rem(&q), r);

        let t = (h-r.clone()) / (q.clone());
        assert!(t < one << 138);
        
        let q_l = Self::from(&q);
        let mut t_l = Self::from(&t);
        t_l.pad_limbs(3);

        let tq_l = Self::calc_quadratic_limbs(&t_l, &q_l);
        let tq_plus_r_l = tq_l + prod.clone();
        
        Self::check_difference_is_zero(h_l, tq_plus_r_l)
    }

    fn verify_x_coordinate_quadratic_is_zero(
        x1: &LimbedInt<F>,
        x2: &LimbedInt<F>,
        y1: &LimbedInt<F>,
        y2: &LimbedInt<F>,
        x3: &LimbedInt<F>,
        v: &LimbedInt<F>,
    ) -> bool {

        let one = BigUint::from(1u64);
        let q: BigUint = (one.clone() << 255) - BigUint::from(19u64);
        let q_l = Self::from(&q);
        let mut q70_l = Self::default();
        for i in 0..q_l.len() {
            q70_l.limbs[i] = q_l.limbs[i] * F::from_u128(1 << 70);
        }

        let x1y2_l = Self::fold_quadratic_limbs(&Self::calc_quadratic_limbs(x1, y2));
        let x2y1_l = Self::fold_quadratic_limbs(&Self::calc_quadratic_limbs(x2, y1));
        let x3v_l = Self::fold_quadratic_limbs(&Self::calc_quadratic_limbs(x3, v));
        let g_l = x1y2_l + x2y1_l - x3.clone() - x3v_l  + q70_l;

        let g = BigUint::from(&g_l);
        assert!(g.clone().rem(q.clone()).is_zero());
        let t = g.clone() / q.clone();
        assert!(t < (one << 72));
        assert!(g == t.clone()*q.clone());

        let t_l = Self::from(&t);
        let tq_l = Self::calc_quadratic_limbs(&t_l, &q_l);
        assert!(t*q == BigUint::from(&tq_l));
        Self::check_difference_is_zero(g_l, tq_l)
    }

    fn verify_y_coordinate_quadratic_is_zero(
        x1: &LimbedInt<F>,
        x2: &LimbedInt<F>,
        y1: &LimbedInt<F>,
        y2: &LimbedInt<F>,
        y3: &LimbedInt<F>,
        v: &LimbedInt<F>,
    ) -> bool {

        let one = BigUint::from(1u64);
        let q: BigUint = (one.clone() << 255) - BigUint::from(19u64);
        let q_l = Self::from(&q);
        let mut q70_l = Self::default();
        for i in 0..q_l.len() {
            q70_l.limbs[i] = q_l.limbs[i] * F::from_u128(1 << 70);
        }

        let x1x2_l = Self::fold_quadratic_limbs(&Self::calc_quadratic_limbs(x1, x2));
        let y1y2_l = Self::fold_quadratic_limbs(&Self::calc_quadratic_limbs(y1, y2));
        let y3v_l = Self::fold_quadratic_limbs(&Self::calc_quadratic_limbs(y3, v));
        let g_l = x1x2_l + y1y2_l + y3v_l - y3.clone()  + q70_l;

        let g = BigUint::from(&g_l);
        assert!(g.clone().rem(q.clone()).is_zero());
        let t = g.clone() / q.clone();
        assert!(t < (one << 72));
        assert!(g == t.clone()*q.clone());

        let t_l = Self::from(&t);
        let tq_l = Self::calc_quadratic_limbs(&t_l, &q_l);
        assert!(t*q == BigUint::from(&tq_l));
        Self::check_difference_is_zero(g_l, tq_l)
    }

    fn verify_ed25519_point_addition(
        p: &AffinePoint,
        q: &AffinePoint,
        r: &AffinePoint,
    ) -> bool {
        let x1 = p.x;
        let y1 = p.y;
        let x2 = q.x;
        let y2 = q.y;
        let x3 = r.x;
        let y3 = r.y;

        let u = D*x1*x2;

        let d_l = LimbedInt::<F>::from(&D);
        let x1_l = LimbedInt::<F>::from(&x1);
        let x2_l = LimbedInt::<F>::from(&x2);
        let u_l = LimbedInt::<F>::from(&u);

        let v = u*y1*y2;
        let y1_l = LimbedInt::<F>::from(&y1);
        let y2_l = LimbedInt::<F>::from(&y2);
        let v_l = LimbedInt::<F>::from(&v);

        let x3_l = LimbedInt::<F>::from(&x3);
        let y3_l = LimbedInt::<F>::from(&y3);
        
        Self::verify_cubic_product(&d_l, &x1_l, &x2_l, &u_l)
             & Self::verify_cubic_product(&u_l, &y1_l, &y2_l, &v_l)
             & Self::verify_x_coordinate_quadratic_is_zero(&x1_l, &x2_l, &y1_l, &y2_l, &x3_l, &v_l)
             & Self::verify_y_coordinate_quadratic_is_zero(&x1_l, &x2_l, &y1_l, &y2_l, &y3_l, &v_l)

    }
}

#[derive(Debug, Clone)]
pub struct LimbedAffinePoint<F: PrimeField + PrimeFieldBits> {
    pub(crate) x: LimbedInt<F>,
    pub(crate) y: LimbedInt<F>,
}

impl<F> Default for LimbedAffinePoint<F>
where
    F: PrimeField + PrimeFieldBits
{
    fn default() -> Self {
        Self {
            x: LimbedInt::<F>::from(&Fe25519::ZERO),
            y: LimbedInt::<F>::from(&Fe25519::ONE),
        }
    }
}

impl<F> From<&AffinePoint> for LimbedAffinePoint<F>
where
    F: PrimeField + PrimeFieldBits
{
    fn from(value: &AffinePoint) -> Self {
        Self {
            x: LimbedInt::<F>::from(&value.x),
            y: LimbedInt::<F>::from(&value.y),
        }
    }
}

impl<F: PrimeField + PrimeFieldBits> LimbedAffinePoint<F> {
    fn verify_ed25519_point_addition(
        p: &Self,
        q: &Self,
        r: &Self,
        u: &LimbedInt<F>,
        v: &LimbedInt<F>,
    ) -> bool {
        let x1_l = &p.x;
        let y1_l = &p.y;
        let x2_l = &q.x;
        let y2_l = &q.y;
        let x3_l = &r.x;
        let y3_l = &r.y;

        let d_l = LimbedInt::<F>::from(&D);
        
        LimbedInt::<F>::verify_cubic_product(&d_l, x1_l, x2_l, u)
             & LimbedInt::<F>::verify_cubic_product(&u, y1_l, y2_l, v)
             & LimbedInt::<F>::verify_x_coordinate_quadratic_is_zero(x1_l, x2_l, y1_l, y2_l, x3_l, v)
             & LimbedInt::<F>::verify_y_coordinate_quadratic_is_zero(x1_l, x2_l, y1_l, y2_l, y3_l, v)

    }
}


#[cfg(test)]
mod tests {
    use crate::curve::Ed25519Curve;

    use super::*;
    use crypto_bigint::{U256, Random};
    use ff::Field;
    use num_bigint::RandBigInt;
    use pasta_curves::Fp;

    #[test]
    fn limbed_int_biguint_roundtrip() {
        let mut rng = rand::thread_rng();
        let big_uint = rng.gen_biguint(256u64);
        let limbed_int = LimbedInt::<Fp>::from(&big_uint);
        assert_eq!(big_uint, BigUint::from(&limbed_int));
    }

    #[test]
    fn limbed_int_fe_roundtrip() {
        let rng = rand::thread_rng();
        let fe = Fe25519::random(rng);
        let limbed_int = LimbedInt::<Fp>::from(&fe);
        assert_eq!(BigUint::from_bytes_le(&fe.to_repr()), BigUint::from(&limbed_int));
    }

    #[test]
    fn limbed_padding() {
        let mut rng = rand::thread_rng();
        let big_uint = rng.gen_biguint(256u64);
        let mut limbed_int = LimbedInt::<Fp>::from(&big_uint);
        assert_eq!(limbed_int.len(), 4);
        limbed_int.pad_limbs(6);
        assert_eq!(limbed_int.len(), 6);
        assert_eq!(limbed_int.limbs[4], Fp::ZERO);
        assert_eq!(limbed_int.limbs[5], Fp::ZERO);
    }

    #[test]
    fn limbed_cubic() {
        let mut rng = rand::thread_rng();
        let a_uint = rng.gen_biguint(256u64);
        let b_uint = rng.gen_biguint(256u64);
        let c_uint = rng.gen_biguint(256u64);
        let a_l = LimbedInt::<Fp>::from(&a_uint);
        let b_l = LimbedInt::<Fp>::from(&b_uint);
        let c_l = LimbedInt::<Fp>::from(&c_uint);

        let cubic_limbed_int = LimbedInt::<Fp>::calc_cubic_limbs(&a_l, &b_l, &c_l);
        assert_eq!(a_uint*b_uint*c_uint, BigUint::from(&cubic_limbed_int));
    }

    #[test]
    fn limbed_cubic_folding() {
        let mut rng = rand::thread_rng();
        let a_uint = rng.gen_biguint(256u64);
        let b_uint = rng.gen_biguint(256u64);
        let c_uint = rng.gen_biguint(256u64);
        let a_l = LimbedInt::<Fp>::from(&a_uint);
        let b_l = LimbedInt::<Fp>::from(&b_uint);
        let c_l = LimbedInt::<Fp>::from(&c_uint);

        let cubic_limbed_int = LimbedInt::<Fp>::calc_cubic_limbs(&a_l, &b_l, &c_l);
        let folded_limbed_int = LimbedInt::<Fp>::fold_cubic_limbs(&cubic_limbed_int);

        let one = BigUint::from(1u64);
        let q: BigUint = (one.clone() << 255) - BigUint::from(19u64);
        let h = BigUint::from(&folded_limbed_int);
        let r = h.clone().rem(&q);
        assert_eq!(BigUint::from(&cubic_limbed_int).rem(&q), r);

        let t = (h-r.clone()) / (q.clone());
        assert!(t < one << 138);
    }

    #[test]
    fn limbed_quadratic() {
        let mut rng = rand::thread_rng();
        let t_uint = rng.gen_biguint(192u64);
        let q_uint = rng.gen_biguint(256u64);
        let t_l = LimbedInt::<Fp>::from(&t_uint);
        let q_l = LimbedInt::<Fp>::from(&q_uint);

        let quadratic_limbed_int = LimbedInt::<Fp>::calc_quadratic_limbs(&t_l, &q_l);
        assert_eq!(t_uint*q_uint, BigUint::from(&quadratic_limbed_int));
    }

    #[test]
    fn limbed_add() {
        let mut rng = rand::thread_rng();
        let a_uint = rng.gen_biguint(192u64);
        let b_uint = rng.gen_biguint(256u64);
        let a_l = LimbedInt::<Fp>::from(&a_uint);
        let b_l = LimbedInt::<Fp>::from(&b_uint);

        let sum_limbed_int = a_l+b_l;
        assert_eq!(a_uint+b_uint, BigUint::from(&sum_limbed_int));
    }

    #[test]
    fn limbed_check_cubic_zero() {
        let one = BigUint::from(1u64);
        let q: BigUint = (one.clone() << 255) - BigUint::from(19u64);
        let zero = BigUint::from(0u64);
        let mut rng = rand::thread_rng();
        let a_uint = rng.gen_biguint_range(&zero, &q);
        let b_uint = rng.gen_biguint_range(&zero, &q);
        let c_uint = rng.gen_biguint_range(&zero, &q);
        let a_l = LimbedInt::<Fp>::from(&a_uint);
        let b_l = LimbedInt::<Fp>::from(&b_uint);
        let c_l = LimbedInt::<Fp>::from(&c_uint);

        let r = (a_uint * b_uint * c_uint).rem(&q);
        let r_l = LimbedInt::<Fp>::from(&r);

        assert!(LimbedInt::<Fp>::verify_cubic_product(&a_l, &b_l, &c_l, &r_l));
    }

    #[test]
    fn limbed_point_addition_verification() {
        let b = Ed25519Curve::basepoint();
        let mut rng = rand::thread_rng();
        let scalar = U256::random(&mut rng);
        let p = Ed25519Curve::scalar_multiplication(&b, &scalar);
        let scalar = U256::random(&mut rng);
        let q = Ed25519Curve::scalar_multiplication(&b, &scalar);
        let r = p.add(q);
        assert!(LimbedInt::<Fp>::verify_ed25519_point_addition(&p, &q, &r));
    }

    #[test]
    fn limbed_affine_point_addition_verification() {
        let b = Ed25519Curve::basepoint();
        let mut rng = rand::thread_rng();
        let scalar = U256::random(&mut rng);
        let p = Ed25519Curve::scalar_multiplication(&b, &scalar);
        let scalar = U256::random(&mut rng);
        let q = Ed25519Curve::scalar_multiplication(&b, &scalar);
        let r = p.add(q);

        let p_l = LimbedAffinePoint::<Fp>::from(&p);
        let q_l = LimbedAffinePoint::<Fp>::from(&q);
        let r_l = LimbedAffinePoint::<Fp>::from(&r);
        let u = D*p.x*q.x;
        let v = u*p.y*q.y;
        let u_l = LimbedInt::<Fp>::from(&u);
        let v_l = LimbedInt::<Fp>::from(&v);
        assert!(LimbedAffinePoint::<Fp>::verify_ed25519_point_addition(&p_l, &q_l, &r_l, &u_l, &v_l));
    }

}