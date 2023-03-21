use std::ops::{Add, Sub};

use ff::{PrimeField, PrimeFieldBits};
use num_bigint::BigUint;
use num_traits::{Zero, One};



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

}


#[cfg(test)]
mod tests {
    use std::ops::Rem;

    use super::*;
    use ff::Field;
    use num_bigint::RandBigInt;
    use pasta_curves::Fp;

    #[test]
    fn limbed_int_roundtrip() {
        let mut rng = rand::thread_rng();
        let big_uint = rng.gen_biguint(256u64);
        let limbed_int = LimbedInt::<Fp>::from(&big_uint);
        assert_eq!(big_uint, BigUint::from(&limbed_int));
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

        let cubic_limbed_int = LimbedInt::<Fp>::calc_cubic_limbs(&a_l, &b_l, &c_l);
        let h_l = LimbedInt::<Fp>::fold_cubic_limbs(&cubic_limbed_int);

        let h = BigUint::from(&h_l);
        let r = h.clone().rem(&q);
        assert_eq!(BigUint::from(&cubic_limbed_int).rem(&q), r);

        let t = (h-r.clone()) / (q.clone());
        assert!(t < one << 138);
        
        let q_l = LimbedInt::<Fp>::from(&q);
        let r_l = LimbedInt::<Fp>::from(&r);
        let mut t_l = LimbedInt::<Fp>::from(&t);
        t_l.pad_limbs(3);

        let tq_l = LimbedInt::<Fp>::calc_quadratic_limbs(&t_l, &q_l);
        let tq_plus_r_l = tq_l + r_l;
        assert!(LimbedInt::<Fp>::check_difference_is_zero(h_l, tq_plus_r_l));
    }


}