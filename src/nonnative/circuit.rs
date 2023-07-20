use std::ops::{Add, Sub, Mul, Rem};
use bellperson::{ConstraintSystem, SynthesisError, LinearCombination, Variable};
use bellperson::gadgets::boolean::{AllocatedBit, Boolean};
use ff::{PrimeField, PrimeFieldBits};
use num_bigint::BigUint;
use num_traits::Zero;

use crate::{nonnative::vanilla::LimbedInt, curve::{AffinePoint, D}};

use super::vanilla::LimbedAffinePoint;


fn mul_lc_with_scalar<F>(
    lc: LinearCombination<F>,
    scalar: F
) -> LinearCombination<F>
where
    F: PrimeField
{
    let mut scaled_lc = LinearCombination::<F>::zero();
    for (var, coeff) in lc.iter() {
        scaled_lc = scaled_lc + (scalar*coeff, var);
    }
    scaled_lc
}

// From fits_in_bits of bellperson-nonnative
fn range_check_lc<F, CS>(
    cs: &mut CS,
    lc_input: &LinearCombination<F>,
    lc_value: F,
    num_bits: usize,
) -> Result<(), SynthesisError>
where
    F: PrimeField + PrimeFieldBits,
    CS: ConstraintSystem<F>,
{
    let value_bits = lc_value.to_le_bits();

    // Allocate all but the first bit.
    let bits: Vec<Variable> = (1..num_bits)
        .map(|i| {
            cs.alloc(
                || format!("bit {i}"),
                || {
                    let r = if value_bits[i] {
                        F::ONE
                    } else {
                        F::ZERO
                    };
                    Ok(r)
                },
            )
        })
        .collect::<Result<_, _>>()?;

    for (i, v) in bits.iter().enumerate() {
        cs.enforce(
            || format!("bit {i} is bit"),
            |lc| lc + *v,
            |lc| lc + CS::one() - *v,
            |lc| lc,
        )
    }

    // Last bit
    cs.enforce(
        || format!("last bit of variable is a bit"),
        |mut lc| {
            let mut f = F::ONE;
            lc = lc + lc_input;
            for v in bits.iter() {
                f = f.double();
                lc = lc - (f, *v);
            }
            lc
        },
        |mut lc| {
            lc = lc + CS::one();
            let mut f = F::ONE;
            lc = lc - lc_input;
            for v in bits.iter() {
                f = f.double();
                lc = lc + (f, *v);
            }
            lc
        },
        |lc| lc,
    );

    Ok(())
}

#[derive(Debug, Clone)]
pub struct AllocatedLimbedInt<F: PrimeField + PrimeFieldBits>
{
    // Modelling limbs as linear combinations is more flexible than
    // modelling them as AllocatedNums. For example, adding will not require
    // allocation of a new AllocatedNum
    limbs: Vec<LinearCombination<F>>,
    value: Option<LimbedInt<F>>,
}

impl<F> Add<&AllocatedLimbedInt<F>> for AllocatedLimbedInt<F>
where
    F: PrimeField + PrimeFieldBits 
 {
    type Output = Self;

    fn add(self, rhs: &AllocatedLimbedInt<F>) -> Self::Output {
        assert!(self.value.is_some());
        assert!(rhs.value.is_some());
        let self_value = self.clone().value.unwrap();
        let rhs_value = rhs.clone().value.unwrap();
        let self_len = self_value.len();
        let rhs_len = rhs_value.len();
        assert_eq!(self.limbs.len(), self_len);
        assert_eq!(rhs.limbs.len(), rhs_len);

        let sum_len = self_len.max(rhs_len);
        let mut sum = Self { 
            limbs: vec![LinearCombination::zero(); sum_len],
            value: Some(self_value + rhs_value),
        };
        for i in 0..sum_len {
            let mut tmp = LinearCombination::<F>::zero();
            if i < self_len {
                tmp = self.limbs[i].clone();
            }
            if i < rhs_len {
                tmp = tmp + &rhs.limbs[i];
            }
            sum.limbs[i] = tmp;
        }
        sum
    }
}

impl<F> Sub<&AllocatedLimbedInt<F>> for AllocatedLimbedInt<F>
where
    F: PrimeField + PrimeFieldBits 
 {
    type Output = Self;

    fn sub(self, rhs: &AllocatedLimbedInt<F>) -> Self::Output {
        assert!(self.value.is_some());
        assert!(rhs.value.is_some());
        let self_value = self.clone().value.unwrap();
        let rhs_value = rhs.clone().value.unwrap();
        let self_len = self_value.len();
        let rhs_len = rhs_value.len();
        assert_eq!(self.limbs.len(), self_len);
        assert_eq!(rhs.limbs.len(), rhs_len);

        let diff_len = self_len.max(rhs_len);
        let mut diff = Self { 
            limbs: vec![LinearCombination::zero(); diff_len],
            value: Some(self_value - rhs_value),
        };
        for i in 0..diff_len {
            let mut tmp = LinearCombination::<F>::zero();
            if i < self_len {
                tmp = self.limbs[i].clone();
            }
            if i < rhs_len {
                tmp = tmp - &rhs.limbs[i];
            }
            diff.limbs[i] = tmp;
        }
        diff
    }
}

impl<F> Mul<&LimbedInt<F>> for AllocatedLimbedInt<F>
where
    F: PrimeField + PrimeFieldBits
{
    type Output = Self;

    fn mul(self, rhs: &LimbedInt<F>) -> Self::Output {
        assert!(self.value.is_some());
        let self_value = self.clone().value.unwrap();
        let self_lc_vec = self.limbs;
        assert_eq!(self_lc_vec.len(), self_value.len());
        let prod_len = self_value.len() + rhs.len() - 1;

        let mut prod_lcs = vec![LinearCombination::<F>::zero(); prod_len];
        for i in 0..self_value.len() {
            for j in 0..rhs.len() {
                prod_lcs[i+j] = prod_lcs[i+j].clone() + &mul_lc_with_scalar(self_lc_vec[i].clone(), rhs.limbs[j]);
            }
        }
        Self {
            limbs: prod_lcs,
            value: Some(LimbedInt::<F>::calc_quadratic_limbs(&self_value, rhs)),
        }
    }
}

pub enum AllocatedOrConstantLimbedInt<F>
where
    F: PrimeField + PrimeFieldBits
{
    Allocated(AllocatedLimbedInt<F>),
    Constant(LimbedInt<F>)
}

impl<F> AllocatedLimbedInt<F>
where
    F: PrimeField + PrimeFieldBits
{
    pub fn alloc_from_limbed_int<CS>(
        cs: &mut CS,
        value: LimbedInt<F>,
        num_limbs: usize,
    ) -> Result<Self, SynthesisError>
    where
        CS: ConstraintSystem<F>,
    {
        if value.len() != num_limbs {
            return Err(SynthesisError::Unsatisfiable);
        }
        let limbs = (0..value.len())
            .map( |i| {
                cs.alloc(
                    || format!("limb {i}"),
                    || Ok(value.limbs[i])
                )
                .map(|v| LinearCombination::<F>::zero() + v)
            })
            .collect::<Result<Vec<_>, _>>()?;
        
        Ok(AllocatedLimbedInt { limbs, value: Some(value) })
    }

    pub fn add_limbed_int<CS>(
        &self,
        limbed_int: &LimbedInt<F>,
    ) -> Result<Self, SynthesisError>
    where
        CS: ConstraintSystem<F>,
    {
        assert!(self.value.is_some());
        let self_value = self.clone().value.unwrap();
        let self_lc_vec = self.limbs.clone();
        assert_eq!(self_lc_vec.len(), self_value.len());
        let sum_len = self_value.len().max(limbed_int.len());

        let mut sum_lcs = vec![LinearCombination::<F>::zero(); sum_len];
        let mut sum_values = LimbedInt::<F>::default();
        for i in 0..sum_len {
            let mut tmp = LinearCombination::<F>::zero();
            let mut tmp_val = F::ZERO;
            if i < self_value.len() {
                tmp = self_lc_vec[i].clone();
                tmp_val = self_value.limbs[i];
            }
            if i < limbed_int.len() {
                tmp = tmp + &LinearCombination::<F>::from_coeff(CS::one(), limbed_int.limbs[i]);
                tmp_val = tmp_val + limbed_int.limbs[i];
            }
            sum_lcs[i] = tmp;
            sum_values.limbs[i] = tmp_val;
        }

        Ok(Self {
            limbs: sum_lcs,
            value: Some(sum_values),
        })
    }
    
    pub fn range_check_limbs<CS: ConstraintSystem<F>>(
        &self,
        cs: &mut CS,
        num_bits: usize,
    ) -> Result<(), SynthesisError> {
        assert!(self.value.is_some());
        let limbed_int = self.clone().value.unwrap();
        assert_eq!(self.limbs.len(), limbed_int.len());

        for i in 0..limbed_int.len() {
            range_check_lc(
                &mut cs.namespace(|| format!("Range check limb {i}")),
                &self.limbs[i],
                limbed_int.limbs[i],
                num_bits
            )?;
        }

        Ok(())
    }

    pub fn check_base_field_membership<CS: ConstraintSystem<F>>(
        &self,
        cs: &mut CS,
    ) -> Result<(), SynthesisError> {
        // Less than 4 limbs is trivially in base field
        assert_eq!(self.limbs.len(), 4);
        let limbed_int = self.clone().value.unwrap();
        assert_eq!(self.limbs.len(), limbed_int.len());

        let max_ls_limb_value: F = F::from(u64::MAX);
        let max_ms_limb_value: F = F::from((1u64 << 63) - 1);

        // range check the limbs, most significant limb occupies 63 bits
        for i in 0..4 {
            let num_bits = if i == 3 {
                63
            } else {
                64
            };

            range_check_lc(
                &mut cs.namespace(|| format!("range check limb {i}")),
                &self.limbs[i],
                limbed_int.limbs[i],
                num_bits,
            )?;
        }

        let equality_bits: Vec<AllocatedBit> = (1..4)
            .map( |i| {
                let max_limb_value = if i == 3 {
                    max_ms_limb_value
                } else {
                    max_ls_limb_value
                };

                let bit = AllocatedBit::alloc(
                    cs.namespace(|| format!("check if limb {i} equals max value")),
                    Some(limbed_int.limbs[i] == max_limb_value),
                );
                bit.unwrap()
            })
            .collect();

        let limbs12_maxed = AllocatedBit::and(
            cs.namespace(|| "limbs 1 and 2 are maximum possible values"),
            &equality_bits[0],
            &equality_bits[1],
        )?;
        let limbs123_maxed = AllocatedBit::and(
            cs.namespace(|| "limbs 1, 2 and 3 are maximum possible values"),
            &limbs12_maxed,
            &equality_bits[2],
        )?;

        let c = F::from(19u64);
        let ls_limb_value = if limbs123_maxed.get_value().unwrap() {
            // Add 18 to the least significant limb. It may or may not overflow the 64 bits
            limbed_int.limbs[0] + c
        } else {
            F::ZERO
        };
        
        let ls_limb_modified = cs.alloc(
            || "modified limb value",
            || Ok(ls_limb_value)
        )?;

        // If all the most significant limbs are equal to their max values, then
        // ls_limb_modified == self.limbs[0] + c
        // Otherwise, ls_limb_modified == 0
        cs.enforce(
            || "check modified limb value is correct",
            |lc| lc + &self.limbs[0] + &LinearCombination::from_coeff(CS::one(), c),
            |lc| lc + limbs123_maxed.get_variable(),
            |lc| lc + ls_limb_modified,
        );

        range_check_lc(
            &mut cs.namespace(|| "range check modified LS limb"),
            &LinearCombination::from_variable(ls_limb_modified),
            ls_limb_value,
            64,
        )?;

        Ok(())
    }

    pub fn alloc_product<CS: ConstraintSystem<F>>(
        &self,
        cs: &mut CS,
        other: &Self,
    ) -> Result<AllocatedLimbedInt<F>, SynthesisError> {
        if self.value.is_none() || other.value.is_none() {
            return Err(SynthesisError::Unsatisfiable);
        }

        let a_l = self.clone().value.unwrap();
        let b_l = other.clone().value.unwrap();
        if self.limbs.len() != a_l.len() || other.limbs.len() != b_l.len() {
            return Err(SynthesisError::Unsatisfiable);
        } 

        let prod = LimbedInt::<F>::calc_quadratic_limbs(&a_l, &b_l);
        let prod_limbs = (0..prod.len())
            .map(|i| {
                cs.alloc(
                    || format!("product limb {i}"),
                    || Ok(prod.limbs[i]),
                )
                .map(|v| LinearCombination::<F>::zero() + v)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let product = AllocatedLimbedInt {
            limbs: prod_limbs,
            value: Some(prod),
        };

        let mut x = F::ZERO;
        for _ in 0..product.limbs.len() {
            x += F::ONE;
            cs.enforce(
                || format!("pointwise product @ {x:?}"),
                |lc| {
                    let mut i = F::ONE;
                    self.limbs.iter().fold(lc, |lc, c| {
                        let r = lc + (i, c);
                        i *= x;
                        r
                    })
                },
                |lc| {
                    let mut i = F::ONE;
                    other.limbs.iter().fold(lc, |lc, c| {
                        let r = lc + (i, c);
                        i *= x;
                        r
                    })
                },
                |lc| {
                    let mut i = F::ONE;
                    product.limbs.iter().fold(lc, |lc, c| {
                        let r = lc + (i, c);
                        i *= x;
                        r
                    })
                },
            )
        }
        
        Ok(product)
    }
    
    pub fn alloc_cubic_product_3var<CS: ConstraintSystem<F>>(
        &self,
        cs: &mut CS,
        b: &Self,
        c: &Self,
    ) -> Result<AllocatedLimbedInt<F>, SynthesisError> {
        if self.value.is_none() || b.value.is_none() || c.value.is_none() {
            return Err(SynthesisError::Unsatisfiable);
        }

        let a_l = self.clone().value.unwrap();
        let b_l = b.clone().value.unwrap();
        let c_l = c.clone().value.unwrap();
        if self.limbs.len() != a_l.len()
            || b_l.limbs.len() != b_l.len()
            || c_l.limbs.len() != c_l.len() {
            return Err(SynthesisError::Unsatisfiable);
        }

        let ab = self.alloc_product(
            &mut cs.namespace(|| "a times b"),
            &b
        )?;
        let abc = ab.alloc_product(
            &mut cs.namespace(|| "ab times c"),
            &c
        );
        abc
    }

    pub fn alloc_cubic_product_2var1const<CS: ConstraintSystem<F>>(
        &self,
        cs: &mut CS,
        b: &Self,
        c: &LimbedInt<F>,
    ) -> Result<AllocatedLimbedInt<F>, SynthesisError> {
        if self.value.is_none() || b.value.is_none() {
            return Err(SynthesisError::Unsatisfiable);
        }

        let a_l = self.clone().value.unwrap();
        let b_l = b.clone().value.unwrap();
        if self.limbs.len() != a_l.len()
            || b_l.limbs.len() != b_l.len()
            || c.limbs.len() != c.len() {
            return Err(SynthesisError::Unsatisfiable);
        }

        let ab = self.alloc_product(
            &mut cs.namespace(|| "a times b"),
            &b
        )?;
        let abc = ab * c;
        Ok(abc)
    }

    pub fn fold_quadratic_limbs(
        &self,
    ) -> Result<AllocatedLimbedInt<F>, SynthesisError> {
        if self.value.is_none() {
            return Err(SynthesisError::Unsatisfiable);
        }

        let f = self.clone().value.unwrap();
        if self.limbs.len() != f.len() || f.len() != 7 {
            return Err(SynthesisError::Unsatisfiable);
        }

        let c1 = F::from(38u64);

        let f_limbs = self.clone().limbs; 
        let mut h_limbs = vec![LinearCombination::<F>::zero(); 4];
        h_limbs[0] = f_limbs[0].clone() + &mul_lc_with_scalar(f_limbs[4].clone(), c1);
        h_limbs[1] = f_limbs[1].clone() + &mul_lc_with_scalar(f_limbs[5].clone(), c1);
        h_limbs[2] = f_limbs[2].clone() + &mul_lc_with_scalar(f_limbs[6].clone(), c1);
        h_limbs[3] = f_limbs[3].clone();

        Ok(AllocatedLimbedInt {
            limbs: h_limbs,
            value: Some(LimbedInt::fold_quadratic_limbs(&f)),
        })
    }

    pub fn fold_cubic_limbs(
        &self,
    ) -> Result<Self, SynthesisError> {
        if self.value.is_none() {
            return Err(SynthesisError::Unsatisfiable);
        }

        let g = self.clone().value.unwrap();
        if self.limbs.len() != g.len() || g.len() != 10 {
            return Err(SynthesisError::Unsatisfiable);
        }

        let c1 = F::from(38u64);
        let c2 = F::from(1444u64);

        let g_limbs = self.clone().limbs; 
        let mut h_limbs = vec![LinearCombination::<F>::zero(); 4];
        h_limbs[0] = g_limbs[0].clone() + &mul_lc_with_scalar(g_limbs[4].clone(), c1)
            + &mul_lc_with_scalar(g_limbs[8].clone(), c2);
        h_limbs[1] = g_limbs[1].clone() + &mul_lc_with_scalar(g_limbs[5].clone(), c1)
            + &mul_lc_with_scalar(g_limbs[9].clone(), c2);
        h_limbs[2] = g_limbs[2].clone() + &mul_lc_with_scalar(g_limbs[6].clone(), c1);
        h_limbs[3] = g_limbs[3].clone() + &mul_lc_with_scalar(g_limbs[7].clone(), c1);

        Ok(AllocatedLimbedInt {
            limbs: h_limbs,
            value: Some(LimbedInt::fold_cubic_limbs(&g)),
        })
    }

    // The ith carry lies between -2^carry_lb_bitwidth+1 and
    // 2^carry_ub_bitwidth-1
    pub fn check_difference_is_zero<CS: ConstraintSystem<F>>(
        self,
        cs: &mut CS,
        other: &Self,
        carry_ub_bitwidth: Vec<usize>,
        carry_lb_bitwidth: Vec<usize>,
        base_bitwidth: usize,
    ) -> Result<(), SynthesisError> {
        let diff = self - other;
        if diff.value.is_none() {
            return Err(SynthesisError::Unsatisfiable);
        }
        let diff_value = diff.value.unwrap();
        let diff_len = diff.limbs.len();
        assert_eq!(carry_ub_bitwidth.len(), diff_len-1);
        assert_eq!(carry_lb_bitwidth.len(), diff_len-1);

        let mut carries: Vec<F> = vec![F::ZERO; diff_len-1];
        let mut carry_variables: Vec<Variable> = vec![];
        let exp = BigUint::from(base_bitwidth as u64);
        let base = F::from(2u64).pow_vartime(exp.to_u64_digits());

        for i in 0..diff_len-1 {
            assert!(carry_ub_bitwidth[i] - base_bitwidth > 0);
            let carry_plus_offset_range_bits = carry_ub_bitwidth[i] - base_bitwidth + 1;
            let offset = F::from(2u64).pow_vartime(&[(carry_lb_bitwidth[i]-base_bitwidth) as u64]);

            if i == 0 {
                let limb_bits = diff_value.limbs[0].to_le_bits();
                let mut coeff = F::ONE;
                // Calculating carries[0] as diff_value.limbs[0] shifted to the right 64 times (discard the 64 LSBs)
                for (j, bit) in limb_bits.into_iter().enumerate() {
                    if  j >= 64 {
                        if bit {
                            carries[0] += coeff;
                        }
                        coeff += coeff;
                    }
                }
                assert_eq!(diff_value.limbs[0], carries[0]*base);

                let cv = cs.alloc(
                    || format!("carry {i}"),
                    || Ok(carries[i])
                );
                assert!(cv.is_ok());
                carry_variables.push(cv.unwrap());

                range_check_lc(
                    &mut cs.namespace(|| format!("Range check carry {i} plus offset")),
                    &(LinearCombination::from_coeff(CS::one(), offset) + carry_variables[i]),
                    carries[i]+offset,
                    carry_plus_offset_range_bits,
                )?;

                cs.enforce(
                    || format!("Enforce carry constraint {i}"),
                    |lc| lc + &LinearCombination::from_coeff(carry_variables[i], base),
                    |lc| lc + CS::one(),
                    |lc| lc + &diff.limbs[i],
                );
            }
            else {
                let limb_bits = (carries[i-1] + diff_value.limbs[i]).to_le_bits();
                let mut coeff = F::ONE;
                // Calculating carries[i] as diff_value.limbs[i] + carries[i-1] shifted to the right 64 times (discard the 64 LSBs)
                for (j, bit) in limb_bits.into_iter().enumerate() {
                    if  j >= 64 {
                        if bit {
                            carries[i] += coeff;
                        }
                        coeff += coeff;
                    }
                }
                assert_eq!(diff_value.limbs[i] + carries[i-1], carries[i]*base);

                let cv = cs.alloc(
                    || format!("carry {i}"),
                    || Ok(carries[i])
                );
                assert!(cv.is_ok());
                carry_variables.push(cv.unwrap());

                range_check_lc(
                    &mut cs.namespace(|| format!("Range check carry {i} plus offset")),
                    &(LinearCombination::from_coeff(CS::one(), offset) + carry_variables[i]),
                    carries[i]+offset,
                    carry_plus_offset_range_bits,
                )?;

                cs.enforce(
                    || format!("Enforce carry constraint {i}"),
                    |lc| lc + &LinearCombination::from_coeff(carry_variables[i], base),
                    |lc| lc + CS::one(),
                    |lc| lc + &(diff.limbs[i].clone() + carry_variables[i-1]),
                );
            }
        }
        assert_eq!(diff_value.limbs[diff_len-1] + carries[diff_len-2], F::ZERO);
        Ok(cs.enforce(
            || format!("Enforce final zero"),
            |lc| lc + &diff.limbs[diff_len-1] + carry_variables[diff_len-2],
            |lc| lc + CS::one(),
            |lc| lc,
        ))
    }

    pub fn reduce_cubic_product<CS: ConstraintSystem<F>>(
        cs: &mut CS,
        a: &Self,
        b: &Self,
        c: &AllocatedOrConstantLimbedInt<F>,
    ) -> Result<Self, SynthesisError> {

        let abc = match c {
            AllocatedOrConstantLimbedInt::Allocated(allocated_c) => a.alloc_cubic_product_3var(
                &mut cs.namespace(|| format!("allocate unreduced cubic product")),
                b,
                allocated_c,
            ).unwrap(),
            AllocatedOrConstantLimbedInt::Constant(constant_c) => a.alloc_cubic_product_2var1const(
                &mut cs.namespace(|| format!("allocate unreduced cubic product")),
                b,
                constant_c,
            ).unwrap(),
        };
        let a_l = a.clone().value.unwrap();
        let b_l = b.clone().value.unwrap();
        let c_l = match c {
            AllocatedOrConstantLimbedInt::Allocated(allocated_c) => allocated_c.clone().value.unwrap(),
            AllocatedOrConstantLimbedInt::Constant(constant_c) => constant_c.clone(),
        };

        let one = BigUint::from(1u64);
        let q: BigUint = (one.clone() << 255) - BigUint::from(19u64);
        let q_l = LimbedInt::<F>::from(&q);
        
        let (t_l, r_l) = LimbedInt::<F>::calc_cubic_product_witness(
            &a_l,
            &b_l,
            &c_l,
        );
        let t = Self::alloc_from_limbed_int(
            &mut cs.namespace(|| "cubic product quotient"),
            t_l,
            3
        )?;
        t.range_check_limbs(
            &mut cs.namespace(|| "range check cubic product quotient"),
            64,
        )?;

        let r = Self::alloc_from_limbed_int(
            &mut cs.namespace(|| "cubic product remainder"),
            r_l,
            4
        )?;
        r.check_base_field_membership(
            &mut cs.namespace(|| "check cubic product remainder is in field"),
        )?;

        let tq = t * &q_l;
        let tq_plus_r = tq + &r;
        let h_l = abc.fold_cubic_limbs().unwrap();

        h_l.check_difference_is_zero(
            &mut cs.namespace(|| "checking difference is zero"),
            &tq_plus_r,
            vec![206, 205, 204, 203, 139],
            vec![129, 131, 131, 131, 130],
            64,
        )?;
        Ok(r)
    }

    pub fn verify_x_coordinate_quadratic_is_zero<CS: ConstraintSystem<F>>(
        cs: &mut CS,
        x1: &Self,
        x2: &Self,
        y1: &Self,
        y2: &Self,
        x3: &Self,
        v: &Self,
    ) -> Result<(), SynthesisError> {
        let one = BigUint::from(1u64);
        let q_uint: BigUint = (one.clone() << 255) - BigUint::from(19u64);
        let q_l = LimbedInt::<F>::from(&q_uint);
        let mut q71_l = LimbedInt::<F>::default();
        let two = F::from(2u64);
        let two_power_71 = two.pow_vartime(&[71u64]);
        for i in 0..q_l.len() {
            q71_l.limbs[i] = q_l.limbs[i] * two_power_71;
        }

        let x1y2 = x1.alloc_product(
            &mut cs.namespace(|| "alloc x1*y2"),
            y2,
        )?;
        let x1y2_folded = x1y2.fold_quadratic_limbs()?;
        let x2y1 = x2.alloc_product(
            &mut cs.namespace(|| "alloc x2*y1"),
            y1,
        )?;
        let x2y1_folded = x2y1.fold_quadratic_limbs()?;
        let x3v = x3.alloc_product(
            &mut cs.namespace(|| "alloc x3*v"),
            v,
        )?;
        let x3v_folded = x3v.fold_quadratic_limbs()?;

        let mut g_al = x1y2_folded + &x2y1_folded - &x3v_folded - x3;
        g_al = g_al.add_limbed_int::<CS>(&q71_l)?;
        
        let g_uint = BigUint::from(&g_al.clone().value.unwrap());
        assert!(g_uint.clone().rem(q_uint.clone()).is_zero());
        let t_uint = g_uint.clone() / q_uint.clone();
        assert!(t_uint < (one << 72));
        assert!(g_uint == t_uint.clone()*q_uint.clone());
        
        let t_l = LimbedInt::<F>::from(&t_uint);
        let t_al = Self::alloc_from_limbed_int(
            &mut cs.namespace(|| "allocate quotient t"),
            t_l,
            2,
        )?;
        t_al.range_check_limbs(
            &mut cs.namespace(|| "range check quotient"),
            64,
        )?;

        let tq_al = t_al * &q_l;

        g_al.check_difference_is_zero(
            &mut cs.namespace(|| "checking difference is zero"),
            &tq_al,
            vec![139, 140, 140, 140],
            vec![128, 130, 130, 130],
            64,
        )
    }

    pub fn verify_y_coordinate_quadratic_is_zero<CS: ConstraintSystem<F>>(
        cs: &mut CS,
        x1: &Self,
        x2: &Self,
        y1: &Self,
        y2: &Self,
        y3: &Self,
        v: &Self,
    ) -> Result<(), SynthesisError> {
        let one = BigUint::from(1u64);
        let q_uint: BigUint = (one.clone() << 255) - BigUint::from(19u64);
        let q_l = LimbedInt::<F>::from(&q_uint);
        let mut q71_l = LimbedInt::<F>::default();
        let two = F::from(2u64);
        let two_power_71 = two.pow_vartime(&[71u64]);
        for i in 0..q_l.len() {
            q71_l.limbs[i] = q_l.limbs[i] * two_power_71;
        }

        let x1x2 = x1.alloc_product(
            &mut cs.namespace(|| "alloc x1*x2"),
            x2,
        )?;
        let x1x2_folded = x1x2.fold_quadratic_limbs()?;
        let y1y2 = y1.alloc_product(
            &mut cs.namespace(|| "alloc y1*y2"),
            y2,
        )?;
        let y1y2_folded = y1y2.fold_quadratic_limbs()?;
        let y3v = y3.alloc_product(
            &mut cs.namespace(|| "alloc y3*v"),
            v,
        )?;
        let y3v_folded = y3v.fold_quadratic_limbs()?;

        let mut g_al = x1x2_folded + &y1y2_folded + &y3v_folded - y3;
        g_al = g_al.add_limbed_int::<CS>(&q71_l)?;
        
        let g_uint = BigUint::from(&g_al.clone().value.unwrap());
        assert!(g_uint.clone().rem(q_uint.clone()).is_zero());
        let t_uint = g_uint.clone() / q_uint.clone();
        assert!(t_uint < (one << 72));
        assert!(g_uint == t_uint.clone()*q_uint.clone());
        
        let t_l = LimbedInt::<F>::from(&t_uint);
        let t_al = Self::alloc_from_limbed_int(
            &mut cs.namespace(|| "allocate quotient t"),
            t_l,
            2,
        )?;
        t_al.range_check_limbs(
            &mut cs.namespace(|| "range check quotient"),
            64,
        )?;

        let tq_al = t_al * &q_l;

        g_al.check_difference_is_zero(
            &mut cs.namespace(|| "checking difference is zero"),
            &tq_al,
            vec![139, 140, 140, 140],
            vec![128, 130, 130, 130],
            64,
        )
    }

    // If condition is true, return a. Otherwise return b.
    // Based on Nova/src/gadgets/utils.rs:conditionally_select
    fn conditionally_select<CS: ConstraintSystem<F>>(
        cs: &mut CS,
        a: &Self,
        b: &Self,
        condition: &Boolean,
    ) -> Result<Self, SynthesisError> {
        assert!(a.value.is_some());
        assert!(b.value.is_some());
        let a_value = a.value.as_ref().unwrap();
        let b_value = b.value.as_ref().unwrap();
        assert_eq!(a_value.len(), b_value.len());

        let res = Self::alloc_from_limbed_int(
            &mut cs.namespace(|| "conditional select result"),
            if condition.get_value().unwrap() {
                a_value.clone()
            } else {
                b_value.clone()
            },
            a_value.len(),
        )?;
        
        // a[i] * condition + b[i]*(1-condition) = c[i] ->
        // a[i] * condition - b[i]*condition = c[i] - b[i]
        for i in 0..a_value.len() {
            cs.enforce(
                || format!("conditional select constraint on limb {i}"),
                |lc| lc + &a.limbs[i] - &b.limbs[i],
                |_| condition.lc(CS::one(), F::ONE),
                |lc| lc + &res.limbs[i] - &b.limbs[i],
            );
        }
        
        Ok(res)
    }
    // If condition0 + 2*condition1 is i, return ai.
    // Based on gnark/frontend/cs/r1cs/api.go:Lookup2
    fn conditionally_select2<CS: ConstraintSystem<F>>(
        cs: &mut CS,
        a0: &Self,
        a1: &Self,
        a2: &Self,
        a3: &Self,
        condition0: &Boolean,
        condition1: &Boolean,
    ) -> Result<Self, SynthesisError> {
        assert!(a0.value.is_some());
        assert!(a1.value.is_some());
        assert!(a2.value.is_some());
        assert!(a3.value.is_some());
        let a0_value = a0.value.as_ref().unwrap();
        let a1_value = a1.value.as_ref().unwrap();
        let a2_value = a2.value.as_ref().unwrap();
        let a3_value = a3.value.as_ref().unwrap();
        assert_eq!(a0_value.len(), a1_value.len());
        assert_eq!(a1_value.len(), a2_value.len());
        assert_eq!(a2_value.len(), a3_value.len());

        let res_value = match (condition1.get_value().unwrap(), condition0.get_value().unwrap()) {
            (false, false) => a0_value.clone(),
            (false,  true) => a1_value.clone(),
            (true,  false) => a2_value.clone(),
            (true,   true) => a3_value.clone(),
        };
        let res = Self::alloc_from_limbed_int(
            &mut cs.namespace(|| "conditional select2 result"),
            res_value,
            a0_value.len(), // all ai's have the same number of limbs
        )?;

        let tmp1_value = match condition1.get_value().unwrap() {
            false => a1_value.clone() - a0_value.clone(),
            true => a3_value.clone() - a2_value.clone(),
        };
        let tmp1 = Self::alloc_from_limbed_int(
            &mut cs.namespace(|| "conditional select2 tmp1 value"),
            tmp1_value.clone(),
            a0_value.len(), // all ai's have the same number of limbs
        )?;

        let mut zero_value = LimbedInt::<F>::default();
        zero_value.pad_limbs(a0_value.len());

        let tmp2_value = match condition0.get_value().unwrap() {
            false => zero_value,
            true => tmp1_value,
        };
        let tmp2 = Self::alloc_from_limbed_int(
            &mut cs.namespace(|| "conditional select2 tmp2 value"),
            tmp2_value,
            a0_value.len(), // all ai's have the same number of limbs
        )?;

        
        // Two-bit lookup can be done with three constraints
        //    (1) (a3 - a2 - a1 + a0) * condition1 = tmp1 - a1 + a0
        //    (2) tmp1 * condition0 = tmp2
        //    (3) (a2 - a0) * condition1 = res - tmp2 - a0 
        for i in 0..a0_value.len() {
            cs.enforce(
                || format!("conditional select2 constraint 1 on limb {i}"),
                |lc| lc + &a3.limbs[i] - &a2.limbs[i] - &a1.limbs[i] + &a0.limbs[i],
                |_| condition1.lc(CS::one(), F::ONE),
                |lc| lc + &tmp1.limbs[i] - &a1.limbs[i] + &a0.limbs[i],
            );
            cs.enforce(
                || format!("conditional select2 constraint 2 on limb {i}"),
                |lc| lc + &tmp1.limbs[i],
                |_| condition0.lc(CS::one(), F::ONE),
                |lc| lc + &tmp2.limbs[i],
            );
            cs.enforce(
                || format!("conditional select2 constraint 3 on limb {i}"),
                |lc| lc + &a2.limbs[i] - &a0.limbs[i],
                |_| condition1.lc(CS::one(), F::ONE),
                |lc| lc + &res.limbs[i] - &tmp2.limbs[i] - &a0.limbs[i],
            );
        }
        
        Ok(res)
    }

    // Adapted from https://docs.rs/bellperson-nonnative/0.4.0/src/bellperson_nonnative/util/gadget.rs.html#104-124
    fn mux_tree<'a, CS>(
        cs: &mut CS,
        mut select_bits: impl Iterator<Item = &'a Boolean> + Clone, // The first bit is taken as the highest order
        inputs: &[Self]
    ) -> Result<Self, SynthesisError>
    where
        CS: ConstraintSystem<F>,
    {
        if let Some(bit) = select_bits.next() {
            if inputs.len() & 1 != 0 {
                return Err(SynthesisError::Unsatisfiable);
            }
            let left_half = &inputs[..(inputs.len() / 2)];
            let right_half = &inputs[(inputs.len() / 2)..];
            let left = AllocatedLimbedInt::mux_tree(&mut cs.namespace(|| "left"), select_bits.clone(), left_half)?;
            let right = AllocatedLimbedInt::mux_tree(&mut cs.namespace(|| "right"), select_bits, right_half)?;
            AllocatedLimbedInt::conditionally_select(&mut cs.namespace(|| "join"),  &right, &left, bit)
        } else {
            if inputs.len() != 1 {
                return Err(SynthesisError::Unsatisfiable);
            }
            Ok(inputs[0].clone())
        }
    }
    
}


#[derive(Debug, Clone)]
pub struct AllocatedAffinePoint<F: PrimeField + PrimeFieldBits> {
    x: AllocatedLimbedInt<F>,
    y: AllocatedLimbedInt<F>,
    value: AffinePoint,
}

impl<F: PrimeField + PrimeFieldBits> AllocatedAffinePoint<F>  {
    
    pub fn get_point(&self) -> AffinePoint {
        self.value
    }

    pub fn alloc_affine_point<CS>(
        cs: &mut CS,
        value: AffinePoint,
    ) -> Result<Self, SynthesisError>
    where
        CS: ConstraintSystem<F>,
    {
        let limbed_affine_point = LimbedAffinePoint::<F>::from(&value);
        let x = AllocatedLimbedInt::<F>::alloc_from_limbed_int(
            &mut cs.namespace(|| "x coordinate"),
            limbed_affine_point.x,
            4
        )?;
        let y = AllocatedLimbedInt::<F>::alloc_from_limbed_int(
            &mut cs.namespace(|| "y coordinate"),
            limbed_affine_point.y,
            4
        )?;
        Ok(Self { x, y, value })
    }

    fn conditionally_select<CS>(
        cs: &mut CS,
        a: &Self,
        b: &Self,
        condition: &Boolean,
    ) -> Result<Self, SynthesisError>
    where
        CS: ConstraintSystem<F>,
    {
        let x = AllocatedLimbedInt::conditionally_select(
            &mut cs.namespace(|| "allocate value of output x coordinate"),
            &a.x,
            &b.x,
            condition,
        )?;

        let y = AllocatedLimbedInt::conditionally_select(
            &mut cs.namespace(|| "allocate value of output y coordinate"),
            &a.y,
            &b.y,
            condition,
        )?;

        let c = condition.get_value().unwrap();
        let value = if c {
            a.value
        } else {
            b.value
        };
        
        Ok(Self { x, y, value })
    }

    fn conditionally_select2<CS>(
        cs: &mut CS,
        a0: &Self,
        a1: &Self,
        a2: &Self,
        a3: &Self,
        condition0: &Boolean,
        condition1: &Boolean,
    ) -> Result<Self, SynthesisError>
    where
        CS: ConstraintSystem<F>,
    {
        let x = AllocatedLimbedInt::conditionally_select2(
            &mut cs.namespace(|| "allocate value of output x coordinate"),
            &a0.x,
            &a1.x,
            &a2.x,
            &a3.x,
            condition0,
            condition1,
        )?;

        let y = AllocatedLimbedInt::conditionally_select2(
            &mut cs.namespace(|| "allocate value of output y coordinate"),
            &a0.y,
            &a1.y,
            &a2.y,
            &a3.y,
            condition0,
            condition1,
        )?;

        let c0 = condition0.get_value().unwrap();
        let c1 = condition1.get_value().unwrap();
        let value = match (c0, c1) {
            (false, false) => a0.value,
            (true,  false) => a1.value,
            (false, true)  => a2.value,
            (true,  true)  => a3.value,
        };
        
        Ok(Self { x, y, value })
    }

    fn conditionally_select_m<CS>(
        cs: &mut CS,
        bits: &[Boolean],
        coords: &[Self],
        m: usize // Number of bits
    ) -> Result<Self, SynthesisError>
    where
        CS: ConstraintSystem<F>
    {
        assert_eq!(bits.len(), m);
        assert_eq!(coords.len(), 1 << m);

        let x_coords = coords
            .iter()
            .map(|c| c.x.clone())
            .collect::<Vec<AllocatedLimbedInt<F>>>()
        ;
        let y_coords = coords
            .iter()
            .map(|c| c.y.clone())
            .collect::<Vec<AllocatedLimbedInt<F>>>()
        ;

        let x = AllocatedLimbedInt::mux_tree(
            &mut cs.namespace(|| "allocate value of output x coordinate"), 
            bits.iter(), 
            &x_coords)
        ?;

        let y = AllocatedLimbedInt::mux_tree(
            &mut cs.namespace(|| "allocate value of output y coordinate"), 
            bits.iter(), 
            &y_coords)
        ?;

        let mut bits_value: Vec<bool> = bits.iter().map(|b| b.get_value().unwrap()).collect();
        bits_value.reverse();
        let mut idx = 0;
        for (i,b) in bits_value.iter().enumerate() {
            if *b {
                idx += 1<<i;
            }
        }
        let value = coords[idx].value.clone();
        Ok(Self { x, y, value})
    }

    fn verify_ed25519_point_addition<CS>(
        cs: &mut CS,
        p: &Self,
        q: &Self,
        r: &Self,
    ) -> Result<(), SynthesisError>
    where
        CS: ConstraintSystem<F>,
    {
        let x1_l = &p.x;
        let y1_l = &p.y;
        let x2_l = &q.x;
        let y2_l = &q.y;
        let x3_l = &r.x;
        let y3_l = &r.y;

        let d_l = LimbedInt::<F>::from(&D);

        let u_l = AllocatedLimbedInt::reduce_cubic_product(
            &mut cs.namespace(|| "allocate d*x1*x2 mod q"),
            &x1_l,
            &x2_l,
            &AllocatedOrConstantLimbedInt::Constant(d_l),
        )?;
        
        let v_l = AllocatedLimbedInt::reduce_cubic_product(
            &mut cs.namespace(|| "allocate u*y1*y2 mod q"),
            &u_l,
            &y1_l,
            &AllocatedOrConstantLimbedInt::<F>::Allocated(y2_l.clone()),
        )?;
        
        
        AllocatedLimbedInt::<F>::verify_x_coordinate_quadratic_is_zero(
            &mut cs.namespace(|| "checking x coordinate quadratic"),
            x1_l, x2_l, y1_l, y2_l, x3_l, &v_l
        )?;
        AllocatedLimbedInt::<F>::verify_y_coordinate_quadratic_is_zero(
            &mut cs.namespace(|| "checking y coordinate quadratic"),
            x1_l, x2_l, y1_l, y2_l, y3_l, &v_l
        )?;
        Ok(())
    }
    
    pub fn ed25519_point_addition<CS>(
        cs: &mut CS,
        p: &Self,
        q: &Self,
    ) -> Result<Self, SynthesisError>
    where
        CS: ConstraintSystem<F>,
    {
        let sum_value = p.value + q.value;
        let sum = Self::alloc_affine_point(
            &mut cs.namespace(|| "allocate sum"),
            sum_value,
        )?;

        sum.x.check_base_field_membership(
            &mut cs.namespace(|| "check x coordinate of sum is in base field")
        )?;
        sum.y.check_base_field_membership(
            &mut cs.namespace(|| "check y coordinate of sum is in base field")
        )?;

        Self::verify_ed25519_point_addition(
            &mut cs.namespace(|| "verify point addition"),
            p,
            q,
            &sum,
        )?;

        Ok(sum)
    }

    pub fn ed25519_point_doubling<CS>(
        cs: &mut CS,
        p: &Self,
    ) -> Result<Self, SynthesisError>
    where
        CS: ConstraintSystem<F>,
    {
        let sum_value = p.value + p.value;
        let sum = Self::alloc_affine_point(
            &mut cs.namespace(|| "allocate sum"),
            sum_value,
        )?;

        sum.x.check_base_field_membership(
            &mut cs.namespace(|| "check x coordinate of sum is in base field")
        )?;
        sum.y.check_base_field_membership(
            &mut cs.namespace(|| "check y coordinate of sum is in base field")
        )?;

        Self::verify_ed25519_point_addition(
            &mut cs.namespace(|| "verify point addition"),
            p,
            p,
            &sum,
        )?;

        Ok(sum)
    }

    pub fn ed25519_scalar_multiplication_old<CS>(
        self,
        cs: &mut CS,
        scalar: Vec<Boolean>,
    ) -> Result<Self, SynthesisError>
    where
        CS: ConstraintSystem<F>,
    {
        assert!(scalar.len() < 254usize); // the largest curve25519 scalar fits in 253 bits
        let identity: AffinePoint = AffinePoint::default();
        
        // No range checks on limbs required as value is known to be (0,1)
        let mut output = Self::alloc_affine_point(
            &mut cs.namespace(|| "allocate initial value of output"),
            identity,
        )?;

        // Remember to avoid field membership checks before calling this function
        self.x.check_base_field_membership(
            &mut cs.namespace(|| "check x coordinate of base point is in base field")
        )?;
        self.y.check_base_field_membership(
            &mut cs.namespace(|| "check y coordinate of base point is in base field")
        )?;

        let mut step_point = self;

        for (i, bit) in scalar.iter().enumerate() {
            let output0 = output.clone();
            let output1 = Self::ed25519_point_addition(
                &mut cs.namespace(|| format!("sum in step {i} if bit is one")),
                &output,
                &step_point,
            )?;

            let output_value = if bit.get_value().unwrap() {
                output1.value
            } else {
                output0.value
            };

            let output_x = AllocatedLimbedInt::conditionally_select(
                &mut cs.namespace(|| format!("conditionally select x coordinate in step {i}")),
                &output1.x,
                &output0.x,
                bit,
            )?;
            let output_y = AllocatedLimbedInt::conditionally_select(
                &mut cs.namespace(|| format!("conditionally select y coordinate in step {i}")),
                &output1.y,
                &output0.y,
                bit,
            )?;

            output = Self {
                x: output_x,
                y: output_y,
                value: output_value,
            };

            step_point = Self::ed25519_point_addition(
                &mut cs.namespace(|| format!("point doubling in step {i}")),
                &step_point,
                &step_point,
            )?;
        }

        Ok(output)
    }

    pub fn ed25519_scalar_multiplication<CS>(
        self,
        cs: &mut CS,
        scalar: Vec<Boolean>,
    ) -> Result<Self, SynthesisError>
    where
        CS: ConstraintSystem<F>,
    {
        let scalar_len = scalar.len();
        assert!(scalar_len < 254usize); // the largest curve25519 scalar fits in 253 bits
        let identity: AffinePoint = AffinePoint::default();
        
        // No range checks on limbs required as value is known to be (0,1)
        let identity_point = Self::alloc_affine_point(
            &mut cs.namespace(|| "allocate identity point"),
            identity,
        )?;
        // Remember to avoid field membership checks before calling this function
        self.x.check_base_field_membership(
            &mut cs.namespace(|| "check x coordinate of base point is in base field")
        )?;
        self.y.check_base_field_membership(
            &mut cs.namespace(|| "check y coordinate of base point is in base field")
        )?;


        let a = Self::ed25519_point_doubling(
            &mut cs.namespace(|| "allocate double the base"),
            &self,
        )?;
        let b = Self::ed25519_point_addition(
            &mut cs.namespace(|| "allocate three times the base"),
            &a,
            &self,
        )?;
        let n = scalar_len - 1;
        assert!(n > 1);

        let mut output = Self::conditionally_select2(
            &mut cs.namespace(|| "allocate initial value of output"),
            &identity_point,
            &self,
            &a,
            &b,
            &scalar[n-1],
            &scalar[n]
        )?;
            
        let mut i: i32 = (n-2) as i32;
        while i > 0 {
            output = Self::ed25519_point_doubling(
                &mut cs.namespace(|| format!("first doubling in iteration {i}")),
                &output,
            )?;
            output = Self::ed25519_point_doubling(
                &mut cs.namespace(|| format!("second doubling in iteration {i}")),
                &output,
            )?;

            let tmp = Self::conditionally_select2(
                &mut cs.namespace(|| format!("allocate tmp value in iteration {i}")),
                &identity_point,
                &self,
                &a,
                &b,
                &scalar[(i-1) as usize],
                &scalar[i as usize]
            )?;

            output = Self::ed25519_point_addition(
                &mut cs.namespace(|| format!("allocate sum of output and tmp in iteration {i}")),
                &output,
                &tmp,
            )?;
                
            i = i-2;
        }

        if n % 2 == 0 {
            output = Self::ed25519_point_doubling(
                &mut cs.namespace(|| "final doubling of output"),
                &output,
            )?;
            let tmp = Self::ed25519_point_addition(
                &mut cs.namespace(|| "final sum of output and base"),
                &output,
                &self,
            )?;
            output = Self::conditionally_select(
                cs,
                &tmp,
                &output,
                &scalar[0],
            )?;
            
        }

        Ok(output)
    }

    pub fn ed25519_scalar_multiplication_m_bit<CS>(
        self,
        cs: &mut CS,
        scalar: Vec<Boolean>,
        m: usize // Number of bits
    ) -> Result<Self, SynthesisError>
    where
        CS: ConstraintSystem<F>,
    {
        let scalar_len = scalar.len();
        assert!(scalar_len < 254usize); // the largest curve25519 scalar fits in 253 bits
        let identity: AffinePoint = AffinePoint::default();

        // No range checks on limbs required as value is known to be (0,1)
        let identity_point = Self::alloc_affine_point(
            &mut cs.namespace(|| "allocate identity point"),
            identity,
        )?;
        // Remember to avoid field membership checks before calling this function
        self.x.check_base_field_membership(
            &mut cs.namespace(|| "check x coordinate of base point is in base field")
        )?;
        self.y.check_base_field_membership(
            &mut cs.namespace(|| "check y coordinate of base point is in base field")
        )?;

        let mut lookup_vec: Vec<AllocatedAffinePoint<F>> = vec![];
        lookup_vec.push(identity_point.clone());
        lookup_vec.push(self.clone());

        for i in 2..(1<<m) {
            let point = Self::ed25519_point_addition(
                &mut cs.namespace(|| format!("allocate {} times the base", i)),
                &lookup_vec[i-1],
                &self,
            )?;
            lookup_vec.push(point);
        }
        assert_eq!(lookup_vec.len(), (1<<m));

        let n = scalar_len - 1;
        // assert!(n > 1);

        let mut lookup_bits: Vec<Boolean> = vec![];
        for i in ((n+1-m)..(n+1)).rev() {
            lookup_bits.push(scalar[i].clone());
        }
        assert_eq!(lookup_bits.len(), m);

        let mut output = Self::conditionally_select_m(
            &mut cs.namespace(|| "allocate initial value of output"), 
            &lookup_bits, 
            &lookup_vec,
            m
        )?;
            
        let mut i: i32 = n as i32 - m as i32 ;
        while i > 0 {

            if i < (m as i32)-1 {

                for j in 0..(i+1) {
                    output = Self::ed25519_point_doubling(
                        &mut cs.namespace(|| format!("{j} doubling in iteration {i}")),
                        &output,
                    )?;
                }

                let mut lookup_bits: Vec<Boolean> = vec![];
                for j in (0..(i+1)).rev() {
                    lookup_bits.push(scalar[j as usize].clone());
                }

                let tmp = Self::conditionally_select_m(
                    &mut cs.namespace(|| format!("allocate tmp value in iteration {i}")),
                    &lookup_bits, 
                    &lookup_vec[0..(1<<(i as usize +1))],
                    i as usize +1,
                )?;

                output = Self::ed25519_point_addition(
                    &mut cs.namespace(|| format!("allocate sum of output and tmp in iteration {i}")),
                    &output,
                    &tmp,
                )?;

                break;
            }

            for j in 0..m {
                output = Self::ed25519_point_doubling(
                    &mut cs.namespace(|| format!("{j} doubling in iteration {i}")),
                    &output,
                )?;
            }
            let mut lookup_bits: Vec<Boolean> = vec![];
            for j in ((i as usize + 1 - m)..(i as usize + 1)).rev() {
                lookup_bits.push(scalar[j as usize].clone());
            }
            assert_eq!(lookup_bits.len(), m);

            let tmp = Self::conditionally_select_m(
                &mut cs.namespace(|| format!("allocate tmp value in iteration {i}")),
                &lookup_bits, 
                &lookup_vec,
                m
            )?;

            output = Self::ed25519_point_addition(
                &mut cs.namespace(|| format!("allocate sum of output and tmp in iteration {i}")),
                &output,
                &tmp,
            )?;
                
            i = i - (m as i32);
        }

        if n % m == 0 {
            output = Self::ed25519_point_doubling(
                &mut cs.namespace(|| "final doubling of output"),
                &output,
            )?;
            let tmp = Self::ed25519_point_addition(
                &mut cs.namespace(|| "final sum of output and base"),
                &output,
                &self,
            )?;
            output = Self::conditionally_select(
                cs,
                &tmp,
                &output,
                &scalar[0],
            )?;
            
        }

        Ok(output)
    }

}

#[cfg(test)]
mod tests {
    use crate::curve::Ed25519Curve;

    use super::*;
    use bellperson::gadgets::test::TestConstraintSystem;
    use crypto_bigint::{U256, Random, Integer};
    use num_traits::Zero;
    use pasta_curves::Fp;
    use num_bigint::{RandBigInt, BigUint};

    #[test]
    fn range_check_linear_combination() {
        let mut cs = TestConstraintSystem::<Fp>::new();
        let mut rng = rand::thread_rng();
        let a_num_bits = 64usize;
        let a_uint = rng.gen_biguint(a_num_bits as u64);
        // let b_uint = rng.gen_biguint(192u64);
        let a_repr_init: Vec<u8> = a_uint.to_bytes_le();
        
        let mut a_repr: [u8; 32] = [0u8; 32];
        for i in 0..a_repr_init.len(){
            a_repr[i] = a_repr_init[i];
        }

        let a_scalar = Fp::from_repr(a_repr);
        assert!(bool::from(a_scalar.is_some()));
        let a = a_scalar.unwrap();

        let a_var = cs.alloc(|| "a variable", || Ok(a));
        assert!(a_var.is_ok());
        let a_var = a_var.unwrap();
        let res = range_check_lc(
            &mut cs.namespace(|| "Check range of a"),
            &LinearCombination::from_variable(a_var),
            a,
            a_num_bits,
        );
        assert!(res.is_ok());

        let b_num_bits = 143usize;
        let b_uint = rng.gen_biguint(b_num_bits as u64);
        let b_repr_init: Vec<u8> = b_uint.to_bytes_le();
        
        let mut b_repr: [u8; 32] = [0u8; 32];
        for i in 0..b_repr_init.len(){
            b_repr[i] = b_repr_init[i];
        }

        let b_scalar = Fp::from_repr(b_repr);
        assert!(bool::from(b_scalar.is_some()));
        let b = b_scalar.unwrap();

        let b_var = cs.alloc(|| "b variable", || Ok(b));
        assert!(b_var.is_ok());
        let b_var = b_var.unwrap();
        let res = range_check_lc(
            &mut cs.namespace(|| "Check range of b"),
            &LinearCombination::from_variable(b_var),
            b,
            b_num_bits,
        );
        assert!(res.is_ok());


        assert!(cs.is_satisfied());
        println!("Num constraints = {:?}", cs.num_constraints());
        println!("Num inputs = {:?}", cs.num_inputs());
    }
    
    #[test]
    fn alloc_limbed_sum() {
        let mut cs = TestConstraintSystem::<Fp>::new();
        let mut rng = rand::thread_rng();
        let a_uint = rng.gen_biguint(255u64);
        let b_uint = rng.gen_biguint(255u64);
        let a_l = LimbedInt::<Fp>::from(&a_uint);
        let b_l = LimbedInt::<Fp>::from(&b_uint);

        let a = AllocatedLimbedInt::<Fp>::alloc_from_limbed_int(
            &mut cs.namespace(|| "alloc a"),
            a_l,
            4
        );
        assert!(a.is_ok());
        let b = AllocatedLimbedInt::<Fp>::alloc_from_limbed_int(
            &mut cs.namespace(|| "alloc b"),
            b_l,
            4
        );
        assert!(b.is_ok());

        let a = a.unwrap();
        let b = b.unwrap();
        let sum = a.clone() + &b;

        for i in 0..sum.limbs.len() {
            cs.enforce(
                || format!("sum {i}"),
                |lc| lc + &a.limbs[i] + &b.limbs[i],
                |lc| lc + TestConstraintSystem::<Fp>::one(),
                |lc| lc + &sum.limbs[i],
            );
        }
        
        assert!(cs.is_satisfied());
        println!("Num constraints = {:?}", cs.num_constraints());
        println!("Num inputs = {:?}", cs.num_inputs());
    }

    #[test]
    fn alloc_limbed_difference() {
        let mut cs = TestConstraintSystem::<Fp>::new();
        let mut rng = rand::thread_rng();
        let x_uint = rng.gen_biguint(255u64);
        let y_uint = rng.gen_biguint(255u64);
        let (a_uint, b_uint) = if x_uint > y_uint {
            (x_uint, y_uint)
        } else {
            (y_uint, x_uint)
        };
        let diff_uint = a_uint.clone() - b_uint.clone();
        let a_l = LimbedInt::<Fp>::from(&a_uint);
        let b_l = LimbedInt::<Fp>::from(&b_uint);
        let diff = LimbedInt::<Fp>::from(&diff_uint);
        println!("{:?}\n{:?}", diff_uint, BigUint::from(&diff));

        let a = AllocatedLimbedInt::<Fp>::alloc_from_limbed_int(
            &mut cs.namespace(|| "alloc a"),
            a_l,
            4
        );
        assert!(a.is_ok());
        let b = AllocatedLimbedInt::<Fp>::alloc_from_limbed_int(
            &mut cs.namespace(|| "alloc b"),
            b_l,
            4
        );
        assert!(b.is_ok());

        let a = a.unwrap();
        let b = b.unwrap();
        let diff = a.clone() - &b;
        let a_val = a.value.unwrap();
        let b_val = b.value.unwrap();
        let diff_val = diff.value.unwrap();

        for i in 0..diff.limbs.len() {
            assert_eq!(a_val.limbs[i]-b_val.limbs[i], diff_val.limbs[i]);
            cs.enforce(
                || format!("sum {i}"),
                |lc| lc + &a.limbs[i] - &b.limbs[i],
                |lc| lc + TestConstraintSystem::<Fp>::one(),
                |lc| lc + &diff.limbs[i],
            );
        }
        
        assert!(cs.is_satisfied());
        println!("Num constraints = {:?}", cs.num_constraints());
        println!("Num inputs = {:?}", cs.num_inputs());
    }
    
    #[test]
    fn alloc_limbed_range_check() {
        let mut cs = TestConstraintSystem::<Fp>::new();
        let mut rng = rand::thread_rng();
        let a_uint = rng.gen_biguint(256u64);
        let a_l = LimbedInt::<Fp>::from(&a_uint);

        let a = AllocatedLimbedInt::<Fp>::alloc_from_limbed_int(
            &mut cs.namespace(|| "alloc a"),
            a_l,
            4
        );
        assert!(a.is_ok());
        let a = a.unwrap();

        let res = a.range_check_limbs(
            &mut cs.namespace(|| "range check limbs"),
            64,
        );
        assert!(res.is_ok());

        assert!(cs.is_satisfied());
        println!("Num constraints = {:?}", cs.num_constraints());
        println!("Num inputs = {:?}", cs.num_inputs());

        let b_uint: BigUint = BigUint::from(u64::MAX);
        let b_l = LimbedInt::<Fp>::from(&b_uint);
        let b = AllocatedLimbedInt::<Fp>::alloc_from_limbed_int(
            &mut cs.namespace(|| "alloc b"),
            b_l,
            1
        );
        assert!(b.is_ok());
        let b = b.unwrap();
        let res = b.range_check_limbs(
            &mut cs.namespace(|| "range check limbs with fewer bits"),
            63,
        );
        assert!(res.is_ok());
        assert!(!cs.is_satisfied());
        println!("Num constraints = {:?}", cs.num_constraints());
        println!("Num inputs = {:?}", cs.num_inputs());

    }

    #[test]
    fn alloc_limbed_field_membership() {
        let mut cs = TestConstraintSystem::<Fp>::new();
        let zero = BigUint::from(0u64);
        let one = BigUint::from(1u64);
        let q_uint: BigUint = (one.clone() << 255) - BigUint::from(19u64);
        let mut rng = rand::thread_rng();
        let a_uint = rng.gen_biguint_range(&zero, &q_uint);
        let a_l = LimbedInt::<Fp>::from(&a_uint);

        let a = AllocatedLimbedInt::<Fp>::alloc_from_limbed_int(
            &mut cs.namespace(|| "alloc a"),
            a_l,
            4
        );
        assert!(a.is_ok());
        let a = a.unwrap();

        let res = a.check_base_field_membership(
            &mut cs.namespace(|| "check field membership"),
        );
        assert!(res.is_ok());

        assert!(cs.is_satisfied());
        println!("Num constraints = {:?}", cs.num_constraints());
        println!("Num inputs = {:?}", cs.num_inputs());

        let b_uint = q_uint.clone() - one;
        let b_l = LimbedInt::<Fp>::from(&b_uint);

        let b = AllocatedLimbedInt::<Fp>::alloc_from_limbed_int(
            &mut cs.namespace(|| "alloc b"),
            b_l,
            4
        );
        assert!(b.is_ok());
        let b = b.unwrap();

        let res = b.check_base_field_membership(
            &mut cs.namespace(|| "check field membership of q-1"),
        );
        assert!(res.is_ok());

        assert!(cs.is_satisfied());
        println!("Num constraints = {:?}", cs.num_constraints());
        println!("Num inputs = {:?}", cs.num_inputs());

        let q_l = LimbedInt::<Fp>::from(&q_uint);

        let q_al = AllocatedLimbedInt::<Fp>::alloc_from_limbed_int(
            &mut cs.namespace(|| "alloc q"),
            q_l,
            4
        );
        assert!(q_al.is_ok());
        let q_al = q_al.unwrap();

        let res = q_al.check_base_field_membership(
            &mut cs.namespace(|| "check field non-membership of q"),
        );
        assert!(res.is_ok());

        assert!(!cs.is_satisfied());
        println!("Num constraints = {:?}", cs.num_constraints());
        println!("Num inputs = {:?}", cs.num_inputs());

    }

    #[test]
    fn alloc_limbed_quadratic_product() {
        let mut cs = TestConstraintSystem::<Fp>::new();
        let mut rng = rand::thread_rng();
        let a_uint = rng.gen_biguint(192u64);
        let b_uint = rng.gen_biguint(256u64);
        let a_l = LimbedInt::<Fp>::from(&a_uint);
        let b_l = LimbedInt::<Fp>::from(&b_uint);

        let a = AllocatedLimbedInt::<Fp>::alloc_from_limbed_int(
            &mut cs.namespace(|| "alloc a"),
            a_l,
            3
        );
        assert!(a.is_ok());
        let b = AllocatedLimbedInt::<Fp>::alloc_from_limbed_int(
            &mut cs.namespace(|| "alloc b"),
            b_l,
            4
        );
        assert!(b.is_ok());
        
        let a = a.unwrap();
        let b = b.unwrap();

        let prod = a.alloc_product(
            &mut cs.namespace(|| "product"),
            &b
        );
        assert!(prod.is_ok());
        assert_eq!(BigUint::from(&prod.unwrap().value.unwrap()), a_uint*b_uint);

        assert!(cs.is_satisfied());
        println!("Num constraints = {:?}", cs.num_constraints());
        println!("Num inputs = {:?}", cs.num_inputs());

    }

    #[test]
    fn alloc_limbed_cubic_product() {
        let mut cs = TestConstraintSystem::<Fp>::new();
        let mut rng = rand::thread_rng();
        let a_uint = rng.gen_biguint(256u64);
        let b_uint = rng.gen_biguint(256u64);
        let c_uint = rng.gen_biguint(256u64);
        let a_l = LimbedInt::<Fp>::from(&a_uint);
        let b_l = LimbedInt::<Fp>::from(&b_uint);
        let c_l = LimbedInt::<Fp>::from(&c_uint);

        let a = AllocatedLimbedInt::<Fp>::alloc_from_limbed_int(
            &mut cs.namespace(|| "alloc a"),
            a_l,
            4
        );
        assert!(a.is_ok());
        let b = AllocatedLimbedInt::<Fp>::alloc_from_limbed_int(
            &mut cs.namespace(|| "alloc b"),
            b_l,
            4
        );
        assert!(b.is_ok());
        let c = AllocatedLimbedInt::<Fp>::alloc_from_limbed_int(
            &mut cs.namespace(|| "alloc c"),
            c_l,
            4
        );
        assert!(c.is_ok());
        
        let a = a.unwrap();
        let b = b.unwrap();
        let c = c.unwrap();

        let prod = a.alloc_cubic_product_3var(
            &mut cs.namespace(|| "product"),
            &b,
            &c,
        );
        assert!(prod.is_ok());
        assert_eq!(BigUint::from(&prod.unwrap().value.unwrap()), a_uint*b_uint*c_uint);

        assert!(cs.is_satisfied());
        println!("Num constraints = {:?}", cs.num_constraints());
        println!("Num inputs = {:?}", cs.num_inputs());

    }

    #[test]
    fn alloc_limbed_fold_quadratic() {
        let mut cs = TestConstraintSystem::<Fp>::new();
        let mut rng = rand::thread_rng();
        let a_uint = rng.gen_biguint(256u64);
        let b_uint = rng.gen_biguint(256u64);
        let a_l = LimbedInt::<Fp>::from(&a_uint);
        let b_l = LimbedInt::<Fp>::from(&b_uint);

        let a = AllocatedLimbedInt::<Fp>::alloc_from_limbed_int(
            &mut cs.namespace(|| "alloc a"),
            a_l,
            4
        );
        assert!(a.is_ok());
        let b = AllocatedLimbedInt::<Fp>::alloc_from_limbed_int(
            &mut cs.namespace(|| "alloc b"),
            b_l,
            4
        );
        assert!(b.is_ok());
        
        let a = a.unwrap();
        let b = b.unwrap();

        let prod = a.alloc_product(
            &mut cs.namespace(|| "product"),
            &b
        );
        assert!(prod.is_ok());

        let h = prod.unwrap().fold_quadratic_limbs().unwrap();
        assert!(h.value.is_some());
        let h_value = h.value.unwrap();
        let one = TestConstraintSystem::<Fp>::one();
        for i in 0..h_value.len() {
            cs.enforce(|| format!("limb {i} equality"),
                |lc| lc + &h.limbs[i],
                |lc| lc + one,
                |lc| lc + (h_value.limbs[i], one),
            );
        }


        assert!(cs.is_satisfied());
        println!("Num constraints = {:?}", cs.num_constraints());
        println!("Num inputs = {:?}", cs.num_inputs());

    }

    #[test]
    fn alloc_limbed_fold_cubic() {
        let mut cs = TestConstraintSystem::<Fp>::new();
        let mut rng = rand::thread_rng();
        let a_uint = rng.gen_biguint(256u64);
        let b_uint = rng.gen_biguint(256u64);
        let c_uint = rng.gen_biguint(256u64);
        let a_l = LimbedInt::<Fp>::from(&a_uint);
        let b_l = LimbedInt::<Fp>::from(&b_uint);
        let c_l = LimbedInt::<Fp>::from(&c_uint);

        let a = AllocatedLimbedInt::<Fp>::alloc_from_limbed_int(
            &mut cs.namespace(|| "alloc a"),
            a_l,
            4
        );
        assert!(a.is_ok());
        let b = AllocatedLimbedInt::<Fp>::alloc_from_limbed_int(
            &mut cs.namespace(|| "alloc b"),
            b_l,
            4
        );
        assert!(b.is_ok());
        let c = AllocatedLimbedInt::<Fp>::alloc_from_limbed_int(
            &mut cs.namespace(|| "alloc c"),
            c_l,
            4
        );
        assert!(c.is_ok());
        
        
        let a = a.unwrap();
        let b = b.unwrap();
        let c = c.unwrap();

        let prod = a.alloc_cubic_product_3var(
            &mut cs.namespace(|| "product"),
            &b,
            &c,
        );
        assert!(prod.is_ok());

        let h = prod.unwrap().fold_cubic_limbs().unwrap();
        assert!(h.value.is_some());
        let h_value = h.value.unwrap();
        let one = TestConstraintSystem::<Fp>::one();
        for i in 0..h_value.len() {
            cs.enforce(|| format!("limb {i} equality"),
                |lc| lc + &h.limbs[i],
                |lc| lc + one,
                |lc| lc + (h_value.limbs[i], one),
            );
        }

        assert!(cs.is_satisfied());
        println!("Num constraints = {:?}", cs.num_constraints());
        println!("Num inputs = {:?}", cs.num_inputs());

    }

    #[test]
    fn alloc_limbed_check_difference() {
        let mut cs = TestConstraintSystem::<Fp>::new();
        let mut rng = rand::thread_rng();
        let a_uint = rng.gen_biguint(256u64);
        let b_uint = rng.gen_biguint(256u64);
        let a_l = LimbedInt::<Fp>::from(&a_uint);
        let b_l = LimbedInt::<Fp>::from(&b_uint);

        let ab_l = LimbedInt::<Fp>::calc_quadratic_limbs(&a_l, &b_l);
        let ab_folded_l = LimbedInt::<Fp>::fold_quadratic_limbs(&ab_l);
        let ab_folded_uint = BigUint::from(&ab_folded_l);

        let one = BigUint::from(1u64);
        let q_uint: BigUint = (one.clone() << 255) - BigUint::from(19u64);
        let r_uint = ab_folded_uint.clone() % q_uint.clone();
        let t_uint = (ab_folded_uint.clone() -r_uint.clone())/q_uint.clone();
        assert!(((ab_folded_uint - t_uint.clone()*q_uint.clone() - r_uint.clone()).is_zero()));

        let q_l = LimbedInt::<Fp>::from(&q_uint);
        let mut t_l = LimbedInt::<Fp>::from(&t_uint);
        t_l.pad_limbs(2);
        let r_l = LimbedInt::<Fp>::from(&r_uint);

        let ab_folded = AllocatedLimbedInt::<Fp>::alloc_from_limbed_int(
            &mut cs.namespace(|| "alloc ab_folded"),
            ab_folded_l,
            4
        );
        assert!(ab_folded.is_ok());
        let t = AllocatedLimbedInt::<Fp>::alloc_from_limbed_int(
            &mut cs.namespace(|| "alloc t"),
            t_l,
            2
        );
        assert!(t.is_ok());
        let r = AllocatedLimbedInt::<Fp>::alloc_from_limbed_int(
            &mut cs.namespace(|| "alloc r"),
            r_l,
            4
        );
        assert!(r.is_ok());
        
        let ab_folded = ab_folded.unwrap();
        let t = t.unwrap();
        let r = r.unwrap();

        let tq = t * &q_l;
        let tq_plus_r = tq + &r;

        let res = ab_folded.check_difference_is_zero(
            &mut cs.namespace(|| "check difference is zero"),
            &tq_plus_r,
            vec![139, 140, 140, 140],
            vec![128, 130, 130, 130],
            64,
        );
        assert!(res.is_ok());

        
        println!("{:?}", cs.which_is_unsatisfied());
        assert!(cs.is_satisfied());
        println!("Num constraints = {:?}", cs.num_constraints());
        println!("Num inputs = {:?}", cs.num_inputs());

    }

    #[test]
    fn alloc_limbed_reduce_cubic() {
        let mut cs = TestConstraintSystem::<Fp>::new();
        let mut rng = rand::thread_rng();
        let a_uint = rng.gen_biguint(256u64);
        let b_uint = rng.gen_biguint(256u64);
        let c_uint = rng.gen_biguint(256u64);
        let a_l = LimbedInt::<Fp>::from(&a_uint);
        let b_l = LimbedInt::<Fp>::from(&b_uint);
        let c_l = LimbedInt::<Fp>::from(&c_uint);
        let (_, r_l) = LimbedInt::<Fp>::calc_cubic_product_witness(&a_l, &b_l, &c_l);

        let a = AllocatedLimbedInt::<Fp>::alloc_from_limbed_int(
            &mut cs.namespace(|| "alloc a"),
            a_l,
            4
        );
        assert!(a.is_ok());
        let b = AllocatedLimbedInt::<Fp>::alloc_from_limbed_int(
            &mut cs.namespace(|| "alloc b"),
            b_l,
            4
        );
        assert!(b.is_ok());
        let c = AllocatedLimbedInt::<Fp>::alloc_from_limbed_int(
            &mut cs.namespace(|| "alloc c"),
            c_l,
            4
        );
        assert!(c.is_ok());
        let r = AllocatedLimbedInt::<Fp>::alloc_from_limbed_int(
            &mut cs.namespace(|| "alloc r"),
            r_l,
            4
        );
        assert!(r.is_ok());
        
        let a = a.unwrap();
        let b = b.unwrap();
        let c = c.unwrap();
        let r = r.unwrap();

        let r_calc = AllocatedLimbedInt::<Fp>::reduce_cubic_product(
            &mut cs.namespace(|| "verify cubic product reduced mod q"),
            &a,
            &b,
            &AllocatedOrConstantLimbedInt::Allocated(c),
        );
        assert!(r_calc.is_ok());
        let r_calc = r_calc.unwrap();

        let one = TestConstraintSystem::<Fp>::one();
        for i in 0..r.limbs.len() {
            cs.enforce(|| format!("r limb {i} equality"),
                |lc| lc + &r.limbs[i],
                |lc| lc + one,
                |lc| lc + &r_calc.limbs[i],
            );
        }

        assert!(cs.is_satisfied());
        println!("Num constraints = {:?}", cs.num_constraints());
        println!("Num inputs = {:?}", cs.num_inputs());

    }

    #[test]
    fn alloc_limbed_conditionally_select() {
        let mut cs = TestConstraintSystem::<Fp>::new();
        let mut rng = rand::thread_rng();
        let a_uint = rng.gen_biguint(256u64);
        let b_uint = rng.gen_biguint(256u64);
        let a_l = LimbedInt::<Fp>::from(&a_uint);
        let b_l = LimbedInt::<Fp>::from(&b_uint);

        let a = AllocatedLimbedInt::<Fp>::alloc_from_limbed_int(
            &mut cs.namespace(|| "alloc a"),
            a_l,
            4
        );
        assert!(a.is_ok());
        let b = AllocatedLimbedInt::<Fp>::alloc_from_limbed_int(
            &mut cs.namespace(|| "alloc b"),
            b_l,
            4
        );
        assert!(b.is_ok());
        
        let a = a.unwrap();
        let b = b.unwrap();
        
        let conditions = vec![false, true];
        for c in conditions {
            let condition = Boolean::constant(c);
            
            let res = AllocatedLimbedInt::<Fp>::conditionally_select(
                &mut cs.namespace(|| format!("conditionally select a or b for condition = {c}")),
                &a,
                &b,
                &condition,
            );
            assert!(res.is_ok());
            let res = res.unwrap();

            let one = TestConstraintSystem::<Fp>::one();
            let res_expected = if c {
                a.clone()
            } else {
                b.clone()
            };
            for i in 0..res.limbs.len() {
                cs.enforce(|| format!("c limb {i} equality for condition = {c}"),
                    |lc| lc + &res.limbs[i],
                    |lc| lc + one,
                    |lc| lc + &res_expected.limbs[i],
                );
            }

            assert!(cs.is_satisfied());
        }
        // Note that the number of constraints for one invocation of conditionally_select will be
        // half the number printed by the below statement minus 4 (the constraints in the test)
        println!("Num constraints = {:?}", cs.num_constraints());
        println!("Num inputs = {:?}", cs.num_inputs());

    }

    #[test]
    fn alloc_limbed_conditionally_select2() {
        let mut cs = TestConstraintSystem::<Fp>::new();
        let mut rng = rand::thread_rng();
        let a0_uint = rng.gen_biguint(256u64);
        let a1_uint = rng.gen_biguint(256u64);
        let a2_uint = rng.gen_biguint(256u64);
        let a3_uint = rng.gen_biguint(256u64);
        let a0_l = LimbedInt::<Fp>::from(&a0_uint);
        let a1_l = LimbedInt::<Fp>::from(&a1_uint);
        let a2_l = LimbedInt::<Fp>::from(&a2_uint);
        let a3_l = LimbedInt::<Fp>::from(&a3_uint);

        let a0 = AllocatedLimbedInt::<Fp>::alloc_from_limbed_int(
            &mut cs.namespace(|| "alloc a0"),
            a0_l,
            4
        );
        assert!(a0.is_ok());
        let a1 = AllocatedLimbedInt::<Fp>::alloc_from_limbed_int(
            &mut cs.namespace(|| "alloc a1"),
            a1_l,
            4
        );
        assert!(a1.is_ok());
        let a2 = AllocatedLimbedInt::<Fp>::alloc_from_limbed_int(
            &mut cs.namespace(|| "alloc a2"),
            a2_l,
            4
        );
        assert!(a2.is_ok());
        let a3 = AllocatedLimbedInt::<Fp>::alloc_from_limbed_int(
            &mut cs.namespace(|| "alloc a3"),
            a3_l,
            4
        );
        assert!(a3.is_ok());
        
        let a0 = a0.unwrap();
        let a1 = a1.unwrap();
        let a2 = a2.unwrap();
        let a3 = a3.unwrap();
        
        let conditions = vec![(false, false), (false, true), (true, false), (true, true)];
        for (c0, c1) in conditions {

            let condition0 = Boolean::constant(c0);
            let condition1 = Boolean::constant(c1);
            
            let res = AllocatedLimbedInt::<Fp>::conditionally_select2(
                &mut cs.namespace(|| format!("conditionally select2 result for conditions = {c0}, {c1}")),
                &a0,
                &a1,
                &a2,
                &a3,
                &condition0,
                &condition1,
            );
            assert!(res.is_ok());
            let res = res.unwrap();

            let one = TestConstraintSystem::<Fp>::one();
            let res_expected = match (c0, c1) {
                (false, false) => a0.clone(),
                (true, false) => a1.clone(),
                (false, true) => a2.clone(),
                (true, true) => a3.clone(),
            };
            for i in 0..res.limbs.len() {
                cs.enforce(|| format!("res limb {i} equality for conditions = {c0}, {c1}"),
                    |lc| lc + &res.limbs[i],
                    |lc| lc + one,
                    |lc| lc + &res_expected.limbs[i],
                );
            }

            assert!(cs.is_satisfied());
        }
        
        // Note that the number of constraints for one invocation of conditionally_select2 will be
        // one-fourth of the number printed by the below statement minus 4 (the constraints in the test)
        println!("Num constraints = {:?}", cs.num_constraints());
        println!("Num inputs = {:?}", cs.num_inputs());

    }

    #[test]
    fn alloc_affine_point_addition_verification() {
        let b = Ed25519Curve::basepoint();
        let mut rng = rand::thread_rng();
        let scalar = U256::random(&mut rng);
        let p = Ed25519Curve::scalar_multiplication(&b, &scalar);
        let scalar = U256::random(&mut rng);
        let q = Ed25519Curve::scalar_multiplication(&b, &scalar);
        let sum_expected_value = p.add(q);

        let mut cs = TestConstraintSystem::<Fp>::new();

        let p_alloc = AllocatedAffinePoint::alloc_affine_point(
            &mut cs.namespace(|| "alloc point p"),
            p,
        );
        assert!(p_alloc.is_ok());
        let p_al = p_alloc.unwrap();

        let q_alloc = AllocatedAffinePoint::alloc_affine_point(
            &mut cs.namespace(|| "alloc point q"),
            q,
        );
        assert!(q_alloc.is_ok());
        let q_al = q_alloc.unwrap();

        let sum_alloc = AllocatedAffinePoint::ed25519_point_addition(
            &mut cs.namespace(|| "adding p and q"),
            &p_al,
            &q_al,
        );
        assert!(sum_alloc.is_ok());
        let sum_al = sum_alloc.unwrap();

        assert_eq!(sum_expected_value, sum_al.value);

        assert!(cs.is_satisfied());
        println!("Num constraints = {:?}", cs.num_constraints());
        println!("Num inputs = {:?}", cs.num_inputs());

    }

    #[test]
    fn alloc_affine_scalar_multiplication_window1() {
        let b = Ed25519Curve::basepoint();
        let mut rng = rand::thread_rng();
       
        let mut scalar = U256::random(&mut rng);
        scalar = scalar >> 3; // scalar now has 253 significant bits
        let p = Ed25519Curve::scalar_multiplication(&b, &scalar);
       
        let mut scalar_vec: Vec<Boolean> = vec![];
        for _i in 0..253 {
            if bool::from(scalar.is_odd()) {
                scalar_vec.push(Boolean::constant(true))
            } else {
                scalar_vec.push(Boolean::constant(false))
            };
            scalar = scalar >> 1;
        }

        let mut cs = TestConstraintSystem::<Fp>::new();

        let b_alloc = AllocatedAffinePoint::alloc_affine_point(
            &mut cs.namespace(|| "allocate base point"),
            b,
        );
        assert!(b_alloc.is_ok());
        let b_al = b_alloc.unwrap();

        let p_alloc = b_al.ed25519_scalar_multiplication_old(
            &mut cs.namespace(|| "scalar multiplication"),
            scalar_vec,
        );
        assert!(p_alloc.is_ok());
        let p_al = p_alloc.unwrap();

        assert_eq!(p, p_al.value);

        assert!(cs.is_satisfied());
        println!("Num constraints = {:?}", cs.num_constraints());
        println!("Num inputs = {:?}", cs.num_inputs());

    }

    #[test]
    fn alloc_affine_scalar_multiplication_window2() {
        let b = Ed25519Curve::basepoint();
        let mut rng = rand::thread_rng();
       
        let mut scalar = U256::random(&mut rng);
        scalar = scalar >> 3; // scalar now has 253 significant bits
        let p = Ed25519Curve::scalar_multiplication(&b, &scalar);
       
        let mut scalar_vec: Vec<Boolean> = vec![];
        for _i in 0..253 {
            if bool::from(scalar.is_odd()) {
                scalar_vec.push(Boolean::constant(true))
            } else {
                scalar_vec.push(Boolean::constant(false))
            };
            scalar = scalar >> 1;
        }

        let mut cs = TestConstraintSystem::<Fp>::new();

        let b_alloc = AllocatedAffinePoint::alloc_affine_point(
            &mut cs.namespace(|| "allocate base point"),
            b,
        );
        assert!(b_alloc.is_ok());
        let b_al = b_alloc.unwrap();

        let p_alloc = b_al.ed25519_scalar_multiplication(
            &mut cs.namespace(|| "scalar multiplication"),
            scalar_vec,
        );
        assert!(p_alloc.is_ok());
        let p_al = p_alloc.unwrap();

        assert_eq!(p, p_al.value);

        assert!(cs.is_satisfied());
        println!("Num constraints = {:?}", cs.num_constraints());
        println!("Num inputs = {:?}", cs.num_inputs());

    }

    #[test]
    fn alloc_affine_scalar_multiplication_window_m() {
        let b = Ed25519Curve::basepoint();
        let mut rng = rand::thread_rng();
       
        let mut scalar = U256::random(&mut rng);
        scalar = scalar >> 3; // scalar now has 253 significant bits
        let p = Ed25519Curve::scalar_multiplication(&b, &scalar);
       
        let mut scalar_vec: Vec<Boolean> = vec![];
        for _i in 0..253 {
            if bool::from(scalar.is_odd()) {
                scalar_vec.push(Boolean::constant(true))
            } else {
                scalar_vec.push(Boolean::constant(false))
            };
            scalar = scalar >> 1;
        }

        for i in 3..7 {
            let mut cs = TestConstraintSystem::<Fp>::new();

            let b_alloc = AllocatedAffinePoint::alloc_affine_point(
                &mut cs.namespace(|| "allocate base point"),
                b,
            );
            assert!(b_alloc.is_ok());
            let b_al = b_alloc.unwrap();
    
            let p_alloc = b_al.ed25519_scalar_multiplication_m_bit(
                &mut cs.namespace(|| "scalar multiplication"),
                scalar_vec.clone(),
                i
            );
            assert!(p_alloc.is_ok());
            let p_al = p_alloc.unwrap();
    
            assert_eq!(p, p_al.value);
    
            assert!(cs.is_satisfied());
            println!("lookup {}, Num constraints = {:?}", i, cs.num_constraints());
        }
        
    }
}