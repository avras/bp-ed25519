use core::ops::{Add, Sub, Neg, Mul};

const NUM_LIMBS: usize = 10;

#[derive(Clone, Debug)]
pub struct FE25519 {
    limbs: [i32; NUM_LIMBS],
}

pub const ZERO: FE25519 = FE25519 {
    limbs: [0i32; NUM_LIMBS]
};

pub const ONE: FE25519 = FE25519 {
    limbs: [1i32, 0i32, 0i32, 0i32, 0i32, 0i32, 0i32, 0i32, 0i32, 0i32]
};

impl Default for FE25519 {
    fn default() -> Self {
        Self { limbs: [0i32; NUM_LIMBS] }
    }
}

// Implementation is a port of supercop-20221122/crypto_sign/ed25519/ref10/fe_add.c
// NOTE: Limbs are not checked for overflow as in the ref10 implementation
impl Add<FE25519> for FE25519 {
    type Output = Self;

    fn add(self, rhs: FE25519) -> Self::Output {
        let mut sum: FE25519 = FE25519::default();
        for i in 0..NUM_LIMBS {
            sum.limbs[i] = self.limbs[i] + rhs.limbs[i];
        }
        sum
    }
}

// Implementation is a port of supercop-20221122/crypto_sign/ed25519/ref10/fe_neg.c
impl Neg for FE25519 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let mut neg: FE25519 = FE25519::default();
        for i in 0..NUM_LIMBS {
            neg.limbs[i] = -self.limbs[i];
        }
        neg
    }
}

// Implementation is a port of supercop-20221122/crypto_sign/ed25519/ref10/fe_sub.c
impl Sub<FE25519> for FE25519 {
    type Output = Self;

    fn sub(self, rhs: FE25519) -> Self::Output {
        let mut difference: FE25519 = FE25519::default();
        for i in 0..NUM_LIMBS {
            difference.limbs[i] = self.limbs[i] - rhs.limbs[i];
        }
        difference
    }
}

// Implementation is a port of supercop-20221122/crypto_sign/ed25519/ref10/fe_mul.c
impl Mul<FE25519> for FE25519 {
    type Output = Self;
    
    fn mul(self, rhs: FE25519) -> Self::Output {
        let mut product: FE25519 = FE25519::default();
        let f0 = self.limbs[0];
        let f1 = self.limbs[1];
        let f2 = self.limbs[2];
        let f3 = self.limbs[3];
        let f4 = self.limbs[4];
        let f5 = self.limbs[5];
        let f6 = self.limbs[6];
        let f7 = self.limbs[7];
        let f8 = self.limbs[8];
        let f9 = self.limbs[9];
        let g0 = rhs.limbs[0];
        let g1 = rhs.limbs[1];
        let g2 = rhs.limbs[2];
        let g3 = rhs.limbs[3];
        let g4 = rhs.limbs[4];
        let g5 = rhs.limbs[5];
        let g6 = rhs.limbs[6];
        let g7 = rhs.limbs[7];
        let g8 = rhs.limbs[8];
        let g9 = rhs.limbs[9];
        let g1_19 = 19 * g1;
        let g2_19 = 19 * g2;
        let g3_19 = 19 * g3;
        let g4_19 = 19 * g4;
        let g5_19 = 19 * g5;
        let g6_19 = 19 * g6;
        let g7_19 = 19 * g7;
        let g8_19 = 19 * g8;
        let g9_19 = 19 * g9;
        let f1_2 = 2 * f1;
        let f3_2 = 2 * f3;
        let f5_2 = 2 * f5;
        let f7_2 = 2 * f7;
        let f9_2 = 2    * f9;
        let f0g0    = (f0   as i64) * (g0    as i64);
        let f0g1    = (f0   as i64) * (g1    as i64);
        let f0g2    = (f0   as i64) * (g2    as i64);
        let f0g3    = (f0   as i64) * (g3    as i64);
        let f0g4    = (f0   as i64) * (g4    as i64);
        let f0g5    = (f0   as i64) * (g5    as i64);
        let f0g6    = (f0   as i64) * (g6    as i64);
        let f0g7    = (f0   as i64) * (g7    as i64);
        let f0g8    = (f0   as i64) * (g8    as i64);
        let f0g9    = (f0   as i64) * (g9    as i64);
        let f1g0    = (f1   as i64) * (g0    as i64);
        let f1g1_2  = (f1_2 as i64) * (g1    as i64);
        let f1g2    = (f1   as i64) * (g2    as i64);
        let f1g3_2  = (f1_2 as i64) * (g3    as i64);
        let f1g4    = (f1   as i64) * (g4    as i64);
        let f1g5_2  = (f1_2 as i64) * (g5    as i64);
        let f1g6    = (f1   as i64) * (g6    as i64);
        let f1g7_2  = (f1_2 as i64) * (g7    as i64);
        let f1g8    = (f1   as i64) * (g8    as i64);
        let f1g9_38 = (f1_2 as i64) * (g9_19 as i64);
        let f2g0    = (f2   as i64) * (g0    as i64);
        let f2g1    = (f2   as i64) * (g1    as i64);
        let f2g2    = (f2   as i64) * (g2    as i64);
        let f2g3    = (f2   as i64) * (g3    as i64);
        let f2g4    = (f2   as i64) * (g4    as i64);
        let f2g5    = (f2   as i64) * (g5    as i64);
        let f2g6    = (f2   as i64) * (g6    as i64);
        let f2g7    = (f2   as i64) * (g7    as i64);
        let f2g8_19 = (f2   as i64) * (g8_19 as i64);
        let f2g9_19 = (f2   as i64) * (g9_19 as i64);
        let f3g0    = (f3   as i64) * (g0    as i64);
        let f3g1_2  = (f3_2 as i64) * (g1    as i64);
        let f3g2    = (f3   as i64) * (g2    as i64);
        let f3g3_2  = (f3_2 as i64) * (g3    as i64);
        let f3g4    = (f3   as i64) * (g4    as i64);
        let f3g5_2  = (f3_2 as i64) * (g5    as i64);
        let f3g6    = (f3   as i64) * (g6    as i64);
        let f3g7_38 = (f3_2 as i64) * (g7_19 as i64);
        let f3g8_19 = (f3   as i64) * (g8_19 as i64);
        let f3g9_38 = (f3_2 as i64) * (g9_19 as i64);
        let f4g0    = (f4   as i64) * (g0    as i64);
        let f4g1    = (f4   as i64) * (g1    as i64);
        let f4g2    = (f4   as i64) * (g2    as i64);
        let f4g3    = (f4   as i64) * (g3    as i64);
        let f4g4    = (f4   as i64) * (g4    as i64);
        let f4g5    = (f4   as i64) * (g5    as i64);
        let f4g6_19 = (f4   as i64) * (g6_19 as i64);
        let f4g7_19 = (f4   as i64) * (g7_19 as i64);
        let f4g8_19 = (f4   as i64) * (g8_19 as i64);
        let f4g9_19 = (f4   as i64) * (g9_19 as i64);
        let f5g0    = (f5   as i64) * (g0    as i64);
        let f5g1_2  = (f5_2 as i64) * (g1    as i64);
        let f5g2    = (f5   as i64) * (g2    as i64);
        let f5g3_2  = (f5_2 as i64) * (g3    as i64);
        let f5g4    = (f5   as i64) * (g4    as i64);
        let f5g5_38 = (f5_2 as i64) * (g5_19 as i64);
        let f5g6_19 = (f5   as i64) * (g6_19 as i64);
        let f5g7_38 = (f5_2 as i64) * (g7_19 as i64);
        let f5g8_19 = (f5   as i64) * (g8_19 as i64);
        let f5g9_38 = (f5_2 as i64) * (g9_19 as i64);
        let f6g0    = (f6   as i64) * (g0    as i64);
        let f6g1    = (f6   as i64) * (g1    as i64);
        let f6g2    = (f6   as i64) * (g2    as i64);
        let f6g3    = (f6   as i64) * (g3    as i64);
        let f6g4_19 = (f6   as i64) * (g4_19 as i64);
        let f6g5_19 = (f6   as i64) * (g5_19 as i64);
        let f6g6_19 = (f6   as i64) * (g6_19 as i64);
        let f6g7_19 = (f6   as i64) * (g7_19 as i64);
        let f6g8_19 = (f6   as i64) * (g8_19 as i64);
        let f6g9_19 = (f6   as i64) * (g9_19 as i64);
        let f7g0    = (f7   as i64) * (g0    as i64);
        let f7g1_2  = (f7_2 as i64) * (g1    as i64);
        let f7g2    = (f7   as i64) * (g2    as i64);
        let f7g3_38 = (f7_2 as i64) * (g3_19 as i64);
        let f7g4_19 = (f7   as i64) * (g4_19 as i64);
        let f7g5_38 = (f7_2 as i64) * (g5_19 as i64);
        let f7g6_19 = (f7   as i64) * (g6_19 as i64);
        let f7g7_38 = (f7_2 as i64) * (g7_19 as i64);
        let f7g8_19 = (f7   as i64) * (g8_19 as i64);
        let f7g9_38 = (f7_2 as i64) * (g9_19 as i64);
        let f8g0    = (f8   as i64) * (g0    as i64);
        let f8g1    = (f8   as i64) * (g1    as i64);
        let f8g2_19 = (f8   as i64) * (g2_19 as i64);
        let f8g3_19 = (f8   as i64) * (g3_19 as i64);
        let f8g4_19 = (f8   as i64) * (g4_19 as i64);
        let f8g5_19 = (f8   as i64) * (g5_19 as i64);
        let f8g6_19 = (f8   as i64) * (g6_19 as i64);
        let f8g7_19 = (f8   as i64) * (g7_19 as i64);
        let f8g8_19 = (f8   as i64) * (g8_19 as i64);
        let f8g9_19 = (f8   as i64) * (g9_19 as i64);
        let f9g0    = (f9   as i64) * (g0    as i64);
        let f9g1_38 = (f9_2 as i64) * (g1_19 as i64);
        let f9g2_19 = (f9   as i64) * (g2_19 as i64);
        let f9g3_38 = (f9_2 as i64) * (g3_19 as i64);
        let f9g4_19 = (f9   as i64) * (g4_19 as i64);
        let f9g5_38 = (f9_2 as i64) * (g5_19 as i64);
        let f9g6_19 = (f9   as i64) * (g6_19 as i64);
        let f9g7_38 = (f9_2 as i64) * (g7_19 as i64);
        let f9g8_19 = (f9   as i64) * (g8_19 as i64);
        let f9g9_38 = (f9_2 as i64) * (g9_19 as i64);
        let mut h0 = f0g0+f1g9_38+f2g8_19+f3g7_38+f4g6_19+f5g5_38+f6g4_19+f7g3_38+f8g2_19+f9g1_38;
        let mut h1 = f0g1+f1g0   +f2g9_19+f3g8_19+f4g7_19+f5g6_19+f6g5_19+f7g4_19+f8g3_19+f9g2_19;
        let mut h2 = f0g2+f1g1_2 +f2g0   +f3g9_38+f4g8_19+f5g7_38+f6g6_19+f7g5_38+f8g4_19+f9g3_38;
        let mut h3 = f0g3+f1g2   +f2g1   +f3g0   +f4g9_19+f5g8_19+f6g7_19+f7g6_19+f8g5_19+f9g4_19;
        let mut h4 = f0g4+f1g3_2 +f2g2   +f3g1_2 +f4g0   +f5g9_38+f6g8_19+f7g7_38+f8g6_19+f9g5_38;
        let mut h5 = f0g5+f1g4   +f2g3   +f3g2   +f4g1   +f5g0   +f6g9_19+f7g8_19+f8g7_19+f9g6_19;
        let mut h6 = f0g6+f1g5_2 +f2g4   +f3g3_2 +f4g2   +f5g1_2 +f6g0   +f7g9_38+f8g8_19+f9g7_38;
        let mut h7 = f0g7+f1g6   +f2g5   +f3g4   +f4g3   +f5g2   +f6g1   +f7g0   +f8g9_19+f9g8_19;
        let mut h8 = f0g8+f1g7_2 +f2g6   +f3g5_2 +f4g4   +f5g3_2 +f6g2   +f7g1_2 +f8g0   +f9g9_38;
        let mut h9 = f0g9+f1g8   +f2g7   +f3g6   +f4g5   +f5g4   +f6g3   +f7g2   +f8g1   +f9g0   ;
        let mut carry0: i64;
        let carry1: i64;
        let carry2: i64;
        let carry3: i64;
        let mut carry4: i64;
        let carry5: i64;
        let carry6: i64;
        let carry7: i64;
        let carry8: i64;
        let carry9: i64;
      
      
        carry0 = (h0 + (1i64 << 25)) >> 26; h1 += carry0; h0 -= carry0 << 26;
        carry4 = (h4 + (1i64 << 25)) >> 26; h5 += carry4; h4 -= carry4 << 26;
      
        carry1 = (h1 + (1i64 << 24)) >> 25; h2 += carry1; h1 -= carry1 << 25;
        carry5 = (h5 + (1i64 << 24)) >> 25; h6 += carry5; h5 -= carry5 << 25;
      
        carry2 = (h2 + (1i64 << 25)) >> 26; h3 += carry2; h2 -= carry2 << 26;
        carry6 = (h6 + (1i64 << 25)) >> 26; h7 += carry6; h6 -= carry6 << 26;
      
        carry3 = (h3 + (1i64 << 24)) >> 25; h4 += carry3; h3 -= carry3 << 25;
        carry7 = (h7 + (1i64 << 24)) >> 25; h8 += carry7; h7 -= carry7 << 25;
      
        carry4 = (h4 + (1i64 << 25)) >> 26; h5 += carry4; h4 -= carry4 << 26;
        carry8 = (h8 + (1i64 << 25)) >> 26; h9 += carry8; h8 -= carry8 << 26;
      
        carry9 = (h9 + (1i64 << 24)) >> 25; h0 += carry9 * 19; h9 -= carry9 << 25;
      
        carry0 = (h0 + (1i64 << 25)) >> 26; h1 += carry0; h0 -= carry0 << 26;
      
        product.limbs[0] = h0 as i32;
        product.limbs[1] = h1 as i32;
        product.limbs[2] = h2 as i32;
        product.limbs[3] = h3 as i32;
        product.limbs[4] = h4 as i32;
        product.limbs[5] = h5 as i32;
        product.limbs[6] = h6 as i32;
        product.limbs[7] = h7 as i32;
        product.limbs[8] = h8 as i32;
        product.limbs[9] = h9 as i32;
        product
    }
}

impl FE25519 {
    // Implementation is a port of supercop-20221122/crypto_sign/ed25519/ref10/fe_sq.c
    pub fn square(&self) -> Self {
        let mut square = FE25519::default();

        let f0 = self.limbs[0];
        let f1 = self.limbs[1];
        let f2 = self.limbs[2];
        let f3 = self.limbs[3];
        let f4 = self.limbs[4];
        let f5 = self.limbs[5];
        let f6 = self.limbs[6];
        let f7 = self.limbs[7];
        let f8 = self.limbs[8];
        let f9 = self.limbs[9];
        let f0_2 = 2 * f0;
        let f1_2 = 2 * f1;
        let f2_2 = 2 * f2;
        let f3_2 = 2 * f3;
        let f4_2 = 2 * f4;
        let f5_2 = 2 * f5;
        let f6_2 = 2 * f6;
        let f7_2 = 2 * f7;
        let f5_38 = 38 * f5;
        let f6_19 = 19 * f6;
        let f7_38 = 38 * f7;
        let f8_19 = 19 * f8;
        let f9_38 = 38 * f9;
        let f0f0    = (f0   as i64) * (f0 as i64);
        let f0f1_2  = (f0_2 as i64) * (f1 as i64);
        let f0f2_2  = (f0_2 as i64) * (f2 as i64);
        let f0f3_2  = (f0_2 as i64) * (f3 as i64);
        let f0f4_2  = (f0_2 as i64) * (f4 as i64);
        let f0f5_2  = (f0_2 as i64) * (f5 as i64);
        let f0f6_2  = (f0_2 as i64) * (f6 as i64);
        let f0f7_2  = (f0_2 as i64) * (f7 as i64);
        let f0f8_2  = (f0_2 as i64) * (f8 as i64);
        let f0f9_2  = (f0_2 as i64) * (f9 as i64);
        let f1f1_2  = (f1_2 as i64) * (f1 as i64);
        let f1f2_2  = (f1_2 as i64) * (f2 as i64);
        let f1f3_4  = (f1_2 as i64) * (f3_2 as i64);
        let f1f4_2  = (f1_2 as i64) * (f4 as i64);
        let f1f5_4  = (f1_2 as i64) * (f5_2 as i64);
        let f1f6_2  = (f1_2 as i64) * (f6 as i64);
        let f1f7_4  = (f1_2 as i64) * (f7_2 as i64);
        let f1f8_2  = (f1_2 as i64) * (f8 as i64);
        let f1f9_76 = (f1_2 as i64) * (f9_38 as i64);
        let f2f2    = (f2   as i64) * (f2 as i64);
        let f2f3_2  = (f2_2 as i64) * (f3 as i64);
        let f2f4_2  = (f2_2 as i64) * (f4 as i64);
        let f2f5_2  = (f2_2 as i64) * (f5 as i64);
        let f2f6_2  = (f2_2 as i64) * (f6 as i64);
        let f2f7_2  = (f2_2 as i64) * (f7 as i64);
        let f2f8_38 = (f2_2 as i64) * (f8_19 as i64);
        let f2f9_38 = (f2   as i64) * (f9_38 as i64);
        let f3f3_2  = (f3_2 as i64) * (f3 as i64);
        let f3f4_2  = (f3_2 as i64) * (f4 as i64);
        let f3f5_4  = (f3_2 as i64) * (f5_2 as i64);
        let f3f6_2  = (f3_2 as i64) * (f6 as i64);
        let f3f7_76 = (f3_2 as i64) * (f7_38 as i64);
        let f3f8_38 = (f3_2 as i64) * (f8_19 as i64);
        let f3f9_76 = (f3_2 as i64) * (f9_38 as i64);
        let f4f4    = (f4   as i64) * (f4 as i64);
        let f4f5_2  = (f4_2 as i64) * (f5 as i64);
        let f4f6_38 = (f4_2 as i64) * (f6_19 as i64);
        let f4f7_38 = (f4   as i64) * (f7_38 as i64);
        let f4f8_38 = (f4_2 as i64) * (f8_19 as i64);
        let f4f9_38 = (f4   as i64) * (f9_38 as i64);
        let f5f5_38 = (f5   as i64) * (f5_38 as i64);
        let f5f6_38 = (f5_2 as i64) * (f6_19 as i64);
        let f5f7_76 = (f5_2 as i64) * (f7_38 as i64);
        let f5f8_38 = (f5_2 as i64) * (f8_19 as i64);
        let f5f9_76 = (f5_2 as i64) * (f9_38 as i64);
        let f6f6_19 = (f6   as i64) * (f6_19 as i64);
        let f6f7_38 = (f6   as i64) * (f7_38 as i64);
        let f6f8_38 = (f6_2 as i64) * (f8_19 as i64);
        let f6f9_38 = (f6   as i64) * (f9_38 as i64);
        let f7f7_38 = (f7   as i64) * (f7_38 as i64);
        let f7f8_38 = (f7_2 as i64) * (f8_19 as i64);
        let f7f9_76 = (f7_2 as i64) * (f9_38 as i64);
        let f8f8_19 = (f8   as i64) * (f8_19 as i64);
        let f8f9_38 = (f8   as i64) * (f9_38 as i64);
        let f9f9_38 = (f9   as i64) * (f9_38 as i64);
        let mut h0 = f0f0  +f1f9_76+f2f8_38+f3f7_76+f4f6_38+f5f5_38;
        let mut h1 = f0f1_2+f2f9_38+f3f8_38+f4f7_38+f5f6_38;
        let mut h2 = f0f2_2+f1f1_2 +f3f9_76+f4f8_38+f5f7_76+f6f6_19;
        let mut h3 = f0f3_2+f1f2_2 +f4f9_38+f5f8_38+f6f7_38;
        let mut h4 = f0f4_2+f1f3_4 +f2f2   +f5f9_76+f6f8_38+f7f7_38;
        let mut h5 = f0f5_2+f1f4_2 +f2f3_2 +f6f9_38+f7f8_38;
        let mut h6 = f0f6_2+f1f5_4 +f2f4_2 +f3f3_2 +f7f9_76+f8f8_19;
        let mut h7 = f0f7_2+f1f6_2 +f2f5_2 +f3f4_2 +f8f9_38;
        let mut h8 = f0f8_2+f1f7_4 +f2f6_2 +f3f5_4 +f4f4   +f9f9_38;
        let mut h9 = f0f9_2+f1f8_2 +f2f7_2 +f3f6_2 +f4f5_2;
        let mut carry0: i64;
        let carry1: i64;
        let carry2: i64;
        let carry3: i64;
        let mut carry4: i64;
        let carry5: i64;
        let carry6: i64;
        let carry7: i64;
        let carry8: i64;
        let carry9: i64;
      
        carry0 = (h0 + (1i64 << 25)) >> 26; h1 += carry0; h0 -= carry0 << 26;
        carry4 = (h4 + (1i64 << 25)) >> 26; h5 += carry4; h4 -= carry4 << 26;
      
        carry1 = (h1 + (1i64 << 24)) >> 25; h2 += carry1; h1 -= carry1 << 25;
        carry5 = (h5 + (1i64 << 24)) >> 25; h6 += carry5; h5 -= carry5 << 25;
      
        carry2 = (h2 + (1i64 << 25)) >> 26; h3 += carry2; h2 -= carry2 << 26;
        carry6 = (h6 + (1i64 << 25)) >> 26; h7 += carry6; h6 -= carry6 << 26;
      
        carry3 = (h3 + (1i64 << 24)) >> 25; h4 += carry3; h3 -= carry3 << 25;
        carry7 = (h7 + (1i64 << 24)) >> 25; h8 += carry7; h7 -= carry7 << 25;
      
        carry4 = (h4 + (1i64 << 25)) >> 26; h5 += carry4; h4 -= carry4 << 26;
        carry8 = (h8 + (1i64 << 25)) >> 26; h9 += carry8; h8 -= carry8 << 26;
      
        carry9 = (h9 + (1i64 << 24)) >> 25; h0 += carry9 * 19; h9 -= carry9 << 25;
      
        carry0 = (h0 + (1i64 << 25)) >> 26; h1 += carry0; h0 -= carry0 << 26;
      
        square.limbs[0] = h0 as i32;
        square.limbs[1] = h1 as i32;
        square.limbs[2] = h2 as i32;
        square.limbs[3] = h3 as i32;
        square.limbs[4] = h4 as i32;
        square.limbs[5] = h5 as i32;
        square.limbs[6] = h6 as i32;
        square.limbs[7] = h7 as i32;
        square.limbs[8] = h8 as i32;
        square.limbs[9] = h9 as i32;

        square
    }

    // Implementation is a port of supercop-20221122/crypto_sign/ed25519/ref10/fe_frombytes.c
    pub fn from_bytes(input: &[u8]) -> Self {
        assert_eq!(input.len(), 32);
        let mut fe = FE25519::default();
        let mut h0 = load_4(&input[0..4]);
        let mut h1 = load_3(&input[4..7]) << 6;
        let mut h2 = load_3(&input[7..10]) << 5;
        let mut h3 = load_3(&input[10..13]) << 3;
        let mut h4 = load_3(&input[13..16]) << 2;
        let mut h5 = load_4(&input[16..20]);
        let mut h6 = load_3(&input[20..23]) << 7;
        let mut h7 = load_3(&input[23..26]) << 5;
        let mut h8 = load_3(&input[26..29]) << 4;
        let mut h9 = (load_3(&input[29..32]) & 8388607) << 2;
        let carry0;
        let carry1;
        let carry2;
        let carry3;
        let carry4;
        let carry5;
        let carry6;
        let carry7;
        let carry8;
        let carry9;
      
        carry9 = (h9 + (1i64 << 24)) >> 25; h0 += carry9 * 19; h9 -= carry9 << 25;
        carry1 = (h1 + (1i64 << 24)) >> 25; h2 += carry1; h1 -= carry1 << 25;
        carry3 = (h3 + (1i64 << 24)) >> 25; h4 += carry3; h3 -= carry3 << 25;
        carry5 = (h5 + (1i64 << 24)) >> 25; h6 += carry5; h5 -= carry5 << 25;
        carry7 = (h7 + (1i64 << 24)) >> 25; h8 += carry7; h7 -= carry7 << 25;
      
        carry0 = (h0 + (1i64 << 25)) >> 26; h1 += carry0; h0 -= carry0 << 26;
        carry2 = (h2 + (1i64 << 25)) >> 26; h3 += carry2; h2 -= carry2 << 26;
        carry4 = (h4 + (1i64 << 25)) >> 26; h5 += carry4; h4 -= carry4 << 26;
        carry6 = (h6 + (1i64 << 25)) >> 26; h7 += carry6; h6 -= carry6 << 26;
        carry8 = (h8 + (1i64 << 25)) >> 26; h9 += carry8; h8 -= carry8 << 26;
      
        fe.limbs[0] = h0 as i32;
        fe.limbs[1] = h1 as i32;
        fe.limbs[2] = h2 as i32;
        fe.limbs[3] = h3 as i32;
        fe.limbs[4] = h4 as i32;
        fe.limbs[5] = h5 as i32;
        fe.limbs[6] = h6 as i32;
        fe.limbs[7] = h7 as i32;
        fe.limbs[8] = h8 as i32;
        fe.limbs[9] = h9 as i32;
        fe
    }

    // Implementation is a port of supercop-20221122/crypto_sign/ed25519/ref10/fe_frombytes.c
    pub fn to_bytes(&self) -> [u8; 32] {
        let mut output = [0u8; 32];
        let mut h0 = self.limbs[0];
        let mut h1 = self.limbs[1];
        let mut h2 = self.limbs[2];
        let mut h3 = self.limbs[3];
        let mut h4 = self.limbs[4];
        let mut h5 = self.limbs[5];
        let mut h6 = self.limbs[6];
        let mut h7 = self.limbs[7];
        let mut h8 = self.limbs[8];
        let mut h9 = self.limbs[9];
        let mut q;
        let carry0;
        let carry1;
        let carry2;
        let carry3;
        let carry4;
        let carry5;
        let carry6;
        let carry7;
        let carry8;
        let carry9;
      
        q = (19 * h9 + (1i32 << 24)) >> 25;
        q = (h0 + q) >> 26;
        q = (h1 + q) >> 25;
        q = (h2 + q) >> 26;
        q = (h3 + q) >> 25;
        q = (h4 + q) >> 26;
        q = (h5 + q) >> 25;
        q = (h6 + q) >> 26;
        q = (h7 + q) >> 25;
        q = (h8 + q) >> 26;
        q = (h9 + q) >> 25;
      
        h0 += 19 * q;
      
        carry0 = h0 >> 26; h1 += carry0; h0 -= carry0 << 26;
        carry1 = h1 >> 25; h2 += carry1; h1 -= carry1 << 25;
        carry2 = h2 >> 26; h3 += carry2; h2 -= carry2 << 26;
        carry3 = h3 >> 25; h4 += carry3; h3 -= carry3 << 25;
        carry4 = h4 >> 26; h5 += carry4; h4 -= carry4 << 26;
        carry5 = h5 >> 25; h6 += carry5; h5 -= carry5 << 25;
        carry6 = h6 >> 26; h7 += carry6; h6 -= carry6 << 26;
        carry7 = h7 >> 25; h8 += carry7; h7 -= carry7 << 25;
        carry8 = h8 >> 26; h9 += carry8; h8 -= carry8 << 26;
        carry9 = h9 >> 25;               h9 -= carry9 << 25;
      
        output[0] = (h0 >> 0) as u8;
        output[1] = (h0 >> 8) as u8;
        output[2] = (h0 >> 16) as u8;
        output[3] = ((h0 >> 24) | (h1 << 2)) as u8;
        output[4] = (h1 >> 6) as u8;
        output[5] = (h1 >> 14) as u8;
        output[6] = ((h1 >> 22) | (h2 << 3)) as u8;
        output[7] = (h2 >> 5) as u8;
        output[8] = (h2 >> 13) as u8;
        output[9] = ((h2 >> 21) | (h3 << 5)) as u8;
        output[10] = (h3 >> 3) as u8;
        output[11] = (h3 >> 11) as u8;
        output[12] = ((h3 >> 19) | (h4 << 6)) as u8;
        output[13] = (h4 >> 2) as u8;
        output[14] = (h4 >> 10) as u8;
        output[15] = (h4 >> 18) as u8;
        output[16] = (h5 >> 0) as u8;
        output[17] = (h5 >> 8) as u8;
        output[18] = (h5 >> 16) as u8;
        output[19] = ((h5 >> 24) | (h6 << 1)) as u8;
        output[20] = (h6 >> 7) as u8;
        output[21] = (h6 >> 15) as u8;
        output[22] = ((h6 >> 23) | (h7 << 3)) as u8;
        output[23] = (h7 >> 5) as u8;
        output[24] = (h7 >> 13) as u8;
        output[25] = ((h7 >> 21) | (h8 << 4)) as u8;
        output[26] = (h8 >> 4) as u8;
        output[27] = (h8 >> 12) as u8;
        output[28] = ((h8 >> 20) | (h9 << 6)) as u8;
        output[29] = (h9 >> 2) as u8;
        output[30] = (h9 >> 10) as u8;
        output[31] = (h9 >> 18) as u8;
        output
    }

}

fn load_3(input: &[u8]) -> i64 {
    assert!(input.len() >= 3);
    let mut result;
    result = input[0] as i64;
    result |= (input[1] as i64) << 8;
    result |= (input[2] as i64) << 16;
    result
}

fn load_4(input: &[u8]) -> i64 {
    assert!(input.len() >= 4);
    let mut result;
    result = input[0] as i64;
    result |= (input[1] as i64) << 8;
    result |= (input[2] as i64) << 16;
    result |= (input[3] as i64) << 24;
    result
}