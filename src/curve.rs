use std::ops::{Add, Neg, Sub};

use crypto_bigint::{U256, Integer};
use ff::{Field, PrimeField};
use crate::field::Fe25519;

const D: Fe25519 =
    Fe25519(U256::from_be_hex("52036cee2b6ffe738cc740797779e89800700a4d4141d8ab75eb4dca135978a3"));

#[derive(Clone, Debug, PartialEq, Eq, Copy)]
pub struct AffinePoint {
    x: Fe25519,
    y: Fe25519,
}

impl Add<AffinePoint> for AffinePoint {
    type Output = AffinePoint;

    fn add(self, rhs: AffinePoint) -> Self::Output {
        Ed25519Curve::add_points(&self, &rhs)
    }
}

impl Sub<AffinePoint> for AffinePoint {
    type Output = AffinePoint;

    fn sub(self, rhs: AffinePoint) -> Self::Output {
        let rhs_neg = -rhs;
        Ed25519Curve::add_points(&self, &rhs_neg)
    }
}

impl Neg for AffinePoint {
    type Output = Self;

    fn neg(self) -> Self::Output {
        AffinePoint {
            x: self.x.neg(),
            y: self.y,
        }
    }
}

impl AffinePoint {
    pub fn is_on_curve(&self) -> bool {
        Ed25519Curve::is_on_curve(self)
    }

    pub fn is_zero(&self) -> bool {
        self.x == Fe25519::ZERO && self.y == Fe25519::ONE
    }

    pub fn double(&self) -> Self {
        Ed25519Curve::add_points(&self, &self)
    }
}

impl Default for AffinePoint {
    fn default() -> Self {
        Self { x: Fe25519::ZERO, y: Fe25519::ONE }
    }
}

pub struct Ed25519Curve;

impl Ed25519Curve {
    pub fn recover_even_x_from_y(y: Fe25519) -> Fe25519 {
        let y_sq = y.square();
        let x_sq = (y_sq - Fe25519::ONE) * (D*y_sq + Fe25519::ONE).invert().unwrap();

        let (is_sqroot, x) = x_sq.sqrt_alt();
        assert!(bool::from(is_sqroot)); // y must correspond to a curve point
        if x.is_even().into() {
            x
        }
        else {
            -x
        }
    }

    pub fn basepoint() -> AffinePoint {
        let y = Fe25519::from(4u64) * Fe25519::from(5u64).invert().unwrap();
        let x = Self::recover_even_x_from_y(y);
        AffinePoint { x, y }
    }

    pub fn is_on_curve(point: &AffinePoint) -> bool {
        let x = point.x;
        let y = point.y;
        let x_sq = x.square();
        let y_sq = y.square();
        let tmp = -x_sq + y_sq - Fe25519::ONE - D*x_sq*y_sq;
        tmp == Fe25519::ZERO
    }

    pub fn add_points(p: &AffinePoint, q: &AffinePoint) -> AffinePoint {
        let x1 = p.x;
        let y1 = p.y;
        let x2 = q.x;
        let y2 = q.y;
        let dx1x2y1y2 = D*x1*x2*y1*y2;
        AffinePoint {
            x: (x1*y2 + x2*y1)*(Fe25519::ONE + dx1x2y1y2).invert().unwrap(),
            y: (x1*x2 + y1*y2)*(Fe25519::ONE - dx1x2y1y2).invert().unwrap(),
        }
    }

    pub fn scalar_multiplication(point: &AffinePoint, scalar: &U256) -> AffinePoint {
        let mut output = AffinePoint::default();
        let mut step_point = *point;
        let mut scaled_scalar = *scalar;
        for _i in 0..256 {
            if bool::from(scaled_scalar.is_odd()) {
                output = output + step_point;
            }
            step_point = step_point.double();
            scaled_scalar = scaled_scalar >> 1;
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const CURVE_ORDER: U256 =
        U256::from_be_hex("1000000000000000000000000000000014def9dea2f79cd65812631a5cf5d3ed");

    fn random_point() -> AffinePoint {
        let mut rng = rand::thread_rng();
        let mut point = AffinePoint::default();
        loop {
            let y = Fe25519::random(&mut rng);
            let y_sq = y.square();
            let x_sq = (y_sq - Fe25519::ONE) * (D*y_sq + Fe25519::ONE).invert().unwrap();

            let (is_sqroot, x) = x_sq.sqrt_alt();
            if bool::from(is_sqroot) == true {
                point.x = x;
                point.y = y;
                break;
            }
        }
        point
    }

    #[test]
    fn point_negation() {
        let p = random_point();
        assert!(Ed25519Curve::is_on_curve(&p));
        let neg_p = -p;
        let sum = p + neg_p;
        assert!(sum.is_zero());
    }

    #[test]
    fn point_addition_difference() {
        let p = random_point();
        assert!(Ed25519Curve::is_on_curve(&p));
        let p2 = p.double();
        let p3 = p+p+p;
        assert_eq!(p2+p, p3);
        assert_eq!(p3-p, p2);
    }

    #[test]
    fn point_scalar_multiplication() {
        let b = Ed25519Curve::basepoint();
        assert!(Ed25519Curve::is_on_curve(&b));
        let scalar = U256::from(6u64);
        let p = Ed25519Curve::scalar_multiplication(&b, &scalar);
        assert_eq!(p, b+b+b+b+b+b);
    }

    #[test]
    fn point_order() {
        let b = Ed25519Curve::basepoint();
        assert!(Ed25519Curve::is_on_curve(&b));
        let scalar: U256 = CURVE_ORDER;
        let p = Ed25519Curve::scalar_multiplication(&b, &scalar);
        assert!(p.is_zero());

        let p = random_point();
        let scalar = CURVE_ORDER << 3; // Accounting for the co-factor
        let p = Ed25519Curve::scalar_multiplication(&p, &scalar);
        assert!(p.is_zero());
    }

}