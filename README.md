# Ed25519 Curve Operations in `bellperson`

[bellperson](https://github.com/filecoin-project/bellperson) gadget for ed22519 curve operations (point addition, scalar multiplication). It is inspired by the approach taken by [circom-ecdsa](https://github.com/0xPARC/circom-ecdsa).

> Experimental code, do not use in production

Scalar multiplication with a 4-bit lookup gives the minimum number of constraints (about 1.3 million).

```
$ cargo t -r window_m -- --nocapture

running 1 test
lookup 3, Num constraints = 1347012
lookup 4, Num constraints = 1294624
lookup 5, Num constraints = 1307404
lookup 6, Num constraints = 1406882
test nonnative::circuit::tests::alloc_affine_scalar_multiplication_window_m ... ok
```

## Repo Organization

- The `scratch` directory contains prototyping code and notes
- The `src` directory has the Rust code
    + The `src/field.rs` and `src/curve.rs` modules have the non-circuit and non-limbed versions of the ed25519 base field and curve.
    + The `src/nonnative/vanilla.rs` module has the non-circuit limbed version of the ed25519 base field. Not all the functions here are used in the gadget. Some are the result of prototyping limbed arithmetic and will be removed in the future.
    + The `src/nonnative/circuit.rs` module has the circuits 


## License

Licensed under MIT license as the `field` module is based on an earlier version of the MIT-licensed `crypto/dalek-ff-group` crate of [Serai](https://github.com/serai-dex/serai).