use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::{Array, Array2, Ix2};

use rustfrc::{binom_slit_2, binom_split};

fn binom_2_test() -> Vec<Array2<i32>> {
    let mut a = Array2::zeros((2000, 2000));
    a.fill(200);
    let a_ars = vec![a.clone(), a.clone(), a.clone(), a.clone(), a.clone(), a.clone()];
    let mut v = Vec::new();
    for a in a_ars {
        let c = binom_slit_2(a).unwrap();
        v.push(c)
    }

    v
    //println!("{:?}", c)
}

fn binom_1_test() -> Vec<Array<i32, Ix2>> {
    let mut a = Array2::zeros((2000, 2000));
    a.fill(200);
    let a_ars = vec![a.clone(), a.clone(), a.clone(), a.clone(), a.clone(), a.clone()];
    let mut v = Vec::new();
    for a in a_ars {
        let c = binom_split(a).unwrap();
        v.push(c)
    }


    v

    //println!("{:?}", c)
}


pub fn binom_2(c: &mut Criterion) {
    c.bench_function("binom 2", |b| b.iter(|| binom_2_test()));
}

pub fn binom_old(c: &mut Criterion) {
    c.bench_function("binom old", |b| b.iter(|| binom_1_test()));
}

criterion_group!(benches, binom_2, binom_old);
criterion_main!(benches);