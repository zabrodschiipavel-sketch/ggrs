//! Векторные примитивы: скалярный эталон + AVX2/FMA. Выбор в рантайме.

use std::sync::OnceLock;

pub fn have_avx2() -> bool {
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    })
}

pub mod scalar {
    pub fn vec_add(a: &[f32], b: &[f32], d: &mut [f32]) {
        for i in 0..d.len() {
            d[i] = a[i] + b[i];
        }
    }
    pub fn vec_mul(a: &[f32], b: &[f32], d: &mut [f32]) {
        for i in 0..d.len() {
            d[i] = a[i] * b[i];
        }
    }
    pub fn vec_scale(d: &mut [f32], s: f32) {
        for x in d.iter_mut() {
            *x *= s;
        }
    }
    pub fn vec_dot(a: &[f32], b: &[f32]) -> f32 {
        let mut sum = 0.0f32;
        for i in 0..a.len() {
            sum += a[i] * b[i];
        }
        sum
    }
}

#[cfg(target_arch = "x86_64")]
pub mod avx2 {
    use std::arch::x86_64::*;

    #[inline]
    unsafe fn hsum256(v: __m256) -> f32 {
        let lo = _mm256_castps256_ps128(v);
        let hi = _mm256_extractf128_ps(v, 1);
        let s = _mm_add_ps(lo, hi);
        let s = _mm_hadd_ps(s, s);
        let s = _mm_hadd_ps(s, s);
        _mm_cvtss_f32(s)
    }

    /// # Safety
    /// Вызывать только при have_avx2().
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn vec_dot(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut i = 0usize;
        while i + 16 <= n {
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i + 8)), _mm256_loadu_ps(pb.add(i + 8)), acc1);
            i += 16;
        }
        while i + 8 <= n {
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), acc0);
            i += 8;
        }
        let mut sum = hsum256(_mm256_add_ps(acc0, acc1));
        while i < n {
            sum += *pa.add(i) * *pb.add(i);
            i += 1;
        }
        sum
    }

    /// # Safety
    /// Вызывать только при have_avx2().
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn vec_add(a: &[f32], b: &[f32], d: &mut [f32]) {
        let n = d.len();
        let (pa, pb, pd) = (a.as_ptr(), b.as_ptr(), d.as_mut_ptr());
        let mut i = 0usize;
        while i + 8 <= n {
            _mm256_storeu_ps(pd.add(i), _mm256_add_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i))));
            i += 8;
        }
        while i < n {
            *pd.add(i) = *pa.add(i) + *pb.add(i);
            i += 1;
        }
    }

    /// # Safety
    /// Вызывать только при have_avx2().
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn vec_mul(a: &[f32], b: &[f32], d: &mut [f32]) {
        let n = d.len();
        let (pa, pb, pd) = (a.as_ptr(), b.as_ptr(), d.as_mut_ptr());
        let mut i = 0usize;
        while i + 8 <= n {
            _mm256_storeu_ps(pd.add(i), _mm256_mul_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i))));
            i += 8;
        }
        while i < n {
            *pd.add(i) = *pa.add(i) * *pb.add(i);
            i += 1;
        }
    }

    /// # Safety
    /// Вызывать только при have_avx2().
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn vec_scale(d: &mut [f32], s: f32) {
        let n = d.len();
        let pd = d.as_mut_ptr();
        let vs = _mm256_set1_ps(s);
        let mut i = 0usize;
        while i + 8 <= n {
            _mm256_storeu_ps(pd.add(i), _mm256_mul_ps(_mm256_loadu_ps(pd.add(i)), vs));
            i += 8;
        }
        while i < n {
            *pd.add(i) *= s;
            i += 1;
        }
    }
}

// ---- диспетчеры ----
pub fn vec_add(a: &[f32], b: &[f32], d: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if have_avx2() {
        unsafe { avx2::vec_add(a, b, d) };
        return;
    }
    scalar::vec_add(a, b, d);
}
pub fn vec_mul(a: &[f32], b: &[f32], d: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if have_avx2() {
        unsafe { avx2::vec_mul(a, b, d) };
        return;
    }
    scalar::vec_mul(a, b, d);
}
pub fn vec_scale(d: &mut [f32], s: f32) {
    #[cfg(target_arch = "x86_64")]
    if have_avx2() {
        unsafe { avx2::vec_scale(d, s) };
        return;
    }
    scalar::vec_scale(d, s);
}
pub fn vec_dot(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if have_avx2() {
        return unsafe { avx2::vec_dot(a, b) };
    }
    scalar::vec_dot(a, b)
}
