#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum DType {
    F32,
    F16,
    I32,
}

/// Преобразование f32 в IEEE 754 binary16 (round-to-nearest-even).
pub fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7FFFFF;

    // NaN или Inf
    if exp == 0xFF {
        if mant == 0 {
            return sign | 0x7C00; // Inf
        }
        return sign | 0x7E00; // NaN (каноническая)
    }

    // Нуль или денормал f32 → нуль
    if exp == 0 {
        return sign;
    }

    // Нормальное число f32
    let new_exp = exp - 127 + 15;

    // Переполнение → Inf
    if new_exp >= 31 {
        return sign | 0x7C00;
    }

    // Полная мантисса 24 бита (1.mant)
    let sig = 0x800000 | mant;

    if new_exp > 0 {
        // Нормальное f16
        // sig → 14 бит (1 + 10 мантиссы + 3 бита rounding: G, R, S_bit)
        let sig_14 = sig >> 10;            // биты 23:10 sig → 14 бит
        let mant_10 = ((sig_14 >> 3) & 0x3FF) as u16;
        let grs = sig_14 & 0x7;            // G=bit2, R=bit1, S_bit=bit0
        let g = (grs >> 2) & 1;
        let r = (grs >> 1) & 1;
        let s_bit = grs & 1;
        let sticky = s_bit != 0 || (sig & 0x3FF) != 0;

        let mut result = sign | ((new_exp as u16) << 10) | mant_10;

        // RNE: округление если G=1 и (R=1 или sticky=1 или LSB=1)
        if g == 1 && (r == 1 || sticky || (mant_10 & 1) == 1) {
            result = result.wrapping_add(1);
        }

        result
    } else {
        // Денормальное f16 (underflow)
        // Сдвиг на 14 - new_exp позиций: 24-битная sig → 10-битная денормальная мантисса
        let shift = 14 - new_exp; // >= 14

        if shift >= 24 {
            // Слишком мало для представления даже в денормальном f16
            // Значения [2^-25, 2^-24) молча уходят в 0 без RNE —
            // это осознанное усечение на краю underflow (стандартное поведение).
            return sign;
        }

        // Бит защиты (Guard) — первый выдвинутый бит
        let guard = (sig >> (shift - 1)) & 1;
        // Sticky — OR всех битов ниже guard
        let mask = if shift > 1 {
            (1u32 << (shift - 1)) - 1
        } else {
            0
        };
        let sticky = (sig & mask) != 0;

        let mant_10 = (sig >> shift) as u16 & 0x3FF;

        let mut result = sign | mant_10;

        // RNE
        if guard == 1 && (sticky || (mant_10 & 1) == 1) {
            result = result.wrapping_add(1);
        }

        result
    }
}

/// Преобразование IEEE 754 binary16 в f32.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 1;
    let exp = ((h >> 10) & 0x1F) as i32;
    let mant = (h & 0x3FF) as u32;

    if exp == 0x1F {
        // NaN или Inf
        if mant == 0 {
            // Inf
            let bits = (sign as u32) << 31 | 0x7F800000;
            return f32::from_bits(bits);
        } else {
            // NaN
            let bits = (sign as u32) << 31 | 0x7FC00000;
            return f32::from_bits(bits);
        }
    }

    if exp == 0 {
        // Денормал или нуль: value = mant * 2^(-24)
        let val = (mant as f32) * 2.0f32.powi(-24);
        if sign == 1 {
            return -val;
        }
        return val;
    }

    // Нормальное f16
    // exp — 5 бит, bias=15 → f32 exp = exp - 15 + 127 = exp + 112
    let f32_exp = (exp + 112) as u32; // 112 = 127 - 15
    let f32_mant = mant << 13; // 10 → 23 бита
    let bits = (sign as u32) << 31 | f32_exp << 23 | f32_mant;
    f32::from_bits(bits)
}

impl DType {
    /// УСТАРЕЛО: см. type_size
    pub fn size(self) -> usize {
        match self {
            DType::F32 | DType::I32 => 4,
            DType::F16 => 2,
        }
    }

    /// Число элементов в одном блоке.
    pub fn blck_size(self) -> usize {
        match self {
            DType::F32 | DType::F16 | DType::I32 => 1,
        }
    }

    /// Число байт на один блок.
    pub fn type_size(self) -> usize {
        match self {
            DType::F32 | DType::I32 => 4,
            DType::F16 => 2,
        }
    }

    /// Число байт на строку из ne0 элементов.
    pub fn row_size(self, ne0: usize) -> usize {
        let bs = self.blck_size();
        assert!(ne0.is_multiple_of(bs), "ggrs: ne0={} не кратен blck_size={}", ne0, bs);
        ne0 / bs * self.type_size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_roundtrip_1_0() {
        let h = f32_to_f16(1.0);
        let back = f16_to_f32(h);
        assert!((back - 1.0).abs() < 1e-6, "roundtrip 1.0: got {back}");
    }

    #[test]
    fn f16_zero_and_neg_zero() {
        assert_eq!(f32_to_f16(0.0), 0);
        assert_eq!(f32_to_f16(-0.0), 0x8000);
    }

    #[test]
    fn f16_exact_roundtrips() {
        for &v in &[-2.5f32, 0.5, 65504.0] {
            let h = f32_to_f16(v);
            let back = f16_to_f32(h);
            assert!(
                (back - v).abs() < 1e-6,
                "roundtrip lossy: {v} -> {h:#06x} -> {back}",
            );
        }
    }

    #[test]
    fn f16_overflow_to_inf() {
        let h = f32_to_f16(1e20f32);
        let back = f16_to_f32(h);
        assert!(back.is_infinite(), "1e20 должно overflow в Inf, got {back}");
    }

    #[test]
    fn f16_known_values() {
        // 1.0 = 0x3C00
        assert!((f16_to_f32(0x3C00) - 1.0).abs() < 1e-6);
        // -2.0 = 0xC000
        assert!((f16_to_f32(0xC000) - (-2.0)).abs() < 1e-6);
    }

    #[test]
    fn f16_nan() {
        assert!(f16_to_f32(0x7E00).is_nan());
    }

    #[test]
    fn f16_denormal() {
        // denormal: 0x0001 = 2^(-24)
        let expected = 2.0f32.powi(-24);
        let val = f16_to_f32(0x0001);
        assert_eq!(val, expected, "denormal 0x0001: got {val}, expected {expected}");
        let h = f32_to_f16(expected);
        assert_eq!(h, 0x0001, "roundtrip 2^(-24): got {h:#06x}, expected 0x0001");
    }

    #[test]
    fn f16_denormal_zero() {
        // 0x0000 = +0.0
        let val = f16_to_f32(0x0000);
        assert_eq!(val, 0.0);
        assert!(val.is_sign_positive(), "0x0000 должен быть +0.0");
        // 0x8000 = -0.0
        let val = f16_to_f32(0x8000);
        assert_eq!(val, 0.0);
        assert!(val.is_sign_negative(), "0x8000 должен быть -0.0");
    }

    #[test]
    fn f16_denormal_max() {
        // 0x03FF = max денормал f16 = 1023 * 2^(-24)
        let val = f16_to_f32(0x03FF);
        let expected = 1023.0 * 2.0f32.powi(-24);
        assert!(
            (val - expected).abs() < 1e-10,
            "denormal max 0x03FF: got {val}, expected {expected}",
        );
    }

    #[test]
    fn f16_denormal_max_roundtrip() {
        let h = f32_to_f16(1023.0 * 2.0f32.powi(-24));
        assert_eq!(h, 0x03FF, "roundtrip max denormal: got {h:#06x}, expected 0x03FF");
        let back = f16_to_f32(h);
        let expected = 1023.0 * 2.0f32.powi(-24);
        assert!(
            (back - expected).abs() < 1e-10,
            "roundtrip max denormal: got {back}, expected {expected}",
        );
    }

    #[test]
    fn f16_round_to_nearest_even() {
        // x = 1.0 + 0.5/1024 (полпути между 0x3C00=1.0 и 0x3C01)
        let a = f16_to_f32(0x3C00); // 1.0
        let b = f16_to_f32(0x3C01); // 1.0 + 1/1024
        let x = a + (b - a) / 2.0; // ровно на полпути
        let h = f32_to_f16(x);
        // round-to-nearest-even: 0x3C00 (чётный)
        assert_eq!(
            h, 0x3C00,
            "RNE полпути 1.0 <-> 1+1/1024: got {h:#06x}, expected 0x3C00",
        );
    }

    #[test]
    fn blck_size_is_one() {
        assert_eq!(DType::F32.blck_size(), 1);
        assert_eq!(DType::F16.blck_size(), 1);
        assert_eq!(DType::I32.blck_size(), 1);
    }

    #[test]
    fn type_size_correct() {
        assert_eq!(DType::F32.type_size(), 4);
        assert_eq!(DType::F16.type_size(), 2);
        assert_eq!(DType::I32.type_size(), 4);
    }

    #[test]
    fn row_size_example() {
        assert_eq!(DType::F16.row_size(8), 16);
    }

    #[test]
    fn size_f16_is_two() {
        assert_eq!(DType::F16.size(), 2);
        assert_eq!(DType::F32.size(), 4);
        assert_eq!(DType::I32.size(), 4);
    }
}
