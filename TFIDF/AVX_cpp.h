/*
 * Cython C++ Shim for AVX
 * Author: George Lok
 *
 * Based on AVX.h by CS205
 */
extern "C" {
  #include <math.h>

  #ifdef __AVX__
  #  include <immintrin.h>
  #else
  #  include "avxintrin-emu.h"
  #endif

  #ifdef __FMA__
  #  include <fmaintrin.h>
  #else
  #  define _mm256_fmadd_ps(a, b, c) _mm256_add_ps(_mm256_mul_ps((a), (b)), (c))
  #  define _mm256_fmsub_ps(a, b, c) _mm256_sub_ps(_mm256_mul_ps((a), (b)), (c))
  #endif
}

//Float
typedef __m256 float8;

//Construction
inline float8 float_to_float8(float v) { return _mm256_set1_ps(v); }

inline float8 make_float8(float v1, float v2, float v3, float v4, float v5, float v6, float v7, float v8) {
  return _mm256_set_ps((v1), (v2), (v3), (v4), (v5), (v6), (v7), (v8));
}
//Arithmetic
inline float8 fmadd(float8 a, float8 b, float8 c) {
  return _mm256_fmadd_ps((a), (b), (c)); // a * b + c  
}       
inline float8 fmsub(float8 a, float8 b, float8 c) {
  return _mm256_fmsub_ps((a), (b), (c)); // a * b - c
}
inline float8 sqrt(float8 val) { return _mm256_sqrt_ps(val); }
inline float8 mul(float8 a, float8 b) { return _mm256_mul_ps((a), (b)); }
inline float8 add(float8 a, float8 b) { return _mm256_add_ps((a), (b)); }
inline float8 sub(float8 a, float8 b) { return _mm256_sub_ps((a), (b)); }
inline float8 div(float8 a, float8 b) { return _mm256_div_ps((a), (b)); }
//Bitwise
inline float8 bitwise_and(float8 a, float8 b) { return _mm256_and_ps((a), (b)); }
inline float8 bitwise_andnot(float8 a, float8 b) { return _mm256_andnot_ps((a), (b)); }
inline float8 bitwise_xor(float8 a, float8 b) { return _mm256_xor_ps((a), (b)); }
//Logical
inline float8 less_than(float8 a, float8 b) { return _mm256_cmp_ps((a), (b), _CMP_LT_OS); }
inline float8 greater_than(float8 a, float8 b) { return _mm256_cmp_ps((a), (b), _CMP_GT_OS); }
//Helpers
inline int signs(float8 a) { return (_mm256_movemask_ps(a) & 255); }
inline void to_mem(float8 reg, float *mem) { _mm256_storeu_ps(mem, reg); }

