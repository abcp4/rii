#ifndef DISTANCE_H
#define DISTANCE_H

// http://koturn.hatenablog.com/entry/2016/07/18/090000
// windows is not supported, but just in case (later someone might implement)
// https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=590,27,2

#ifdef __aarch64__
#  include  <arm_neon.h>


// These fast L2 squared distance codes (SSE and AVX) are from the Faiss library:
// https://github.com/facebookresearch/faiss/blob/master/utils.cpp
//
// Based on them, AVX512 implementation is also prepared.
// But it doesn't seem drastically fast. Only slightly faster than AVX:
// (runtime) REF >> SSE >= AVX ~ AVX512

namespace rii {

// From Faiss.
// Reference implementation
float fvec_L2sqr_ref(const float *x, const float *y, size_t d)
{
    size_t i;
    float res_ = 0;
    for (i = 0; i < d; i++) {
        const float tmp = x[i] - y[i];
        res_ += tmp * tmp;
    }
    return res_;
}




// ========================= Distance functions ============================

static const std::string g_simd_architecture = "arm64";

#if defined(__aarch64__)

float fvec_L2sqr (const float * x,
                  const float * y,
                  size_t d)
{
    if (d & 3) return fvec_L2sqr_ref (x, y, d);
    float32x4_t accu = vdupq_n_f32 (0);
    for (size_t i = 0; i < d; i += 4) {
        float32x4_t xi = vld1q_f32 (x + i);
        float32x4_t yi = vld1q_f32 (y + i);
        float32x4_t sq = vsubq_f32 (xi, yi);
        accu = vfmaq_f32 (accu, sq, sq);
    }
    float32x4_t a2 = vpaddq_f32 (accu, accu);
    return vdups_laneq_f32 (a2, 0) + vdups_laneq_f32 (a2, 1);
}
#endif // 


} // namespace rii

#endif // DISTANCE_H
