RII for ARM. Just included header arm_neon and function fvec_L2sqr for arm from https://github.com/efficient/faiss-learned-termination/blob/master/utils_simd.cpp .

Beware of aarch64: libgomp.so.1: cannot allocate memory in static TLS block. Fix: https://github.com/opencv/opencv/issues/14884
