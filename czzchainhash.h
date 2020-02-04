#pragma once
#include "cuda_runtime.h"
__device__ void fchainhash(uint64_t nonce, uint8_t digs[DGST_SIZE]);
__global__ void compute(uint64_t nonce_start);
