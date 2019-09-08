#include "gpu_var.h"
#include "hash.h"

__constant__ uint8_t kInput[HEAD_SIZE];
__constant__ uint8_t kTarget[TARG_SIZE];
__device__ uint8_t gOutput[DGST_SIZE];

__device__ uint8_t gTable[524288 * 64];
