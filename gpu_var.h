#include "hash.h"
extern __device__ uint8_t gTable[524288 * 64];
extern __constant__ uint8_t kInput[HEAD_SIZE];
extern __constant__ uint8_t kTarget[TARG_SIZE];
//extern __constant__ uint8_t kTarget2[TARG_SIZE];
extern __device__ uint8_t gOutput[DGST_SIZE];
extern __device__ uint64_t gFoundIdx;
extern __constant__ int kXor[256];
extern __constant__ uint8_t czPerm[256];
extern __constant__ uint8_t czBox[256];
extern __constant__ uint64_t kPerm[2048];