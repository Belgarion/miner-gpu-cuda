#ifndef LIBCZA_H_
#define LIBCZA_H_

#include <stddef.h>
#include <stdint.h>

#define CZA_DATA_SIZE			8
#define CZA_KEYSBUFF_SIZE		56

__device__ void cza_key_set(const uint8_t cw[CZA_DATA_SIZE], uint8_t key[CZA_KEYSBUFF_SIZE]);

__device__ int cza_dec(const uint8_t key[CZA_KEYSBUFF_SIZE], uint8_t *data, uint8_t *input);

#endif