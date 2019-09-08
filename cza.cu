#include "cza.h"
#include "gpu_var.h"

__constant__ int kXor[256];
__constant__ uint8_t czPerm[256];
__constant__ uint8_t czBox[256];
__constant__ uint64_t kPerm[2048];

__device__ uint64_t cza_load_le64(const uint8_t *p)
{
	return (uint64_t)(((uint64_t)p[7] << 56) |
		((uint64_t)p[6] << 48) |
		((uint64_t)p[5] << 40) |
		((uint64_t)p[4] << 32) |
		((uint64_t)p[3] << 24) |
		((uint64_t)p[2] << 16) |
		((uint64_t)p[1] << 8) |
		(uint64_t)p[0]
		);
}

__device__ uint64_t cza_key_permute_block(uint64_t k)
{
	uint64_t n = 0;
	int i;

	for (i = 0; i < 8; i++)
	{
		n |= kPerm[i*256 + (k & 0xff)];
		k >>= 8;
	}
	return n;
}

__device__ void cza_key_set(const uint8_t cw[CZA_DATA_SIZE], uint8_t key[CZA_KEYSBUFF_SIZE])
{
	uint64_t k[7];
	int i, j;

	k[6] = (uint64_t)(((uint64_t)cw[7] << 56) | ((uint64_t)cw[6] << 48) |
		((uint64_t)cw[5] << 40) | ((uint64_t)cw[4] << 32) | ((uint64_t)cw[3] << 24) |
		((uint64_t)cw[2] << 16) | ((uint64_t)cw[1] << 8) | (uint64_t)cw[0]	);
	
	for (i = 6; i > 0; i--)
	{
		k[i - 1] = cza_key_permute_block(k[i]);
	}
	
	for (i = 0; i < 7; i++)
		for (j = 0; j < 8; j++)
			key[i * 8 + j] = (uint8_t)((k[i] >> (j * 8)) ^ i);
}

__device__ void cza_word_dec(const uint8_t key[CZA_KEYSBUFF_SIZE],  uint8_t *in)
{
	unsigned int	i = CZA_KEYSBUFF_SIZE;
	uint8_t	W[CZA_DATA_SIZE];

	for (int k = 0; k < CZA_DATA_SIZE; k++)
	{
		W[k] = in[k];
	}

	while (i--)
	{
		uint8_t	L;
		uint8_t	S;

		S = czBox[key[i] ^ W[6]];

		L = W[7] ^ S;
		W[7] = W[6];
		W[6] = W[5] ^ czPerm[S];
		W[5] = W[4];
		W[4] = W[3] ^ L;
		W[3] = W[2] ^ L;
		W[2] = W[1] ^ L;
		W[1] = W[0];

		W[0] = L;
	}

	for (int k = 0; k < CZA_DATA_SIZE; k++)
	{
		in[k] = W[k];
		//printf("0x%02x\n", in[k]);
	}

}

__device__ void cza_xor_64(uint8_t *b, const uint8_t *a)
{
	for (int i = 0; i < 8; i++)
	{
		b[i] ^= a[i];
	}
		
}


__device__ int cza_dec(const uint8_t key[CZA_KEYSBUFF_SIZE], uint8_t *data, uint8_t *in)
{
	int r = 0;

	cza_word_dec(key, data);

	for (int i = 8; i < 256; i += 8)
	{
		cza_xor_64(data + i - 8, data + i);
		cza_word_dec(key, data + i);
	}

	for (int i = 0; i < 256; i++)
	{
		if (data[i] != 0 && in[i] != 0)
			r ^= kXor[in[i] & data[i] & 0xFF];
	}

	return r;
}