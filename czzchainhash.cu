#include "gpu_var.h"
#include "hash.h"
#include "sha3.h"
#include "cza.h"

__device__ uint64_t gFoundIdx = uint64(-1);

__device__ int byteCpy(byte datIn[256], byte datOut[256])
{
	for (int k = 0; k < 256; k += 4)
	{
		datOut[k] = datIn[k + 3];
		datOut[k + 1] = datIn[k + 2];
		datOut[k + 2] = datIn[k + 1];
		datOut[k + 3] = datIn[k];
	}

	return 0;
}

__device__ int shift(uint8_t out[256], uint8_t in[256], int sf)
{
	int sf_i = sf / 8;
	int sf_r = sf % 8;
	int mask = (1 << sf_r) - 1;
	int bits = (8 - sf_r);
	int res;
	
	if (sf_i > 0)
	{
		for (int k = 0; k < 256; k++)
		{
			int ks = ((k + sf_i) % 256);
			out[k] = in[ks];
		}
	}
	else
	{
		for (int k = 0; k < 256; k++)
			out[k] = in[k];
	}
	res = (out[0] & mask) << bits;
	//printf("res is 0x%02x [0x%02x\t0x%02x\t0x%02x]\n", res, out[0], mask, bits);
	for (int k = 0; k < 255; k++)
	{
		uint8_t val = (out[k + 1] & mask) << bits;
		out[k] = (out[k] >> sf_r) + val;
	}
	out[255] = (out[255] >> sf_r) + res;

	return 0;
}

__device__ int scramble(uint8_t permute_in[256])
{

	uint8_t *ptbl;
	uint8_t permute_out[TS_SIZE] = { 0 };
	

	for (int k = 0; k < 64; k++)
	{
		int sf, bs;
		uint64_t it;
		sf = permute_in[0] & 0x7f;
		bs = permute_in[255] >> 4;
		if (k == 2)
			k = k;
		it = (k / 16) * 16 + bs;
		ptbl = gTable + it * 524288;	
		uint8_t cw[CZA_DATA_SIZE] = {
		0x0, 0xe0, 0x1b, 0x02, 0xc9, 0xe0, 0x45, 0xee
		};
		cw[0] = it;
		//printf("============ stage %d\n", k);
		//printf("sf and bs is %d\t%d\n", sf, bs);
		uint8_t key[CZA_KEYSBUFF_SIZE] = { 0 };
		uint8_t data[TS_SIZE] = {0};
		//printf("table is 0x%02x\n", ptbl[0]);
		for (int kk = 0; kk < 2048; kk++)
		{
			int r;
			
			cza_key_set(cw, key);
			
			for (int kkk = 0; kkk < TS_SIZE; kkk++)
			{
				data[kkk] = ptbl[kkk];
			}
			for (int kkk = 0; kkk < CZA_DATA_SIZE; kkk++)
			{
				cw[kkk] = ptbl[kkk];
			}

			//printf("0x%")
			r = cza_dec(key, data, permute_in);
			permute_out[kk / 8] |= (r << (kk % 8));
			ptbl += 256;
		}

		shift(permute_in, permute_out, sf);
		
		for (int kk = 0; kk < 256; kk++)
		{
			permute_out[kk] = 0;
		}
		
		
		//printf("%d 0x%02x\n", k, permute_in[0]);
	}


	return 0;
}

__device__ int byteReverse(uint8_t sha512_out[64])
{
	for (int k = 0; k < 32; k++)
	{
		uint8_t temp = sha512_out[k];
		sha512_out[k] = sha512_out[63 - k];
		sha512_out[63 - k] = temp;
	}

	return 0;
}

int convertLE(uint8_t header[HEAD_SIZE])
{
	int wz = HEAD_SIZE / 4;

	for (int k = 0; k < wz; k++)
	{
		uint8_t temp[4];
		temp[0] = header[k * 4 + 3];
		temp[1] = header[k * 4 + 2];
		temp[2] = header[k * 4 + 1];
		temp[3] = header[k * 4 + 0];
		header[k * 4 + 0] = temp[0];
		header[k * 4 + 1] = temp[1];
		header[k * 4 + 2] = temp[2];
		header[k * 4 + 3] = temp[3];
	}
	return 0;
}

__device__ int convertWD(uint8_t header[HEAD_SIZE])
{
	uint8_t temp[HEAD_SIZE];
	int wz = HEAD_SIZE / 4;
	for (int k = 0; k < wz; k++)
	{
		int i = 7 - k;
		temp[k * 4] = header[i * 4];
		temp[k * 4 + 1] = header[i * 4 + 1];
		temp[k * 4 + 2] = header[i * 4 + 2];
		temp[k * 4 + 3] = header[i * 4 + 3];
	}
	for (int k = 0; k < HEAD_SIZE; k++)
	{
		header[k] = temp[k];
	}
	return 0;
}

__device__ int compare(uint8_t dgst[DGST_SIZE], uint8_t target[TARG_SIZE])
{
	for (int k = TARG_SIZE - 1; k >= 0; k--)
	{
		int dif = (int)dgst[k] - (int)target[k];
		if (dif > 0)
			return 0;
		if (dif < 0)
			return 1;
	}
	
	return 0;
}


__global__ void compute(uint64 nonce_start)
{
	uint8_t digs[DGST_SIZE];
	const uint64 offset = gridDim.x * blockDim.x;
	nonce_start += threadIdx.x + blockIdx.x * blockDim.x;
	printf("CZZ from block %d, thread %d\n", blockIdx.x, threadIdx.x);
	//printrf("blockIdx %d\n", blockIdx.x);
	//printrf("blockDim %d\n", blockDim.x);
	fchainhash(nonce_start, digs);
	printf("finished\n");
#if 0
	while (nonce_start < gFoundIdx)
	{
		fchainhash(nonce_start, digs);

		if (compare(digs, kTarget) == 1)
		{
			atomicMin((unsigned long long int*)&gFoundIdx, unsigned long long int(nonce_start));
			break;
		}
		// Get result here
		//printf("Current nonce : %llu\n", nonce_start);
		nonce_start += offset;
	}
#endif
}

__device__ void fchainhash(uint64 nonce, uint8_t digs[DGST_SIZE])
{

	uint8_t seed[64] = { 0 };
	uint8_t output[DGST_SIZE] = { 0 };

	uint32 val0 = (uint32)(nonce & 0xFFFFFFFF);
	uint32 val1 = (uint32)(nonce >> 32);
	for (int k = 3; k >= 0; k--)
	{
		seed[k] = val0 & 0xFF;
		val0 >>= 8;
	}
	
	for (int k = 7; k >= 4; k--)
	{
		seed[k] = val1 & 0xFF;
		val1 >>= 8;
	}
	
	for (int k = 0; k < HEAD_SIZE; k++)
	{
		seed[k+8] = kInput[k];
	}

	uint8_t sha512_out[64];
	sha3(seed, 64, sha512_out, 64);
	byteReverse(sha512_out);

	uint8_t permute_in[256] = { 0 };
	for (int k = 0; k < 4; k++)
	{
		for (int kk = 0; kk < 64; kk++)
		{
			permute_in[k*64+ kk] = sha512_out[kk];
		}
	}

	scramble(permute_in);

	uint8_t dat_in[256];
	
	byteCpy(permute_in, dat_in);

	//unsigned char output[64];
	sha3(dat_in, 256, output, 32);
	// reverse byte
	for (int k = 0; k < DGST_SIZE; k++)
	{
		digs[k] = output[DGST_SIZE - k - 1];
	}

}

