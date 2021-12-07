/*
 * UCEncryptionLEA.c
 *
 *  Created on: 2021. 9. 2.
 *      Author: jrkim
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <stdint.h>

#include <uem_common.h>

#include <UCBasic.h>

#define BLOCK_SIZE 16

const static uint32_t DELTA[4] = {0xc3efe9db, 0x88c4d604, 0xe789f229, 0xc6f98763};

static inline uint32_t rot32l1(uint32_t x) {
	return (x << 1) | (x >> 31);
}

static inline uint32_t rot32r1(uint32_t x) {
	return (x >> 1) | (x << 31);
}

static inline uint32_t rot32l8(uint32_t x) {
	return (x << 8) | (x >> 24);
}

static inline uint32_t rot32r8(uint32_t x) {
	return (x >> 8) | (x << 24);
}

static inline uint32_t rot32l3(uint32_t x) {
	return rot32l1(rot32l1(rot32l1(x)));	
}

static inline uint32_t rot32r3(uint32_t x) {
	return rot32r1(rot32r1(rot32r1(x)));	
}

static inline uint32_t rot32l5(uint32_t x) {
	return rot32r3(rot32l8(x));
}

static inline uint32_t rot32l9(uint32_t x) {
	return rot32l1(rot32l8(x));
}

static inline uint32_t rot32r5(uint32_t x) {
	return rot32l3(rot32r8(x));
}

static inline uint32_t rot32r9(uint32_t x) {
	return rot32r1(rot32r8(x));
}

static inline uint32_t rot32l2(uint32_t x) {
	return rot32l1(rot32l1(x));
}

static inline uint32_t rot32l4(uint32_t x) {
	return rot32l2(rot32l2(x));
}

static inline uint32_t rot32l6(uint32_t x) {
	return rot32r1(rot32r1(rot32l8(x)));
}

static inline uint32_t rot32l11(uint32_t x) {
	return rot32l3(rot32l8(x));
}

static void xor(uint8_t* iv, uint8_t* data, uint32_t nKeyLen) {
	uint8_t i = 0;
	uint32_t *tIv = (uint32_t *)iv;
	uint32_t *tData = (uint32_t *)data;

	for(i = 0; i < nKeyLen/sizeof(uint32_t); i++) {
		tData[i] ^= tIv[i];
	}
}

static void plusOne(uint8_t* iv, uint32_t nKeyLen) {
	uint8_t i = 0;
	
	for(i = 0; i < nKeyLen; i++) {
		iv[i] = (iv[i] + 1) % 256;
	}
}

uem_result UCEncryptionLEA_GenerateRoundKey(uint8_t* userkey, uint8_t* roundkey, uint8_t nRoundNum)
{
	uem_result result = ERR_UEM_UNKNOWN;
	uint32_t* rk = (uint32_t*) roundkey;
	uint32_t* t = (uint32_t*) userkey;
	uint32_t delta[4] = {DELTA[0], DELTA[1], DELTA[2], DELTA[3]};
	uint32_t tmp;
	uint32_t ridx = 0;
	uint8_t i;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(userkey, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif

	uint32_t t0 = t[0];
	uint32_t t1 = t[1];
	uint32_t t2 = t[2];
	uint32_t t3 = t[3];

	for (i = 0; i < nRoundNum; ++i) {
		tmp = delta[i & 3];
		
		t0 = rot32l1(t0 + tmp);
		t1 = rot32l3(t1 + rot32l1(tmp));
		t2 = rot32l6(t2 + rot32l2(tmp));
		t3 = rot32l11(t3 + rot32l3(tmp));
		delta[i & 3] = rot32l4(tmp);
		
		rk[ridx++] = t1;
		rk[ridx++] = t3;
		rk[ridx++] = t2;
		rk[ridx++] = t0;
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UCEncryptionLEA_Encode(uint8_t* roundkey, uint8_t* data, uint32_t nRoundNum)
{
	uem_result result = ERR_UEM_UNKNOWN;
	uint32_t* block = (uint32_t*) data;
	const uint32_t* rk = (const uint32_t*) roundkey;
	uint8_t i;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(roundkey, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	IFVARERRASSIGNGOTO(data, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif

	uint32_t b0 = block[0];
	uint32_t b1 = block[1];
	uint32_t b2 = block[2];
	uint32_t b3 = block[3];
	
	for (i = 0; i < nRoundNum/sizeof(uint32_t); ++i) {
		b3 = rot32r3(((b2 ^ rk[1]) + (b3 ^ rk[0])));
		b2 = rot32r5(((b1 ^ rk[2]) + (b2 ^ rk[0])));
		b1 = rot32l9(((b0 ^ rk[3]) + (b1 ^ rk[0])));
		rk += 4;
		
		b0 = rot32r3(((b3 ^ rk[1]) + (b0 ^ rk[0])));
		b3 = rot32r5(((b2 ^ rk[2]) + (b3 ^ rk[0])));
		b2 = rot32l9(((b1 ^ rk[3]) + (b2 ^ rk[0])));
		rk += 4;
		
		b1 = rot32r3(((b0 ^ rk[1]) + (b1 ^ rk[0])));
		b0 = rot32r5(((b3 ^ rk[2]) + (b0 ^ rk[0])));
		b3 = rot32l9(((b2 ^ rk[3]) + (b3 ^ rk[0])));
		rk += 4;
		
		b2 = rot32r3(((b1 ^ rk[1]) + (b2 ^ rk[0])));
		b1 = rot32r5(((b0 ^ rk[2]) + (b1 ^ rk[0])));
		b0 = rot32l9(((b3 ^ rk[3]) + (b0 ^ rk[0])));
		rk += 4;
	}
	
	block[0] = b0;
	block[1] = b1;
	block[2] = b2;
	block[3] = b3;	

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UCEncryptionLEA_Decode(uint8_t* roundkey, uint8_t* data, uint32_t nRoundNum)
{
	uem_result result = ERR_UEM_UNKNOWN;
	uint32_t* block = (uint32_t*) data;
	const uint32_t* rk = (const uint32_t*) roundkey;
	uint8_t i;
#ifdef ARGUMENT_CHECK
	IFVARERRASSIGNGOTO(roundkey, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
	IFVARERRASSIGNGOTO(data, NULL, result, ERR_UEM_INVALID_PARAM, _EXIT);
#endif

	uint32_t b0 = block[0];
	uint32_t b1 = block[1];
	uint32_t b2 = block[2];
	uint32_t b3 = block[3];
	
	rk += 92;
	for (i = 0; i < nRoundNum/sizeof(uint32_t); ++i) {
		b0 = (rot32r9(b0) - (b3 ^ rk[3])) ^ rk[0];
		b1 = (rot32l5(b1) - (b0 ^ rk[2])) ^ rk[0];
		b2 = (rot32l3(b2) - (b1 ^ rk[1])) ^ rk[0];
		rk -= 4;
		
		b3 = (rot32r9(b3) - (b2 ^ rk[3])) ^ rk[0];
		b0 = (rot32l5(b0) - (b3 ^ rk[2])) ^ rk[0];
		b1 = (rot32l3(b1) - (b0 ^ rk[1])) ^ rk[0];
		rk -= 4;

		b2 = (rot32r9(b2) - (b1 ^ rk[3])) ^ rk[0];
		b3 = (rot32l5(b3) - (b2 ^ rk[2])) ^ rk[0];
		b0 = (rot32l3(b0) - (b3 ^ rk[1])) ^ rk[0];
		rk -= 4;

		b1 = (rot32r9(b1) - (b0 ^ rk[3])) ^ rk[0];
		b2 = (rot32l5(b2) - (b1 ^ rk[2])) ^ rk[0];
		b3 = (rot32l3(b3) - (b2 ^ rk[1])) ^ rk[0];
		rk -= 4;		
	}
	
	block[0] = b0;
	block[1] = b1;
	block[2] = b2;
	block[3] = b3;

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}

uem_result UCEncryptionLEA_EncodeOnCTRMode(uint32_t *roundkey, uint8_t *iv, uem_uint8 *pData, uint32_t nDataLen, uint32_t nRoundNum)
{
	uem_result result = ERR_UEM_UNKNOWN;
	uem_uint32 nBlockNum = 0, nRemainSize = 0, nLoc = 0;
	uem_uint8 tiv[BLOCK_SIZE] = {0, };
	int i = 0;

	nBlockNum = nDataLen / BLOCK_SIZE;
	nRemainSize = nDataLen % BLOCK_SIZE;
	UC_memcpy(tiv, iv, sizeof(uem_uint8) * BLOCK_SIZE);

	for(i = 0; i < nBlockNum; i++) {
		xor(tiv, pData + nLoc, BLOCK_SIZE);
		nLoc += sizeof(uem_uint8) * BLOCK_SIZE;

		if(nBlockNum > 1) {
			plusOne(tiv, BLOCK_SIZE);

			result = UCEncryptionLEA_Encode((uem_uint8*)roundkey, tiv, nRoundNum);
			ERRIFGOTO(result, _EXIT);
		}
	}
	
	if(nRemainSize != 0) {
		uem_uint8 tData[BLOCK_SIZE] = {0, };
		
		UC_memcpy(tData, pData + nLoc, sizeof(uem_uint8)*nRemainSize);	
		xor(tiv, tData, BLOCK_SIZE);
		UC_memcpy(pData + nLoc, tData, sizeof(uem_uint8)*nRemainSize);
	}

	result = ERR_UEM_NOERROR;
_EXIT:
	return result;
}
