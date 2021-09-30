#ifndef BSWAPS_H_
#define BSWAPS_H_

// partial byte swap
// swap first N bytes in buffer
// optimized for small buffer sizes < 32 bytes

#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>

#include <iostream>

#include <arm_neon.h>

void pp(uint8_t* buf) {
  for (int j = 0; j < 16; j++) {
    std::cout << " " << int(buf[j]);
  }
  std::cout << std::endl;
}

void bswap_2(uint8_t* buf) {
  std::swap(buf[0], buf[1]);
}

void bswap_3(uint8_t* buf) {
  std::swap(buf[0], buf[2]);
}

void bswap_4(uint8_t* buf) {
  std::swap(buf[0], buf[3]);
  std::swap(buf[1], buf[2]);
}

void bswap_5(uint8_t* buf) {
  std::swap(buf[0], buf[4]);
  std::swap(buf[1], buf[3]);
}

void bswap_6(uint8_t* buf) {
  std::swap(buf[0], buf[5]);
  std::swap(buf[1], buf[4]);
  std::swap(buf[2], buf[3]);
}

void bswap_7(uint8_t* buf) {
  std::swap(buf[0], buf[6]);
  std::swap(buf[1], buf[5]);
  std::swap(buf[2], buf[4]);
}

void bswap_8_(uint8_t* buf) {
  uint64_t* a = reinterpret_cast<uint64_t*>(buf);
  *a = __builtin_bswap64(*a);
}

void bswap_8(uint8_t* buf) {
  std::swap(buf[0], buf[7]);
  std::swap(buf[1], buf[6]);
  std::swap(buf[2], buf[5]);
  std::swap(buf[3], buf[4]);
}

void bswap_9(uint8_t* buf) {
  std::swap(buf[0], buf[8]);
  std::swap(buf[1], buf[7]);
  std::swap(buf[2], buf[6]);
  std::swap(buf[3], buf[5]);
}

void bswap_10(uint8_t* buf) {
  std::swap(buf[0], buf[9]);
  std::swap(buf[1], buf[8]);
  std::swap(buf[2], buf[7]);
  std::swap(buf[3], buf[6]);
  std::swap(buf[4], buf[5]);
}

void bswap_11(uint8_t* buf) {
  std::swap(buf[0], buf[10]);
  std::swap(buf[1], buf[9]);
  std::swap(buf[2], buf[8]);
  std::swap(buf[3], buf[7]);
  std::swap(buf[4], buf[6]);
}

void bswap_12(uint8_t* buf) {
  std::swap(buf[0], buf[11]);
  std::swap(buf[1], buf[10]);
  std::swap(buf[2], buf[9]);
  std::swap(buf[3], buf[8]);
  std::swap(buf[4], buf[7]);
  std::swap(buf[5], buf[6]);
}

void bswap_13(uint8_t* buf) {
  std::swap(buf[0], buf[12]);
  std::swap(buf[1], buf[11]);
  std::swap(buf[2], buf[10]);
  std::swap(buf[3], buf[9]);
  std::swap(buf[4], buf[8]);
  std::swap(buf[5], buf[7]);
}

void bswap_14(uint8_t* buf) {
  std::swap(buf[0], buf[13]);
  std::swap(buf[1], buf[12]);
  std::swap(buf[2], buf[11]);
  std::swap(buf[3], buf[10]);
  std::swap(buf[4], buf[9]);
  std::swap(buf[5], buf[8]);
  std::swap(buf[6], buf[7]);
}

void bswap_15(uint8_t* buf) {
  std::swap(buf[0], buf[14]);
  std::swap(buf[1], buf[13]);
  std::swap(buf[2], buf[12]);
  std::swap(buf[3], buf[11]);
  std::swap(buf[4], buf[10]);
  std::swap(buf[5], buf[9]);
  std::swap(buf[6], buf[8]);
}

void bswap_16(uint8_t* buf) {
  std::swap(buf[0], buf[15]);
  std::swap(buf[1], buf[14]);
  std::swap(buf[2], buf[13]);
  std::swap(buf[3], buf[12]);
  std::swap(buf[4], buf[11]);
  std::swap(buf[5], buf[10]);
  std::swap(buf[6], buf[9]);
  std::swap(buf[7], buf[8]);
}

using bswap_hardcoded_fn = void(*)(uint8_t*);

static bswap_hardcoded_fn bswap_hardcoded[] = {
  nullptr,
  nullptr,
  bswap_2,
  bswap_3,
  bswap_4,
  bswap_5,
  bswap_6,
  bswap_7,
  bswap_8,
  bswap_9,
  bswap_10,
  bswap_11,
  bswap_12,
  bswap_13,
  bswap_14,
  bswap_15,
  bswap_16
};


static bswap_hardcoded_fn bswap_hardcoded_[] = {
  nullptr,
  nullptr,
  bswap_2,
  bswap_3,
  bswap_4,
  bswap_5,
  bswap_6,
  bswap_7,
  bswap_8_,
  bswap_9,
  bswap_10,
  bswap_11,
  bswap_12,
  bswap_13,
  bswap_14,
  bswap_15,
  bswap_16
};

void bswap_naive(uint8_t* buf, int64_t n) {
  int64_t i, j;
  uint8_t t;
  for (i = 0, j = n - 1; i < j; i++, j--) {
    t = buf[i];
    buf[i] = buf[j];
    buf[j] = t;
  } 
}

static const uint64_t bswap8b_masks[] = {
  0xffffffffffffffffULL,
  0xffffffffffffff00ULL,
  0xffffffffffff0000ULL,
  0xffffffffff000000ULL,
  0xffffffff00000000ULL,
  0xffffff0000000000ULL,
  0xffff000000000000ULL,
  0xff00000000000000ULL,
  0x0000000000000000ULL,
};

void bswap8b(uint8_t* buf, int64_t n) {
  uint64_t* a = (uint64_t*)buf; 
  *a = (__builtin_bswap64(*a) >> (64 - n * 8)) | ((*a) & bswap8b_masks[n]);
}

uint64_t bswap8(uint64_t a, int64_t n) {
  return (__builtin_bswap64(a) >> (64 - n * 8)) | (a & bswap8b_masks[n]);
}

// 8-16
void bswap16b(uint64_t* a, int64_t n) {
  uint64_t tmp = __builtin_bswap64(a[0]);
  a[0] = bswap8(a[1], n - 8);
  a[1] = tmp;
}

uint8x16_t neon_lookuptable[16];

void load_neon_table() {
  uint8_t buf[16];
  for (int i = 0; i < 16; i++) {
    for (int j = 0; j < i; j++) {
      buf[j] = i - j - 1;
    }
    for (int j = i; j < 16; j++) {
      buf[j] = j;
    }
    neon_lookuptable[i] = vld1q_u8(buf);
  }
}

void bswap16bneon(uint8_t* a, int64_t n) {
  uint8x16x4_t v;
  uint8x16_t r;
  v.val[0] = vld1q_u8(a);
  r = vqtbl4q_u8(v, neon_lookuptable[n]);
  vst1q_u8(a, r);
}

int32_t topswops_unrolled_(uint8_t* a) {
  int32_t res = 0;
  while (a[0] != 1) {
    bswap_hardcoded_[a[0]](a);
    res++;
  }
  return res;
}


int32_t topswops_unrolled(uint8_t* a) {
  int32_t res = 0;
  while (a[0] != 1) {
    bswap_hardcoded[a[0]](a);
    res++;
  }
  return res;
}

int32_t topswops_naive(uint8_t* a) {
  int32_t res = 0;
  while (a[0] != 1) {
    bswap_naive(a, a[0]);
    res++;
  }
  return res;
}

int32_t topswops_neon2(uint8_t* a) {
  uint8x16x4_t v;
  v.val[0] = vld1q_u8(a);
  int32_t res = 0;
  size_t n = a[0];

  while (n != 1) {
    v.val[0] = vqtbl4q_u8(v, neon_lookuptable[n]);
    res++;
    vst1q_u8(a, v.val[0]);
    n = a[0];
  }
  return res;
}


int32_t topswops_neon(uint8_t* a) {
  int32_t res = 0;
  //printf("val = 0x%" PRIx64 "\n", A[0]);
  //printf("val = 0x%" PRIx64 "\n", A[1]);
  while (a[0] != 1) {
    bswap16bneon(a, a[0]);
    //printf("val = 0x%" PRIx64 "\n", A[0]);
    //printf("val = 0x%" PRIx64 "\n\n", A[1]);
    res++;
  }
  return res;
}

// TODO doesn't work
int32_t topswops_better(uint8_t* a) {
  int32_t res = 0;
  uint64_t* A = reinterpret_cast<uint64_t*>(a);
  printf("val = 0x%" PRIx64 "\n", A[0]);
  printf("val = 0x%" PRIx64 "\n", A[1]);
  while (a[0] != 1) {
    if (a[0] >= 8)
      bswap16b(A, a[0]);
    else 
      bswap8b(a, a[0]);
    printf("val = 0x%" PRIx64 "\n", A[0]);
    printf("val = 0x%" PRIx64 "\n\n", A[1]);
    res++;
    if (res > 30) break;
  }
  return res;
}

#endif
