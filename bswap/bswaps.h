#ifndef BSWAPS_H_
#define BSWAPS_H_

// partial byte swap
// swap first N bytes in buffer
// optimized for small buffer sizes < 32 bytes

#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>

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

#endif
