#include "bench.h"
#include "hero-target.h"
#include <assert.h>
#include <errno.h>        // error codes
#include <stdint.h>       // uint8_t
#include <stdio.h>        // printf()
#include <stdlib.h>       // rand(), srand()
#include <string.h>       // memcmp()

#pragma omp declare target
#include "aes.h"
#include "aes.c"
#pragma omp end declare target

/**
 * Randomize an array of bytes.
 */
void rand_bytes(uint8_t* const arr, const size_t n_bytes)
{
  assert(arr != NULL);

  for (uint8_t* wptr = arr; wptr < arr + n_bytes; ++wptr)
    *wptr = rand();
}

/**
 * Free byte buffers.
 */
void free_byte_bufs(uint8_t** const bufs, const unsigned n)
{
  assert(bufs != NULL);

  for (uint8_t** buf = bufs; buf < bufs + n; ++buf)
    free(*buf);
  free(bufs);
}

void print_aes_block(const uint8_t* const block)
{
  for (const uint8_t* byte = block; byte < block + AES_BLOCKLEN - 1; ++byte)
    printf("%02x ", *byte);
  printf("%02x\n", *(block + AES_BLOCKLEN - 1));
}

/**
 * Allocate byte buffers.
 */
uint8_t** alloc_byte_bufs(const unsigned n, const size_t len)
{
  uint8_t** const bufs = malloc(n * sizeof(uint8_t*));
  if (bufs == NULL)
    return NULL;

  for (uint8_t** buf = bufs; buf < bufs + n; ++buf) {
    *buf = malloc(len * sizeof(uint8_t));
    if (*buf == NULL) {
      // Free already allocated buffers and return.
      free_byte_bufs(bufs, bufs-buf);
      return NULL;
    }
  }

  return bufs;
}

/**
 * Compare byte buffers.
 *
 * @return  Number of buffers that are not equal.
 */
unsigned memcmp_byte_bufs(uint8_t** const s1s, uint8_t** const s2s, const unsigned n,
    const size_t len)
{
  assert(s1s != NULL);
  assert(s2s != NULL);

  const uint8_t** s1 = (const uint8_t**)s1s;
  const uint8_t** s2 = (const uint8_t**)s2s;
  unsigned cnt_ne = 0;
  for (unsigned i = 0; i < n; ++i) {
    if (memcmp(*s1, *s2, len) != 0)
      cnt_ne++;
  }
  return cnt_ne;
}

/**
 * Initialize AES contexts with pseudo-random initialization vectors.
 */
struct AES_ctx** init_AES_ctxs(const unsigned n, const uint8_t* const key)
{
  assert(key != NULL);

  struct AES_ctx** const ctxs = (struct AES_ctx**)alloc_byte_bufs(n, sizeof(struct AES_ctx));
  if (ctxs == NULL)
    return NULL;

  for (struct AES_ctx** ctx = ctxs; ctx < ctxs + n; ++ctx) {
    uint8_t iv[AES_BLOCKLEN];
    rand_bytes((uint8_t*)&iv, AES_BLOCKLEN);
    AES_init_ctx_iv(*ctx, key, &iv);
  }

  return ctxs;
}

/**
 * Initialize IVs from given AES contexts.
 */
uint8_t** init_ivs(struct AES_ctx** const ctxs, const unsigned n)
{
  assert(ctxs != NULL);

  uint8_t** const ivs = alloc_byte_bufs(n, AES_BLOCKLEN * sizeof(uint8_t));
  if (ivs == NULL)
    return NULL;

  uint8_t** iv = ivs;
  for (struct AES_ctx** ctx = ctxs; ctx < ctxs + n; ++ctx)
    memcpy(*(iv++), (*ctx)->Iv, AES_BLOCKLEN);

  return ivs;
}

void AES_ctx_set_ivs(struct AES_ctx** const ctxs, uint8_t** const ivs, const unsigned n)
{
  assert(ctxs != NULL);
  assert(ivs != NULL);

  uint8_t** iv = ivs;
  for (struct AES_ctx** ctx = ctxs; ctx < ctxs + n; ++ctx)
    AES_ctx_set_iv(*ctx, *(iv++));
}

void free_AES_ctxs(struct AES_ctx** const ctxs, const unsigned n)
{
  free_byte_bufs((uint8_t**)ctxs, n);
}

size_t pad_str_len(const size_t str_len)
{
  const size_t pad_len = AES_BLOCKLEN - str_len % AES_BLOCKLEN;
  if (pad_len == AES_BLOCKLEN)
    return str_len;
  else
    return str_len + pad_len;
}

/**
 * Initialize plaintext strings with pseudo-random data.
 *
 * Returned plaintext strings are multiples of 16 bytes in length, if necessary padded with PKCS7.
 */
uint8_t** init_plains(const unsigned n, const size_t str_len)
{
  const size_t padded_str_len = pad_str_len(str_len);
  uint8_t** const plains = alloc_byte_bufs(n, padded_str_len);
  if (plains == NULL)
    return NULL;

  for (uint8_t** plain = plains; plain < plains + n; ++plain) {
    rand_bytes(*plain, str_len);
    // Pad with the PKCS7 scheme, if necessary.
    const uint8_t n_pad = padded_str_len - str_len;
    if (n_pad > 0) {
      uint8_t* const bptr = *plain + str_len;
      uint8_t* const eptr = bptr + n_pad;
      for (uint8_t* c = bptr; c < eptr; ++c)
        *c = n_pad;
    }
  }

  return plains;
}

void print_usage()
{
  printf("aes [num. of plaintexts] [len. of each plaintext] [seed]\n");
}

int main(int argc, char *argv[])
{
  printf("HERO AES CBC encryption benchmark started.\n");

  // Configure through command-line arguments.
  unsigned n_strs = 1024;   // number of plaintext strings
  size_t str_len  = 128;    // length of plaintext string in bytes
  unsigned seed   = time(0);
  if (argc == 2 && !strcmp(argv[1], "-h")) {
    print_usage();
    return 0;
  }
  switch (argc) {
    case 4:   seed = atoi(argv[3]);
    case 3:   str_len = atoi(argv[2]);
    case 2:   n_strs = atoi(argv[1]);
    case 1:   break;
    default:  print_usage();
              return 0;
  }

  printf("n_strs = %u, str_len = %u, seed = 0x%08x\n", n_strs, str_len, seed);

  // Seed pseudo-random number generator.
  srand(seed);

  // Generate pseudo-random key.
  uint8_t* const key = (uint8_t*)malloc(AES_KEYLEN * sizeof(uint8_t));
  if (key == NULL) {
    printf("ERROR: malloc for `key` failed!\n");
    return -ENOMEM;
  }
  rand_bytes(key, AES_KEYLEN);

  // Initialize AES contexts.
  struct AES_ctx** const ctxs = init_AES_ctxs(n_strs, key);
  if (ctxs == NULL) {
    printf("ERROR: Failed to initialize AES contexts!\n");
    free(key);
    return -ENOMEM;
  }

  // Initialize (unmodifiable copy of) IVs.
  uint8_t** const ivs = init_ivs(ctxs, n_strs);
  if (ivs == NULL) {
    printf("ERROR: Failed to initialize IVs!\n");
    free_AES_ctxs(ctxs, n_strs);
    free(key);
    return -ENOMEM;
  }

  // Initialize plaintexts with pseudo-random data.
  uint8_t** const plains = init_plains(n_strs, str_len);
  if (plains == NULL) {
    printf("ERROR: Failed to initialize plaintext strings!\n");
    free(ivs);
    free_AES_ctxs(ctxs, n_strs);
    free(key);
    return -ENOMEM;
  }
  const size_t padded_str_len = pad_str_len(str_len);

  // Allocate buffers for ciphertext and decrypted texts.
  uint8_t** const ciphers = alloc_byte_bufs(n_strs, padded_str_len);
  if (ciphers == NULL) {
    printf("ERROR: Failed to allocate memory for ciphertext strings!\n");
    free(key);
    free_AES_ctxs(ctxs, n_strs);
    free_byte_bufs(plains, n_strs);
    return -ENOMEM;
  }
  uint8_t** decrypteds = alloc_byte_bufs(n_strs, padded_str_len);
  if (decrypteds == NULL) {
    printf("ERROR: Failed to allocate memory for decrypted strings!\n");
    free(key);
    free_AES_ctxs(ctxs, n_strs);
    free_byte_bufs(plains, n_strs);
    free_byte_bufs(ciphers, n_strs);
    return -ENOMEM;
  }

  /**
   * Execute on host
   */

  printf("IVs:\n");
  for (unsigned i = 0; i < n_strs; ++i)
    print_aes_block(ivs[i]);

  bench_start("Host: Encryption");
  #pragma omp parallel //firstprivate(ctxs, ciphers, plains, n_strs, str_len)
  {
    #pragma omp parallel for
    for (unsigned i = 0; i < n_strs; ++i) {
      memcpy(ciphers[i], plains[i], padded_str_len * sizeof(uint8_t));
      //AES_CBC_encrypt_buffer(ctxs[i], ciphers[i], padded_str_len);
    }
  }
  bench_stop();

  printf("IVs:\n");
  for (unsigned i = 0; i < n_strs; ++i)
    print_aes_block(ivs[i]);

  // Reset IVs.
  AES_ctx_set_ivs(ctxs, ivs, n_strs);

  printf("IVs:\n");
  for (unsigned i = 0; i < n_strs; ++i)
    print_aes_block(ivs[i]);

  bench_start("Host: Decryption");
  #pragma omp parallel //firstprivate(ctxs, decrpyteds, ciphers, n_strs, str_len)
  {
    #pragma omp parallel for
    for (unsigned i = 0; i < n_strs; ++i) {
      memcpy(decrypteds[i], ciphers[i], padded_str_len * sizeof(uint8_t));
      //AES_CBC_decrypt_buffer(ctxs[i], decrypteds[i], padded_str_len);
    }
  }
  bench_stop();
  const unsigned n_mismatches_host = memcmp_byte_bufs(decrypteds, plains, n_strs, str_len);
  if (n_mismatches_host != 0) {
    printf("ERROR: %u out of %u decrypted ciphertexts do NOT match the given plaintext!\n",
        n_mismatches_host, n_strs);
  }

  /**
   * Execute on PULP
   */

  // Make sure PULP is booted (speeds up the following actual target).
  unsigned tmp_1 = 1;
  unsigned tmp_2 = 2;
  #pragma omp target device(0) map(to: tmp_1) map(from: tmp_2)
  {
    tmp_2 = tmp_1;
  }
  tmp_1 = tmp_2;

  // Copy-based execution does not work out of the box because we would have to offload a linked
  // data structure.

  bench_start("PULP Parallel, SVM, DMA: Encryption");
  //#pragma omp target device(0) \
  //    map(to: plains[0:n_strs-1], n_strs, padded_str_len) \
  //    map(tofrom: ctxs[0:n_strs-1]) \
  //    map(from: ciphers[0:n_strs-1])
  //{
  //  unsigned sync = 0;

  //  #pragma omp parallel default(none) \
  //      shared(n_strs, sync) num_threads(2)
  //  {
  //    // Spawn the miss-handler thread
  //    if (omp_get_thread_num() == 0) {
  //      do {
  //        const int ret = hero_handle_rab_misses();
  //        if (!(ret == 0 || ret == -ENOENT)) {
  //          printf("RAB miss handling returned error: %d!\n", -ret);
  //          break;
  //        }
  //      } while (sync == 0);
  //    }

  //    // Worker threads
  //    else {
  //      // Read arguments to local memory.
  //      const uint8_t** const plains_loc = (const uint8_t**)hero_tryread((unsigned*)&plains);
  //      const unsigned n_strs_loc = hero_tryread((unsigned*)&n_strs);
  //      const unsigned str_len_loc = hero_tryread((unsigned*)&padded_str_len);
  //      struct AES_ctx** const ctxs_loc = (struct AES_ctx**)hero_tryread((unsigned*)&ctxs);
  //      uint8_t** const ciphers_loc = (uint8_t**)hero_tryread((unsigned*)&ciphers);

  //      // Allocate memory for local buffers.
  //      struct AES_ctx* ctx[7];
  //      uint8_t* cipher[7];
  //      for (unsigned i = 0; i < 7; ++i) {
  //        ctx[i] = (struct AES_ctx*)hero_l1malloc(sizeof(struct AES_ctx));
  //        if (ctx[i] == NULL)
  //          printf("Failed to allocate buffer for AES context!\n");
  //        cipher[i] = (uint8_t*)hero_l1malloc(str_len_loc * sizeof(uint8_t));
  //        if (cipher[i] == NULL)
  //          printf("Failed to allocate buffer for ciphertext!\n");
  //      }

  //      #pragma omp parallel for \
  //          firstprivate(plains_loc, n_strs_loc, str_len_loc, ctxs_loc, ciphers_loc, ctx, cipher)
  //      for (unsigned i = 0; i < n_strs_loc; ++i) {
  //        const unsigned tid = omp_get_thread_num();

  //        // Read plaintext and AES context into buffer.
  //        // TODO: double-buffering
  //        const uint8_t* const plain_ptr = (uint8_t*)hero_tryread((unsigned*)(plains_loc + i));
  //        struct AES_ctx* const ctx_ptr = (struct AES_ctx*)hero_tryread((unsigned*)(ctxs_loc + i));
  //        const hero_dma_job_t dma_plain
  //            = hero_dma_memcpy_async(cipher[tid], plain_ptr, str_len_loc * sizeof(uint8_t));
  //        const hero_dma_job_t dma_ctx_in
  //            = hero_dma_memcpy_async(ctx[tid], ctx_ptr, sizeof(struct AES_ctx));
  //        hero_dma_wait(dma_plain);
  //        hero_dma_wait(dma_ctx_in);

  //        AES_CBC_encrypt_buffer(ctx[i], cipher[i], str_len_loc);

  //        // Write ciphertext and AES context from buffer.
  //        uint8_t* const cipher_ptr = (uint8_t*)hero_tryread((unsigned*)(ciphers_loc + i));
  //        const hero_dma_job_t dma_cipher
  //            = hero_dma_memcpy_async(cipher_ptr, cipher[tid], str_len_loc * sizeof(uint8_t));
  //        const hero_dma_job_t dma_ctx_out
  //            = hero_dma_memcpy_async(ctx_ptr, ctx[tid], sizeof(struct AES_ctx));
  //        hero_dma_wait(dma_cipher);
  //        hero_dma_wait(dma_ctx_out);
  //      }

  //      // Deallocate local buffers.
  //      for (unsigned i = 0; i < 7; ++i) {
  //        hero_l1free(ctx[i]);
  //        hero_l1free(cipher[i]);
  //      }

  //      // Tell the miss-handler thread that we are done.
  //      sync = 1;
  //    }
  //  }
  //}
  bench_stop();

  // free memory
  free_byte_bufs(decrypteds, n_strs);
  free_byte_bufs(ciphers, n_strs);
  free_byte_bufs(plains, n_strs);
  free_AES_ctxs(ctxs, n_strs);
  free(key);

  return 0;
}
