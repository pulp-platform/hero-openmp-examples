#include <stddef.h> // size_t
#include <stdint.h> // uint32_t

int compare_matrices(const uint32_t* const a, const uint32_t* const b,
    const size_t n);

void init_matrices(uint32_t* const a, uint32_t* const b, uint32_t* const c,
    uint32_t* const d, const size_t n);

void alloc_matrices(uint32_t** a, uint32_t** b, uint32_t** c, uint32_t** d,
    const size_t n);
