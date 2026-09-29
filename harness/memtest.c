// User-space RAM pattern test (memtester-style). memtest <GiB> <passes> <threads>
// Each thread owns a slice; patterns: address-as-data, inverse, walking ones, xorshift random (write then verify).
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>

static uint64_t *buf; static size_t words; static int nthr, passes;
static volatile unsigned long long errors;

static inline uint64_t xs(uint64_t *s) { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; return *s; }

static void check(uint64_t *p, uint64_t exp, const char *pat, int pass) {
    if (*p != exp) {
        unsigned long long e = __sync_add_and_fetch(&errors, 1);
        if (e <= 20) fprintf(stderr, "ERR pass %d %s at %p: got %016llx want %016llx (xor %016llx)\n", pass, pat,
                             (void *)p, (unsigned long long)*p, (unsigned long long)exp, (unsigned long long)(*p ^ exp));
    }
}

static void *run(void *arg) {
    long t = (long)arg; size_t lo = words * t / nthr, hi = words * (t + 1) / nthr;
    for (int pass = 0; pass < passes; pass++) {
        for (size_t i = lo; i < hi; i++) buf[i] = (uint64_t)&buf[i];
        for (size_t i = lo; i < hi; i++) check(&buf[i], (uint64_t)&buf[i], "addr", pass);
        for (size_t i = lo; i < hi; i++) buf[i] = ~(uint64_t)&buf[i];
        for (size_t i = lo; i < hi; i++) check(&buf[i], ~(uint64_t)&buf[i], "~addr", pass);
        for (int b = 0; b < 64; b += 9) {
            uint64_t v = 1ULL << b;
            for (size_t i = lo; i < hi; i++) buf[i] = (i & 1) ? v : ~v;
            for (size_t i = lo; i < hi; i++) check(&buf[i], (i & 1) ? v : ~v, "walk", pass);
        }
        for (int r = 0; r < 3; r++) {
            uint64_t s = 0x9E3779B97F4A7C15ULL ^ (t * 0x1000193ULL + pass * 131 + r * 7919), s0 = s;
            for (size_t i = lo; i < hi; i++) buf[i] = xs(&s);
            s = s0;
            for (size_t i = lo; i < hi; i++) check(&buf[i], xs(&s), "rand", pass);
        }
        if (t == 0) { fprintf(stderr, "pass %d done, errors so far %llu\n", pass, (unsigned long long)errors); }
    }
    return NULL;
}

int main(int argc, char **argv) {
    size_t gib = atol(argv[1]); passes = atoi(argv[2]); nthr = atoi(argv[3]);
    words = gib * (1ULL << 30) / 8;
    buf = mmap(NULL, words * 8, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
    if (buf == MAP_FAILED) { perror("mmap"); return 2; }
    pthread_t th[64];
    for (long t = 0; t < nthr; t++) pthread_create(&th[t], NULL, run, (void *)t);
    for (int t = 0; t < nthr; t++) pthread_join(th[t], NULL);
    printf("RESULT %zu GiB x %d passes x %d threads: %llu errors\n", gib, passes, nthr, (unsigned long long)errors);
    return errors ? 1 : 0;
}
