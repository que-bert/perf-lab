// User-space RAM pattern test (memtester-style). memtest <GiB> <passes> <threads>
// Each thread owns a slice; patterns: address-as-data, inverse, walking ones, xorshift random (write then verify).
// Run as root to get physical addresses (via /proc/self/pagemap); unprivileged runs report phys 0.
// At the end, prints one memmap=4K$<page> kernel option per distinct bad physical page.
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

static uint64_t *buf; static size_t words; static int nthr, passes, pagemap_fd = -1;
static volatile unsigned long long errors;
static pthread_mutex_t bad_mu = PTHREAD_MUTEX_INITIALIZER;
static uint64_t bad_pages[256]; static uint64_t bad_bits[256]; static int nbad;

static inline uint64_t xs(uint64_t *s) { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; return *s; }

static uint64_t phys_of(const void *v) {
    uint64_t ent = 0;
    if (pagemap_fd < 0 || pread(pagemap_fd, &ent, 8, ((uintptr_t)v / 4096) * 8) != 8) return 0;
    if (!(ent >> 63) || !(ent & ((1ULL << 55) - 1))) return 0;
    return (ent & ((1ULL << 55) - 1)) * 4096 + (uintptr_t)v % 4096;
}

static void check(uint64_t *p, uint64_t exp, const char *pat, int pass) {
    uint64_t got = *p;
    if (got != exp) {
        unsigned long long e = __sync_add_and_fetch(&errors, 1);
        uint64_t pa = phys_of(p);
        if (e <= 40) fprintf(stderr, "ERR pass %d %s at %p phys %#llx: got %016llx want %016llx (xor %016llx)\n", pass,
                             pat, (void *)p, (unsigned long long)pa, (unsigned long long)got, (unsigned long long)exp,
                             (unsigned long long)(got ^ exp));
        pthread_mutex_lock(&bad_mu);
        int i = 0;
        while (i < nbad && bad_pages[i] != (pa & ~4095ULL)) i++;
        if (i == nbad && nbad < 256) bad_pages[nbad++] = pa & ~4095ULL;
        if (i < 256) bad_bits[i] |= got ^ exp;
        pthread_mutex_unlock(&bad_mu);
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
    // MAP_LOCKED keeps pages resident; with sysctl vm.compact_unevictable_allowed=0 they also stay put physically.
    buf = mmap(NULL, words * 8, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE | MAP_LOCKED, -1, 0);
    if (buf == MAP_FAILED) buf = mmap(NULL, words * 8, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
    if (buf == MAP_FAILED) { perror("mmap"); return 2; }
    pagemap_fd = open("/proc/self/pagemap", O_RDONLY);
    if (!phys_of(buf)) fprintf(stderr, "note: no physical addresses (run as root to get them)\n");
    // Optional extra args: known-bad physical addresses; report whether this run's buffer covers each one.
    for (int a = 4; a < argc && phys_of(buf); a++) {
        uint64_t want = strtoull(argv[a], NULL, 0) & ~4095ULL; size_t hit = 0;
        for (size_t off = 0; off < words * 8 && !hit; off += 4096)
            if ((phys_of((char *)buf + off) & ~4095ULL) == want) hit = off + 1;
        fprintf(stderr, "target %#llx: %s\n", (unsigned long long)want, hit ? "covered by this run" : "NOT covered");
    }
    pthread_t th[64];
    for (long t = 0; t < nthr; t++) pthread_create(&th[t], NULL, run, (void *)t);
    for (int t = 0; t < nthr; t++) pthread_join(th[t], NULL);
    printf("RESULT %zu GiB x %d passes x %d threads: %llu errors, %d distinct pages\n", gib, passes, nthr,
           (unsigned long long)errors, nbad);
    for (int i = 0; i < nbad; i++)
        printf("BAD page %#llx bits %016llx  memmap=4K$%#llx\n", (unsigned long long)bad_pages[i],
               (unsigned long long)bad_bits[i], (unsigned long long)bad_pages[i]);
    return errors ? 1 : 0;
}
