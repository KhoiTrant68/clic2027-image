/* Interleaved rANS, byte-identical to clic27/entropy/rans.py (PREC = 16, L = 2^16, 16-bit renormalisation).
 *
 * Build: gcc -O3 -shared -fPIC -o _rans.so _rans.c   (done by clic27.gc.rans_c.build())
 * Stream: [u32 final state per lane, little endian][u16 words].
 * Symbols are assigned round-robin to lanes; step k covers symbols [k*lanes, min(n, (k+1)*lanes)).
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define PREC 16
#define RL (1u << 16)
#define MASK ((1u << PREC) - 1)

/* Returns number of bytes written to out (out must hold 4*lanes + 2*n + 16 bytes), or -1 on a zero-frequency symbol. */
int64_t rans_encode(const int64_t *sym, const int64_t *tab, int64_t n, const uint32_t *cdf, int64_t width1,
                    int64_t lanes, uint8_t *out) {
    uint64_t *x = (uint64_t *)malloc(sizeof(uint64_t) * (lanes ? lanes : 1));
    uint16_t *words = (uint16_t *)malloc(sizeof(uint16_t) * (n + 1));
    int64_t steps = (n + lanes - 1) / lanes;
    int64_t *chunk = (int64_t *)malloc(sizeof(int64_t) * (steps + 1));
    int64_t nw = 0;
    for (int64_t j = 0; j < lanes; j++) x[j] = RL;
    for (int64_t k = steps - 1; k >= 0; k--) {
        int64_t lo = k * lanes, hi = lo + lanes < n ? lo + lanes : n;
        chunk[k] = nw;
        for (int64_t i = lo; i < hi; i++) {
            const uint32_t *c = cdf + tab[i] * width1;
            uint64_t st = c[sym[i]], fr = (uint64_t)c[sym[i] + 1] - c[sym[i]];
            if (fr == 0) { free(x); free(words); free(chunk); return -1; }
            uint64_t xs = x[i - lo];
            if (xs >= (((uint64_t)RL >> PREC) << 16) * fr) { words[nw++] = (uint16_t)(xs & 0xFFFF); xs >>= 16; }
            x[i - lo] = ((xs / fr) << PREC) + xs % fr + st;
        }
    }
    chunk[steps] = nw;  /* chunk k spans [chunk[k], end_k) where end_k = start of the chunk written just before it */
    int64_t pos = 0;
    for (int64_t j = 0; j < lanes; j++) {
        uint32_t v = (uint32_t)x[j];
        memcpy(out + pos, &v, 4); pos += 4;
    }
    /* chunks were written for k = steps-1 ... 0; output them in order k = 0 ... steps-1 */
    for (int64_t k = 0; k < steps; k++) {
        int64_t s = chunk[k];
        int64_t e = (k == 0) ? nw : chunk[k - 1];
        for (int64_t i = s; i < e; i++) { memcpy(out + pos, &words[i], 2); pos += 2; }
    }
    free(x); free(words); free(chunk);
    return pos;
}

/* Returns 0 on success, -1 on integrity failure. */
int64_t rans_decode(const uint8_t *data, int64_t nbytes, const int64_t *tab, int64_t n, const uint32_t *cdf,
                    int64_t width1, int64_t lanes, int64_t *out) {
    if (nbytes < 4 * lanes) return -1;
    uint64_t *x = (uint64_t *)malloc(sizeof(uint64_t) * (lanes ? lanes : 1));
    for (int64_t j = 0; j < lanes; j++) { uint32_t v; memcpy(&v, data + 4 * j, 4); x[j] = v; }
    const uint8_t *w = data + 4 * lanes;
    int64_t nw = (nbytes - 4 * lanes) / 2, pos = 0;
    int64_t steps = (n + lanes - 1) / lanes;
    for (int64_t k = 0; k < steps; k++) {
        int64_t lo = k * lanes, hi = lo + lanes < n ? lo + lanes : n;
        for (int64_t i = lo; i < hi; i++) {
            const uint32_t *c = cdf + tab[i] * width1;
            uint64_t xs = x[i - lo], slot = xs & MASK;
            int64_t a = 0, b = width1 - 2;
            while (a < b) { int64_t m = (a + b + 1) >> 1; if (c[m] <= slot) a = m; else b = m - 1; }
            uint64_t st = c[a], fr = (uint64_t)c[a + 1] - c[a];
            xs = fr * (xs >> PREC) + slot - st;
            if (xs < RL) {
                if (pos >= nw) { free(x); return -1; }
                uint16_t v; memcpy(&v, w + 2 * pos, 2); pos++;
                xs = (xs << 16) | v;
            }
            x[i - lo] = xs;
            out[i] = a;
        }
    }
    int ok = pos == nw;
    for (int64_t j = 0; j < lanes; j++) ok &= x[j] == RL;
    free(x);
    return ok ? 0 : -1;
}
