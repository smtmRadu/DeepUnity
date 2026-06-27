#ifndef DEEPUNITY_KVCACHE_INCLUDED
#define DEEPUNITY_KVCACHE_INCLUDED

// ============================================================================
// Shared KV-cache (de)quantization for ALL DeepUnity LLM compute shaders.
// Selected by a `#pragma multi_compile _ KV_FP16 KV_INT8` declared in the .compute.
// INCLUDE THIS *AFTER* the `head_dim` / `num_heads_kv` uniforms are declared
// (the INT8 readers derive (token,head) from the flat element index using them).
//
//   (no keyword) FP32 : K/V/kv_cache are StructuredBuffer<float>. 4 B/elem.
//   KV_FP16           : packed 2 halves / uint (f16tof32 / f32tof16). 2 B/elem, ~lossless.
//   KV_INT8           : asymmetric uint8, 4 / uint, + per-(token,head) fp16 scale&zp
//                       (packed scale|zp<<16). x ~= (q - zp) * scale. 1 B/elem.
//
// Call sites use KV_READ_K(e) / KV_READ_V(e) to read element e of the K / V cache,
// and (in WriteCache) KV_WRITE_* helpers. The K/V cache buffers and the matching
// scale/zp buffers are declared HERE so the two models never duplicate them.
// ============================================================================

#if defined(KV_FP16) || defined(KV_INT8)
    #define KV_PACKED 1
    StructuredBuffer<uint>      K;          // packed K cache (read)
    StructuredBuffer<uint>      V;          // packed V cache (read)
    RWStructuredBuffer<uint>    kv_cache;   // packed write target (bound to the K or V buffer per dispatch)
#else
    StructuredBuffer<float>     K;
    StructuredBuffer<float>     V;
    RWStructuredBuffer<float>   kv_cache;
#endif
StructuredBuffer<float>         kv_new;     // freshly-computed K or V for the new tokens (fp32 activations)

// ----------------------------------------------------------------- FP16 path
#if defined(KV_FP16)
float kv_unpack16(StructuredBuffer<uint> buf, uint e)
{
    uint w = buf[e >> 1];
    return f16tof32((e & 1u) == 0u ? (w & 0xffffu) : (w >> 16));
}
#define KV_READ_K(e) kv_unpack16(K, (e))
#define KV_READ_V(e) kv_unpack16(V, (e))

// pack two consecutive (even-aligned) cache elements v0,v1 at flat index `e` (e&1==0) into one uint
#define KV_WRITE2(buf, e, v0, v1) (buf)[(e) >> 1] = (f32tof16(v0) | (f32tof16(v1) << 16))

// ----------------------------------------------------------------- INT8 path
#elif defined(KV_INT8)
StructuredBuffer<uint>          k_scale_zp;   // [tokens * num_heads_kv] : (scale_half | zp_half<<16)
StructuredBuffer<uint>          v_scale_zp;

// dequantize element e of an asymmetric-uint8 cache `buf` with per-(token,head) scale/zp `sz`.
// (token,head) = e / head_dim (the cache is laid out [pos, head, d], so e/head_dim selects the row).
float kv_unpack8(StructuredBuffer<uint> buf, StructuredBuffer<uint> sz, uint e)
{
    uint q  = (buf[e >> 2] >> ((e & 3u) * 8u)) & 0xffu;     // unpack uint8 (4 / uint)
    uint w  = sz[e / head_dim];                             // (token,head) row
    float scale = f16tof32(w & 0xffffu);
    float zp    = f16tof32(w >> 16);
    return ((float)q - zp) * scale;
}
#define KV_READ_K(e) kv_unpack8(K, k_scale_zp, (e))
#define KV_READ_V(e) kv_unpack8(V, v_scale_zp, (e))

// WRITE side. The quantizing WriteCache (per-(token,head) min/max reduction + pack) writes the
// scale/zp through this RW alias — bound per dispatch to the same buffer the matching read uses
// (k_scale_zp for the K write, v_scale_zp for the V write). The uint8 cache itself is written
// through the shared `kv_cache` RW buffer (declared above). The read above (kv_unpack8) is the
// single source of truth for the layout this MUST reproduce: element e lives in byte (e&3) of
// uint e>>2, and its (token,head) scale/zp lives at sz[e / head_dim]. Because head_dim is a
// multiple of 4, a uint (4 consecutive head-dim elements 4u..4u+3) never straddles a row, so
// each thread can own one whole uint and overwrite it outright — no read-modify-write race.
RWStructuredBuffer<uint>        kv_scale_zp_w;   // bound to k_scale_zp or v_scale_zp for the write

// Pack a scale and a zero-point (both fp32, zp already rounded+clamped to [0,255]) into one uint
// exactly as kv_unpack8 unpacks it: low half = scale, high half = zp.
#define KV_PACK_SCALEZP(scale, zp) (f32tof16(scale) | (f32tof16((float)(zp)) << 16))

// Threads per group for the INT8 quantizing WriteCache: ONE group per (token, kv-head), the group
// cooperatively reduces min/max over the head_dim values then packs them 4-per-uint. 64 threads ==
// head_dim/4 for head_dim 256 (one uint per thread); the kernels still loop so any head_dim works.
// Used as the [numthreads] of WriteCacheFull / WriteCacheSliding in the INT8 variant (the FP32/FP16
// variants of those kernels keep their own element-per-thread [numthreads(256)]).
#define KV_WRITE_INT8_THREADS 64

// ----------------------------------------------------------------- FP32 path
#else
#define KV_READ_K(e) (K[e])
#define KV_READ_V(e) (V[e])
#define KV_WRITE1(buf, e, v) (buf)[e] = (v)
#endif

#endif // DEEPUNITY_KVCACHE_INCLUDED
