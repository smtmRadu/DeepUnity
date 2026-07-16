#!/usr/bin/env python3
"""
Kokoro-82M voice cloning by STYLE-VECTOR SEARCH. Kokoro has no reference-audio cloning —
a "voice" is just a [510, 1, 256] voicepack of per-phoneme-length style rows (StyleTTS2
split: [:128] acoustic/timbre for the decoder, [128:] prosody for the predictor). Cloning
a target speaker therefore = searching the 256-dim style vector whose SYNTHESIS sounds
like the reference audio.

A single speaker-embedding cosine is trivially GAMED by the optimizer (0.93 "similarity"
that sounds nothing like the target). Fitness here follows kvoicewalk
(github.com/RobViren/kvoicewalk): the HARMONIC mean of three (0,1] terms, so no component
can collapse while the score climbs:
  t  TARGET   ensemble cosine (Resemblyzer GE2E + speechbrain ECAPA-TDNN) between the
              reference and 3 different probe texts synthesized with the candidate,
              RMS-normalized before embedding (embedders are level-sensitive);
  s  SELF     mean pairwise cosine between the candidate's 3 probe renditions — a real
              voice sounds like itself across texts, embedder-gaming degenerates don't;
  f  FEATURE  1/(1+d), d = scaled distance of low-level stats vs the reference (voiced
              log-F0 mean/std, log spectral centroid/rolloff, MFCC mean/std) — blocks
              "passes the checks, sounds like a metal basket of tools down the stairs".
ANTI-HACK GATE: WavLM-base-plus-sv is held OUT of the loss entirely; after fitting it
scores init vs final on the unseen test paragraph (report.json holdout_wavlm.improved —
if false, the ensemble was hacked and the run is honest about it).

Search: (1) rank all stock voicepacks by ensemble target cosine; (2) INTERPOLATION START —
random simplex blends of the top-5 packs (perfectly on-manifold), best blend = CMA init;
(3) sep-CMA-ES over a whitened offset (vector = blend + z * per-dim std of stock packs,
weak L2 anchor to the blend) with restarts-on-plateau, best-so-far kept across restarts,
until --budget fitness evals or --max-wall seconds.

Outputs into --out: style_256.npy, voicepack_510x256.pt (row tiled to the stock
[510,1,256] shape -> drops into KPipeline(voice=tensor) and import_kokoro.py),
reference.wav / init_sample.wav / cloned_sample.wav (A/B, same test paragraph),
report.json (ranking, blend, fitness parts, holdout verdict, history, wall time).

Runs on WSL in the `kokoro` env (kokoro 0.9.4 + misaki + resemblyzer + speechbrain +
librosa + cma; torchaudio must be the +cu128 build):
    python clone_voice.py /mnt/c/.../Ansbach_4-15s.mp3 --device cuda [--budget 2200]
"""
import argparse, itertools, json, os, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.normpath(os.path.join(HERE, *[".."] * 6))   # .../DeepUnity repo root
SR_KOKORO, SR_EMB, RMS = 24000, 16000, 0.05

# The 28 stock English v1.0 voicepacks (hexgrad/Kokoro-82M/voices, VOICES.md). bf_/bm_
# packs are usable in the lang_code="a" pipeline (packs are language-blind).
VOICES = ("af_alloy af_aoede af_bella af_heart af_jessica af_kore af_nicole af_nova "
          "af_river af_sarah af_sky am_adam am_echo am_eric am_fenrir am_liam am_michael "
          "am_onyx am_puck am_santa bf_alice bf_emma bf_isabella bf_lily bm_daniel "
          "bm_fable bm_george bm_lewis").split()

# 3 phonetically diverse fitting probes (each must stay ONE KPipeline chunk) and the A/B
# test paragraph (same lighthouse passage as bench_reference.py).
PROBES = ["The old lighthouse keeper climbed the spiral stairs every evening at dusk.",
          "Nobody expected the ancient door to open, yet it swung wide without a sound.",
          "Seven wild geese flew over the frozen river just before the storm arrived."]
TEXT = ("The old lighthouse keeper climbed the spiral stairs every evening at dusk. "
        "He lit the great lamp and watched the beam sweep across the darkening waves; "
        "ships far at sea counted on that light to find their way home safely.")


def rms_norm(wav):
    w = np.asarray(wav, np.float32)
    return w * (RMS / max(float(np.sqrt(np.mean(w * w))), 1e-5))


def features(wav24):
    """Low-level stats @24k: voiced log-F0 mean/std, log centroid/rolloff mean,
    MFCC c1..c12 mean+std. NaN-free; degenerate audio -> zeros (huge distance)."""
    import librosa
    f0 = librosa.yin(wav24, fmin=50, fmax=400, sr=SR_KOKORO, frame_length=2048, hop_length=512)
    rms = librosa.feature.rms(y=wav24, frame_length=2048, hop_length=512)[0]
    n = min(len(f0), len(rms))
    voiced = (rms[:n] > 0.3 * np.median(rms[:n])) & (f0[:n] > 52) & (f0[:n] < 392)
    lf = np.log(f0[:n][voiced]) if voiced.sum() >= 5 else np.zeros(1)
    cen = np.log(librosa.feature.spectral_centroid(y=wav24, sr=SR_KOKORO).clip(min=1.0))
    rol = np.log(librosa.feature.spectral_rolloff(y=wav24, sr=SR_KOKORO).clip(min=1.0))
    mf = librosa.feature.mfcc(y=wav24, sr=SR_KOKORO, n_mfcc=13)[1:]
    return {"f0m": float(lf.mean()), "f0s": float(lf.std()), "cen": float(cen.mean()),
            "rol": float(rol.mean()), "mfm": mf.mean(axis=1), "mfs": mf.std(axis=1)}


def feat_dist(c, r):
    """Scaled feature distance (each term ~1 for a clearly different voice)."""
    return float(np.mean([abs(c["f0m"] - r["f0m"]) / 0.35, abs(c["f0s"] - r["f0s"]) / 0.15,
                          abs(c["cen"] - r["cen"]) / 0.40, abs(c["rol"] - r["rol"]) / 0.40,
                          np.linalg.norm(c["mfm"] - r["mfm"]) / 50.0,
                          np.linalg.norm(c["mfs"] - r["mfs"]) / 25.0]))


def main():
    ap = argparse.ArgumentParser(description="Kokoro voice cloning by CMA-ES style search")
    ap.add_argument("reference_audio", help="target speaker sample (mp3/wav, >=5 s)")
    ap.add_argument("--out", default=None, help="output dir (default ProbeLogs/voice_clone_<ref>)")
    ap.add_argument("--text", default=TEXT, help="A/B test paragraph for the output samples")
    ap.add_argument("--budget", type=int, default=2200, help="max fitness evals (incl. blends)")
    ap.add_argument("--max-wall", type=float, default=2400, help="max search seconds")
    ap.add_argument("--interp", type=int, default=200, help="top-5 simplex blends to try as init")
    ap.add_argument("--popsize", type=int, default=16)
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--sigma0", type=float, default=0.5, help="CMA sigma in whitened units")
    ap.add_argument("--anchor", type=float, default=0.005, help="L2 weight on offset from blend")
    ap.add_argument("--patience", type=int, default=15, help="plateau iters before CMA restart")
    ap.add_argument("--restarts", type=int, default=8, help="max CMA restarts")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--voices", default=None, help="comma list overriding the stock-voice scan")
    a = ap.parse_args()

    import cma, librosa, soundfile as sf, torch
    from loguru import logger
    logger.remove()                                     # silence kokoro's per-voice chatter
    from kokoro import KPipeline
    from resemblyzer import VoiceEncoder, preprocess_wav
    from speechbrain.inference.speaker import EncoderClassifier

    refname = os.path.splitext(os.path.basename(a.reference_audio))[0]
    out = a.out or os.path.join(PROJECT, "ProbeLogs", f"voice_clone_{refname}")
    os.makedirs(out, exist_ok=True)
    t_start = time.perf_counter()
    rng = np.random.default_rng(a.seed)
    print(f"[out]    {out}")

    # ---- embedders (fitness ensemble; WavLM stays OUT until the holdout gate) -----------
    res_enc = VoiceEncoder(a.device, verbose=False)
    ecapa = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.path.expanduser("~/.cache/speechbrain/spkrec-ecapa-voxceleb"),
        run_opts={"device": f"{a.device}:0" if a.device == "cuda" else a.device})

    def embed_res(wav24):
        w = preprocess_wav(rms_norm(wav24), source_sr=SR_KOKORO)
        if len(w) < 8000:                               # VAD ate (almost) everything
            w = librosa.resample(rms_norm(wav24), orig_sr=SR_KOKORO, target_sr=SR_EMB)
        return res_enc.embed_utterance(w)               # L2-normalized [256]

    def embed_ecapa(wavs24):                            # list -> [B, 192] L2-normalized
        ws = [librosa.resample(rms_norm(w), orig_sr=SR_KOKORO, target_sr=SR_EMB) for w in wavs24]
        T = max(len(w) for w in ws)
        batch = torch.zeros(len(ws), T)
        for i, w in enumerate(ws):
            batch[i, :len(w)] = torch.from_numpy(w)
        lens = torch.tensor([len(w) / T for w in ws])
        e = ecapa.encode_batch(batch, wav_lens=lens).squeeze(1).cpu().numpy()
        return e / np.linalg.norm(e, axis=1, keepdims=True)

    # ---- reference: 24 kHz wav copy + ensemble embeddings + feature stats ---------------
    ref24, _ = librosa.load(a.reference_audio, sr=SR_KOKORO, mono=True)
    sf.write(os.path.join(out, "reference.wav"), ref24, SR_KOKORO)
    ref_res, ref_eca, ref_feat = embed_res(ref24), embed_ecapa([ref24])[0], features(ref24)
    print(f"[ref]    {a.reference_audio}  {len(ref24)/SR_KOKORO:.1f}s  "
          f"logF0 {ref_feat['f0m']:.2f}±{ref_feat['f0s']:.2f}  device {a.device}")

    # ---- pipeline + fixed probe phonemes (G2P once, then raw model() per candidate) -----
    pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", device=a.device)
    model = pipeline.model
    probe_ps = [list(pipeline(p, voice="af_heart"))[0].phonemes for p in PROBES]
    rows = [len(ps) - 1 for ps in probe_ps]             # voicepack rows KPipeline would use

    def synth(ps, vec):
        """probe audio (np float32 @24k) for one 256-dim style vector, deterministic."""
        torch.manual_seed(a.seed)                       # freeze NSF/SineGen random phases
        ref_s = torch.from_numpy(np.asarray(vec, np.float32)).reshape(1, 256)
        return model(ps, ref_s).numpy()

    evals = [0]

    def fitness(vec):
        """harmonic_mean(target, self, feature) in (0,1] + parts, over the 3 probes."""
        evals[0] += 1
        wavs = [synth(ps, vec) for ps in probe_ps]
        res = [embed_res(w) for w in wavs]
        eca = embed_ecapa(wavs)
        t_cos = float(np.mean([ref_res @ e for e in res]) + np.mean(eca @ ref_eca)) / 2
        pair = list(itertools.combinations(range(len(wavs)), 2))
        s_cos = float(np.mean([res[i] @ res[j] for i, j in pair])
                      + np.mean([eca[i] @ eca[j] for i, j in pair])) / 2
        fe = [features(w) for w in wavs]
        favg = {k: np.mean([f[k] for f in fe], axis=0) for k in fe[0]}
        t, s = max(t_cos, 0.05), max(s_cos, 0.05)
        f = 1.0 / (1.0 + feat_dist(favg, ref_feat))
        return 3.0 / (1 / t + 1 / s + 1 / f), {"t": round(t_cos, 4), "s": round(s_cos, 4),
                                               "f": round(f, 4)}

    # ---- 1) rank stock voicepacks by ensemble target cosine (probe 0 only, cheap) -------
    names = a.voices.split(",") if a.voices else VOICES
    packs, scores = {}, []
    for name in names:
        try:
            pack = pipeline.load_voice(name)            # [510, 1, 256], cpu
        except Exception as e:                          # offline / unknown voice
            print(f"[scan]   {name}: SKIP ({e})")
            continue
        packs[name] = pack
        wav = synth(probe_ps[0], pack[rows[0]].numpy())
        sim = (float(ref_res @ embed_res(wav)) + float(embed_ecapa([wav])[0] @ ref_eca)) / 2
        scores.append((sim, name))
    scores.sort(reverse=True)
    assert scores, "no voicepacks could be scored"
    for sim, name in scores[:5]:
        print(f"[scan]   {sim:.4f}  {name}")
    init_name = scores[0][1]
    vertex = {n: np.mean([packs[n][r].numpy().reshape(256) for r in rows], axis=0)
              .astype(np.float64) for _, n in scores[:5]}

    # ---- 2) interpolation start: simplex blends of the top-5 (stay on-manifold) ---------
    top5 = [n for _, n in scores[:5]]
    cands = [np.eye(5)[i] for i in range(5)]                        # pure voices
    cands += [(np.eye(5)[i] + np.eye(5)[j]) / 2 for i, j in itertools.combinations(range(5), 2)]
    cands += list(rng.dirichlet(np.ones(5), size=max(a.interp - len(cands), 0)))
    V = np.stack([vertex[n] for n in top5])                         # [5, 256]
    best_H, best_vec, best_parts, stock_best = -1.0, None, None, None
    blend_w = None
    for k, w in enumerate(cands):
        H, parts = fitness(w @ V)
        if k < 5 and (stock_best is None or H > stock_best[0]):
            stock_best = (H, top5[k], parts)
        if H > best_H:
            best_H, best_vec, best_parts, blend_w = H, w @ V, parts, w
    blend = {"H": round(best_H, 4), **best_parts,
             "weights": {n: round(float(w), 3) for n, w in zip(top5, blend_w) if w > 0.01}}
    print(f"[stock]  H {stock_best[0]:.4f}  {stock_best[1]}  {stock_best[2]}")
    print(f"[blend]  H {blend['H']:.4f}  {best_parts}  {blend['weights']}")
    blend_vec, blend_H = best_vec.copy(), best_H

    # ---- 3) sep-CMA-ES in whitened offset space, restarts on plateau --------------------
    dim_std = np.stack([p[rows[0]].numpy().reshape(256) for p in packs.values()]) \
                .std(axis=0).astype(np.float64).clip(min=1e-3)

    def cost(vec):
        H, parts = fitness(vec)
        z_total = (vec - blend_vec) / dim_std
        return -H + a.anchor * float(np.mean(z_total * z_total)), H, parts

    history, it, n_restart = [], 0, 0
    for n_restart in range(a.restarts + 1):
        if evals[0] >= a.budget or time.perf_counter() - t_start >= a.max_wall:
            break
        center = best_vec.copy()                        # restart centered on best-so-far
        es = cma.CMAEvolutionStrategy(np.zeros(256), a.sigma0 * 0.75 ** n_restart, {
            "popsize": a.popsize, "seed": a.seed + 1 + n_restart,
            "CMA_diagonal": True, "verbose": -9})
        stale = 0
        while stale < a.patience:
            if evals[0] >= a.budget or time.perf_counter() - t_start >= a.max_wall:
                break
            it += 1
            zs = es.ask()
            trip = [cost(center + z * dim_std) for z in zs]
            es.tell(zs, [c for c, _, _ in trip])
            i = int(np.argmax([H for _, H, _ in trip]))
            if trip[i][1] > best_H + 1e-4:
                best_H, best_vec, best_parts = trip[i][1], center + zs[i] * dim_std, trip[i][2]
                stale = 0
            else:
                stale += 1
            history.append(round(best_H, 4))
            print(f"[iter {it:3d}] r{n_restart} evals {evals[0]:4d}  best H {best_H:.4f}  "
                  f"{best_parts}  sigma {es.sigma:.3f}", flush=True)
        else:
            print(f"[restart] plateau ({a.patience} iters) -> restart {n_restart + 1}")
            continue
        break                                           # budget/wall exhausted

    # ---- 4) outputs ----------------------------------------------------------------------
    vec32 = best_vec.astype(np.float32)
    np.save(os.path.join(out, "style_256.npy"), vec32)
    pack = torch.from_numpy(np.tile(vec32, (510, 1, 1)))            # stock [510, 1, 256]
    torch.save(pack, os.path.join(out, "voicepack_510x256.pt"))
    blend_pack = torch.from_numpy(np.tile(blend_vec.astype(np.float32), (510, 1, 1)))

    def speak(voice):                                   # test paragraph through KPipeline
        torch.manual_seed(a.seed)
        return np.concatenate([r.audio.numpy() for r in pipeline(a.text, voice=voice)])

    test_wavs, test_sims = {}, {}
    for fname, voice, key in (("init_sample.wav", init_name, "init"),
                              (None, blend_pack, "blend"),
                              ("cloned_sample.wav", pack, "cloned")):
        wav = speak(voice)
        test_wavs[key] = wav
        if fname:
            sf.write(os.path.join(out, fname), wav, SR_KOKORO)
        test_sims[key] = round((float(ref_res @ embed_res(wav))
                                + float(embed_ecapa([wav])[0] @ ref_eca)) / 2, 4)

    # ---- 5) anti-hacking gate: held-out WavLM-base-plus-sv on the unseen test text ------
    from transformers import AutoFeatureExtractor, WavLMForXVector
    fe_w = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
    wavlm = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(a.device).eval()

    def wavlm_embed(wavs24):
        ws = [librosa.resample(rms_norm(w), orig_sr=SR_KOKORO, target_sr=SR_EMB) for w in wavs24]
        inp = fe_w(ws, sampling_rate=SR_EMB, return_tensors="pt", padding=True)
        with torch.no_grad():
            e = wavlm(input_values=inp.input_values.to(a.device),
                      attention_mask=inp.attention_mask.to(a.device)).embeddings.cpu().numpy()
        return e / np.linalg.norm(e, axis=1, keepdims=True)

    hw = wavlm_embed([ref24, test_wavs["init"], test_wavs["blend"], test_wavs["cloned"]])
    holdout = {"init": round(float(hw[0] @ hw[1]), 4), "blend": round(float(hw[0] @ hw[2]), 4),
               "final": round(float(hw[0] @ hw[3]), 4)}
    holdout["improved"] = bool(holdout["final"] > holdout["init"] + 0.01)

    wall = time.perf_counter() - t_start
    report = {
        "reference": a.reference_audio, "device": a.device, "torch": torch.__version__,
        "top5": [{"voice": n, "target_cos": round(s, 4)} for s, n in scores[:5]],
        "init_voice": init_name, "blend": blend,
        "fitness": {"stock_best": {"H": round(stock_best[0], 4), "voice": stock_best[1],
                                   **stock_best[2]},
                    "blend_H": round(blend_H, 4),
                    "final": {"H": round(best_H, 4), **best_parts}},
        "manifold_ceiling": bool(best_H - blend_H < 0.01),
        "test_sims_ensemble": test_sims, "holdout_wavlm": holdout,
        "evals": evals[0], "iterations": it, "restarts": n_restart, "popsize": a.popsize,
        "budget": a.budget, "sigma0": a.sigma0, "anchor": a.anchor,
        "wall_s": round(wall, 1), "history": history,
    }
    with open(os.path.join(out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print(f"\nDone - stock H {stock_best[0]:.4f} ({init_name}) -> blend {blend_H:.4f} -> "
          f"fitted {best_H:.4f} | test ensemble {test_sims['init']} -> {test_sims['cloned']} | "
          f"holdout WavLM {holdout['init']} -> {holdout['final']} "
          f"({'improved' if holdout['improved'] else 'NOT improved - ensemble hacked?'}) | "
          f"{evals[0]} evals, {wall:.0f}s -> {out}")


if __name__ == "__main__":
    main()
