#!/usr/bin/env python3
# =============================================================================
# reproduce_results.py
#
# PAPER TYPE CLASSIFICATION (D10 STEP 0):
#   [BENCH] Benchmark / measurement study — measures real cryptographic
#           library implementations (PQClean via pqcrypto 0.4.0; OpenSSL via
#           cryptography 46.0.6), plus an algorithmic selection framework
#           evaluated directly on the measured data.
#
# D10 ITEM MAPPING ([BENCH] column):
#   - "Download/load dataset"      -> import/install the systems under test
#                                     (pip install pqcrypto cryptography numpy
#                                      matplotlib adjustText) — no external
#                                      dataset. adjustText is optional.
#   - "Preprocessing"              -> warm-up runs + timer calibration.
#   - "Define model architecture"  -> define the schemes/suites under test.
#   - "Train model"                -> run benchmark iterations (no ML model
#                                     is trained anywhere in this paper; no
#                                     training loops, ROC curves, confusion
#                                     matrices, or loss curves exist, per the
#                                     CRITICAL RULE in D10 Step 1).
#   - "Evaluate on test set"       -> collect timing / size / memory
#                                     measurements.
#   - "Metrics (tables)"           -> latency distributions, bytes, RSS,
#                                     selection outcomes.
#   - "Figures"                    -> box plots, stacked bars, Pareto
#                                     scatter, tail-latency scatter.
#
# Every numerical value in the paper's Results tables is printed to stdout
# by this script, labeled "Table N | ...", and saved under results/*.csv.
# Every figure in the paper is saved under figures/*.png.
#
# Usage:  python reproduce_results.py

# Classification and mapping. The study is a benchmark/measurement paper: there is no dataset to download and no model to train. The pipeline’s stages map onto the standard experimental skeleton as follows: system-under-test installation replaces dataset acquisition (pip install pqcrypto cryptography numpy matplotlib, plus the optional adjustText for figure-label placement); warm-up and timer calibration replace preprocessing; the suite and profile definitions replace model specification; the timed measurement loops replace training and evaluation; and the CSV/figure emission stages produce the tables and figures. No machine-learning scaffolding of any kind (training curves, confusion matrices, ROC analyses) exists anywhere in the pipeline, because none would be meaningful for this study type.
# Environment. Ubuntu 24.04 (x86-64), Python 3.12.3, single logical CPU (Intel Xeon at 2.10 GHz), 3.9 GB RAM; package versions pqcrypto 0.4.0 (PQClean reference-C implementations), cryptography 46.0.6 (OpenSSL backend), numpy 2.4.4, matplotlib 3.10.8. The pipeline writes results/platform_manifest.json recording the platform, package versions, timer resolution (1 ns), random seed (20260716), repetition-tier policy, bootstrap configuration, memory-measurement method, and every link-rate and payload scenario parameter, so that any re-run is self-describing.
# Invocation and outputs. A single command, python reproduce_results.py, executes all stages in order with progress printed per stage and per algorithm; total runtime on the reference platform is roughly 200 s, dominated by SPHINCS+-128s signing and the per-algorithm memory subprocesses. Every value reported in Tables 2–9 is printed to standard output with an explicit Table N | prefix and simultaneously persisted to results/ as sizes.csv, kem_bench.csv, sig_bench.csv, memory.csv, handshake.csv, selection.csv, baselines.csv, ablation.csv, and robustness.csv, together with raw_samples.npz (every individual timing sample, for audit and re-analysis) and selection_rankings.json (full per-profile score rankings). Figures 1–5 are written to figures/ as 300-dpi PNG files by the same run. Because timing results are platform-dependent, exact latency values will differ across machines while the size and fragmentation values (Table 2; byte columns of Table 6) reproduce exactly; the selection outcomes are expected to reproduce on comparable hardware, and the weight-sweep robustness results indicate how much latitude they have.
# Determinism and statistics. Bootstrap confidence intervals are seeded and exactly reproducible given identical samples; timing samples themselves are inherently stochastic, which is why the raw archive ships with the summary statistics. The adaptive repetition tiers are recorded per row in the n column of the benchmark CSVs.


# =============================================================================

import csv
import gc
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from itertools import product

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
try:                                    # optional: collision-free figure labels
    from adjustText import adjust_text
    _HAVE_ADJUSTTEXT = True
except ImportError:
    _HAVE_ADJUSTTEXT = False


def _declutter_labels(ax, texts):
    """Spread annotation labels so they do not overlap each other or the data
    markers, adding thin leader lines from each label back to its point. Uses
    adjustText when installed; otherwise the labels remain at their points."""
    if texts and _HAVE_ADJUSTTEXT:
        adjust_text(texts, ax=ax,
                    arrowprops=dict(arrowstyle="-", color="0.45", lw=0.7),
                    expand=(1.7, 2.1), force_text=(0.7, 1.1),
                    force_static=(0.35, 0.55),
                    only_move={"static": "xy", "text": "xy", "explode": "xy"})


# ----------------------------- Configuration --------------------------------

SEED = 20260716                  # RNG seed for bootstrap resampling
WARMUP = 5                       # warm-up iterations before timing
BOOTSTRAP_B = 2000               # bootstrap resamples for median 95% CI
MSG = b"\x5a" * 64               # 64-byte message (transcript-hash-sized)
OUT_RESULTS = "results"
OUT_FIGURES = "figures"

# Adaptive repetition tiers: (single-shot upper bound in ms, repetitions)
REP_TIERS = [(1.0, 400), (10.0, 150), (100.0, 40), (1000.0, 15), (float("inf"), 8)]

# Link-layer scenario parameters (stated as such in the paper; these are
# INPUTS to the analysis, not measured results):
LINK_RATES_BPS = {               # nominal application-visible data rates
    "ethernet_100M": 100e6,
    "lte_m_1M": 1e6,
    "wpan_250k": 250e3,
    "nb_iot_62k5": 62.5e3,
}
PAYLOAD_IPV6 = 1232              # 1280 B IPv6 minimum MTU - 40 (IPv6) - 8 (UDP)
PAYLOAD_6LOWPAN = 81             # conservative usable payload per 802.15.4 frame

rng = np.random.default_rng(SEED)

# ----------------------------- Systems under test ---------------------------

from pqcrypto.kem import (
    ml_kem_512, ml_kem_768, ml_kem_1024,
    hqc_128, hqc_192, hqc_256,
    mceliece348864,
)
from pqcrypto.sign import (
    ml_dsa_44, ml_dsa_65, ml_dsa_87,
    falcon_512, falcon_1024,
    sphincs_sha2_128s_simple, sphincs_sha2_128f_simple,
)
from cryptography.hazmat.primitives.asymmetric import x25519, ec, ed25519, rsa, padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

# --- Classical KEM adapters (DHKEM-style, cf. RFC 9180 construction) --------

def x25519_keygen():
    sk = x25519.X25519PrivateKey.generate()
    pk = sk.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return pk, sk

def x25519_encaps(pk_bytes):
    eph = x25519.X25519PrivateKey.generate()
    ss = eph.exchange(x25519.X25519PublicKey.from_public_bytes(pk_bytes))
    ct = eph.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return ct, ss

def x25519_decaps(sk, ct):
    return sk.exchange(x25519.X25519PublicKey.from_public_bytes(ct))

def _hkdf(ss):
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None,
                info=b"hybrid-kem").derive(ss)

def hybrid_keygen():
    pk_x, sk_x = x25519_keygen()
    pk_k, sk_k = ml_kem_768.generate_keypair()
    return (pk_x + pk_k), (sk_x, sk_k)

def hybrid_encaps(pk):
    pk_x, pk_k = pk[:32], pk[32:]
    ct_x, ss_x = x25519_encaps(pk_x)
    ct_k, ss_k = ml_kem_768.encrypt(pk_k)
    return (ct_x + ct_k), _hkdf(ss_x + ss_k)

def hybrid_decaps(sk, ct):
    sk_x, sk_k = sk
    ct_x, ct_k = ct[:32], ct[32:]
    ss_x = x25519_decaps(sk_x, ct_x)
    ss_k = ml_kem_768.decrypt(sk_k, ct_k)
    return _hkdf(ss_x + ss_k)

# --- Classical signature adapters -------------------------------------------

def ed25519_keygen():
    sk = ed25519.Ed25519PrivateKey.generate()
    pk = sk.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return pk, sk

def ecdsa_keygen():
    sk = ec.generate_private_key(ec.SECP256R1())
    pk = sk.public_key().public_bytes(
        serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)
    return pk, sk

def rsa_keygen():
    sk = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pk = sk.public_key().public_bytes(
        serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
    return pk, sk

# KEM registry: name -> (keygen, encaps, decaps, NIST_level, family)
# level 0 = classically secure only (no quantum resistance)
KEMS = {
    "ML-KEM-512":      (ml_kem_512.generate_keypair,  ml_kem_512.encrypt,  ml_kem_512.decrypt,  1, "lattice"),
    "ML-KEM-768":      (ml_kem_768.generate_keypair,  ml_kem_768.encrypt,  ml_kem_768.decrypt,  3, "lattice"),
    "ML-KEM-1024":     (ml_kem_1024.generate_keypair, ml_kem_1024.encrypt, ml_kem_1024.decrypt, 5, "lattice"),
    "HQC-128":         (hqc_128.generate_keypair,     hqc_128.encrypt,     hqc_128.decrypt,     1, "code"),
    "HQC-192":         (hqc_192.generate_keypair,     hqc_192.encrypt,     hqc_192.decrypt,     3, "code"),
    "HQC-256":         (hqc_256.generate_keypair,     hqc_256.encrypt,     hqc_256.decrypt,     5, "code"),
    "McEliece-348864": (mceliece348864.generate_keypair, mceliece348864.encrypt, mceliece348864.decrypt, 1, "code"),
    "X25519":          (x25519_keygen,                x25519_encaps,       x25519_decaps,       0, "classical"),
    "X25519+ML-KEM-768": (hybrid_keygen,              hybrid_encaps,       hybrid_decaps,       3, "hybrid"),
}

# Signature registry: name -> (keygen, sign, verify, NIST_level, family)
def _mk_sign(mod):
    return (mod.generate_keypair, mod.sign, mod.verify)

SIGS = {
    "ML-DSA-44":        (*_mk_sign(ml_dsa_44),  2, "lattice"),
    "ML-DSA-65":        (*_mk_sign(ml_dsa_65),  3, "lattice"),
    "ML-DSA-87":        (*_mk_sign(ml_dsa_87),  5, "lattice"),
    "Falcon-512":       (*_mk_sign(falcon_512), 1, "lattice"),
    "Falcon-1024":      (*_mk_sign(falcon_1024),5, "lattice"),
    "SPHINCS+-128s":    (*_mk_sign(sphincs_sha2_128s_simple), 1, "hash"),
    "SPHINCS+-128f":    (*_mk_sign(sphincs_sha2_128f_simple), 1, "hash"),
    "Ed25519":  (ed25519_keygen,
                 lambda sk, m: sk.sign(m),
                 lambda pk, m, s: ed25519.Ed25519PublicKey.from_public_bytes(pk).verify(s, m),
                 0, "classical"),
    "ECDSA-P256": (ecdsa_keygen,
                   lambda sk, m: sk.sign(m, ec.ECDSA(hashes.SHA256())),
                   lambda pk, m, s: ec.EllipticCurvePublicKey.from_encoded_point(
                       ec.SECP256R1(), pk).verify(s, m, ec.ECDSA(hashes.SHA256())),
                   0, "classical"),
    "RSA-2048": (rsa_keygen,
                 lambda sk, m: sk.sign(m, padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                                                      salt_length=32), hashes.SHA256()),
                 lambda pk, m, s: serialization.load_der_public_key(pk).verify(
                     s, m, padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                                       salt_length=32), hashes.SHA256()),
                 0, "classical"),
}

# ----------------------------- Benchmark harness ----------------------------

def choose_reps(single_shot_ms):
    for bound, reps in REP_TIERS:
        if single_shot_ms < bound:
            return reps
    return 8

def bench(fn, *args):
    """Time fn(*args) with adaptive repetitions; returns list of seconds."""
    for _ in range(WARMUP):
        fn(*args)
    t0 = time.perf_counter()
    fn(*args)
    single_ms = (time.perf_counter() - t0) * 1e3
    reps = choose_reps(single_ms)
    times = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(reps):
            t0 = time.perf_counter_ns()
            fn(*args)
            times.append((time.perf_counter_ns() - t0) / 1e9)
    finally:
        if gc_was_enabled:
            gc.enable()
    return times

def summarize(times_s):
    us = np.array(sorted(t * 1e6 for t in times_s))
    med = float(np.median(us))
    boot = np.median(rng.choice(us, size=(BOOTSTRAP_B, len(us)), replace=True), axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "n": len(us),
        "median_us": med,
        "q1_us": float(np.percentile(us, 25)),
        "q3_us": float(np.percentile(us, 75)),
        "p95_us": float(np.percentile(us, 95)),
        "p99_us": float(np.percentile(us, 99)),
        "mean_us": float(np.mean(us)),
        "std_us": float(np.std(us, ddof=1)) if len(us) > 1 else 0.0,
        "min_us": float(us.min()),
        "max_us": float(us.max()),
        "cv_pct": float(100 * np.std(us, ddof=1) / np.mean(us)) if len(us) > 1 else 0.0,
        "ci95_lo_us": float(lo),
        "ci95_hi_us": float(hi),
    }

def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

# ----------------------------- Stage 1: KEM benchmarks ----------------------

def stage_kem_benchmarks():
    print("\n[Stage 1] KEM micro-benchmarks", flush=True)
    rows, sizes, samples = [], [], {}
    for name, (kg, enc, dec, lvl, fam) in KEMS.items():
        print(f"  - {name} ...", flush=True)
        pk, sk = kg()
        ct, ss = enc(pk)
        ss2 = dec(sk, ct)
        d1 = ss if isinstance(ss, bytes) else bytes(ss)
        d2 = ss2 if isinstance(ss2, bytes) else bytes(ss2)
        assert d1 == d2, f"{name}: shared-secret mismatch"
        pk_len = len(pk) if isinstance(pk, bytes) else len(pk[0]) + 0  # bytes for all
        sizes.append({"algo": name, "kind": "KEM", "level": lvl, "family": fam,
                      "pk_B": len(pk) if isinstance(pk, bytes) else -1,
                      "sk_or_priv": "opaque" if not isinstance(sk, bytes) else len(sk),
                      "ct_or_sig_B": len(ct), "ss_B": len(d1)})
        for op, fn, args in [("keygen", kg, ()),
                             ("encaps", enc, (pk,)),
                             ("decaps", dec, (sk, ct))]:
            ts = bench(fn, *args)
            samples[f"KEM|{name}|{op}"] = np.array(ts)
            s = summarize(ts)
            s.update({"algo": name, "op": op, "level": lvl, "family": fam})
            rows.append(s)
            print(f"      {op:7s}: median {s['median_us']:>10.1f} us  "
                  f"(n={s['n']}, p99 {s['p99_us']:.1f} us)", flush=True)
    return rows, sizes, samples

# ----------------------------- Stage 2: signature benchmarks ----------------

def stage_sig_benchmarks():
    print("\n[Stage 2] Signature micro-benchmarks", flush=True)
    rows, sizes, samples = [], [], {}
    for name, (kg, sg, vf, lvl, fam) in SIGS.items():
        print(f"  - {name} ...", flush=True)
        pk, sk = kg()
        sig = sg(sk, MSG)
        vf(pk, MSG, sig)  # raises on failure
        # Falcon and ECDSA produce variable-length signatures; record the
        # median over 100 fresh signatures for those schemes
        if name in ("Falcon-512", "Falcon-1024", "ECDSA-P256"):
            sig_lens = [len(sg(sk, MSG)) for _ in range(100)]
            sig_B = int(statistics.median(sig_lens))
        else:
            sig_B = len(sig)
        sizes.append({"algo": name, "kind": "SIG", "level": lvl, "family": fam,
                      "pk_B": len(pk), "sk_or_priv": "opaque" if not isinstance(sk, bytes) else len(sk),
                      "ct_or_sig_B": sig_B, "ss_B": ""})
        for op, fn, args in [("keygen", kg, ()),
                             ("sign", sg, (sk, MSG)),
                             ("verify", vf, (pk, MSG, sig))]:
            ts = bench(fn, *args)
            samples[f"SIG|{name}|{op}"] = np.array(ts)
            s = summarize(ts)
            s.update({"algo": name, "op": op, "level": lvl, "family": fam})
            rows.append(s)
            print(f"      {op:7s}: median {s['median_us']:>10.1f} us  "
                  f"(n={s['n']}, p99 {s['p99_us']:.1f} us)", flush=True)
    return rows, sizes, samples

# ----------------------------- Stage 3: memory footprint --------------------

_MEM_SNIPPET = r"""
import sys
def vmrss_kb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1])
    return -1
rss_start = vmrss_kb()
mode, kind, name = sys.argv[1], sys.argv[2], sys.argv[3]
MSG = b"\x5a" * 64
PQ_KEM = {"ML-KEM-512":"ml_kem_512","ML-KEM-768":"ml_kem_768",
          "ML-KEM-1024":"ml_kem_1024","HQC-128":"hqc_128","HQC-192":"hqc_192",
          "HQC-256":"hqc_256","McEliece-348864":"mceliece348864"}
PQ_SIG = {"ML-DSA-44":"ml_dsa_44","ML-DSA-65":"ml_dsa_65","ML-DSA-87":"ml_dsa_87",
          "Falcon-512":"falcon_512","Falcon-1024":"falcon_1024",
          "SPHINCS+-128s":"sphincs_sha2_128s_simple",
          "SPHINCS+-128f":"sphincs_sha2_128f_simple"}
import importlib
if kind == "kem":
    if name in PQ_KEM:
        m = importlib.import_module("pqcrypto.kem." + PQ_KEM[name])
        kg, enc, dec = m.generate_keypair, m.encrypt, m.decrypt
    else:
        from cryptography.hazmat.primitives.asymmetric import x25519
        from cryptography.hazmat.primitives import serialization, hashes
        from cryptography.hazmat.primitives.kdf.hkdf import HKDF
        def xkg():
            sk = x25519.X25519PrivateKey.generate()
            return sk.public_key().public_bytes(serialization.Encoding.Raw,
                serialization.PublicFormat.Raw), sk
        def xenc(pk):
            e = x25519.X25519PrivateKey.generate()
            ss = e.exchange(x25519.X25519PublicKey.from_public_bytes(pk))
            return e.public_key().public_bytes(serialization.Encoding.Raw,
                serialization.PublicFormat.Raw), ss
        def xdec(sk, ct):
            return sk.exchange(x25519.X25519PublicKey.from_public_bytes(ct))
        if name == "X25519":
            kg, enc, dec = xkg, xenc, xdec
        else:  # X25519+ML-KEM-768
            mk = importlib.import_module("pqcrypto.kem.ml_kem_768")
            def hk():
                p1, s1 = xkg(); p2, s2 = mk.generate_keypair()
                return p1 + p2, (s1, s2)
            def he(pk):
                c1, s1 = xenc(pk[:32]); c2, s2 = mk.encrypt(pk[32:])
                return c1 + c2, HKDF(algorithm=hashes.SHA256(), length=32,
                    salt=None, info=b"h").derive(s1 + s2)
            def hd(sk, ct):
                s1 = xdec(sk[0], ct[:32]); s2 = mk.decrypt(sk[1], ct[32:])
                return HKDF(algorithm=hashes.SHA256(), length=32, salt=None,
                    info=b"h").derive(s1 + s2)
            kg, enc, dec = hk, he, hd
    if mode == "active":
        for _ in range(2):
            pk, sk = kg(); ct, ss = enc(pk); dec(sk, ct)
else:
    if name in PQ_SIG:
        m = importlib.import_module("pqcrypto.sign." + PQ_SIG[name])
        kg, sg, vf = m.generate_keypair, m.sign, m.verify
    else:
        from cryptography.hazmat.primitives.asymmetric import ec, ed25519, rsa, padding
        from cryptography.hazmat.primitives import hashes, serialization
        if name == "Ed25519":
            def kg():
                sk = ed25519.Ed25519PrivateKey.generate()
                return sk.public_key().public_bytes(serialization.Encoding.Raw,
                    serialization.PublicFormat.Raw), sk
            sg = lambda sk, m_: sk.sign(m_)
            vf = lambda pk, m_, s: ed25519.Ed25519PublicKey.from_public_bytes(pk).verify(s, m_)
        elif name == "ECDSA-P256":
            def kg():
                sk = ec.generate_private_key(ec.SECP256R1())
                return sk.public_key().public_bytes(serialization.Encoding.X962,
                    serialization.PublicFormat.UncompressedPoint), sk
            sg = lambda sk, m_: sk.sign(m_, ec.ECDSA(hashes.SHA256()))
            vf = lambda pk, m_, s: ec.EllipticCurvePublicKey.from_encoded_point(
                ec.SECP256R1(), pk).verify(s, m_, ec.ECDSA(hashes.SHA256()))
        else:  # RSA-2048
            def kg():
                sk = rsa.generate_private_key(public_exponent=65537, key_size=2048)
                return sk.public_key().public_bytes(serialization.Encoding.DER,
                    serialization.PublicFormat.SubjectPublicKeyInfo), sk
            pss = padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=32)
            sg = lambda sk, m_: sk.sign(m_, pss, hashes.SHA256())
            vf = lambda pk, m_, s: serialization.load_der_public_key(pk).verify(
                s, m_, pss, hashes.SHA256())
rss_import = vmrss_kb()
rss_post = rss_import
if kind == "kem":
    for _ in range(2):
        pk, sk = kg(); ct, ss = enc(pk); dec(sk, ct)
        rss_post = max(rss_post, vmrss_kb())
else:
    for _ in range(2):
        pk, sk = kg(); s = sg(sk, MSG); vf(pk, MSG, s)
        rss_post = max(rss_post, vmrss_kb())
print(rss_start, rss_import, rss_post)
"""

def stage_memory():
    print("\n[Stage 3] Resident-set (VmRSS) memory footprint, three-point sampling",
          flush=True)
    rows = []
    env = dict(os.environ)
    for kind, reg in [("kem", KEMS), ("sig", SIGS)]:
        for name in reg:
            out = subprocess.run([sys.executable, "-c", _MEM_SNIPPET,
                                  "active", kind, name],
                                 capture_output=True, text=True, env=env,
                                 timeout=600)
            if out.returncode != 0:
                raise RuntimeError(out.stderr[-400:])
            start_kb, imp_kb, post_kb = map(int, out.stdout.split())
            rows.append({"algo": name, "kind": kind.upper(),
                         "start_MB": round(start_kb / 1024.0, 2),
                         "import_MB": round(imp_kb / 1024.0, 2),
                         "postop_MB": round(post_kb / 1024.0, 2),
                         "lib_delta_MB": round((imp_kb - start_kb) / 1024.0, 2),
                         "op_delta_MB": round(max(post_kb - imp_kb, 0) / 1024.0, 2),
                         "footprint_MB": round((post_kb - start_kb) / 1024.0, 2)})
            r = rows[-1]
            print(f"  - {name:18s} start {r['start_MB']:6.1f}  import "
                  f"{r['import_MB']:6.1f}  post-op {r['postop_MB']:6.1f}  "
                  f"lib {r['lib_delta_MB']:5.2f}  op {r['op_delta_MB']:5.2f}  "
                  f"footprint {r['footprint_MB']:5.2f} MB", flush=True)
    return rows

# ----------------------------- Stage 4: handshake composition ---------------
# Cryptographic-payload model of a TLS 1.3-style mutual-agreement handshake
# with server-only authentication. Counts ONLY the cryptographic objects that
# scale with algorithm choice (protocol framing and X.509 encoding overhead
# are suite-independent constants and are excluded; this is stated in the
# paper). CA signature assumed to use the same signature scheme as the leaf.
#
#   B_kex  = |pk_KEM|  (ClientHello key_share)  +  |ct_KEM| (ServerHello)
#   B_auth = |pk_SIG|  (leaf certificate key)   +  |sig| (CA signature on
#            the certificate)                    +  |sig| (CertificateVerify)
#
# Per-suite compute path (server-authenticated handshake):
#   client:  KEM keygen + KEM decaps + 2 x verify (cert chain + CertVerify)
#   server:  KEM encaps + 1 x sign   (CertificateVerify; CA signing offline)

SUITE_TIERS = {
    1: (["ML-KEM-512", "HQC-128", "McEliece-348864"],
        ["Falcon-512", "ML-DSA-44", "SPHINCS+-128s", "SPHINCS+-128f"]),
    3: (["ML-KEM-768", "HQC-192"], ["ML-DSA-65"]),
    5: (["ML-KEM-1024", "HQC-256"], ["ML-DSA-87", "Falcon-1024"]),
}
HYBRID_SUITES = [("X25519+ML-KEM-768", "ML-DSA-65"),
                 ("X25519+ML-KEM-768", "ECDSA-P256")]
CLASSICAL_SUITES = [("X25519", "Ed25519"), ("X25519", "ECDSA-P256"),
                    ("X25519", "RSA-2048")]

def build_suites():
    # kex_tier: post-quantum security category of the key exchange
    # auth_tier: post-quantum security category of the authentication
    # (classical primitives contribute tier 0 on their axis)
    suites = []
    for lvl, (kems, sigs) in SUITE_TIERS.items():
        for k, g in product(kems, sigs):
            suites.append({"kem": k, "sig": g, "kex_tier": KEMS[k][3],
                           "auth_tier": SIGS[g][3], "class": "pq"})
    for k, g in HYBRID_SUITES:
        suites.append({"kem": k, "sig": g, "kex_tier": 3,
                       "auth_tier": SIGS[g][3], "class": "hybrid"})
    for k, g in CLASSICAL_SUITES:
        suites.append({"kem": k, "sig": g, "kex_tier": 0, "auth_tier": 0,
                       "class": "classical"})
    return suites

def stage_handshake(kem_rows, sig_rows, size_rows, mem_rows):
    print("\n[Stage 4] Handshake cryptographic-payload composition", flush=True)
    sz = {r["algo"]: r for r in size_rows}
    mem = {r["algo"]: r["footprint_MB"] for r in mem_rows}
    lat = {(r["algo"], r["op"]): r for r in kem_rows + sig_rows}
    rows = []
    for s in build_suites():
        k, g = s["kem"], s["sig"]
        b_kex = sz[k]["pk_B"] + sz[k]["ct_or_sig_B"]
        b_auth = sz[g]["pk_B"] + 2 * sz[g]["ct_or_sig_B"]
        b_total = b_kex + b_auth
        med = (lat[(k, "keygen")]["median_us"] + lat[(k, "encaps")]["median_us"]
               + lat[(k, "decaps")]["median_us"] + lat[(g, "sign")]["median_us"]
               + 2 * lat[(g, "verify")]["median_us"]) / 1e3   # ms
        p99 = (lat[(k, "keygen")]["p99_us"] + lat[(k, "encaps")]["p99_us"]
               + lat[(k, "decaps")]["p99_us"] + lat[(g, "sign")]["p99_us"]
               + 2 * lat[(g, "verify")]["p99_us"]) / 1e3      # ms (composition bound)
        m = max(mem[k], mem[g])
        row = {"suite": f"{k} / {g}", "kem": k, "sig": g,
               "kex_tier": s["kex_tier"], "auth_tier": s["auth_tier"],
               "class": s["class"], "bytes_kex": b_kex, "bytes_auth": b_auth,
               "bytes_total": b_total,
               "frag_ipv6": math.ceil(b_total / PAYLOAD_IPV6),
               "frag_6lowpan": math.ceil(b_total / PAYLOAD_6LOWPAN),
               "t_crypto_med_ms": round(med, 3),
               "t_crypto_p99_ms": round(p99, 3),
               "mem_MB": round(m, 2)}
        for lname, rate in LINK_RATES_BPS.items():
            t_net = b_total * 8 / rate * 1e3  # ms serialization delay
            row[f"t_net_{lname}_ms"] = round(t_net, 3)
            row[f"t_total_med_{lname}_ms"] = round(med + t_net, 3)
            row[f"t_total_p99_{lname}_ms"] = round(p99 + t_net, 3)
        rows.append(row)
        print(f"  - {row['suite']:34s} bytes {b_total:>7d}  "
              f"t_crypto med {med:8.2f} ms  p99 {p99:8.2f} ms", flush=True)
    return rows

# ----------------------------- Stage 5: selection framework -----------------
# Feasibility constraints + weighted-normalized scoring over the measured
# suite metrics (Eq. 4-6 in the paper). Weights and budgets are scenario
# parameters (inputs), stated per profile.

PROFILES = {
    "P-A industrial-Ethernet": dict(rate="ethernet_100M", tau_ms=20, mem_MB=512,
                                    min_kex=3, min_auth=3,
                                    max_frag=None, frag_basis="frag_ipv6",
                                    w=(0.60, 0.20, 0.20)),
    "P-B LTE-M gateway":       dict(rate="lte_m_1M", tau_ms=300, mem_MB=128,
                                    min_kex=1, min_auth=1,
                                    max_frag=None, frag_basis="frag_ipv6",
                                    w=(0.45, 0.40, 0.15)),
    "P-C NB-IoT gateway":      dict(rate="nb_iot_62k5", tau_ms=3000, mem_MB=64,
                                    min_kex=1, min_auth=1,
                                    max_frag=None, frag_basis="frag_ipv6",
                                    w=(0.35, 0.50, 0.15)),
    "P-D 6LoWPAN node":        dict(rate="wpan_250k", tau_ms=2000, mem_MB=32,
                                    min_kex=1, min_auth=1,
                                    max_frag=96, frag_basis="frag_6lowpan",
                                    w=(0.30, 0.55, 0.15)),
}

def feasible(row, p, use_tail=True):
    if row["kex_tier"] < p["min_kex"]:
        return False
    if row["auth_tier"] < p["min_auth"]:
        return False
    tkey = f"t_total_{'p99' if use_tail else 'med'}_{p['rate']}_ms"
    if row[tkey] > p["tau_ms"]:
        return False
    if row["mem_MB"] > p["mem_MB"]:
        return False
    if p["max_frag"] is not None and row[p["frag_basis"]] > p["max_frag"]:
        return False
    return True

def score_and_select(hs_rows, p, use_tail=True, weights=None, use_bytes=True,
                     use_mem=True, time_basis="total"):
    w_t, w_b, w_m = weights if weights else p["w"]
    q = "p99" if use_tail else "med"
    tkey = (f"t_total_{q}_{p['rate']}_ms" if time_basis == "total"
            else f"t_crypto_{q}_ms")
    feas = [r for r in hs_rows if feasible(r, p, use_tail)]
    if not feas:
        return None, []
    def norm(vals):
        lo, hi = min(vals), max(vals)
        return [(v - lo) / (hi - lo) if hi > lo else 0.0 for v in vals]
    T = norm([r[tkey] for r in feas])
    B = norm([r["bytes_total"] for r in feas])
    M = norm([r["mem_MB"] for r in feas])
    scored = []
    for i, r in enumerate(feas):
        sc = w_t * T[i] + (w_b * B[i] if use_bytes else 0.0) \
             + (w_m * M[i] if use_mem else 0.0)
        scored.append((sc, r["bytes_total"], r))
    scored.sort(key=lambda x: (x[0], x[1]))
    return scored[0][2], [(round(sc, 4), r["suite"]) for sc, _, r in scored]

def pareto_front(hs_rows, p, use_tail=True):
    tkey = f"t_total_{'p99' if use_tail else 'med'}_{p['rate']}_ms"
    feas = [r for r in hs_rows if feasible(r, p, use_tail)]
    front = []
    for r in feas:
        dominated = any(
            (o[tkey] <= r[tkey] and o["bytes_total"] <= r["bytes_total"]
             and o["mem_MB"] <= r["mem_MB"]
             and (o[tkey] < r[tkey] or o["bytes_total"] < r["bytes_total"]
                  or o["mem_MB"] < r["mem_MB"]))
            for o in feas)
        if not dominated:
            front.append(r["suite"])
    return front

def stage_selection(hs_rows):
    print("\n[Stage 5] Multi-criteria suite selection per device profile", flush=True)
    sel_rows, rank_dump = [], {}
    for pname, p in PROFILES.items():
        chosen, ranking = score_and_select(hs_rows, p)
        front = pareto_front(hs_rows, p)
        tkey = f"t_total_p99_{p['rate']}_ms"
        n_feas = sum(feasible(r, p) for r in hs_rows)
        sel_rows.append({
            "profile": pname, "chosen_suite": chosen["suite"] if chosen else "NONE",
            "n_feasible": n_feas, "n_pareto": len(front),
            "t_total_p99_ms": chosen[tkey] if chosen else "",
            "bytes_total": chosen["bytes_total"] if chosen else "",
            "mem_MB": chosen["mem_MB"] if chosen else "",
            "pareto_set": "; ".join(front)})
        rank_dump[pname] = ranking
        print(f"  - {pname:26s} -> {sel_rows[-1]['chosen_suite']:34s} "
              f"({n_feas} feasible, {len(front)} Pareto)", flush=True)
    return sel_rows, rank_dump

# ----------------------------- Stage 6: baselines + ablation ----------------

def stage_baselines_ablation(hs_rows):
    print("\n[Stage 6] Baseline policies and framework ablation", flush=True)
    by_suite = {r["suite"]: r for r in hs_rows}
    base_rows, abl_rows = [], []

    def eval_policy(pname, p, row):
        tkey = f"t_total_p99_{p['rate']}_ms"
        viol = []
        if row["kex_tier"] < p["min_kex"]:
            viol.append("kex-tier")
        if row["auth_tier"] < p["min_auth"]:
            viol.append("auth-tier")
        if row[tkey] > p["tau_ms"]:
            viol.append("latency")
        if row["mem_MB"] > p["mem_MB"]:
            viol.append("memory")
        if p["max_frag"] is not None and row[p["frag_basis"]] > p["max_frag"]:
            viol.append("fragmentation")
        return row[tkey], row["bytes_total"], viol

    static = by_suite["X25519+ML-KEM-768 / ML-DSA-65"]
    for pname, p in PROFILES.items():
        chosen, _ = score_and_select(hs_rows, p)
        tkey = f"t_total_p99_{p['rate']}_ms"
        pq = [r for r in hs_rows if r["kex_tier"] >= p["min_kex"]
              and r["auth_tier"] >= p["min_auth"]]
        cmin = min(pq, key=lambda r: r["t_crypto_p99_ms"])
        bmin = min(pq, key=lambda r: r["bytes_total"])
        # transitional posture: PQ key exchange required, classical
        # authentication temporarily permitted (auth_tier constraint relaxed)
        p_trans = dict(p, min_auth=0)
        trans, _ = score_and_select(hs_rows, p_trans)
        for pol, row in [("framework", chosen), ("static-default", static),
                         ("compute-min", cmin), ("bytes-min", bmin),
                         ("transitional-auth", trans)]:
            t, b, viol = eval_policy(pname, p, row)
            base_rows.append({"profile": pname, "policy": pol,
                              "suite": row["suite"], "t_total_p99_ms": t,
                              "bytes_total": b,
                              "violations": ",".join(viol) if viol else "none"})
            print(f"  - {pname:26s} {pol:14s} -> {row['suite']:34s} "
                  f"viol=[{','.join(viol) if viol else 'none'}]", flush=True)

    # Ablations of the framework's components and constraints
    variants = {
        "full": (dict(), None),
        "A1 no-bytes-term": (dict(use_bytes=False), None),
        "A2 median-not-p99": (dict(use_tail=False), None),
        "A3 no-memory-term": (dict(use_mem=False), None),
        "A4 equal-weights": (dict(weights=(1/3, 1/3, 1/3)), None),
        "A5 crypto-time-only": (dict(time_basis="crypto"), None),
        "A6 no-frag-constraint": (dict(), dict(max_frag=None)),
    }
    rob_rows = []
    for pname, p in PROFILES.items():
        full_choice = None
        for vname, (kw, pmod) in variants.items():
            pv = dict(p, **pmod) if pmod else p
            chosen, _ = score_and_select(hs_rows, pv, **kw)
            suite = chosen["suite"] if chosen else "NONE"
            if vname == "full":
                full_choice = suite
            abl_rows.append({"profile": pname, "variant": vname, "chosen": suite,
                             "changed": "no" if suite == full_choice else "yes"})
        # Weight-sweep robustness: fraction of the weight simplex (0.05 grid)
        # on which the modal suite is selected
        counts = {}
        n_grid = 0
        for i in range(0, 21):
            for j in range(0, 21 - i):
                wt, wb = i / 20.0, j / 20.0
                wm = 1.0 - wt - wb
                chosen, _ = score_and_select(hs_rows, p, weights=(wt, wb, wm))
                if chosen:
                    counts[chosen["suite"]] = counts.get(chosen["suite"], 0) + 1
                    n_grid += 1
        modal = max(counts, key=counts.get)
        rob_rows.append({"profile": pname, "modal_suite": modal,
                         "modal_share_pct": round(100 * counts[modal] / n_grid, 1),
                         "n_distinct": len(counts), "grid_points": n_grid,
                         "all_counts": "; ".join(f"{k}:{v}" for k, v in
                                                 sorted(counts.items(),
                                                        key=lambda x: -x[1]))})
        print(f"  - {pname:26s} weight-sweep modal: {modal} "
              f"({rob_rows[-1]['modal_share_pct']}% of simplex, "
              f"{len(counts)} distinct)", flush=True)
    return base_rows, abl_rows, rob_rows

# ----------------------------- Stage 7: figures -----------------------------

def _box(ax, samples, keys, labels):
    data = [samples[k] * 1e6 for k in keys]  # us
    ax.boxplot(data, tick_labels=labels, showfliers=True, whis=(1, 99))
    ax.set_yscale("log")
    ax.grid(True, which="both", axis="y", alpha=0.3)
    for tick in ax.get_xticklabels():
        tick.set_rotation(60)
        tick.set_ha("right")

def stage_figures(samples_kem, samples_sig, hs_rows, sel_rows):
    print("\n[Stage 7] Figure generation", flush=True)
    os.makedirs(OUT_FIGURES, exist_ok=True)

    # Figure 1: KEM operation latency distributions
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    kem_names = list(KEMS.keys())
    for ax, op in zip(axes, ["keygen", "encaps", "decaps"]):
        _box(ax, samples_kem, [f"KEM|{n}|{op}" for n in kem_names], kem_names)
        ax.set_title(f"KEM {op}")
    axes[0].set_ylabel("Latency (\u00b5s, log scale)")
    fig.suptitle("KEM operation latency distributions (whiskers: p1\u2013p99)")
    fig.tight_layout()
    fig.savefig(f"{OUT_FIGURES}/fig1_kem_latency.png", dpi=300)
    plt.close(fig)

    # Figure 2: signature operation latency distributions
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    sig_names = list(SIGS.keys())
    for ax, op in zip(axes, ["keygen", "sign", "verify"]):
        _box(ax, samples_sig, [f"SIG|{n}|{op}" for n in sig_names], sig_names)
        ax.set_title(f"Signature {op}")
    axes[0].set_ylabel("Latency (\u00b5s, log scale)")
    fig.suptitle("Signature operation latency distributions (whiskers: p1\u2013p99)")
    fig.tight_layout()
    fig.savefig(f"{OUT_FIGURES}/fig2_sig_latency.png", dpi=300)
    plt.close(fig)

    # Figure 3: handshake cryptographic payload per suite (stacked)
    rows = sorted(hs_rows, key=lambda r: r["bytes_total"])
    names = [r["suite"] for r in rows]
    kex = np.array([r["bytes_kex"] for r in rows])
    auth = np.array([r["bytes_auth"] for r in rows])
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(y, kex, label="Key exchange (pk + ct)")
    ax.barh(y, auth, left=kex, label="Authentication (cert pk + 2 sigs)")
    ax.set_yticks(y, names, fontsize=7.5)
    ax.set_xscale("log")
    ax.axvline(PAYLOAD_IPV6, ls="--", c="k", lw=1)
    ax.text(PAYLOAD_IPV6 * 1.05, len(rows) - 1.2, "IPv6 min-MTU payload (1232 B)",
            rotation=90, va="top", fontsize=7)
    ax.set_xlabel("Handshake cryptographic payload (bytes, log scale)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, axis="x", which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT_FIGURES}/fig3_handshake_bytes.png", dpi=300)
    plt.close(fig)

    # Figure 4: bytes vs total p99 handshake time on LTE-M, Pareto + selections
    p = PROFILES["P-B LTE-M gateway"]
    tkey = f"t_total_p99_{p['rate']}_ms"
    front = set(pareto_front(hs_rows, p))
    chosen = {r["chosen_suite"] for r in sel_rows}
    colors = {"pq": "tab:blue", "hybrid": "tab:green", "classical": "tab:gray"}
    fig, ax = plt.subplots(figsize=(9.2, 6.6))
    for r in hs_rows:                                    # infeasible drawn first
        if not feasible(r, p):
            ax.scatter(r["bytes_total"], r[tkey], marker="x", c="tab:red",
                       s=46, linewidths=1.4, alpha=0.75, zorder=2)
    labels = []
    for r in hs_rows:
        if not feasible(r, p):
            continue
        if r["suite"] in chosen:
            m, sz, z = "*", 340, 6
        elif r["suite"] in front:
            m, sz, z = "o", 110, 5
        else:
            m, sz, z = "s", 60, 4
        ax.scatter(r["bytes_total"], r[tkey], marker=m, s=sz,
                   c=colors[r["class"]], edgecolors="white", linewidths=1.1, zorder=z)
        if r["suite"] in front or r["suite"] in chosen:
            labels.append(ax.text(r["bytes_total"], r[tkey], r["suite"],
                                  fontsize=9, zorder=7,
                                  bbox=dict(boxstyle="round,pad=0.18", fc="white",
                                            ec="0.7", lw=0.5, alpha=0.9)))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Handshake cryptographic payload (bytes)")
    ax.set_ylabel("Total handshake time, p99 bound @ 1 Mb/s (ms)")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title("Feasible cipher suites on profile P-B (LTE-M gateway)")
    ax.margins(x=0.14, y=0.18)
    _declutter_labels(ax, labels)
    role = [
        Line2D([], [], marker="*", ls="", mfc="0.4", mec="white", ms=17,
               label="Selected on \u22651 profile"),
        Line2D([], [], marker="o", ls="", mfc="0.4", mec="white", ms=9,
               label="Pareto-optimal (P-B)"),
        Line2D([], [], marker="s", ls="", mfc="0.4", mec="white", ms=8,
               label="Feasible, dominated"),
        Line2D([], [], marker="x", ls="", mec="tab:red", mew=1.6, ms=9,
               label="Infeasible on P-B"),
    ]
    cls = [
        Line2D([], [], marker="o", ls="", mfc="tab:blue", mec="white", ms=9,
               label="Post-quantum"),
        Line2D([], [], marker="o", ls="", mfc="tab:green", mec="white", ms=9,
               label="Hybrid (X25519+ML-KEM)"),
    ]
    leg1 = ax.legend(handles=role, loc="upper left", fontsize=8.5, frameon=True,
                     framealpha=0.92, title="Marker shape", title_fontsize=9)
    ax.add_artist(leg1)
    ax.legend(handles=cls, loc="lower right", fontsize=8.5, frameon=True,
              framealpha=0.92, title="Marker colour", title_fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT_FIGURES}/fig4_pareto_pb.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Figure 5: tail behavior of signing (p50 vs p99)
    fig, ax = plt.subplots(figsize=(8.4, 7.0))
    cmap = plt.get_cmap("tab10")
    labels = []
    for i, n in enumerate(SIGS):
        arr = samples_sig[f"SIG|{n}|sign"] * 1e3  # ms
        p50, p99 = np.percentile(arr, [50, 99])
        ax.scatter(p50, p99, s=95, color=cmap(i % 10), edgecolors="black",
                   linewidths=0.7, zorder=4)
        labels.append(ax.text(p50, p99, n, fontsize=9.5, zorder=6,
                              bbox=dict(boxstyle="round,pad=0.18", fc="white",
                                        ec="0.7", lw=0.5, alpha=0.9)))
    lim = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
           max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lim, lim, ls="--", c="gray", lw=1.2, zorder=1,
            label="p99 = p50 (no tail)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Signing latency p50 (ms)")
    ax.set_ylabel("Signing latency p99 (ms)")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title("Median vs. tail signing latency per signature scheme")
    ax.legend(loc="lower right", fontsize=9, frameon=True, framealpha=0.92)
    _declutter_labels(ax, labels)
    fig.tight_layout()
    fig.savefig(f"{OUT_FIGURES}/fig5_sign_tail.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  - 5 figures written to figures/", flush=True)

# ----------------------------- Stage 8: table printing ----------------------

def print_tables(kem_rows, sig_rows, size_rows, mem_rows, hs_rows,
                 sel_rows, base_rows, abl_rows):
    print("\n[Stage 8] Table values (paper mapping)", flush=True)
    for r in size_rows:
        print(f"Table 2 | {r['algo']} ({r['kind']}, L{r['level']}): "
              f"pk={r['pk_B']} B, ct/sig={r['ct_or_sig_B']} B")
    for r in kem_rows:
        print(f"Table 3 | {r['algo']} {r['op']}: median={r['median_us']:.1f} us, "
              f"IQR=[{r['q1_us']:.1f},{r['q3_us']:.1f}], p99={r['p99_us']:.1f} us, "
              f"CV={r['cv_pct']:.1f}%, n={r['n']}")
    for r in sig_rows:
        print(f"Table 4 | {r['algo']} {r['op']}: median={r['median_us']:.1f} us, "
              f"IQR=[{r['q1_us']:.1f},{r['q3_us']:.1f}], p99={r['p99_us']:.1f} us, "
              f"CV={r['cv_pct']:.1f}%, n={r['n']}")
    for r in mem_rows:
        print(f"Table 5 | {r['algo']}: lib={r['lib_delta_MB']} MB, "
              f"op={r['op_delta_MB']} MB, footprint={r['footprint_MB']} MB")
    for r in hs_rows:
        print(f"Table 6 | {r['suite']}: bytes={r['bytes_total']}, "
              f"frag6lowpan={r['frag_6lowpan']}, "
              f"t_crypto med/p99={r['t_crypto_med_ms']}/{r['t_crypto_p99_ms']} ms, "
              f"t_total p99 @NB-IoT={r['t_total_p99_nb_iot_62k5_ms']} ms")
    for r in sel_rows:
        print(f"Table 7 | {r['profile']}: chosen={r['chosen_suite']}, "
              f"feasible={r['n_feasible']}, pareto={r['n_pareto']}")
    for r in base_rows:
        print(f"Table 8 | {r['profile']} / {r['policy']}: {r['suite']} "
              f"violations={r['violations']}")
    for r in abl_rows:
        print(f"Table 9 | {r['profile']} / {r['variant']}: {r['chosen']} "
              f"(changed={r['changed']})")
    for r in rob_rows:
        print(f"Table 9 | {r['profile']} weight-sweep: modal={r['modal_suite']} "
              f"share={r['modal_share_pct']}% distinct={r['n_distinct']}")

# ----------------------------- Main -----------------------------------------

def main():
    t_start = time.time()
    os.makedirs(OUT_RESULTS, exist_ok=True)
    os.makedirs(OUT_FIGURES, exist_ok=True)

    manifest = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "cpu_model": next((l.split(":")[1].strip()
                           for l in open("/proc/cpuinfo") if "model name" in l), "?"),
        "n_logical_cpus": os.cpu_count(),
        "packages": {p: importlib.metadata.version(p)
                     for p in ["pqcrypto", "cryptography", "numpy", "matplotlib"]},
        "timer_resolution_ns": time.get_clock_info("perf_counter").resolution * 1e9,
        "memory_metric": "three-point VmRSS (/proc/self/status): process start, post-import, max post-operation",
        "seed": SEED, "warmup": WARMUP, "rep_tiers": REP_TIERS,
        "bootstrap_B": BOOTSTRAP_B,
        "link_rates_bps": LINK_RATES_BPS,
        "payload_ipv6_B": PAYLOAD_IPV6, "payload_6lowpan_B": PAYLOAD_6LOWPAN,
    }
    with open(f"{OUT_RESULTS}/platform_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("[Stage 0] Platform manifest written.", flush=True)

    try:
        kem_rows, kem_sizes, kem_samples = stage_kem_benchmarks()
    except Exception as e:
        print(f"FATAL in Stage 1 (KEM benchmarks): {e!r}"); raise
    try:
        sig_rows, sig_sizes, sig_samples = stage_sig_benchmarks()
    except Exception as e:
        print(f"FATAL in Stage 2 (signature benchmarks): {e!r}"); raise
    size_rows = kem_sizes + sig_sizes
    try:
        mem_rows = stage_memory()
    except Exception as e:
        print(f"FATAL in Stage 3 (memory): {e!r}"); raise
    try:
        hs_rows = stage_handshake(kem_rows, sig_rows, size_rows, mem_rows)
    except Exception as e:
        print(f"FATAL in Stage 4 (handshake composition): {e!r}"); raise
    try:
        sel_rows, rank_dump = stage_selection(hs_rows)
    except Exception as e:
        print(f"FATAL in Stage 5 (selection): {e!r}"); raise
    try:
        base_rows, abl_rows, rob_rows = stage_baselines_ablation(hs_rows)
    except Exception as e:
        print(f"FATAL in Stage 6 (baselines/ablation): {e!r}"); raise
    try:
        stage_figures(kem_samples, sig_samples, hs_rows, sel_rows)
    except Exception as e:
        print(f"FATAL in Stage 7 (figures): {e!r}"); raise

    # Persist everything
    bench_fields = ["algo", "op", "level", "family", "n", "median_us", "q1_us",
                    "q3_us", "p95_us", "p99_us", "mean_us", "std_us", "min_us",
                    "max_us", "cv_pct", "ci95_lo_us", "ci95_hi_us"]
    write_csv(f"{OUT_RESULTS}/kem_bench.csv", kem_rows, bench_fields)
    write_csv(f"{OUT_RESULTS}/sig_bench.csv", sig_rows, bench_fields)
    write_csv(f"{OUT_RESULTS}/sizes.csv", size_rows,
              ["algo", "kind", "level", "family", "pk_B", "sk_or_priv",
               "ct_or_sig_B", "ss_B"])
    write_csv(f"{OUT_RESULTS}/memory.csv", mem_rows,
              ["algo", "kind", "start_MB", "import_MB", "postop_MB",
               "lib_delta_MB", "op_delta_MB", "footprint_MB"])
    write_csv(f"{OUT_RESULTS}/handshake.csv", hs_rows, list(hs_rows[0].keys()))
    write_csv(f"{OUT_RESULTS}/selection.csv", sel_rows, list(sel_rows[0].keys()))
    write_csv(f"{OUT_RESULTS}/baselines.csv", base_rows, list(base_rows[0].keys()))
    write_csv(f"{OUT_RESULTS}/ablation.csv", abl_rows, list(abl_rows[0].keys()))
    write_csv(f"{OUT_RESULTS}/robustness.csv", rob_rows, list(rob_rows[0].keys()))
    np.savez_compressed(f"{OUT_RESULTS}/raw_samples.npz",
                        **{k: v for k, v in {**kem_samples, **sig_samples}.items()})
    with open(f"{OUT_RESULTS}/selection_rankings.json", "w") as f:
        json.dump(rank_dump, f, indent=2)

    print_tables(kem_rows, sig_rows, size_rows, mem_rows, hs_rows,
                 sel_rows, base_rows, abl_rows)
    print(f"\nDONE in {time.time() - t_start:.1f} s. "
          f"CSVs in {OUT_RESULTS}/, figures in {OUT_FIGURES}/.", flush=True)

if __name__ == "__main__":
    main()
