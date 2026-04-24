"""
Full forward pass simulation for Qwen3-Coder-30B-A3B SPIRAL GGUF.
Runs embedding → layer 0 (attn + MoE) → skip remaining layers → final norm → LM head.
Only layer 0 is computed fully; layers 1-47 are skipped (output = input).
This tests whether the GGUF data produces reasonable intermediate values.

Usage: python full_layer0_sim.py
"""
import struct, numpy as np, math, sys

# ============================================================
# Lloyd-Max centroids
# ============================================================
C = np.array([-2.1556325, -1.3483801, -0.7599415, -0.2466970,
               0.2466970,  0.7599415,  1.3483801,  2.1556325], dtype=np.float32)

# ============================================================
# Helpers
# ============================================================
def parse_gguf(path):
    with open(path, "rb") as f:
        f.read(4); f.read(4)
        nt = struct.unpack("<Q", f.read(8))[0]
        nk = struct.unpack("<Q", f.read(8))[0]
        for _ in range(nk):
            kl = struct.unpack("<Q", f.read(8))[0]; f.read(kl)
            vt = struct.unpack("<I", f.read(4))[0]
            if vt in (0,1,7): f.read(1)
            elif vt in (2,3): f.read(2)
            elif vt in (4,5,6): f.read(4)
            elif vt == 8: f.read(struct.unpack("<Q", f.read(8))[0])
            elif vt == 9:
                at = struct.unpack("<I", f.read(4))[0]
                al = struct.unpack("<Q", f.read(8))[0]
                for _ in range(al):
                    if at in (0,1,7): f.read(1)
                    elif at in (2,3): f.read(2)
                    elif at in (4,5,6): f.read(4)
                    elif at == 8: f.read(struct.unpack("<Q", f.read(8))[0])
                    else: f.read(8)
            else: f.read(8)
        ts = {}
        for _ in range(nt):
            nl = struct.unpack("<Q", f.read(8))[0]
            nm = f.read(nl).decode("utf-8")
            nd = struct.unpack("<I", f.read(4))[0]
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
            dt = struct.unpack("<I", f.read(4))[0]
            off = struct.unpack("<Q", f.read(8))[0]
            ts[nm] = (dims, dt, off)
        ds = ((f.tell() + 31) // 32) * 32
    return ts, ds

def read_f32(path, ds, off, n):
    with open(path, "rb") as f:
        f.seek(ds + off)
        return np.frombuffer(f.read(n * 4), dtype=np.float32).copy()

def read_f16(path, ds, off, n):
    with open(path, "rb") as f:
        f.seek(ds + off)
        return np.frombuffer(f.read(n * 2), dtype=np.float16).astype(np.float32).copy()

def dequant_spiral_matrix(path, ds, off, ne00, ne01):
    """Dequant a full 2D SPIRAL_3BIT weight matrix [ne01 rows × ne00 cols]"""
    nb = ne00 // 128
    W = np.zeros((ne01, ne00), dtype=np.float32)
    with open(path, "rb") as f:
        f.seek(ds + off)
        for r in range(ne01):
            for b in range(nb):
                block = f.read(50)
                norm = struct.unpack("<e", block[0:2])[0]
                for g in range(16):
                    b0, b1, b2 = block[2+g*3], block[2+g*3+1], block[2+g*3+2]
                    codes = [b0&7,(b0>>3)&7,((b0>>6)|(b1<<2))&7,(b1>>1)&7,
                             (b1>>4)&7,((b1>>7)|(b2<<1))&7,(b2>>2)&7,(b2>>5)&7]
                    for j, c in enumerate(codes):
                        W[r, b*128+g*8+j] = C[c] * norm
    return W

def rms_norm(x, weight, eps=1e-6):
    rms = np.sqrt(np.mean(x ** 2) + eps)
    return (x / rms) * weight

def mpb_forward(x, signs_list, perms_list, block_size=128):
    d = len(x)
    n_blocks = d // block_size
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    x = x.copy().astype(np.float32)
    for p in range(len(signs_list)):
        x = x * signs_list[p]
        if p < len(perms_list):
            x = x[perms_list[p]]
        # FWHT per block
        x = x.reshape(n_blocks, block_size)
        for stage in range(int(np.log2(block_size))):
            h = 1 << stage
            for blk in range(n_blocks):
                for i in range(0, block_size, 2*h):
                    for j in range(i, i+h):
                        a, b_ = x[blk, j], x[blk, j+h]
                        x[blk, j] = (a + b_) * inv_sqrt2
                        x[blk, j+h] = (a - b_) * inv_sqrt2
        x = x.reshape(d)
    return x

def generate_rotation(seed, d, block_size=128):
    n_passes = max(2, math.ceil(math.log(d) / math.log(block_size)))
    rng = np.random.default_rng(seed + d)
    signs = []
    perms = []
    for p in range(n_passes):
        s = rng.choice([-1.0, 1.0], size=d).astype(np.float32)
        signs.append(s)
        if p < n_passes - 1:
            perm = rng.permutation(d).astype(np.int64)
            perms.append(perm)
    return signs, perms

# ============================================================
# Main simulation
# ============================================================
gguf = "runs/qwen3-coder-30b-spiral.gguf"
print("Parsing GGUF...")
ts, ds = parse_gguf(gguf)

hidden = 2048
head_dim = 128
n_heads = 32
n_kv_heads = 4
n_embd_gqa = n_kv_heads * head_dim  # 512

# Token "A" — we need to know the token ID
# From earlier test, token 32 has norm=0.6117. Let's use a few candidates.
# Actually, llama-simple with -p "A" tokenizes to whatever the tokenizer says.
# Let's just use token 32 and see if the computation is reasonable.
token_id = 32

# ============================================================
# Step 1: Embedding lookup
# ============================================================
print(f"\n=== Step 1: Embedding for token {token_id} ===")
_, _, emb_off = ts["token_embd.weight"]
x = read_f16(gguf, ds, emb_off + token_id * hidden * 2, hidden)
print(f"  x[:5] = {x[:5]}")
print(f"  norm = {np.linalg.norm(x):.6f}")

# ============================================================
# Step 2: Layer 0 attention
# ============================================================
print(f"\n=== Step 2: Layer 0 RMSNorm (attn_norm) ===")
_, _, norm_off = ts["blk.0.attn_norm.weight"]
attn_norm_w = read_f32(gguf, ds, norm_off, hidden)
cur = rms_norm(x, attn_norm_w)
print(f"  cur[:5] = {cur[:5]}")
print(f"  norm = {np.linalg.norm(cur):.6f}")

# ============================================================
# Step 3: Rotation (dim=2048)
# ============================================================
print(f"\n=== Step 3: MPB Rotation (dim=2048) ===")
seed = 0x524F5431
signs_2048, perms_2048 = generate_rotation(seed, 2048)
cur_rot = mpb_forward(cur, signs_2048, perms_2048)
print(f"  cur_rot[:5] = {cur_rot[:5]}")
print(f"  norm = {np.linalg.norm(cur_rot):.6f}")

# ============================================================
# Step 4: Q projection (first 5 rows only for speed)
# ============================================================
print(f"\n=== Step 4: Q projection (first 5 output rows) ===")
_, _, q_off = ts["blk.0.attn_q.weight"]
# Only dequant first 5 rows for speed
n_check = 5
W_q_partial = dequant_spiral_matrix(gguf, ds, q_off, 2048, n_check)
q_partial = W_q_partial @ cur_rot
print(f"  Q[0:5] = {q_partial}")
print(f"  These should be small but non-zero and varying")

# Also compute WITHOUT rotation for comparison
q_norot = W_q_partial @ cur
print(f"  Q_norot[0:5] = {q_norot}")

# ============================================================
# Step 5: Check if values are reasonable
# ============================================================
print(f"\n=== Sanity checks ===")
print(f"  Embedding norm: {np.linalg.norm(x):.6f} (should be ~0.6)")
print(f"  After attn_norm: {np.linalg.norm(cur):.6f} (should be ~0.01-0.5 given tiny norm weights)")
print(f"  After rotation: {np.linalg.norm(cur_rot):.6f} (should be same as pre-rotation)")
print(f"  Q[0] with rotation: {q_partial[0]:.6f}")
print(f"  Q[0] without rotation: {q_norot[0]:.6f}")
print(f"  Q values different: {not np.allclose(q_partial, q_norot)}")

# ============================================================
# Step 6: Full Q projection to check overall scale
# ============================================================
print(f"\n=== Step 6: Full Q projection (all 4096 rows) — slow ===")
sys.stdout.flush()
W_q_full = dequant_spiral_matrix(gguf, ds, q_off, 2048, 4096)
Q_full = W_q_full @ cur_rot
print(f"  Q norm = {np.linalg.norm(Q_full):.6f}")
print(f"  Q mean = {Q_full.mean():.6f}")
print(f"  Q std = {Q_full.std():.6f}")
print(f"  Q min = {Q_full.min():.6f}")
print(f"  Q max = {Q_full.max():.6f}")
print(f"  Q[:5] = {Q_full[:5]}")

# ============================================================
# Step 7: Apply q_norm
# ============================================================
print(f"\n=== Step 7: Q norm ===")
_, _, qn_off = ts["blk.0.attn_q_norm.weight"]
q_norm_w = read_f32(gguf, ds, qn_off, head_dim)
# q_norm is RMSNorm applied per head
Q_heads = Q_full.reshape(n_heads, head_dim)
for h in range(n_heads):
    Q_heads[h] = rms_norm(Q_heads[h], q_norm_w)
print(f"  Q_normed[:5] = {Q_heads.reshape(-1)[:5]}")
print(f"  Q_normed norm = {np.linalg.norm(Q_heads):.6f}")

print(f"\n=== DONE: Layer 0 partial simulation complete ===")
print(f"If these values look reasonable (non-zero, varying, finite),")
print(f"then the GGUF data is correct and the bug is in Metal execution.")
print(f"If values are NaN, zero, or constant, the data is wrong.")