"""
Verify: does rotating the input to the MoE router change expert selection?

If yes: the rotation before build_moe_ffn is the bug.
If no: the rotation doesn't affect expert selection and the bug is elsewhere.
"""
import struct, numpy as np, math

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

def rms_norm(x, weight, eps=1e-6):
    rms = np.sqrt(np.mean(x ** 2) + eps)
    return (x / rms) * weight

def fwht_block(x, block_size=128):
    d = len(x)
    n_blocks = d // block_size
    x = x.reshape(n_blocks, block_size).copy()
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    for stage in range(int(np.log2(block_size))):
        h = 1 << stage
        for blk in range(n_blocks):
            for i in range(0, block_size, 2*h):
                for j in range(i, i+h):
                    a, b = x[blk, j], x[blk, j+h]
                    x[blk, j] = (a + b) * inv_sqrt2
                    x[blk, j+h] = (a - b) * inv_sqrt2
    return x.reshape(d)

def mpb_forward(x, signs_list, perms_list, block_size=128):
    x = x.copy()
    for p in range(len(signs_list)):
        x = x * signs_list[p]
        if p < len(perms_list):
            x = x[perms_list[p]]
        x = fwht_block(x, block_size)
    return x

def generate_rotation(seed, d):
    n_passes = max(2, math.ceil(math.log(d) / math.log(128)))
    rng = np.random.default_rng(seed + d)
    signs, perms = [], []
    for p in range(n_passes):
        signs.append(rng.choice([-1.0, 1.0], size=d).astype(np.float32))
        if p < n_passes - 1:
            perms.append(rng.permutation(d).astype(np.int64))
    return signs, perms

# ============================================================
gguf = "runs/qwen3-coder-30b-spiral.gguf"
ts, ds = parse_gguf(gguf)
hidden = 2048
seed = 0x524F5431

# Step 1: Get the activation AFTER attn (use embedding + norm as proxy)
token_id = 32
x = read_f16(gguf, ds, ts["token_embd.weight"][2] + token_id * hidden * 2, hidden)

# Apply attn_norm
attn_norm_w = read_f32(gguf, ds, ts["blk.0.attn_norm.weight"][2], hidden)
cur = rms_norm(x, attn_norm_w)

# For this test, we skip the actual attention computation and use
# the post-attn-norm activation as if it were the FFN input.
# The point is: does rotation change the router's expert selection?

# Apply ffn_norm (this is what the router actually sees)
ffn_norm_w = read_f32(gguf, ds, ts["blk.0.ffn_norm.weight"][2], hidden)
ffn_input = rms_norm(x, ffn_norm_w)  # simplified: using embedding directly
print(f"FFN input (after ffn_norm): norm={np.linalg.norm(ffn_input):.6f}")
print(f"  first5 = {ffn_input[:5]}")

# Step 2: Rotate the FFN input
signs_2048, perms_2048 = generate_rotation(seed, 2048)
ffn_rotated = mpb_forward(ffn_input, signs_2048, perms_2048)
print(f"\nRotated FFN input: norm={np.linalg.norm(ffn_rotated):.6f}")
print(f"  first5 = {ffn_rotated[:5]}")

# Step 3: Read router weights (F32, shape [2048, 128])
router_dims, router_dt, router_off = ts["blk.0.ffn_gate_inp.weight"]
print(f"\nRouter weights: dims={router_dims}, dtype={router_dt}")
n_experts = router_dims[1]  # 128
W_router = read_f32(gguf, ds, router_off, hidden * n_experts).reshape(n_experts, hidden)
print(f"  W_router shape: {W_router.shape}")

# Step 4: Compute router logits with UNROTATED input (CORRECT)
logits_correct = W_router @ ffn_input
probs_correct = np.exp(logits_correct) / np.sum(np.exp(logits_correct))
top8_correct = np.argsort(probs_correct)[-8:][::-1]
print(f"\nCORRECT (unrotated) router:")
print(f"  logits range: [{logits_correct.min():.4f}, {logits_correct.max():.4f}]")
print(f"  top-8 experts: {top8_correct}")
print(f"  top-8 probs: {probs_correct[top8_correct]}")

# Step 5: Compute router logits with ROTATED input (BUGGY)
logits_buggy = W_router @ ffn_rotated
probs_buggy = np.exp(logits_buggy) / np.sum(np.exp(logits_buggy))
top8_buggy = np.argsort(probs_buggy)[-8:][::-1]
print(f"\nBUGGY (rotated) router:")
print(f"  logits range: [{logits_buggy.min():.4f}, {logits_buggy.max():.4f}]")
print(f"  top-8 experts: {top8_buggy}")
print(f"  top-8 probs: {probs_buggy[top8_buggy]}")

# Step 6: Compare
overlap = len(set(top8_correct) & set(top8_buggy))
print(f"\n=== VERDICT ===")
print(f"Expert selection overlap: {overlap}/8")
if overlap < 4:
    print(f"CONFIRMED: Rotation corrupts expert selection!")
    print(f"The router receives rotated input but expects unrotated.")
    print(f"FIX: Move rotation from qwen3moe.cpp (before build_moe_ffn)")
    print(f"     to inside build_moe_ffn (after router, before expert matmuls)")
elif overlap == 8:
    print(f"Expert selection is IDENTICAL — rotation does NOT affect routing.")
    print(f"The bug is elsewhere.")
else:
    print(f"Partial overlap — rotation partially corrupts routing.")
    print(f"This contributes to the problem but may not be the only issue.")