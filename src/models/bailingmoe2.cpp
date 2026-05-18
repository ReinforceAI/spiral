#include "models.h"

// Spiral weight quantization family detection (mirrors llama-graph.cpp).
// Recognizes all v3+v4 weight quant types so the bailingmoe2-specific
// rotation Hook A fires for both legacy SPIRAL_3BIT and the new
// SPIRAL_INT4/INT5 types. Excludes SPIRAL_PQ2 (KV cache type — handled
// separately by K cache decode kernels).
//
// Sarvam-30B-Bharat-1 uses bailingmoe2 architecture with fused QKV (single
// wqkv tensor of shape [n_embd + 2 * n_embd_gqa, n_embd]) — Q, K, V are
// views of one matmul output. A single rotation before wqkv covers all
// three. The o_proj rotation (Hook D) fires automatically inside
// build_attn after the llama-graph.cpp v3-only stale-guard fix.
static inline bool is_spiral_quant_weight(ggml_type t) {
    return t == GGML_TYPE_SPIRAL_3BIT
        || t == GGML_TYPE_SPIRAL_INT4
        || t == GGML_TYPE_SPIRAL_INT5;
}

llm_build_bailingmoe2::llm_build_bailingmoe2(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    const int n_transformer_layers = n_layer - hparams.nextn_predict_layers;
    for (int il = 0; il < n_transformer_layers; ++il) {
        ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self_attention
        {
            // Spiral Hook A: rotate activation before fused QKV projection.
            // Sarvam-30B has fused QKV (single wqkv weight), unlike qwen35moe
            // which splits into wq/wk/wv. One rotation covers Q, K, V which are
            // all views of the wqkv matmul output. Rotation is a graceful no-op
            // if codebook params aren't registered for cur->ne[0] = n_embd, so
            // this is safe to add before .spiralcb sidecar loading is wired up.
            // Per-hook env kill switch for diagnosis: SPIRAL_NO_HOOK_A=1 skips.
            static const bool no_hook_a = (getenv("SPIRAL_NO_HOOK_A") != nullptr);
            if (!no_hook_a && is_spiral_quant_weight(model.layers[il].wqkv->type)) {
                cur = spiral_rotate_activation(cur, cur->ne[0]);
            }

            cur = build_lora_mm(model.layers[il].wqkv, cur);
            cb(cur, "wqkv", il);

            ggml_tensor * Qcur = ggml_view_3d(ctx0, cur, n_embd_head, n_head, n_tokens, n_embd_head * sizeof(float),
                                              cur->nb[1], 0 * sizeof(float) * (n_embd));
            ggml_tensor * Kcur = ggml_view_3d(ctx0, cur, n_embd_head, n_head_kv, n_tokens, n_embd_head * sizeof(float),
                                              cur->nb[1], 1 * sizeof(float) * (n_embd));
            ggml_tensor * Vcur = ggml_view_3d(ctx0, cur, n_embd_head, n_head_kv, n_tokens, n_embd_head * sizeof(float),
                                              cur->nb[1], 1 * sizeof(float) * (n_embd + n_embd_gqa));

            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);

            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                 ext_factor, attn_factor, beta_fast, beta_slow);

            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
            cb(Kcur, "Kcur_normed", il);

            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                 ext_factor, attn_factor, beta_fast, beta_slow);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            // Spiral Hook D fires automatically inside build_attn for o_proj
            // (wo). The llama-graph.cpp fix replaced the stale v3-only type
            // guard `wo->type == GGML_TYPE_SPIRAL_3BIT` with
            // `is_spiral_quant_weight(wo->type)`, so SPIRAL_INT4 weights now
            // trigger the rotation correctly. No model-side hook needed here.
            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].bo,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f / sqrtf(float(n_embd_head)), il);
        }

        if (il == n_transformer_layers - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * sa_out = ggml_add(ctx0, cur, inpSA);
        cb(sa_out, "sa_out", il);

        // MoE branch
        cur = build_norm(sa_out, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        if (static_cast<uint32_t>(il) < hparams.n_layer_dense_lead) {
            // Dense FFN path (Sarvam: L=0 only). Spiral rotation hooks inside
            // build_ffn fire automatically for SPIRAL_INT4/INT5 weights via
            // the llama-graph.cpp fix; for Sarvam's bf16 dense L0 the hooks
            // are no-ops by type guard.
            cur = build_ffn(cur,
                    model.layers[il].ffn_up, NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            // MoE path (Sarvam: L>=1). Routed expert rotations fire
            // automatically inside build_moe_ffn for SPIRAL_INT4 weights —
            // those hooks were already correct (used is_spiral_quant_weight
            // pre-fix) since the v4 launch with Qwen36.
            ggml_tensor * moe_out = build_moe_ffn(cur,
                model.layers[il].ffn_gate_inp,
                model.layers[il].ffn_up_exps,
                model.layers[il].ffn_gate_exps,
                model.layers[il].ffn_down_exps,
                model.layers[il].ffn_exp_probs_b,
                n_expert, n_expert_used,
                LLM_FFN_SILU, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il);
            cb(moe_out, "ffn_moe_out", il);

            {
                // Shared expert path. Rotation hooks inside build_ffn fire
                // automatically for SPIRAL_INT4/INT5 via the llama-graph.cpp
                // fix; for Sarvam's bf16 shared experts the hooks are no-ops.
                ggml_tensor * ffn_shexp =
                    build_ffn(cur,
                        model.layers[il].ffn_up_shexp, NULL, NULL,
                        model.layers[il].ffn_gate_shexp, NULL, NULL,
                        model.layers[il].ffn_down_shexp, NULL, NULL,
                        NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(ffn_shexp, "ffn_shexp", il);

                cur = ggml_add(ctx0, moe_out, ffn_shexp);
                cb(cur, "ffn_out", il);
            }
        }

        cur = ggml_add(ctx0, cur, sa_out);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
