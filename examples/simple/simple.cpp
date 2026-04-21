#include "llama.h"
#include <clocale>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static void print_usage(int, char ** argv) {
    printf("\nexample usage:\n");
    printf("\n    %s -m model.gguf [-n n_predict] [-ngl n_gpu_layers] [prompt]\n", argv[0]);
    printf("\n");
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    // path to the model gguf file
    std::string model_path;
    // prompt to generate text from
    std::string prompt = "Hello my name is";
    // number of layers to offload to the GPU
    int ngl = 99;
    // number of tokens to predict
    int n_predict = 32;

    // parse command line arguments

    {
        int i = 1;
        for (; i < argc; i++) {
            if (strcmp(argv[i], "-m") == 0) {
                if (i + 1 < argc) {
                    model_path = argv[++i];
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-n") == 0) {
                if (i + 1 < argc) {
                    try {
                        n_predict = std::stoi(argv[++i]);
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-ngl") == 0) {
                if (i + 1 < argc) {
                    try {
                        ngl = std::stoi(argv[++i]);
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else {
                // prompt starts here
                break;
            }
        }
        if (model_path.empty()) {
            print_usage(argc, argv);
            return 1;
        }
        if (i < argc) {
            prompt = argv[i++];
            for (; i < argc; i++) {
                prompt += " ";
                prompt += argv[i];
            }
        }
    }

    // load dynamic backends

    ggml_backend_load_all();

    // initialize the model

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);

    if (model == NULL) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    // tokenize the prompt

    // find the number of tokens in the prompt
    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);

    // allocate space for the tokens and tokenize the prompt
    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        fprintf(stderr, "%s: error: failed to tokenize the prompt\n", __func__);
        return 1;
    }

    // initialize the context

    llama_context_params ctx_params = llama_context_default_params();
    // n_ctx is the context size
    ctx_params.n_ctx = n_prompt + n_predict - 1;
    // n_batch is the maximum number of tokens that can be processed in a single call to llama_decode
    ctx_params.n_batch = n_prompt;
    // enable performance counters
    ctx_params.no_perf = false;

    llama_context * ctx = llama_init_from_model(model, ctx_params);

    if (ctx == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    // initialize the sampler

    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);

    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // print the prompt token-by-token

    for (auto id : prompt_tokens) {
        char buf[128];
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        if (n < 0) {
            fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
            return 1;
        }
        std::string s(buf, n);
        printf("%s", s.c_str());
    }

    // prepare a batch for the prompt

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

    if (llama_model_has_encoder(model)) {
        if (llama_encode(ctx, batch)) {
            fprintf(stderr, "%s : failed to eval\n", __func__);
            return 1;
        }

        llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
        if (decoder_start_token_id == LLAMA_TOKEN_NULL) {
            decoder_start_token_id = llama_vocab_bos(vocab);
        }

        batch = llama_batch_get_one(&decoder_start_token_id, 1);
    }

    // main loop

    const auto t_main_start = ggml_time_us();
    int n_decode = 0;
    llama_token new_token_id;

    for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + n_predict; ) {
        // evaluate the current batch with the transformer model
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
            return 1;
        }

        // SPIRAL DEBUG: dump logits after decode
        {
            static int logit_dump = 0;
            if (logit_dump < 2) {
                const float * logits = llama_get_logits(ctx);
                if (logits) {
                    int n_vocab = llama_vocab_n_tokens(vocab);
                    float max_val = logits[0];
                    int max_idx = 0;
                    float min_val = logits[0];
                    double sum = 0;
                    int nan_count = 0;
                    int zero_count = 0;
                    for (int i = 0; i < n_vocab; i++) {
                        if (logits[i] != logits[i]) { nan_count++; continue; }
                        if (logits[i] == 0.0f) zero_count++;
                        if (logits[i] > max_val) { max_val = logits[i]; max_idx = i; }
                        if (logits[i] < min_val) { min_val = logits[i]; }
                        sum += logits[i];
                    }
                    fprintf(stderr, "SPIRAL_LOGITS[%d]: n_vocab=%d mean=%.6f min=%.4f max=%.4f max_idx=%d nan=%d zero=%d\n",
                        logit_dump, n_vocab, (float)(sum/n_vocab), min_val, max_val, max_idx, nan_count, zero_count);
                    fprintf(stderr, "  first 10: %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
                        logits[0], logits[1], logits[2], logits[3], logits[4],
                        logits[5], logits[6], logits[7], logits[8], logits[9]);
                    // Check if all logits are the same
                    bool all_same = true;
                    for (int i = 1; i < std::min(n_vocab, 1000); i++) {
                        if (logits[i] != logits[0]) { all_same = false; break; }
                    }
                    fprintf(stderr, "  all_same=%s\n", all_same ? "YES (degenerate!)" : "no");
                } else {
                    fprintf(stderr, "SPIRAL_LOGITS[%d]: logits pointer is NULL!\n", logit_dump);
                }
                logit_dump++;
            }
        }

        n_pos += batch.n_tokens;

        // sample the next token
        {
            new_token_id = llama_sampler_sample(smpl, ctx, -1);

            // SPIRAL DEBUG: what did the sampler pick?
            {
                static int sample_log = 0;
                if (sample_log < 10) {
                    bool is_eog = llama_vocab_is_eog(vocab, new_token_id);
                    char tbuf[128];
                    memset(tbuf, 0, sizeof(tbuf));
                    int tn = llama_token_to_piece(vocab, new_token_id, tbuf, sizeof(tbuf), 0, true);
                    fprintf(stderr, "SPIRAL_SAMPLE[%d]: token_id=%d is_eog=%d n_bytes=%d", 
                        sample_log, new_token_id, is_eog ? 1 : 0, tn);
                    if (tn > 0) {
                        fprintf(stderr, " hex=");
                        for (int bi = 0; bi < tn && bi < 20; bi++) {
                            fprintf(stderr, "%02x", (unsigned char)tbuf[bi]);
                        }
                        fprintf(stderr, " text='%.*s'", tn, tbuf);
                    } else {
                        fprintf(stderr, " (EMPTY)");
                    }
                    fprintf(stderr, "\n");
                    sample_log++;
                }
            }

            // is it an end of generation?
            if (llama_vocab_is_eog(vocab, new_token_id)) {
                break;
            }

            char buf[128];
            int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
            if (n < 0) {
                fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
                return 1;
            }
            std::string s(buf, n);
            printf("%s", s.c_str());
            fflush(stdout);

            // prepare the next batch with the sampled token
            batch = llama_batch_get_one(&new_token_id, 1);

            n_decode += 1;
        }
    }

    printf("\n");

    const auto t_main_end = ggml_time_us();

    fprintf(stderr, "%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
            __func__, n_decode, (t_main_end - t_main_start) / 1000000.0f, n_decode / ((t_main_end - t_main_start) / 1000000.0f));

    fprintf(stderr, "\n");
    llama_perf_sampler_print(smpl);
    llama_perf_context_print(ctx);
    fprintf(stderr, "\n");

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);

    return 0;
}