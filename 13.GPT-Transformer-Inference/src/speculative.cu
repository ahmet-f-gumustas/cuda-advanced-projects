#include "../include/transformer.h"
#include "../include/transformer_kernels.cuh"
#include "../include/kv_cache.h"
#include "../include/cuda_utils.h"
#include <cuda_fp16.h>
#include <vector>
#include <iostream>
#include <cstring>

// ============================================================================
// SpeculativeDecoder
// ============================================================================
// Implements speculative decoding:
//   1. Draft model generates K candidate tokens autoregressively (greedy).
//   2. Target model verifies each draft token sequentially.
//   3. Accept tokens up to the first mismatch; reject the rest.
//
// Speedup analysis:
//   If acceptance_rate = α (fraction of draft tokens accepted):
//   Expected tokens per target call ≈ 1 / (1 - α^K)
//   For α=0.8, K=4 → ~3.36 tokens per target call (vs 1.0 without speculative).
//
// Note: True parallel verification (batched prefill) would require additional
// complexity. This implementation verifies sequentially for clarity.
// ============================================================================

class SpeculativeDecoder {
public:
    SpeculativeDecoder(TransformerModel* draft_model,
                       TransformerModel* target_model,
                       int K = 4)
        : draft_model_(draft_model)
        , target_model_(target_model)
        , K_(K)
    {
        // Allocate logit buffers
        int vocab = target_model_->config().vocab_size;
        d_draft_logits_  = cudaMallocDevice<float>(vocab);
        d_target_logits_ = cudaMallocDevice<float>(vocab);
        d_out_token_     = cudaMallocDevice<int>(1);
        d_max_scratch_   = cudaMallocDevice<float>(1);

        // Create KV caches for both models
        const auto& tc = target_model_->config();
        const auto& dc = draft_model_->config();

        target_kv_ = KVCache::create(tc.n_layers, tc.n_kv_heads, tc.max_seq_len, tc.head_dim);
        draft_kv_  = KVCache::create(dc.n_layers, dc.n_kv_heads, dc.max_seq_len, dc.head_dim);
    }

    ~SpeculativeDecoder() {
        cudaFree(d_draft_logits_);
        cudaFree(d_target_logits_);
        cudaFree(d_out_token_);
        cudaFree(d_max_scratch_);
        target_kv_.destroy();
        draft_kv_.destroy();
    }

    // Prefill both models with a shared prompt
    void prefill(const std::vector<int>& prompt_ids) {
        draft_model_->prefill(prompt_ids, draft_kv_);
        target_model_->prefill(prompt_ids, target_kv_);
        base_pos_ = (int)prompt_ids.size();
    }

    // Run one speculative step: draft K tokens, verify with target
    // Returns accepted tokens (1..K+1 tokens added to output)
    std::vector<int> step(int last_token) {
        std::vector<int> draft_tokens;
        draft_tokens.reserve(K_);

        // ── Draft phase ──────────────────────────────────────────────────
        int cur_token = last_token;
        int draft_pos = base_pos_;

        for (int i = 0; i < K_; ++i) {
            draft_model_->forward(cur_token, d_draft_logits_, draft_kv_, draft_pos);
            draft_kv_.current_pos++;

            // Greedy sample from draft
            int smem_sz = PT_BLOCK_SIZE * 2 * sizeof(float);
            argmax_kernel<<<1, PT_BLOCK_SIZE, smem_sz>>>(
                d_draft_logits_, d_out_token_, d_max_scratch_,
                draft_model_->config().vocab_size);
            CUDA_CHECK_LAST_ERROR();
            CUDA_CHECK(cudaDeviceSynchronize());

            int next;
            CUDA_CHECK(cudaMemcpy(&next, d_out_token_, sizeof(int), cudaMemcpyDeviceToHost));
            draft_tokens.push_back(next);
            cur_token = next;
            draft_pos++;
        }

        // ── Verification phase ──────────────────────────────────────────
        // Run target model on each draft token; check if it agrees (greedy)
        std::vector<int> accepted;
        int verify_token = last_token;
        int verify_pos   = base_pos_;

        for (int i = 0; i < K_; ++i) {
            target_model_->forward(verify_token, d_target_logits_, target_kv_, verify_pos);
            target_kv_.current_pos++;

            // Greedy target prediction
            int smem_sz = PT_BLOCK_SIZE * 2 * sizeof(float);
            argmax_kernel<<<1, PT_BLOCK_SIZE, smem_sz>>>(
                d_target_logits_, d_out_token_, d_max_scratch_,
                target_model_->config().vocab_size);
            CUDA_CHECK_LAST_ERROR();
            CUDA_CHECK(cudaDeviceSynchronize());

            int target_next;
            CUDA_CHECK(cudaMemcpy(&target_next, d_out_token_, sizeof(int), cudaMemcpyDeviceToHost));

            if (target_next == draft_tokens[i]) {
                // Accept
                accepted.push_back(target_next);
                verify_token = target_next;
                verify_pos++;
            } else {
                // Reject: take target's prediction instead
                accepted.push_back(target_next);
                // Roll back draft KV cache to rejection point
                draft_kv_.current_pos = base_pos_ + (int)accepted.size();
                break;
            }
        }

        // If all K draft tokens accepted, run target one more time for bonus token
        if ((int)accepted.size() == K_) {
            target_model_->forward(verify_token, d_target_logits_, target_kv_, verify_pos);
            target_kv_.current_pos++;

            int smem_sz = PT_BLOCK_SIZE * 2 * sizeof(float);
            argmax_kernel<<<1, PT_BLOCK_SIZE, smem_sz>>>(
                d_target_logits_, d_out_token_, d_max_scratch_,
                target_model_->config().vocab_size);
            CUDA_CHECK_LAST_ERROR();
            CUDA_CHECK(cudaDeviceSynchronize());

            int bonus;
            CUDA_CHECK(cudaMemcpy(&bonus, d_out_token_, sizeof(int), cudaMemcpyDeviceToHost));
            accepted.push_back(bonus);
        }

        base_pos_ += (int)accepted.size();
        return accepted;
    }

    // Reset caches for a new prompt
    void reset() {
        draft_kv_.reset();
        target_kv_.reset();
        base_pos_ = 0;
    }

    // Run full generation loop with speculative decoding, print stats
    std::vector<int> generate(const std::vector<int>& prompt_ids,
                               int max_new_tokens,
                               int eos_id = 0)
    {
        reset();
        prefill(prompt_ids);

        std::vector<int> output;
        int last_token = prompt_ids.back();
        int total_draft = 0, total_accepted = 0;

        while ((int)output.size() < max_new_tokens) {
            auto batch = step(last_token);
            total_draft    += K_;
            total_accepted += (int)batch.size();

            for (int tok : batch) {
                if (tok == eos_id) goto done;
                output.push_back(tok);
                last_token = tok;
                if ((int)output.size() >= max_new_tokens) goto done;
            }
        }
    done:
        float acceptance_rate = (total_draft > 0)
            ? (float)total_accepted / (float)total_draft : 0.0f;
        std::cout << "[Speculative] Total draft=" << total_draft
                  << " accepted=" << total_accepted
                  << " rate=" << acceptance_rate * 100.0f << "%"
                  << " tokens_per_step=" << (float)total_accepted / ((float)total_draft / K_)
                  << std::endl;
        return output;
    }

private:
    TransformerModel* draft_model_;
    TransformerModel* target_model_;
    int               K_;
    int               base_pos_ = 0;

    KVCache  target_kv_;
    KVCache  draft_kv_;
    float*   d_draft_logits_;
    float*   d_target_logits_;
    int*     d_out_token_;
    float*   d_max_scratch_;
};

// ============================================================================
// Standalone speculative decoding benchmark
// ============================================================================

void run_speculative_benchmark(int K,
                                int max_tokens,
                                const TransformerConfig& target_cfg,
                                const TransformerConfig& draft_cfg)
{
    std::cout << "\n=== Speculative Decoding Benchmark (K=" << K << ") ===" << std::endl;

    TransformerModel target(target_cfg);
    TransformerModel draft(draft_cfg);

    target.initRandom(42);
    draft.initRandom(42);

    SpeculativeDecoder decoder(&draft, &target, K);

    // Dummy prompt: "Hello" as char tokens
    std::vector<int> prompt = {'H', 'e', 'l', 'l', 'o', ' '};

    CudaTimer timer;
    timer.start();
    auto out = decoder.generate(prompt, max_tokens, 0 /*eos*/);
    timer.stop();

    float ms = timer.elapsed();
    float tokens_per_sec = (float)out.size() / (ms / 1000.0f);

    std::cout << "  Generated " << out.size() << " tokens in "
              << ms << " ms  →  " << tokens_per_sec << " tokens/sec" << std::endl;
}
