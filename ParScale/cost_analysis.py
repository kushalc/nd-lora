import numpy as np
import json
import os
from llm_analysis.analysis import LLMAnalysis, get_gpu_config_by_name, ModelConfig, ActivationRecomputation, BYTES_FP16


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # General model config
    parser.add_argument('--hidden_size', type=int, required=True)
    parser.add_argument('--intermediate_size', type=int, required=True)
    parser.add_argument('--num_hidden_layers', type=int, default=36)
    parser.add_argument('--num_attention_heads', type=int, default=16)
    parser.add_argument('--max_position_embeddings', type=int, default=2048)
    parser.add_argument('--num_key_value_heads', type=int, default=2)
    parser.add_argument('--vocab_size', type=int, default=151936)

    # Parscale config
    parser.add_argument('--P', type=int, default=1) # Number of parallel streams
    parser.add_argument('--parscale_prefix_tokens', type=int, default=48) # Number of prefix tokens

    # Data config
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--input_length', type=int, default=64)
    parser.add_argument('--output_length', type=int, default=64)

    # GPU config
    parser.add_argument('--gpu_config', type=str, default="a100-sxm-80gb")
    parser.add_argument('--flops_efficiency', type=float, default=0.7) # Recommended by llm-analysis
    parser.add_argument('--hbm_memory_efficiency', type=float, default=0.9) # Recommended by llm-analysis

    args = parser.parse_args()
    p = args.P
    model_config = ModelConfig(
        name="", 
        num_layers=args.num_hidden_layers, 
        n_head=args.num_attention_heads, 
        hidden_dim=args.hidden_size, vocab_size=args.vocab_size, 
        max_seq_len=args.max_position_embeddings + (args.parscale_prefix_tokens if p > 1 else 0), 
        num_key_value_heads=args.num_key_value_heads, 
        ffn_embed_dim=args.intermediate_size, 
        mlp_gated_linear_units=True
    )
    gpu_config = get_gpu_config_by_name("a100-sxm-80gb")
    gpu_config.mem_per_GPU_in_GB = 10000

    analysis = LLMAnalysis(
        model_config,
        gpu_config,
        flops_efficiency=0.7,
        hbm_memory_efficiency=0.9,
    )
    seq_len = args.input_length + (args.parscale_prefix_tokens if p > 1 else 0)
    summary_dict = analysis.inference(
        batch_size_per_gpu=args.batch_size * p,
        seq_len=seq_len,
        num_tokens_to_generate=args.output_length,
    )

    # We consider the influence of the aggregation layer. 
    aggregate_param = (args.hidden_size + 1) * args.hidden_size * p if p > 1 else 0
    aggregate_param_vs_fwd_param = aggregate_param / analysis.get_num_params_per_layer_mlp()
    aggregate_latency = aggregate_param_vs_fwd_param * analysis.get_latency_fwd_per_layer_mlp(args.batch_size, args.input_length + args.output_length) if p > 1 else 0
    aggregate_memory = aggregate_param * analysis.dtype_config.weight_bits / 8

    prefill_activation_memory_per_gpu = max(
        # Each layer's activation memory will increase by P times
        analysis.get_activation_memory_per_layer(
            args.batch_size * p,
            seq_len,
            is_inference=True,
            layernorm_dtype_bytes=BYTES_FP16,
        ),
        # The embedding's activation memory will not participate in parallel and independent of P.
        analysis.get_activation_memory_output_embedding(
            args.batch_size, seq_len
        )
    )

    # Since we use batch_size * p as the new batch size, the latency for llm-analysis assumes the embedding latency is also computed in this new batch size. However, ParScale will not increase the computation for embedding.
    # Therefore, we should make a fix toward it. 
    embedding_latency_estimate_for_embedding = (
        analysis.get_latency_fwd_input_embedding(args.batch_size * p, args.input_length + args.output_length, dtype_bytes=analysis.dtype_config.embedding_bits) + 
        analysis.get_latency_fwd_output_embedding_loss(args.batch_size * p, args.input_length + args.output_length)
    )
    embedding_latency_real_for_embedding = (
        analysis.get_latency_fwd_input_embedding(args.batch_size, args.input_length + args.output_length, dtype_bytes=analysis.dtype_config.embedding_bits) + 
        analysis.get_latency_fwd_output_embedding_loss(args.batch_size, args.input_length + args.output_length)
    )

    total_memory = (
        summary_dict['kv_cache_memory_per_gpu'] + 
        summary_dict['weight_memory_per_gpu'] + 
        aggregate_memory + 
        prefill_activation_memory_per_gpu
    )
    total_latency = (
        summary_dict['total_latency'] + aggregate_latency
        - embedding_latency_estimate_for_embedding
        + embedding_latency_real_for_embedding
    )
    print(f"Memory: {total_memory / 2**30:.3f}GB; Latency: {total_latency:.3f}s")