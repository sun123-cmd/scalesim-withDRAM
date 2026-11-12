"""Weight-free GEMM shapes for the default OpenPI pi0.5 sample_actions path.

The operation families are grouped by identical mathematical shape. ``count``
includes transformer depth, camera views, attention heads, denoising steps, and
batch instances where applicable. Each entry means count independent products
A[m, k] @ B[k, n]. The list describes a workload, not an execution scheduler.

Derived from the pinned official OpenPI sources linked in SOURCE_URLS. Dense
projections, both attention products, and the patch Conv lowered to GEMM are
covered. Normalization, softmax, RoPE, nonlinearities, embedding-table gathers,
image preprocessing, and Euler/elementwise updates are outside the GEMM model.
No checkpoint or tensor framework is needed to generate these shapes.
"""

OPENPI_COMMIT = "215abfb217dbac7d5f1273282331b9b1866c0479"
_SOURCE_BASE = "https://github.com/Physical-Intelligence/openpi/blob/" + OPENPI_COMMIT
SOURCE_URLS = {
    name: _SOURCE_BASE + "/src/openpi/models/" + name + ".py"
    for name in ("pi0", "pi0_config", "gemma", "siglip", "model")
}

MODEL_CONFIG = {
    "name": "pi0.5",
    "image_height": 224,
    "image_width": 224,
    "patch_size": 14,
    "vision_width": 1152,
    "vision_layers": 27,
    "vision_mlp": 4304,
    "vision_heads": 16,
    "language_width": 2048,
    "language_layers": 18,
    "language_mlp": 16384,
    "expert_width": 1024,
    "expert_layers": 18,
    "expert_mlp": 4096,
    "gemma_heads": 8,
    "gemma_kv_heads": 1,
    "gemma_head_dim": 256,
    "action_dim": 32,
    "dtype": "bfloat16",
}

EXCLUDED_OPERATIONS = [
    "image preprocessing and tokenization",
    "token embedding-table gathers",
    "sinusoidal timestep embedding",
    "LayerNorm, RMSNorm, and AdaRMS elementwise application",
    "softmax, RoPE, GELU, SiLU, and attention masks",
    "elementwise gates, residual additions, and Euler integration",
]


def build_operations(cameras=3, prompt_tokens=200, action_horizon=50,
                     denoise_steps=10, batch_size=1):
    """Return all GEMM families for one action-chunk generation.

    Pi0Config(pi05=True) supplies a 200-token padded prompt, 50 action tokens,
    32 action channels, and three camera slots. sample_actions defaults to ten
    Euler steps. State is encoded in the prompt, so no extra state token is
    appended. Image and language prefix KV is constructed once and reused by
    every action-expert step.
    """
    arguments = {"cameras": cameras, "prompt_tokens": prompt_tokens,
                 "action_horizon": action_horizon, "denoise_steps": denoise_steps,
                 "batch_size": batch_size}
    for name, value in arguments.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        minimum = 0 if name == "prompt_tokens" else 1
        if value < minimum:
            raise ValueError(f"{name} must be at least {minimum}")

    operations = []

    def add(name, m, n, k, count, stage):
        operations.append({"name": name, "m": m, "n": n, "k": k,
                           "count": count, "stage": stage})

    # siglip.decode_variant('So400m/14') and pi0.Pi0.__init__. At 224x224,
    # the non-overlapping Conv stem emits 16x16 patches and no CLS token.
    image_tokens = (224 // 14) ** 2
    images = cameras * batch_size
    vision_blocks = images * 27
    add("vision.patch_embedding", image_tokens, 1152, 14 * 14 * 3, images, "vision")
    # Flax MHSA has distinct query/key/value dense maps, each 1152->1152.
    add("vision.blocks.qkv", image_tokens, 1152, 1152, 3 * vision_blocks, "vision")
    add("vision.blocks.qk", image_tokens, image_tokens, 1152 // 16,
        16 * vision_blocks, "vision")
    add("vision.blocks.av", image_tokens, 1152 // 16, image_tokens,
        16 * vision_blocks, "vision")
    add("vision.blocks.attention_output", image_tokens, 1152, 1152, vision_blocks, "vision")
    add("vision.blocks.mlp_up", image_tokens, 4304, 1152, vision_blocks, "vision")
    add("vision.blocks.mlp_down", image_tokens, 1152, 4304, vision_blocks, "vision")
    # pool_type='none', num_classes=2048. Only the returned image-token head
    # is live in Pi0.embed_prefix; the diagnostic x_2d branch has no consumer.
    add("vision.multimodal_projector", image_tokens, 2048, 1152, images, "vision")

    prefix_tokens = cameras * image_tokens + prompt_tokens
    prefix_blocks = batch_size * 18
    # gemma.Attention uses MQA. There are eight 256-wide Q heads, while
    # K and V each contain a single shared 256-wide head.
    add("prefix.blocks.q", prefix_tokens, 8 * 256, 2048, prefix_blocks, "prefix")
    add("prefix.blocks.kv", prefix_tokens, 256, 2048, 2 * prefix_blocks, "prefix")
    add("prefix.blocks.qk", prefix_tokens, prefix_tokens, 256, 8 * prefix_blocks, "prefix")
    add("prefix.blocks.av", prefix_tokens, 256, prefix_tokens, 8 * prefix_blocks, "prefix")
    add("prefix.blocks.attention_output", prefix_tokens, 2048, 8 * 256, prefix_blocks, "prefix")
    # gemma.FeedForward has two separate up maps (gate and value) and one down.
    add("prefix.blocks.mlp_gate_and_up", prefix_tokens, 16384, 2048,
        2 * prefix_blocks, "prefix")
    add("prefix.blocks.mlp_down", prefix_tokens, 2048, 16384, prefix_blocks, "prefix")

    steps = batch_size * denoise_steps
    expert_blocks = steps * 18
    # pi0.embed_suffix computes these again on every denoising iteration.
    add("denoise.action_input", action_horizon, 1024, 32, steps, "denoise")
    add("denoise.time_mlp", 1, 1024, 1024, 2 * steps, "denoise")
    # Two conditioned norms per block, plus Module.final_norms for the expert.
    # Every conditioned RMSNorm projects one time vector to scale/shift/gate.
    add("denoise.adarms_modulation", 1, 3 * 1024, 1024,
        (2 * 18 + 1) * steps, "denoise")
    add("denoise.blocks.q", action_horizon, 8 * 256, 1024, expert_blocks, "denoise")
    add("denoise.blocks.kv", action_horizon, 256, 1024, 2 * expert_blocks, "denoise")
    key_tokens = prefix_tokens + action_horizon
    add("denoise.blocks.qk", action_horizon, key_tokens, 256,
        8 * expert_blocks, "denoise")
    add("denoise.blocks.av", action_horizon, 256, key_tokens,
        8 * expert_blocks, "denoise")
    add("denoise.blocks.attention_output", action_horizon, 1024, 8 * 256,
        expert_blocks, "denoise")
    add("denoise.blocks.mlp_gate_and_up", action_horizon, 4096, 1024,
        2 * expert_blocks, "denoise")
    add("denoise.blocks.mlp_down", action_horizon, 1024, 4096, expert_blocks, "denoise")
    add("denoise.action_output", action_horizon, 32, 1024, steps, "denoise")
    return operations


def main():
    """Export the full-sized mathematical workload manifest as JSON."""
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cameras", type=int, default=3)
    parser.add_argument("--prompt-tokens", type=int, default=200)
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--denoise-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    configuration = {"cameras": args.cameras, "prompt_tokens": args.prompt_tokens,
                     "action_horizon": args.action_horizon,
                     "denoise_steps": args.denoise_steps, "batch_size": args.batch_size}
    try:
        operations = build_operations(**configuration)
    except ValueError as error:
        parser.error(str(error))
    configuration.update(MODEL_CONFIG)
    configuration["image_tokens_per_camera"] = 256
    configuration["prefix_tokens"] = args.cameras * 256 + args.prompt_tokens
    manifest = {"model": "pi0.5", "source_commit": OPENPI_COMMIT,
                "sources": SOURCE_URLS, "configuration": configuration,
                "excluded_operations": EXCLUDED_OPERATIONS, "operations": operations}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    total_macs = sum(op["m"] * op["n"] * op["k"] * op["count"] for op in operations)
    print(f"Exported {len(operations)} operation families, {total_macs:,} MACs to {args.output}")


if __name__ == "__main__":
    main()
