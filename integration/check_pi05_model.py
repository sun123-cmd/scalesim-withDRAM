"""Shape and repetition invariants for the pinned default pi0.5 workload."""
import unittest

from pi05_model import build_operations


class Pi05ModelShapes(unittest.TestCase):
    def operations(self, **kwargs):
        return {op["name"]: op for op in build_operations(**kwargs)}

    def test_default_camera_and_prefix_token_counts(self):
        ops = self.operations()
        self.assertEqual(ops["vision.patch_embedding"]["m"], 256)
        self.assertEqual(ops["vision.patch_embedding"]["count"], 3)
        self.assertEqual(ops["prefix.blocks.q"]["m"], 968)
        self.assertEqual(ops["denoise.blocks.qk"]["n"], 1018)

    def test_transformer_depths_heads_and_mqa(self):
        ops = self.operations()
        self.assertEqual(ops["vision.blocks.mlp_up"]["count"], 3 * 27)
        self.assertEqual(ops["vision.blocks.qkv"]["count"], 3 * 3 * 27)
        self.assertEqual(ops["prefix.blocks.q"]["count"], 18)
        self.assertEqual(ops["prefix.blocks.kv"]["count"], 2 * 18)
        self.assertEqual(ops["prefix.blocks.kv"]["n"], 256)
        self.assertEqual(ops["denoise.blocks.q"]["count"], 10 * 18)
        self.assertEqual(ops["denoise.blocks.q"]["n"], 2048)
        self.assertEqual(ops["denoise.adarms_modulation"]["count"], 10 * (2 * 18 + 1))

    def test_attention_matrix_dimensions_match(self):
        ops = self.operations()
        for stage in ("vision", "prefix", "denoise"):
            qk = ops[f"{stage}.blocks.qk"]
            av = ops[f"{stage}.blocks.av"]
            self.assertEqual(qk["m"], av["m"])
            self.assertEqual(qk["n"], av["k"])
            self.assertEqual(qk["k"], av["n"])
            self.assertEqual(qk["count"], av["count"])

    def test_batch_and_step_multiplicity(self):
        base = self.operations()
        batched = self.operations(batch_size=2)
        for name, op in base.items():
            self.assertEqual(batched[name]["count"], 2 * op["count"])
            self.assertEqual(tuple(batched[name][dim] for dim in ("m", "n", "k")),
                             tuple(op[dim] for dim in ("m", "n", "k")))
        single = self.operations(denoise_steps=1)
        for name, op in base.items():
            self.assertEqual(single[name]["count"] * (10 if op["stage"] == "denoise" else 1),
                             op["count"])

    def test_complete_positive_integer_gemm_families(self):
        ops = build_operations()
        self.assertEqual(len(ops), 26)
        self.assertEqual(len({op["name"] for op in ops}), 26)
        for op in ops:
            for field in ("m", "n", "k", "count"):
                self.assertIs(type(op[field]), int)
                self.assertGreater(op[field], 0)
        self.assertEqual(sum(op["m"] * op["n"] * op["k"] * op["count"] for op in ops),
                         2512199286784)

    def test_invalid_inputs_fail(self):
        for options in ({"cameras": 0}, {"prompt_tokens": -1}, {"action_horizon": 0},
                        {"denoise_steps": 0}, {"batch_size": True}):
            with self.assertRaises(ValueError):
                build_operations(**options)


if __name__ == "__main__":
    unittest.main()
