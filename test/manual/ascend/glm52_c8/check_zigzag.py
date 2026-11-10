"""Probe an already running GLM-5.2 A5 server configured for zigzag PCP C8."""

import argparse

from run_model import DRAFTS, add_probe_args, dump, http, probe, workload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:31266")
    parser.add_argument(
        "--tokenizer", required=True, help="Local GLM checkpoint directory"
    )
    parser.add_argument(
        "--case", choices=("pcp", "pcp-nextn-bf16", "pcp-nextn-fp8"), required=True
    )
    parser.add_argument("--cp-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=531, help="Prompt generation seed")
    add_probe_args(parser, require_cache_hit=True)
    args = parser.parse_args()
    if args.cp_size < 2 or args.long_tokens <= 2048 or args.new_tokens < 8:
        parser.error("use --cp-size >= 2, --long-tokens > 2048 and --new-tokens >= 8")

    url = args.url.rstrip("/")
    info = http(f"{url}/server_info", 10)
    expected = {
        "enable_prefill_cp": True,
        "attn_cp_size": args.cp_size,
        "cp_strategy": "zigzag",
        "dcp_size": 1,
        "kv_cache_dtype": "fp8_e4m3",
    }
    _, _, draft = args.case.partition("-")
    if draft:
        expected["speculative_draft_kv_cache_dtype"] = DRAFTS[draft]
    for key, value in expected.items():
        if info.get(key) != value:
            raise SystemExit(f"{key}={info.get(key)!r}, expected {value!r}")
    algorithm = info.get("speculative_algorithm")
    allowed = ("NEXTN", "EAGLE") if draft else (None, "NONE")
    if algorithm not in allowed:
        raise SystemExit(f"{args.case}: unexpected speculative_algorithm={algorithm!r}")

    args.out_dir.mkdir(parents=True, exist_ok=False)
    dump(args.out_dir / "server_info.json", info)
    prompts = workload(args, args.tokenizer, seed=args.seed)

    def check_uncached_tokens(name, response):
        # Zigzag requires at least 2 * cp_size uncached tokens per request.
        rows = response if isinstance(response, list) else [response]
        for row in rows:
            meta = row["meta_info"]
            cached = meta.get("cached_tokens", 0)
            if name == "long_fresh" and cached:
                raise AssertionError(
                    "long_fresh hit an existing prefix; use a new --seed"
                )
            if meta["prompt_tokens"] - cached < 2 * args.cp_size:
                raise AssertionError(f"{name}: too few uncached tokens for zigzag PCP")

    probe(
        url,
        args,
        prompts,
        args.out_dir,
        expect_spec=bool(draft),
        check_extra=check_uncached_tokens,
    )
    print(f"PASS zigzag PCP{args.cp_size} C8 {args.case}: {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
