# LTX-2 self-hosting POC kit

A minimal kit for renting a single GPU pod and running the
[LTX-2](https://huggingface.co/Lightricks/LTX-2) video-generation model
(Lightricks) for evaluation. Designed for quick, reproducible measurements
of quality, latency, VRAM, and batching headroom on H100-class GPUs before
committing to a production self-host architecture.

## What's in here

```
poc/
├── pod/                      # runs on the rented GPU pod
│   ├── server.py             # FastAPI: /healthz /readyz /info /generate
│   ├── bootstrap.sh          # one-command pod setup (idempotent)
│   └── requirements.txt
└── runner/                   # runs on your local machine
    ├── run_pod_eval.py       # drives the 40-prompt eval against the pod
    ├── prompts.py            # 40-prompt evaluation set (6 categories, 6 languages)
    ├── configs.yaml          # eval matrix (variant × batch size × prompt subset)
    └── requirements.txt
```

## Quick start

See [`poc/README.md`](./poc/README.md) for the full operator playbook,
including the recommended pod template, env vars, expected log timeline,
GO/NO-GO criteria, and troubleshooting.

In one sentence: rent an H100 80 GB pod on a GPU cloud (e.g. RunPod), point
its container start command at `poc/pod/bootstrap.sh`, wait ~20 min for the
bootstrap to download LTX-2 weights and start the FastAPI server, then run
`python run_pod_eval.py --config distilled_fp8_batch1` from your laptop.

## License

The code in this repository is MIT-licensed. The LTX-2 model weights are
distributed by Lightricks under their own license — see the
[LTX-2 model card](https://huggingface.co/Lightricks/LTX-2) for terms.
