# Kimi-K3 four-node full-model PCP accuracy validation

This profile validates the full checkpoint without PD disaggregation or layer
truncation.  The same launch command must be run on all four nodes.  The
launcher detects the local IP and assigns node ranks in this order:

| Node | Default IP | Rank |
| --- | --- | ---: |
| master/API | `192.168.25.209` | 0 |
| worker | `192.168.25.212` | 1 |
| worker | `192.168.25.216` | 2 |
| worker | `192.168.25.217` | 3 |

The default model path is `/home/weights/Kimi-K3-W4A8`.  Override it on every
node with `MODEL_PATH=/another/path` when necessary.  If the real subnet is
not `192.168.25.*`, set the same comma-separated `CLUSTER_NODES` value on all
nodes.

## 1. PCP-off baseline A1

On all four nodes:

```bash
cd /home/q00886407/sgl/sglang-kimiK3
./run_4node_full_pcp_off.sh
```

After `http://192.168.25.209:30000/health` is ready, run on node 209:

```bash
cd /home/q00886407/sgl/sglang-kimiK3
export MODEL_PATH=/home/weights/Kimi-K3-W4A8
export BASE_URL=http://192.168.25.209:30000
export RESULT_DIR=$PWD/logs/kimi_k3_4node_full_accuracy
./scripts/run_kimi_k3_pcp_validation_suite.sh collect-accuracy A1
```

Stop the service on all nodes, restart the identical PCP-off profile, and
collect A2.  This independently restarted baseline measures natural numerical
variation:

```bash
./scripts/run_kimi_k3_pcp_validation_suite.sh collect-accuracy A2
```

## 2. PCP A2A/all-gather candidate B

Stop the A2 service, then run on all four nodes:

```bash
./run_4node_full_pcp_a2a.sh
```

Collect on node 209 after the service is ready:

```bash
./scripts/run_kimi_k3_pcp_validation_suite.sh collect-accuracy B
```

## 3. PCP FLA/ring candidate C

Stop the B service, then run on all four nodes:

```bash
./run_4node_full_pcp_fla_ring.sh
```

Collect on node 209 after the service is ready:

```bash
./scripts/run_kimi_k3_pcp_validation_suite.sh collect-accuracy C
```

## 4. Compare

On node 209, with the same `RESULT_DIR` exported:

```bash
./scripts/run_kimi_k3_pcp_validation_suite.sh compare-accuracy
```

The comparison checks repeat stability, prefill logprob differences, first
generated token equality, output-token match rate, and output logprob
differences.  These runs validate model computation but do not cover PD KV
transfer, Mooncake/SMEM sessions, or PD router behavior.

## Topology

The default configurations intentionally keep attention TP at 16:

| Profile | TP | DP | CP | Attention TP |
| --- | ---: | ---: | ---: | ---: |
| PCP off | 64 | 4 | 1 | 16 |
| PCP A2A | 64 | 1 | 4 | 16 |
| PCP FLA/ring | 64 | 1 | 4 | 16 |

CP ranks are carved out of the global TP world; `TP64/DP1/CP4` still launches
64 processes, not 256.
