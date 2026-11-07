# M6: Hardware Latency Scaling ($K=1..5$), Fast Burst Detector & 57,344-Round Stress Evaluation

## 1. Deterministic Hardware Cycle Scaling (`250 MHz`, `4.0 ns` Clock)

| BP Iterations $K$ | BP Core Cycles (`2 + 5K`) | BP Core Latency (`ns`) | Total AXI4-Stream Cycles (`7 + 5K`) | Total AXI4-Stream Latency (`ns`) |
| :---: | :---: | :---: | :---: | :---: |
| $K = 1$ | **`7`** | **`28.0 ns`** | **`12`** | **`48.0 ns`** |
| $K = 2$ | **`12`** | **`48.0 ns`** | **`17`** | **`68.0 ns`** |
| $K = 3$ | **`17`** | **`68.0 ns`** | **`22`** | **`88.0 ns`** |
| $K = 4$ | **`22`** | **`88.0 ns`** | **`27`** | **`108.0 ns`** |
| $K = 5$ | **`27`** | **`108.0 ns`** | **`32`** | **`128.0 ns`** |

## 2. Fast 16-Round Sliding-Subwindow Burst Detector Verification
- **Implementation**: `avid_fast_detector_step()` in `software/src/control/causal_scheduler.cpp`
- **Contract Bound**: `16`-round subwindow + `4`-round debounce (`<= 20` rounds = `20 us` @ `1 MHz`)
- **Measured Trigger Lag in Unit Test**: **`9` rounds (`9.0 us <= 20.0 us`)**

## Artifacts
- [`m6_timing_stress_and_fast_detector.json`](m6_timing_stress_and_fast_detector.json)
- [`m6_rtl_latency_k1_to_k5.csv`](m6_rtl_latency_k1_to_k5.csv)
- [`m6_57344_longrun_stress.csv`](m6_57344_longrun_stress.csv)
