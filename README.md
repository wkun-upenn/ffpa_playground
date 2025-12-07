# FFPA-Attn Performance Benchmarks on Lambda.ai

This document summarizes the FFPA-Attn benchmark results which can be viewed in `ffpa-attn/tests/tmp/`

## 🚀 1. Speedup — FFPA+ACC+F16+L1 vs SDPA EA

![FFPA+ACC+F16+L1 Speedup](tests/tmp/NVIDIA_GH200_480GB_ffpa+acc+f16+L1_Speedup.png)

### 📝 Interpretation

- **Speedup increases** as D drops from 1024 → 576, peaking at **~2.19×**.
- FFPA’s L1 design benefits most when:
  - Tensor Core tile alignment is optimal  
  - Shared memory footprint fits SM capacity  
  - Warp scheduling reaches high occupancy  
- Very small D (e.g., 320) lowers arithmetic intensity → SDPA catches up.

---

## 🔵 2. Speedup — FFPA+ACC+F32+L1 vs SDPA EA

![FFPA+ACC+F32+L1 Speedup](tests/tmp/NVIDIA_GH200_480GB_ffpa+acc+f32+L1_Speedup.png)

### 📝 Interpretation

- FP32 accumulation reduces peak throughput (vs F16), but still delivers **1.6–2.16×** speedups.
- Best region: **D ≈ 448–576**.
- Slow-downs at the extremes:
  - Very large D → register pressure, lower occupancy  
  - Very small D → low compute/bandwidth ratio  

---

## 📈 3. TFLOPS Comparison — FFPA L1 Variants vs SDPA EA

![FFPA L1 vs SDPA EA TFLOPS](tests/tmp/NVIDIA_GH200_480GB.png)

### 📝 Key Observations

- SDPA EA baseline: **92–110 TFLOPS**.
- FFPA+ACC+F16 reaches **180–215 TFLOPS** — the overall highest throughput.
- FFPA+ACC+F32 reaches **150–205 TFLOPS** — slightly lower due to FP32 accumulator overhead.
- Sharp jump at **D = 576**:
  - Perfect Tensor Core tiling  
  - SMEM alignment matches GH200 banking  
  - Maximum warp occupancy  

---

## 🧠 Why the Trends Look Like This

### ✔ GPU Architecture Factors (NVIDIA GH200 / H100)

- **Tensor Core MMA Shape Alignment**  
  Performance peaks when D aligns with natural tile sizes (64, 128, 256 multiples).

- **Shared Memory bank swizzling**  
  FFPA uses SMEM aggressively; bank conflicts appear for unlucky D sizes.

- **Register Pressure & Warp Occupancy**  
  Large D increases register usage → reduces active warps → lower throughput.

- **Arithmetic Intensity**  
  Small D reduces compute per byte → SDPA becomes more competitive.

- **fp16 vs fp32 accumulation**  
  fp32 accumulation uses more registers → slightly lower throughput than fp16.

---

## 📦 Summary

- FFPA L1 consistently outperforms SDPA EA, often by **2×** on GH200.
- FP16 accumulation version is the fastest overall.
- Architectural sweet spots: **D ≈ 512–640**.
- The trends reflect deep interactions between Tensor Cores, shared memory capacity, bank layout, register file pressure, and occupancy.

# Installation
This guide documents every step required to build, install, and benchmark **ffpa-attn** on a **Lambda Cloud H100 (GH200, ARM64)** instance.

Works on:
- Ubuntu 22.04 ARM64  
- CUDA 12.8  
- Python 3.10  
- PyTorch 2.5.1 (cu124 ARM64)

---

## 1. Connect to Lambda Instance (from Mac)

```bash
ssh -i ~/.ssh/lambda ubuntu@<INSTANCE_IP>
```

Check GPU:

```bash
nvidia-smi
```

---

## 2. Sync Local Project → Cloud

From **Mac**:

```bash
rsync -avz --progress /Users/lymtics/Documents/ffpa-attn/ \
    ubuntu@<INSTANCE_IP>:~/ffpa-attn/
```

Sync back from **cloud → Mac**:

```bash
rsync -avz --progress \
    ubuntu@<INSTANCE_IP>:~/ffpa-attn/ \
    /Users/lymtics/Documents/ffpa-attn/
```

---

## 3. Create a Virtual Environment (Cloud)

```bash
cd ~/ffpa-attn
python3 -m venv .venv
source .venv/bin/activate
```

Verify:

```bash
which python
which pip
```

Expected:

```
~/ffpa-attn/.venv/bin/python
~/ffpa-attn/.venv/bin/pip
```

---

## 4. Install Build Dependencies

```bash
pip install --upgrade pip
pip install pybind11 packaging ninja numpy
```

---

## 5. Install PyTorch for CUDA 12.x (ARM64)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Validate install:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## 6. Configure CUDA 12.8

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"

which nvcc
nvcc --version
```

Expected:

```
Cuda compilation tools, release 12.8
```

---

## 7. Build ffpa-attn Wheel (ARM64)

```bash
cd ~/ffpa-attn
python setup.py bdist_wheel
```

Output wheel appears in:

```
dist/ffpa_attn-0.0.2.1-cp310-cp310-linux_aarch64.whl
```

---

## 8. Install the Wheel

```bash
pip install dist/ffpa_attn-*.whl
```

Verify:

```bash
pip show ffpa-attn
```

---

## 9. Run Benchmark Test

```bash
cd tests
python3 test_ffpa_attn.py --B 1 --H 48 --N 8192 --D 320 --show-all
```

If rebuild needed:

```bash
rm -rf build dist *.egg-info
python setup.py bdist_wheel
pip install dist/ffpa_attn-*.whl
```

---

## 10. Troubleshooting

### Missing pybind11
```
fatal error: pybind11/pybind11.h: No such file or directory
```

Fix:
```bash
pip install pybind11
```

---

### `no kernel image is available for execution on the device`
Kernel compiled for wrong architecture.

Ensure NVCC gencode includes:

```
-gencode arch=compute_90,code=sm_90
```

---

## 11. Sync Cloud → Local Again

From your **Mac**:

```bash
rsync -avz --progress ubuntu@<INSTANCE_IP>:~/ffpa-attn/ \
    /Users/lymtics/Documents/ffpa-attn/
```

---

## 12. Lambda Cloud Instance Notes

| Action      | Can Restart? | Files Persist? |
|-------------|--------------|----------------|
| **Stop**    | ✅ Yes       | ✅ Yes         |
| **Terminate** | ❌ No       | ❌ No (unless stored on volume) |

Use rsync or attached volumes to persist work.

---

## Done 🎉
You now have a reproducible build and test workflow for ffpa-attn on Lambda Cloud GH200 + H100 (ARM64).