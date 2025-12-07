import json
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==============================================================
# 1. Load JSON results
# ==============================================================

JSON_PATH = Path("nvbandwidth_results.json")

with JSON_PATH.open("r") as f:
    data = json.load(f)

tests = data["nvbandwidth"]["testcases"]


def get_sum(name: str):
    """Return numeric 'sum' for a testcase, or None."""
    for t in tests:
        if t.get("name") == name and t.get("status") == "Passed":
            return float(t["sum"])
    return None


def avg(values):
    v = [x for x in values if x is not None]
    return sum(v) / len(v) if v else None


# ==============================================================
# 2. Extract metrics and map to conceptual tiers
# ==============================================================

# HBM (on-GPU DRAM)
hbm_local = get_sum("device_local_copy")  # GB/s

# PCIe copies via CE (host <-> HBM, unidirectional)
h2d_ce = get_sum("host_to_device_memcpy_ce")
d2h_ce = get_sum("device_to_host_memcpy_ce")
pcie_ce_uni = avg([h2d_ce, d2h_ce])

# PCIe copies via CE (host <-> HBM, bidirectional)
h2d_ce_bi = get_sum("host_to_device_bidirectional_memcpy_ce")
d2h_ce_bi = get_sum("device_to_host_bidirectional_memcpy_ce")
pcie_ce_bi = avg([h2d_ce_bi, d2h_ce_bi])

# PCIe copies via CUDA kernels / SM path (unidirectional)
h2d_sm = get_sum("host_to_device_memcpy_sm")
d2h_sm = get_sum("device_to_host_memcpy_sm")
pcie_sm_uni = avg([h2d_sm, d2h_sm])

# PCIe copies via CUDA kernels / SM path (bidirectional)
h2d_sm_bi = get_sum("host_to_device_bidirectional_memcpy_sm")
d2h_sm_bi = get_sum("device_to_host_bidirectional_memcpy_sm")
pcie_sm_bi = avg([h2d_sm_bi, d2h_sm_bi])

# SMEM / SRAM throughput from your separate tool (GB/s)
SMEM_THROUGHPUT_GBPS = 73526.90

labels = []
values = []

# HBM
if hbm_local is not None:
    labels.append("HBM\n(device_local_copy)")
    values.append(hbm_local)

# PCIe CE, uni- and bi-directional
if pcie_ce_uni is not None:
    labels.append("PCIe CE\nH↔D uni")
    values.append(pcie_ce_uni)

if pcie_ce_bi is not None:
    labels.append("PCIe CE\nH↔D bidi")
    values.append(pcie_ce_bi)

# PCIe SM, uni- and bi-directional
if pcie_sm_uni is not None:
    labels.append("PCIe SM\nH↔D uni")
    values.append(pcie_sm_uni)

if pcie_sm_bi is not None:
    labels.append("PCIe SM\nH↔D bidi")
    values.append(pcie_sm_bi)

# SMEM / SRAM
labels.append("SMEM / SRAM\n(kernel)")
values.append(SMEM_THROUGHPUT_GBPS)

# If nothing is available, bail out gracefully
if not labels:
    raise RuntimeError("No valid metrics found in nvbandwidth_results.json")

values = np.array(values, dtype=float)

# ==============================================================
# 3. Prepare output directory
# ==============================================================

OUTPUT_DIR = Path("graphs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = OUTPUT_DIR / "memory_subsystems_bar_log.png"

# ==============================================================
# 4. Bar chart (similar style to plot_speedup_bar, log-scale)
# ==============================================================

fig, ax = plt.subplots(figsize=(16, 9))
plt.subplots_adjust(left=0.06, right=0.99, top=0.93, bottom=0.15)

x = range(len(values))

# Random colored bars (like your speedup plot)
random.seed(0)
for i, v in enumerate(values):
    random_color = (random.random(), random.random(), random.random())
    ax.bar(i, v, color=random_color)

ax.set_xlabel("Subsystem / Path", fontsize=15, fontweight="bold")
ax.set_xticks(list(x))
ax.set_xticklabels(labels, fontsize=12, fontweight="bold", rotation=15, ha="right")

ax.set_ylabel("Throughput (GB/s, log scale)", fontsize=15, fontweight="bold")
ax.set_title("H100 Memory & Link Throughput (nvbandwidth + SMEM kernel)", fontsize=16, fontweight="bold")

# Log scale for large SMEM vs others
ax.set_yscale("log")

y_min = max(values.min() * 0.5, 1e-1)
y_max = values.max() * 2.0
ax.set_ylim(y_min, y_max)

ax.yaxis.set_tick_params(labelsize=12)

# Annotate each bar with its numeric value (rounded)
for i, v in enumerate(values):
    ax.text(
        i,
        v * 1.05,  # slightly above bar in log scale
        f"{v:.1f}",
        ha="center",
        fontsize=12,
        fontweight="bold",
        rotation=90,
    )

ax.grid(True, which="both", axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig(out_path, dpi=300)
plt.close()

print(f"✔ Log-scale memory subsystem bar chart saved to: {out_path}")