import json
import matplotlib.pyplot as plt

with open("results_polished.json") as f:
    polished = json.load(f)
with open("results.json") as f:
    unpolished = json.load(f)

widths = [r["width"] for r in polished]
d_half = [r["d"] / 2 for r in polished]
llc    = [r["llc_mean"] for r in polished]
llc_raw = [r["llc_mean"] for r in unpolished]

SYS_ERR = 10   # realistic uncertainty from the seed and polishing checks (chain spread alone is 1-5)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

# Left: LLC vs the regular-model prediction d/2, log scales
ax1.plot(widths, d_half, "o--", color="gray", label="d/2 (regular model)")
ax1.errorbar(widths, llc, yerr=SYS_ERR, fmt="o-", capsize=4, label="estimated LLC")
ax1.set_xscale("log", base=2)
ax1.set_yscale("log")
ax1.set_xlabel("hidden width")
ax1.set_ylabel("learning coefficient")
ax1.set_title("LLC vs. regular-model value")
ax1.legend()

# Right: LLC alone, linear scale, polished vs. original models
ax2.errorbar(widths, llc, yerr=SYS_ERR, fmt="o-", capsize=4, label="polished (final)")
ax2.plot(widths, llc_raw, "o--", alpha=0.4, color="gray", label="unpolished (original run)")
ax2.set_xscale("log", base=2)
ax2.set_ylim(0, 170)
ax2.set_xlabel("hidden width")
ax2.set_ylabel("estimated LLC")
ax2.set_title("LLC is roughly flat in width")
ax2.legend()

plt.tight_layout()
plt.savefig("llc_vs_width.png", dpi=150)
plt.show()