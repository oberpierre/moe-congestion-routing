"""Is the affinity-blind round-robin baseline a lucky draw?

The baseline gives token i the experts {(i*K+m) mod E}. Because i*K mod E has period
E/gcd(K,E), only that many distinct expert bundles exist, so the assignment is structured
rather than random and could pair favourably with expert specialization. This compares it
against random balanced assignments of the same family, randomizing both which tokens take
which bundle and which experts compose one.
"""

import csv
import glob
import math
import sys

import numpy as np

from moe_congestion_routing.metrics.probe_series import read_dump

DRAWS = 400
UNIT = 16384

rows = []
for path in sorted(glob.glob("artifacts/exp1/control/control-trunk/probes/*/iter_0000000.npz")):
    asset = path.split("probes/")[1].split("/")[0]
    dump = read_dump(path)
    scores = dump.router_scores()
    rng = np.random.default_rng(0)
    for layer_index, layer in enumerate(dump.layer_numbers):
        a = scores[layer_index][:UNIT]
        n, e = a.shape
        k = dump.topk
        tok = np.arange(n)[:, None]
        idx = (np.arange(n)[:, None] * k + np.arange(k)[None, :]) % e
        fixed = float(a[tok, idx].sum())
        draws = np.array(
            [float(a[tok, rng.permutation(e)[idx][rng.permutation(n)]].sum()) for _ in range(DRAWS)]
        )
        rows.append(
            {
                "asset": asset,
                "layer": layer,
                "step": dump.step,
                "bundles": e // math.gcd(k, e),
                "roundrobin_affinity": f"{fixed:.4f}",
                "random_mean": f"{draws.mean():.4f}",
                "random_sd": f"{draws.std(ddof=1):.4f}",
                "z_score": f"{(fixed - draws.mean()) / draws.std(ddof=1):.4f}",
                "percentile": f"{(draws < fixed).mean() * 100:.2f}",
                "draws": DRAWS,
            }
        )
        print(rows[-1]["asset"][:28], layer, rows[-1]["z_score"], flush=True)

out = sys.argv[1]
with open(out, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

z = np.array([float(r["z_score"]) for r in rows])
print(
    f"\n{len(rows)} cells: mean z {z.mean():+.3f}, "
    f"sd {z.std(ddof=1):.3f}, |z|>2: {(abs(z) > 2).sum()}"
)
print(f"wrote {out}")
