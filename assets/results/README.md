# Committed results

**This directory is authoritative.** Every CSV here is the version any figure, notebook or write-up
cites. `notebooks/alflb_price_internalization.ipynb` reads from here and from nowhere else.

The scripts that produced these numbers write to `artifacts/game/` under working names
(`kappa_trajectory.csv`, `spread_trajectory_mac.csv`), and `artifacts/` is gitignored, so those
files are machine-local and not recoverable. Each row of `MANIFEST.csv` records the working file
its committed copy came from in the `source` column, which makes the two detectable when they
drift apart:

```bash
uv run python - <<'PY'
import csv, hashlib, pathlib
for row in csv.DictReader(open("assets/results/MANIFEST.csv")):
    src = pathlib.Path(row["source"].split(" ")[0])
    dst = pathlib.Path("assets/results") / row["file"]
    if not src.is_file():
        print(f"{row['file']}: source absent on this machine")
        continue
    same = hashlib.sha256(src.read_bytes()).hexdigest() == hashlib.sha256(dst.read_bytes()).hexdigest()
    print(f"{row['file']}: {'identical' if same else 'DRIFTED from ' + str(src)}")
PY
```

A mismatch means a script was rerun and the committed copy was not refreshed. Resolve it by copying
the working file over the committed one, because the committed name is a label rather than a second
measurement. Three entries are expected not to match, so treat only the others as findings:

| File | Why |
|---|---|
| `price-stability_strided-probe_sequence_parts4.csv` | its working file is not on this machine |
| `internalization_tail-probe_all-steps.csv` | copied with `dual_*` renamed to `bias_price_*` |
| `crossprobe_per-expert-prices_step500.csv` | exported from `crossprobe_duals.npz` rather than copied |

`MANIFEST.csv` also carries the tier (`headline`, `supporting`, `provenance`, `superseded`), the
question each file answers, and what replaced it. Read that before citing anything here.
