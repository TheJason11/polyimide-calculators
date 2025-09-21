````markdown
# Polyimide Calculators (Zeus Polyimide Process Suite)

A set of production tools for polyimide-coated wire and related processes, built with Streamlit.

**Modules**
- **Wire Diameter Predictor** — tension-aware stretch model with per-run alpha calibration, reverse calculator, and CSV logging.
- **Runtime Calculator** — estimates runtime from finished footage and line speed.
- **Copper Wire Converter** — feet ⇄ pounds.
- **Coated Copper Converter** — two-material linear density and length/weight conversions.
- **PAA Usage** — solution usage for a given ID/wall/length, with scrap/hold-up/allowance.
- **Anneal Temp Estimator** — kNN + monotonic guard on diameter/dwell.

---

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
````

---

## GitHub save (run history)

The app logs Wire Diameter Predictor runs to **`wire_diameter_runs.csv`** at the repo root.
To enable **remote saves** from the app:

1. Create a GitHub **personal access token**

   * **Fine-grained:** select this repo and grant **Repository → Contents: Read and write**
   * or **Classic:** scope `repo`
2. Add the following **secrets** (see next section), restart the app, and click **Save Run**.

> The app also writes a local CSV as a fallback. On Streamlit Cloud the local file is ephemeral, so GitHub saves are the durable history.

---

## Secrets

Provide these four values as TOML:

```toml
[gh]
token = "YOUR_GITHUB_TOKEN"
owner = "TheJason11"
repo  = "polyimide-calculators"
branch= "main"
```

* **Local dev:** create `.streamlit/secrets.toml` alongside `app.py` (do **not** commit this file).
* **Streamlit Cloud:** App → Settings → **Secrets** → paste the TOML.

> **Private repo note:** the history panel reads from GitHub. For private repos, switch the loader to use the authenticated Contents API (the app already writes with auth). If history looks empty on a private repo, either make the repo public or patch the loader to use an authenticated read.

---

## Data files

* `wire_diameter_runs.csv` — run history (the app will create/append). **Schema:**

```csv
date_time,starting_diameter_in,passes,line_speed_fpm,annealer_height_ft,annealer_temp_F,payoff_type,payoff_tension_lb,pred_final_od_in,actual_final_od_in,alpha_used,notes
```

**Example rows:**

```csv
2025-09-20 10:30:00,0.0823,30,8,14,825,"22"" Spool",10,0.08110,0.08100,0.26,benchmark run
2025-09-20 11:00:00,0.0415,10,15,14,825,Large,2,0.04110,0.04100,0.70,benchmark run
```

* `annealing_dataset_clean.csv` — source for the anneal temp estimator (`wire_dia,speed,annealer_ht,anneal_t`).

---

## Wire Diameter Predictor: how to use

1. **Fill inputs:** Start OD, Passes, Speed (FPM), Annealer Height (ft), Anneal Temp (°F), **Payoff Type** (`Large` or `22" Spool`), and **Payoff Tension (lbf)**.
2. **Alpha calibration:** Use **Quick Calibrate** with a known run (start & actual final) and click **Solve Alpha**.

   * Practical starting points: `α_Large ≈ 0.70`, `α_22" ≈ 0.25–0.27`.
3. **Predict** the final OD or use the **Reverse Calculator** to find the required start OD.
4. **Save Run** to log inputs, prediction, and `alpha_used` to the CSV (and GitHub if secrets are set).

**Model sketch**

* Stretch scales with **passes**, **temperature above 650 °F**, **dwell** (height/speed), and **tension-derived stress**.
* Payoff type multiplies effective tension (22″ typically stretches more at the same indicated pounds).
* `alpha` is a global scale; calibrate from your own runs.
* Optional residual learning applies small, local bias corrections once you’ve logged enough similar runs.

---

## Typical workflows

* **Day to day:** Calibrate α via Quick Calibrate for the current payoff type, then predict or reverse-solve. Save runs with actual finals to build history.
* **22″ vs Large diverge:** keep different α values per payoff type, or tune tension constants (σ\_ref, 22″ multiplier) in the advanced panel.
* **Private repo:** use an authenticated read for history (or keep the repo public).

---

## Requirements

See `requirements.txt`. Minimal set:

```
streamlit>=1.36,<2
pandas>=2.0,<3
numpy>=1.26,<2
scikit-learn>=1.3,<2
scipy>=1.10,<2
requests>=2.31,<3
```

---

## Contributing

* Don’t commit `.streamlit/secrets.toml`.
* If you change column names, update `WIRE_COLS` and the history table.
* PRs welcome for: auto-calibration (fit α per payoff + tension constants), authenticated history read, additional converters.

---

## License

MIT

```
```
