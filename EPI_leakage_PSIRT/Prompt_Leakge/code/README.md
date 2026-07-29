## Files

| File | Purpose |
|------|---------|
| `LLM_infer.py` | Runs Phi-3 inference inside TDX, signals BMC via MSR |
| `OOB.sh` | Runs on BMC — collects EPI samples during inference |
| `train_bilstm.py` | Trains the Bi-LSTM classifier on collected traces |
| `bilstm_model.pt` | Pre-trained model weights |
| `breast_cancer_epi_traces.csv` | Target-class EPI traces (5,000 rows) |
| `background_epi_traces.csv` | Background-class EPI traces (5,189 rows) |


## How to Run

### Step 1 — Reset MSR (in-band machine)

Run once before starting any collection session:

```bash
sudo wrmsr -p 0 0x610 0x878d2000158af0
```

---

### Step 2 — Start OOB Collection (BMC machine)

Open a terminal on the BMC and run **before** starting inference:

```bash
# For breast-cancer prompts
bash OOB.sh > breast_cancer_run1.txt

# For background prompts
bash OOB.sh > background_run1.txt
```

The script waits silently until it detects the START signal, then writes one EPI delta value per line.

---

### Step 3 — Run Inference (in-band machine)

```bash
sudo taskset -c 0 python3 LLM_infer.py "Your prompt here"
```

Examples:

```bash
# Target topic
sudo taskset -c 0 python3 LLM_infer.py "What is breast cancer in simple words?"

# Background
sudo taskset -c 0 python3 LLM_infer.py "Can deleted pictures on Instagram be recovered?"
```

> `taskset -c 0` pins inference to core 0, keeping the EPI signal clean.

Full dataset collection:
- **Breast cancer:** 100 prompts x 50 queries = 5,000 traces
- **Background:** 5,000 prompts x 1 query = 5,000 traces

---

### Step 4 — Build the Dataset

Convert `.txt` trace files into CSV:

```python
import pandas as pd, glob

rows = []
for fpath in sorted(glob.glob("traces/breast_cancer_*.txt")):
    vals = open(fpath).read().split()
    rows.append({
        "class_label":  "breast_cancer",
        "prompt_text":  "your prompt here",
        "trace_length": len(vals),
        "epi_trace":    " ".join(vals)
    })

pd.DataFrame(rows).to_csv("breast_cancer_epi_traces.csv", index=False)
```

Do the same for background traces using `class_label = "background"`.

---

### Step 5 — Train the Classifier

```bash
python3 train_bilstm.py
```


