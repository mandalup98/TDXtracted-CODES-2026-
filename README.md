# TDXtracted

**Systematic Analysis and Exploitation of In-Band and Out-of-Band Package-Scoped Performance Telemetry in TDX**

Upasana Mandal, Shubhi Shukla, Nimish Mishra, Sarani Bhattacharya, Paritosh Saxena, Debdeep Mukhopadhyay
*Indian Institute of Technology Kharagpur; Microsoft*
Accepted at **CODES+ISSS 2026**

This repository is the artifact accompanying the paper above. It contains the code and datasets used to (1) establish and systematically characterize in-band PMC-based leakage from Intel TDX Trust Domains, and (2) reverse-engineer and exploit the out-of-band Efficient Performance Indicator (EPI) channel exposed via the BMC/PECI management interface.

> **Hardware note for reviewers:** Full end-to-end reproduction (data *collection*) requires an Intel Xeon Sapphire Rapids server with TDX enabled and authenticated BMC/PECI access — this is not commodity/cloud-rentable hardware. All raw traces and datasets used in the paper are included in this repository, so the *analysis and classification pipeline* (training/evaluating the classifiers behind every table and figure in the paper) can be reproduced without specialized hardware.

## Directory Structure

The repository is organized to mirror the paper: **Part 1** covers in-band leakage (Sec. IV–V), **Part 2** covers out-of-band leakage (Sec. VI–VII).

```
TDXtracted-CODES-2026-/
├── Exploitation of In-Band Performance Telemetry/
│   ├── Establishing Leakage/
│   ├── Systematic Analysis and Process Identification/
│   └── Real-World ML Use Cases Leveraging In-Band Performance Telemetry/
│       ├── Class_Leakage_Attack/
│       └── Leakage of Model Architecture/
│
├── Exploitation of Out-of-Band Performance Telemetry/
│   ├── PSIRT_Report__Intel_TDX__OOB_EPI_side_channel.pdf
│   ├── Leakage of Transformer Model Architecture/
│   ├── Leakage of LLM Model Architecture/
│   └── Prompt-Topic Leakage/
│
├── README.md
├── REQUIREMENTS.md
├── INSTALL.md
├── STATUS.md
├── LICENSE
└── requirements.txt
```

### Part 1 — Exploitation of In-Band Performance Telemetry (Sec. IV–V)

| Folder | What's inside |
|---|---|
| `Establishing Leakage` | Establishement of the in-band leakage |
| `Systematic Analysis` | Full 258-event PMC sweep; UnixBench & SPEC2017 workload classification |
| `Real-World ML Use Cases.../Leakage of DNN Model Architecture` | DNN family / sub-family fingerprinting |
| `Real-World ML Use Cases.../Class_Leakage_Attack` | Inference-output / class leakage on CIFAR-10 and CIFAR-100 |

### Part 2 — Exploitation of Out-of-Band Performance Telemetry (Sec. VI–VII)

| Folder  What's inside |
|---|---|
| `Leakage of Transformer Model Architecture` | ViT and language-Transformer fingerprinting via EPI |
| `Leakage of LLM Model Architecture` |  LLM fingerprinting via EPI |
| `Prompt-Topic Leakage` | Leakge of LLM prompt semantic |

Each leaf folder generally contains a `code/` subfolder (data-collection + classifier scripts) and a `dataset/` subfolder (raw perf/PECI traces and CSVs used to produce the paper's tables/figures).

## Quick Links

- 📄 Paper (camera-ready PDF): see submission portal / `paper.pdf` in this repo
- 📋 [`REQUIREMENTS.md`](REQUIREMENTS.md) — hardware & software prerequisites
- ⚙️ [`INSTALL.md`](INSTALL.md) — setup and how to run each component



