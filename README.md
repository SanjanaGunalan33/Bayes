# BayesRCA — Bayesian Root Cause Analysis from Telemetry

BayesRCA is a probabilistic root cause analysis system built using the **RCAEval benchmark**. It analyzes telemetry changes around fault-injection events and ranks likely infrastructure fault types using a trained **discrete Bayesian / Naive Bayes model**.

The project contains the final training pipeline, frozen model parameters, evaluation artifacts, and an interactive web interface for Bayesian inference.

---

## Overview

The final model focuses on the metric-observable infrastructure faults in **RCAEval RE1 and RE2**.

### Pipeline

```text
RCAEval Cases
      ↓
Telemetry Extraction
      ↓
Pre/Post Injection Comparison
      ↓
Service-Level Change Features
      ↓
Low / Medium / High Discretization
      ↓
Discrete Naive Bayes Model
      ↓
Posterior Fault Ranking
```

The model ranks six native RCAEval fault classes:

- CPU
- Memory
- Disk
- Delay
- Loss
- Socket

---

## Dataset

The final model uses **RCAEval RE1 + RE2**.

| Dataset Information | Cases |
|---|---:|
| Initial RE1 + RE2 cases | 645 |
| Excluded cases | 2 |
| Final usable cases | 643 |

Two incidents were excluded because their telemetry timestamps did not provide sufficient observations on both sides of the recorded fault-injection time.

### Why RE3 is not included

RCAEval RE3 contains code-level fault cases represented as F1–F5. These should not simply be mapped to the infrastructure fault categories used by RE1 and RE2.

The current version therefore focuses specifically on **metric-observable infrastructure diagnosis using RE1 and RE2**.

---

## Telemetry Features

The final model uses seven telemetry feature families:

1. CPU change
2. Memory change
3. Disk I/O change
4. Socket change
5. Workload change
6. Latency P50 change
7. Latency P90 change

For each incident, telemetry is divided into **pre-injection** and **post-injection** observations using the recorded fault-injection timestamp.

The change in each available metric is calculated between these two periods, and the strongest service-level change is retained for each telemetry family.

Feature extraction is performed independently of the ground-truth fault label.

### Missing Telemetry

Not every telemetry family is available for every RCAEval environment.

Missing telemetry is therefore represented as **unavailable** rather than being interpreted as zero.

During inference, unavailable features do not contribute a likelihood term.

---

## Feature Discretization

The continuous telemetry changes are converted into three discrete evidence states:

```text
LOW
MEDIUM
HIGH
```

The state thresholds are calculated from the **training split only**, using the 33rd and 67th percentiles of each feature.

This prevents information from the validation or evaluation splits from influencing the learned discretization thresholds.

---

## Train / Validation / Evaluation Split

The **643 usable incidents** are divided approximately 80/10/10 using a deterministic class-wise split.

| Split | Cases | Purpose |
|---|---:|---|
| Training | 514 | Learn model parameters |
| Validation | 62 | Model-development decisions |
| Evaluation | 67 | Final frozen-model evaluation |

The split maintains representation of the six fault classes.

### Training data is used to learn

- Fault-class priors
- Feature discretization thresholds
- Conditional probability tables

### Validation data is used for

- Model-development and feature-selection decisions

### Evaluation data is used for

- Measuring the final frozen model

---

## Bayesian Model

BayesRCA uses a **discrete Naive Bayes model**.

The conceptual Bayesian network is:

```text
                         Fault Type
                             │
        ┌─────────┬──────────┼──────────┬─────────┐
        │         │          │          │         │
       CPU      Memory     Disk I/O    Socket   Workload
        │
        └──────────── Latency P50 / P90
```

More precisely, the fault type is a single categorical parent variable and the seven telemetry features are evidence variables conditioned on that fault type.

For evidence \(E\) and candidate fault \(F\), inference ranks faults according to:

```text
P(F | E) ∝ P(F) × ∏ P(Ei | F)
```

where:

- `P(F)` is the learned prior probability of the fault.
- `P(Ei | F)` is the learned conditional probability of an observed telemetry state given that fault.

The model parameters are estimated from the **training split only**.

Laplace smoothing is applied when estimating the conditional probabilities.

---

## Fault Classes

The model produces a posterior ranking across:

```text
CPU
Memory
Disk
Delay
Loss
Socket
```

Instead of returning only a single classification, BayesRCA retains the posterior distribution so that multiple plausible causes can be inspected.

This is useful when different infrastructure faults produce overlapping telemetry patterns.

---

## Evaluation Results

The final frozen model produced the following results on the **67-case evaluation split**:

| Metric | Result |
|---|---:|
| Top-1 Accuracy | **70.1%** |
| Top-3 Accuracy | **100.0%** |
| Mean Reciprocal Rank (MRR) | **0.841** |

### Interpreting the metrics

**Top-1 Accuracy**

The highest-ranked predicted fault matches the ground-truth fault.

**Top-3 Accuracy**

The ground-truth fault occurs somewhere among the three highest-ranked predictions.

Therefore, the **100% Top-3 result should not be interpreted as 100% classification accuracy**.

**Mean Reciprocal Rank**

MRR measures how highly the correct root cause tends to appear in the ranked output.

---

## Web Application

The project includes an interactive frontend connected to the **same frozen Bayesian model produced by the training pipeline**.

Users provide a state for each telemetry feature:

```text
Low
Medium
High
Unavailable
```

The browser inference engine then calculates posterior probabilities across the six fault classes.

### The interface provides

- Telemetry evidence input
- Posterior root-cause ranking
- Prior probability comparison
- Evidence compatibility information
- Observed evidence summary
- Investigation guidance
- Bayesian network visualization
- Analysis history

The frontend does not require or use the true fault label to calculate its posterior ranking.

---

## Project Structure

```text
BayesRCA/
│
├── index.html
├── style.css
├── app.js
├── bayesian-engine.js
├── trained_model_params.js
│
├── train_bayes_rca.py
├── cases.parquet
│
├── bayes_rca_model.json
├── bayes_rca_evaluation.json
├── bayes_rca_split.csv
├── bayes_rca_exclusions.csv
│
├── README.md
└── .gitignore
```

### Important Files

#### `train_bayes_rca.py`

Final model pipeline containing:

- RCAEval case selection
- Telemetry extraction
- Pre/post injection processing
- Feature construction
- Dataset splitting
- Training-derived discretization
- Bayesian parameter estimation
- Inference
- Evaluation
- Frontend model export

#### `bayes_rca_model.json`

Contains the frozen trained model, including:

- Fault classes
- Feature definitions
- Discretization thresholds
- Fault priors
- Conditional probability tables

#### `trained_model_params.js`

JavaScript representation of the frozen trained model used directly by the browser inference engine.

#### `bayesian-engine.js`

Implements browser-side Bayesian inference using the exported trained parameters.

#### `bayes_rca_evaluation.json`

Stores the final model evaluation results.

#### `bayes_rca_split.csv`

Stores the exact assignment of usable RCAEval incidents to the training, validation, and evaluation splits.

#### `bayes_rca_exclusions.csv`

Records incidents excluded because sufficient pre/post telemetry was unavailable.

---

## Running the Web Application

The frontend uses JavaScript modules, so it should be served through a local HTTP server rather than opening `index.html` directly.

### Using Python

From the project directory:

```bash
python -m http.server 8000
```

Then open:

```text
http://localhost:8000
```

in a web browser.

---

## Training

The final training pipeline is:

```text
train_bayes_rca.py
```

It performs:

```text
Load RCAEval metadata
        ↓
Select RE1 + RE2 cases
        ↓
Load telemetry
        ↓
Separate pre/post injection observations
        ↓
Extract service-level telemetry changes
        ↓
Validate usable incidents
        ↓
Create train/validation/evaluation split
        ↓
Learn discretization thresholds from training data
        ↓
Learn Bayesian priors and CPTs
        ↓
Evaluate frozen model
        ↓
Export model for frontend inference
```

Raw RCAEval telemetry is **not committed to this repository**. The local telemetry cache is intentionally kept outside version control.

---

## Design Decisions

### Why Bayesian inference?

Infrastructure faults often produce overlapping symptoms.

For example, increased latency may result from:

- Network packet loss
- Dependency delay
- Resource pressure
- Other system degradation

Instead of treating every telemetry pattern as deterministic, Bayesian inference allows the system to rank competing hypotheses based on the available evidence.

### Why a discrete model?

The dataset is relatively small, and an interpretable discrete model makes it possible to inspect:

- Priors
- Evidence states
- Conditional probabilities
- Posterior rankings

This provides a transparent baseline before moving to more complex models.

### Why use pre/post changes?

Absolute metric values can differ significantly between services and environments.

Comparing telemetry before and after the recorded injection time allows each incident to use its own pre-fault behavior as a baseline.

### Why use the strongest service-level change?

Many injected faults affect a particular service rather than the entire application.

Using the strongest service-level change helps preserve localized anomalies that could disappear when averaging across all services.

---

## Limitations

The current implementation has several important limitations:

- The model covers **RCAEval RE1 and RE2**, not RE3.
- The Naive Bayes model assumes conditional independence between telemetry features given the fault class.
- Some telemetry families are unavailable in particular RCAEval environments.
- Socket faults have fewer training examples than the other fault categories.
- Disk I/O and socket telemetry availability differs between benchmark environments.
- Posterior values are probabilities under the learned model assumptions and should not automatically be interpreted as calibrated real-world failure probabilities.
- The reported metrics apply specifically to the stated 67-case evaluation split.
- Additional external or cross-environment testing would be required before making broader performance claims.

---

## Future Work

Possible extensions include:

- Service-level fault localization
- Integration of logs and distributed traces
- Support for RCAEval RE3 code-level faults
- Probability calibration
- Cross-environment evaluation
- Comparison with discriminative machine-learning models
- Better modelling of correlated telemetry features
- More detailed uncertainty and confidence analysis

---

## Purpose

BayesRCA explores how an interpretable probabilistic model can transform system telemetry into **ranked root-cause hypotheses** while maintaining separation between training, validation, evaluation, and browser-side inference.

The current implementation provides an end-to-end pipeline from RCAEval telemetry processing to an interactive Bayesian RCA interface.