# Grid Event Intelligence Platform

Operator-facing early warning system for electric grid disturbances.  
This project learns precursor patterns from waveform data, weather, and outage context to rank feeders/assets by short-term failure risk.

## What We Are Building

The platform is designed as a utility-grade event intelligence stack, not just a fault classifier.

- Learn robust signal representations with self-supervised pretraining on raw oscillograms.
- Detect and classify grid events (faults, anomalies, disturbance patterns).
- Produce short-horizon risk rankings for feeders/assets.
- Explain alerts by highlighting the signal segments and features that drove model confidence.
- Support maintenance decisions with a queue simulator that estimates avoided failures vs false alarms.

## Core Capabilities

1. **Signal Understanding**
   - Ingest and preprocess oscillogram/time-series data.
   - Train representation models for unlabeled disturbance data.

2. **Event Intelligence**
   - Event classification model.
   - Anomaly detection model for out-of-distribution behavior.

3. **Risk Ranking**
   - Fuse event scores + weather + outage context.
   - Rank feeders/assets by short-term risk.

4. **Decision Support**
   - Alert explanation panel (attribution on waveform segments).
   - Maintenance queue simulation for operational trade-offs.

## Data Sources (Public Path)

- Real-world oscillogram dataset: https://www.nature.com/articles/s41597-026-06587-8
- Fingrid power system state API: https://data.fingrid.fi/en/datasets/209
- NASA POWER weather API: https://power.larc.nasa.gov/docs/services/api/temporal/daily/

## Suggested Build Phases

1. **Foundation**
   - Define data schemas and ingestion pipelines.
   - Build baseline preprocessing and exploratory notebooks.

2. **Modeling**
   - Train baseline event classifier.
   - Add anomaly detector.
   - Add self-supervised pretraining and compare gains.

3. **Risk Layer**
   - Engineer fusion features (signal + weather + outage context).
   - Train and calibrate short-term risk ranking model.

4. **Explainability + Simulation**
   - Add attribution/explanation outputs for alerts.
   - Implement maintenance queue simulator and evaluation metrics.

5. **Operationalization**
   - Package reproducible training/inference pipelines.
   - Add monitoring hooks for data drift and model performance.

## Success Criteria

- Reliable event detection and anomaly sensitivity on held-out data.
- Meaningful feeder/asset prioritization quality (ranking metrics and incident outcomes).
- Actionable explanations operators can trust.
- Clear trade-off visibility between missed failures and false alarms.

## Project Intent

This portfolio project demonstrates:

- Time-series deep learning
- Self-supervised learning
- Anomaly detection
- Feature attribution / explainability
- MLOps and streaming inference thinking

