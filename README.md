# Geospatial Real Estate Valuation

Real estate valuation engine using spatial embeddings and Graph Neural Networks (GNNs)
to model how neighboring properties influence a home's price — going beyond
traditional tabular models like XGBoost or linear regression.

## Problem
Standard models ignore spatial context. This project builds a valuation engine that
captures neighborhood-level price dependencies using spatial embeddings, KNN graphs,
and GNNs.

## Goal
Beat XGBoost/linear regression baselines on MAPE (Mean Absolute Percentage Error).

## Team
- Mani
- Manoj
- Suriya
- Swarup

## Structure
- `data/` — raw and processed datasets (gitignored)
- `notebooks/` — exploratory analysis and model development
- `src/` — reusable Python modules (data processing, models)
- `app/` — Streamlit deployment app

## Roadmap
- Week 1: Data acquisition, cleaning, spatial visualization
- Week 2: Feature engineering, XGBoost baseline
- Week 3: KNN graph construction, spatial embeddings
- Week 4: GNN/attention model, comparison, Streamlit deployment
