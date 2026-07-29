# geospatial-real-estate-valuation
"Real estate valuation engine using spatial embeddings and GNNs to model neighborhood price dependencies."

PROJECT 3: GEOSPATIAL REAL ESTATE VALUATION
Repo: github.com/Swarup20/geospatial-real-estate-valuation
Team: Mani, Manoj, Suriya, Swarup
Branches: dev-mani, dev-manoj, dev-suriya, dev-swarup

===========================================
WEEK 1 — Data Acquisition & Exploration
Goal: Clean geospatial dataset + first visualizations
===========================================
Swarup (Lead) - Source dataset (King County Housing or similar), initial GeoPandas cleaning
Mani           - Haversine distance calculations, lat/long validation
Manoj          - Exploratory data analysis - price distributions, missing values
Suriya         - Build interactive maps (Folium/Kepler.gl) showing spatial price trends

===========================================
WEEK 2 — Feature Engineering & Baseline Model
Goal: XGBoost baseline + documented weaknesses
===========================================
Mani (Lead)   - Feature engineering (house age, distance to city center, amenity proximity)
Manoj         - Train/tune XGBoost baseline model
Suriya        - Evaluate baseline - MAPE/RMSE, error analysis by region
Swarup        - Document failure cases (e.g. gentrifying neighborhoods) with visuals

===========================================
WEEK 3 — Graph Construction & Embeddings
Goal: KNN graph + spatial embeddings
===========================================
Suriya (Lead) - Build KNN graph (houses = nodes, edges = nearest neighbors)
Swarup        - Generate spatial embeddings from graph structure
Mani          - Validate graph quality (edge weights, neighbor sanity checks)
Manoj         - Visualize the graph structure + embedding clusters

===========================================
WEEK 4 — GNN/Attention Model & Deployment
Goal: Final model + Streamlit app
===========================================
Manoj (Lead)  - Build GNN/attention-based valuation model (PyTorch/DGL)
Suriya        - Compare final model MAPE vs XGBoost baseline
Swarup        - Build Streamlit app skeleton
Mani          - Add price-disparity map + top-5-neighbors feature to app

===========================================
RULES
===========================================
- Everyone commits daily to their own dev-<name> branch
- Commit format: feat: <what you did today> (fixes #<issue number>)
- Merge to main every 1-2 days via Pull Request
- No last-day mega-commits (auto-disqualification risk)
