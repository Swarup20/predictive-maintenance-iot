# 🏠 Geospatial Real Estate Valuation using Spatial Embeddings & Graph Neural Networks

## 📌 Project Overview

Traditional Automated Valuation Models (AVMs) estimate house prices primarily using tabular features such as the number of bedrooms, bathrooms, and square footage. However, these models often fail to capture one of the most important factors influencing property value—the surrounding neighbourhood.

This project develops an advanced **Geospatial Real Estate Valuation System** that leverages **Spatial Embeddings**, **K-Nearest Neighbor (KNN) Graphs**, and **Graph Neural Networks (GNNs)** to model spatial relationships between properties and produce highly accurate house price predictions.

The objective is to demonstrate that incorporating spatial dependencies significantly improves prediction accuracy compared to traditional machine learning models.

---

# 🎯 Objectives

- Build an intelligent property valuation model.
- Capture neighbourhood influence using graph-based learning.
- Compare traditional ML models with Graph Neural Networks.
- Visualize predicted property values on an interactive map.
- Deploy an end-to-end valuation dashboard.

---

# 🚀 Problem Statement

House prices depend not only on the property's characteristics but also on:

- Nearby properties
- Schools
- Hospitals
- Shopping centres
- Parks
- Transportation
- Socio-economic conditions

Traditional regression models cannot naturally learn these spatial relationships.

This project addresses the problem by modelling houses as nodes in a graph where neighbouring properties influence each other's valuation.

---

# 👥 User Personas

## Real Estate Appraiser

**Needs**

- Accurate property valuation
- Explainable comparable properties
- Reduced valuation bias

**Workflow**

- Enter property location
- View predicted price
- Display Top 5 influential neighbouring properties

---

## Investment Strategist

**Needs**

- Detect undervalued locations
- Identify investment hotspots
- Understand regional pricing trends

**Workflow**

- View valuation heatmaps
- Compare neighbourhoods
- Analyse future investment opportunities

---

# 🛠 Tech Stack

### Programming

- Python 3.x

### Data Processing

- Pandas
- NumPy

### Geospatial Processing

- GeoPandas
- Shapely
- Geopy

### Machine Learning

- Scikit-Learn
- XGBoost

### Deep Learning

- PyTorch
- PyTorch Geometric (or DGL)

### Graph Algorithms

- NetworkX
- KNN Graph Construction

### Visualization

- Matplotlib
- Seaborn
- Folium
- Plotly

### Deployment

- Streamlit

### Version Control

- Git
- GitHub

---

# 📂 Dataset

Recommended Dataset:

**King County House Sales Dataset**

Dataset contains:

- House Price
- Latitude
- Longitude
- Bedrooms
- Bathrooms
- Living Area
- Lot Size
- Floors
- Waterfront
- View
- Condition
- Grade
- Year Built
- Renovation Year

---

# 📊 Workflow

```
Raw Housing Dataset
        │
        ▼
Data Cleaning
        │
        ▼
Geospatial Processing
        │
        ▼
Feature Engineering
        │
        ▼
Baseline ML Model (XGBoost)
        │
        ▼
Graph Construction
        │
        ▼
Spatial Embeddings
        │
        ▼
Graph Neural Network
        │
        ▼
Performance Comparison
        │
        ▼
Streamlit Dashboard
```

---

# 📅 Project Timeline

## Week 1 – Geospatial Data Processing

### Tasks

- Import dataset
- Clean missing values
- Remove outliers
- Process Latitude & Longitude
- Calculate Haversine distance
- Plot interactive maps using Folium

### Deliverables

- Clean dataset
- Geographic visualizations

---

## Week 2 – Baseline Machine Learning

### Feature Engineering

- House age
- Distance from city centre
- Property size
- Number of rooms

### Model

- XGBoost Regressor

### Evaluation

- RMSE
- MAE
- MAPE
- R² Score

---

## Week 3 – Graph Construction

### Tasks

- Create graph representation
- Each house = Node
- K nearest houses = Edges

### Techniques

- K-Nearest Neighbors
- Spatial Embeddings
- NetworkX

---

## Week 4 – Graph Neural Network

### Model

- Graph Neural Network (GNN)
- Graph Attention Network (GAT)

### Tasks

- Train graph model
- Compare with XGBoost
- Generate valuation predictions

### Deployment

- Streamlit Dashboard

---

# 📈 Model Comparison

| Model | Features Used | Expected Performance |
|--------|---------------|---------------------|
| Linear Regression | Tabular | Baseline |
| XGBoost | Engineered Features | Better |
| Graph Neural Network | Spatial + Tabular | Best |

---

# 📊 Evaluation Metrics

- Mean Absolute Percentage Error (MAPE)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

---

# 🌍 Key Features

- Geospatial Data Processing
- Haversine Distance Calculation
- KNN Graph Construction
- Spatial Embeddings
- Graph Neural Networks
- Interactive Maps
- Explainable Neighbour Influence
- Streamlit Deployment

---

# 📁 Project Structure

```
Geospatial-RealEstate-Valuation/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── graph_builder.py
│   ├── train_xgboost.py
│   ├── train_gnn.py
│   ├── evaluation.py
│   └── utils.py
│
├── app/
│   └── streamlit_app.py
│
├── models/
│
├── outputs/
│   ├── maps/
│   ├── graphs/
│   └── predictions/
│
├── requirements.txt
│
├── README.md
│
└── LICENSE
```

---

# ▶️ Installation

```bash
git clone https://github.com/yourusername/Geospatial-RealEstate-Valuation.git

cd Geospatial-RealEstate-Valuation

pip install -r requirements.txt
```

---

# ▶️ Run the Project

### Train Baseline Model

```bash
python src/train_xgboost.py
```

### Train Graph Neural Network

```bash
python src/train_gnn.py
```

### Launch Dashboard

```bash
streamlit run app/streamlit_app.py
```

---

# 📌 Expected Outputs

- Accurate property valuation
- Spatial price heatmaps
- Comparable neighbouring properties
- Interactive geospatial dashboard
- Improved prediction accuracy over traditional ML models

---

# 🔮 Future Enhancements

- Real-time property valuation
- Integration with live real estate APIs
- Satellite imagery analysis
- Temporal price forecasting
- Investment recommendation engine
- Explainable AI (XAI) for valuation transparency

---

# 📚 Learning Outcomes

Through this project, the following concepts are explored:

- Geospatial Data Analysis
- Feature Engineering
- Spatial Embeddings
- Graph Theory
- Graph Neural Networks
- Attention Mechanisms
- Property Valuation Models
- Interactive Geospatial Visualization
- Streamlit Deployment
- End-to-End Machine Learning Pipeline

---

# 👨‍💻 Author

**Suriya V G**

B.Tech Artificial Intelligence & Data Science

SNS College of Engineering

---

# ⭐ Acknowledgements

- King County Housing Dataset
- Scikit-Learn
- XGBoost
- PyTorch
- PyTorch Geometric
- GeoPandas
- Folium
- Streamlit
