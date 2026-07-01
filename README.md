# 🏢 AI-Powered Warehouse Item & Zone Efficiency Audit System

An enterprise-grade, full-stack machine learning pipeline and API that monitors, evaluates, and ranks warehouse operational performance at the Item-Zone-Month level. By combining data-driven objective weighting (**CRITIC**) with strategic human priorities (**Group AHP**), the engine feeds a custom **Weighted K-Means & Geometric TOPSIS Piecewise Scoring Engine** to track labor, throughput, and space bottlenecks.

---

## 🚀 Key Features
* **Multi-Source Data Ingestion:** Automates pipeline extractions from underlying warehouse management tracking databases (`tbl_pick_summary`, `tblspace`, `tbl_putaway_summary`).
* **Dual-Track Weighting Engine:** Balanced multi-criteria decision-making using **CRITIC** (statistical contrast/correlation) and **Group AHP** (expert pairwise consensus metrics).
* **Advanced Unsupervised ML Clustering:** Implements **Weighted K-Means Clustering** to segment warehouse operations into sharp, definitive efficiency boundaries (High, Moderate, Low).
* **Geometric TOPSIS Scoring:** Replaces rigid clustering with a smooth, continuous 0–100 piecewise score curve based on mathematical proximity to ideal ("Dream") and anti-ideal ("Nightmare") warehouse conditions.
* **Production-Ready Enterprise API:** Powered by **FastAPI** with optimized database integration for Microsoft SQL Server Express utilizing `pyodbc` fast bulk inserts.

---

## 📐 Mathematical & Architectural Framework

### 1. Data Preparation & Normalization
The system extracts **9 operational performance parameters** (including *Pick Rate, Total Throughput, Picker Performance, Space Utilization, Equipment Pacing, Putaway Speeds, and Accuracy*). To maintain absolute mathematical fairness across varying native metrics (such as raw durations vs percentages), all inputs are normalized via Min-Max scaling onto a strict $0.0$ to $1.0$ baseline.

### 2. The Unsupervised Model: Weighted K-Means
Because warehouse data doesn't come with pre-defined "efficiency grades," a supervised approach is impossible. The engine leverages an unsupervised **K-Means Model** ($k=3$) to partition data into crisp operational tiers. 

To ensure critical metrics are prioritized over secondary ones, a **Weighted K-Means matrix** is constructed. The normalized data matrix is multiplied by the square root of the final weights ($\sqrt{\text{Weights}}$) before clustering, which geometrically stretches the vector space to naturally guide the algorithm's boundaries according to business realities.

### 3. The TOPSIS-Piecewise Scoring Engine
Traditional clustering assigns strict binary groups. To generate granular rankings, the pipeline introduces a continuous **TOPSIS-inspired Piecewise Scoring Engine**. The calculated K-Means cluster centers act as spatial anchors:

*   **The High-Efficiency Center ($t_1$):** Represents the *Ideal Solution* (The Dream Item profile).
*   **The Low-Efficiency Center ($t_3$):** Represents the *Anti-Ideal Solution* (The Nightmare Bottleneck profile).

Using multi-dimensional geometric distances ($d_1, d_2, d_3$) away from these centers, items are dynamically mapped onto a continuous 0 to 100 score distribution:

*   **Tier 3 (Low Efficiency) | Range [0.0 - 33.33]:** Evaluates geometric proximity to the low anchor relative to the moderate tier.
    $$ \text{Score} = 33.33 \times \left(1 - \frac{d_3}{d_3 + d_2 + 1\text{e}^{-5}}\right) $$
*   **Tier 2 (Moderate Efficiency) | Range [33.33 - 66.66]:** Evaluates median performance, adjusting the score step seamlessly depending on whether the asset leans lower ($d_3 < d_1$) or higher.
*   **Tier 1 (High Efficiency) | Range [66.66 - 100.0]:** Evaluates closeness to elite performance peaks. As an item nears perfect alignment with the high anchor ($d_1 \to 0$), its score climbs directly to 100.0.
    $$ \text{Score} = 66.66 + \left(33.34 \times \frac{d_2}{d_1 + d_2 + 1\text{e}^{-5}}\right) $$

### 4. Operational Safety Guardrails
The system includes programmatic hard-stops to capture statistical anomalies:
*   **True Zombie Zone:** Any item exhibiting frozen pick/putaway activity ($\le 5.0\%$) is forcefully hard-coded to a score of **0.0** and tagged as Low Efficiency to immediately flag dead stock for relocation.
*   **Perfect World Zone:** Items exceeding elite milestones simultaneously across all primary tracks are instantly fast-tracked to a clean **100.0** score.

---

## 🛠️ Technology Stack
* **Language:** Python 3.12.3
* **API Framework:** FastAPI, Uvicorn
* **Data & Machine Learning:** Pandas, NumPy, Scikit-Learn (KMeans), Joblib
* **Database Layer:** SQLAlchemy, PyODBC, Microsoft SQL Server / SQL Express
* **Database Management:** SQL Server Management Studio (SSMS)

---

## 📦 Project Setup & Installation

### 1. Environment Activation
It is highly recommended to isolate this production stack to ensure version-level compatibility with internal server environments:
```bash
# Create a virtual environment bubble
python -m venv venv

# Activate the virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

```

### 2. Dependency Ingestion

Since internal servers often have strict network limitations and cached package repositories, install the directly verified library footprint using this exact command:

```bash
pip install pandas==2.3.3 scikit-learn==1.3.2 numpy==1.26.4 joblib==1.3.2 fastapi uvicorn sqlalchemy pyodbc

```

### 3. Running the Server Locally

Launch the asynchronous Uvicorn engine to start your API endpoint:

```bash
python main.py

```

Once initialized, navigate your web browser to `http://localhost:8000/docs` to interact with the Swagger UI documentation and execute the processing endpoints on-demand.

---

## 📊 Database Output Schema

Calculated output arrays are systematically piped directly back into your target warehouse server instance, writing or replacing the **`tbl_item_efficiency_scores`** table with the following parameters:

| Column Name | Type | Description |
| --- | --- | --- |
| `itemCode` | `VARCHAR` | Unique identifier for the warehouse stock SKU. |
| `month` | `VARCHAR` | The transaction grouping month (YYYY-MM). |
| `storageSectionId` | `VARCHAR` | The physical zone or structural warehouse sector. |
| `CRITIC_Efficiency_Score` | `FLOAT` | Purely statistical data-driven efficiency score. |
| `CRITIC_Efficiency_Tier` | `VARCHAR` | Text classification tier based on the objective track. |
| `AHP_Efficiency_Score` | `FLOAT` | Human business strategy-driven efficiency score. |
| `AHP_Efficiency_Tier` | `VARCHAR` | Text classification tier based on the subjective track. |
| `Hybrid_Efficiency_Score` | `FLOAT` | The final blended model metric (50/50 balance). |
| `Hybrid_Efficiency_Tier` | `VARCHAR` | Final production operational tier tag. |
| `score_year` | `INT` | Calendar year of execution context filter. |
| `last_calculated_date` | `DATE` | Data auditing timestamp tracking data freshness. |

---

*Developed at Tekclover as a scalable, enterprise-grade solution for warehouse intelligence and supply chain optimization.*

---

## Author

**Adityaa SS**

Machine Learning Engineer | Data-Driven Systems & Predictive Modeling

GitHub: [https://github.com/4dh11](https://github.com/4dh11)

LinkedIn: [https://www.linkedin.com/in/adityaa-ss-30233b2b3/](https://www.linkedin.com/in/adityaa-ss-30233b2b3/)

---
