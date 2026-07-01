"""
Warehouse Item-Zone Efficiency Pipeline — Production FastAPI Service
=====================================================================
Endpoint : POST /api/warehouse/update-efficiency
Reads    : tbl_pick_summary, tblspace, putaway_summary  (MS SQL Express)
Writes   : tbl_item_efficiency_scores                   (MS SQL Express)
Math     : UNCHANGED from inference notebook — Sections 3, 4, 5 are identical
"""

import warnings
warnings.filterwarnings('ignore')

import re
import json
import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from sqlalchemy import create_engine, text
from contextlib import asynccontextmanager


# =====================================================================
# CONFIGURATION — change only these for a new environment
# =====================================================================
MODEL_PATH   = 'hybrid_production_model.pkl'
SCORE_YEAR   = 2026
DB_SERVER    = r'localhost\SQLEXPRESS'
DB_NAME      = 'WarehouseDB'
DB_DRIVER    = 'ODBC+Driver+17+for+SQL+Server'

CONNECTION_STRING = (
    f"mssql+pyodbc://@{DB_SERVER}/{DB_NAME}"
    f"?driver={DB_DRIVER}&trusted_connection=yes"
)


# =====================================================================
# STARTUP — load model artifacts once at server start
# =====================================================================
model_artifacts = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        payload = joblib.load(MODEL_PATH)
        model_artifacts['scaler']                         = payload['scaler']
        model_artifacts['features_to_cluster']            = payload['features_to_cluster']
        model_artifacts['throughput_95th_ceiling_global'] = payload['throughput_95th_ceiling_global']
        model_artifacts['putaway_75th_ceiling_global']    = payload['putaway_75th_ceiling_global']
        model_artifacts['warehouse_median_rate']          = payload['warehouse_median_rate']
        model_artifacts['d2s_baseline_seconds']           = payload['d2s_baseline_seconds']
        model_artifacts['sku_median_rates']               = payload['sku_median_rates']
        model_artifacts['machine_baselines']              = payload['machine_baselines']
        model_artifacts['models']                         = payload['models']
        model_artifacts['weights']                        = payload['weights']
        print(f"Model artifacts loaded from '{MODEL_PATH}'")
    except FileNotFoundError:
        raise RuntimeError(
            f"CRITICAL: '{MODEL_PATH}' not found. "
            "Run the training notebook first to generate the model artifact."
        )
    yield
    model_artifacts.clear()


app = FastAPI(
    title="Warehouse Efficiency API",
    description="Item-zone-month ML efficiency scoring pipeline",
    version="2.0.0",
    lifespan=lifespan
)


# =====================================================================
# ZONE VALIDATION — DO NOT MODIFY
# =====================================================================
def is_valid_warehouse_zone(zone_str):
    if pd.isna(zone_str):
        return False
    zone_str = str(zone_str).strip().upper()
    named_exceptions = [
        'AA', 'AB', 'AC', 'AD', 'AE', 'AF', 'AG', 'AH', 'AI', 'AJ', 'AK', 'AL',
        'BA', 'BB', 'BC', 'BD', 'BE', 'BF', 'BG', 'BH', 'BI', 'BJ',
        'DP', 'LOC'
    ]
    if zone_str in named_exceptions:
        return True
    return bool(re.match(r'^[A-Z]{2}$', zone_str))


# =====================================================================
# PIECEWISE SCORING — DO NOT MODIFY
# =====================================================================
def calculate_track_scores(df, X_scaled, kmeans_model, weight_vector,
                           score_col, tier_col):
    X_weighted = X_scaled * np.sqrt(weight_vector)
    labels    = kmeans_model.predict(X_weighted)
    distances = kmeans_model.transform(X_weighted)

    center_means    = kmeans_model.cluster_centers_.mean(axis=1)
    sorted_clusters = np.argsort(center_means)
    t3_idx = sorted_clusters[0]
    t2_idx = sorted_clusters[1]
    t1_idx = sorted_clusters[2]

    cluster_names = {
        t1_idx: 'Tier 1: High Efficiency',
        t2_idx: 'Tier 2: Moderate Efficiency',
        t3_idx: 'Tier 3: Low Efficiency'
    }

    d1 = distances[:, t1_idx]
    d2 = distances[:, t2_idx]
    d3 = distances[:, t3_idx]

    scores = np.zeros(len(df))
    c_t3       = (labels == t3_idx)
    c_t2_lower = (labels == t2_idx) & (d3 < d1)
    c_t2_upper = (labels == t2_idx) & (d3 >= d1)
    c_t1       = (labels == t1_idx)

    scores[c_t3]       = 33.33 * (1 - (d3[c_t3] / (d3[c_t3] + d2[c_t3] + 1e-5)))
    scores[c_t2_lower] = 33.33 + (16.67 * (d3[c_t2_lower] / (d2[c_t2_lower] + d3[c_t2_lower] + 1e-5)))
    scores[c_t2_upper] = 50.00 + (16.66 * (1 - (d2[c_t2_upper] / (d1[c_t2_upper] + d2[c_t2_upper] + 1e-5))))
    scores[c_t1]       = 66.66 + (33.34 * (d2[c_t1] / (d1[c_t1] + d2[c_t1] + 1e-5)))

    pk_rate = df['Item_Pick_Rate_Score'].values
    ob_thru = df['Item_Throughput_Score'].values
    d2s     = df['Item_D2S_Score'].values
    pa_thru = df['Item_Putaway_Throughput_Score'].values

    zombie_mask  = (pk_rate <= 5.0)  & (ob_thru <= 5.0)  & (d2s <= 10.0) & (pa_thru <= 10.0)
    perfect_mask = (pk_rate >= 82.0) & (ob_thru >= 89.0) & (d2s >= 85.0) & (pa_thru >= 85.0)

    scores[zombie_mask]  = 0.0
    scores[perfect_mask] = 100.0

    df[score_col] = scores
    df[tier_col]  = np.vectorize(cluster_names.get)(labels)
    return df


# =====================================================================
# DATABASE HELPERS
# =====================================================================
def get_engine():
    return create_engine(CONNECTION_STRING, fast_executemany=True)


def load_table1(conn, year):
    return pd.read_sql(text(f"""
        SELECT summary_id, pickDate, [count], itemCode, itemDescription,
               pickQty, ordQty, pickedStorageBin, pickerId, pickHeNo,
               storageSectionId, totalTimeSeconds, perQtyTime
        FROM tbl_pick_summary
        WHERE YEAR(pickDate) = {year}
    """), conn)


def load_table2(conn, year):
    return pd.read_sql(text(f"""
        SELECT space_id, st_sec_id, totalCount, usedCount,
               createdDate, utilization, itemCode
        FROM tblspace
        WHERE YEAR(createdDate) = {year}
    """), conn)


def load_table3(conn, year):
    return pd.read_sql(text(f"""
        SELECT summary_id, putawayDate, [count], itemCode, putAwayQty,
               ordQty, storageBin, storageSectionId, totalTimeinSeconds,
               perQtyTime, item_text, proposed_st_bin
        FROM putaway_summary
        WHERE YEAR(putawayDate) = {year}
    """), conn)


# =====================================================================
# MAIN ENDPOINT
# =====================================================================
@app.post("/api/warehouse/update-efficiency")
def update_efficiency():
    try:
        # --- Section 1: Extract artifacts ---
        scaler                         = model_artifacts['scaler']
        features_to_cluster            = model_artifacts['features_to_cluster']
        throughput_95th_ceiling_global = model_artifacts['throughput_95th_ceiling_global']
        putaway_75th_ceiling_global    = model_artifacts['putaway_75th_ceiling_global']
        warehouse_median_rate          = model_artifacts['warehouse_median_rate']
        d2s_baseline_seconds           = model_artifacts['d2s_baseline_seconds']
        sku_median_rates               = model_artifacts['sku_median_rates']
        machine_baselines              = model_artifacts['machine_baselines']
        models  = model_artifacts['models']
        weights = model_artifacts['weights']

        critic_model   = models['data']
        ahp_model      = models['strategy']
        hybrid_model   = models['hybrid']
        critic_weights = np.array(weights['data'])
        ahp_weights    = np.array(weights['strategy'])
        hybrid_weights = np.array(weights['hybrid'])

        # --- Section 2: Load data from SQL ---
        engine = get_engine()
        with engine.connect() as conn:
            table1 = load_table1(conn, SCORE_YEAR)
            table2 = load_table2(conn, SCORE_YEAR)
            table3 = load_table3(conn, SCORE_YEAR)

        print(f"SQL loaded — Pick: {len(table1):,} | Space: {len(table2):,} | Putaway: {len(table3):,}")

        # Parse dates
        table1['pickDate']     = pd.to_datetime(table1['pickDate'],    errors='coerce')
        table2['createdDate']  = pd.to_datetime(table2['createdDate'], errors='coerce')
        table3['putaway_Date'] = pd.to_datetime(table3['putawayDate'], format='%Y-%m-%d', errors='coerce')

        # Zone validation
        table1['storageSectionId'] = table1['storageSectionId'].astype(str).str.strip().str.upper()
        table2['st_sec_id']        = table2['st_sec_id'].astype(str).str.strip().str.upper()
        table3['storageSectionId'] = table3['storageSectionId'].astype(str).str.strip().str.upper()

        table1 = table1[table1['storageSectionId'].apply(is_valid_warehouse_zone)].copy()
        table2 = table2[table2['st_sec_id'].apply(is_valid_warehouse_zone)].copy()
        table3 = table3[table3['storageSectionId'].apply(is_valid_warehouse_zone)].copy()

        # Operational sanity filters
        table1 = table1[
            (table1['totalTimeSeconds'] >= 30) &
            (table1['totalTimeSeconds'] <= 7200) &
            (table1['pickQty'] >= 0) &
            (table1['ordQty'] > 0)
        ].copy()
        table2 = table2[table2['totalCount'] > 0].copy()
        table3 = table3[
            (table3['totalTimeinSeconds'] > 0) &
            (table3['totalTimeinSeconds'] <= 3600) &
            (table3['putAwayQty'] > 0)
        ].copy()

        # Month column
        table1['month'] = table1['pickDate'].dt.to_period('M').astype(str)
        table2['month'] = table2['createdDate'].dt.to_period('M').astype(str)
        table3['month'] = table3['putaway_Date'].dt.to_period('M').astype(str)

        # String cleaning
        table1['itemCode'] = table1['itemCode'].astype(str).str.strip()
        table2['itemCode'] = table2['itemCode'].astype(str).str.strip()
        table3['itemCode'] = table3['itemCode'].astype(str).str.strip()
        table1['pickHeNo'] = table1['pickHeNo'].astype(str).str.strip().str.upper()
        table3['storageBin']      = table3['storageBin'].astype(str).str.upper().str.strip()
        table3['proposed_st_bin'] = table3['proposed_st_bin'].astype(str).str.upper().str.strip()

        # --- Section 3: Engineer 8 parameters --- DO NOT MODIFY ---

        # P1: Item Pick Rate Score
        pr_df = table1.groupby(['itemCode', 'month', 'storageSectionId']).agg(
            item_picks=('pickQty', 'sum'),
            item_time =('totalTimeSeconds', 'sum')
        ).reset_index()
        pr_df['item_pick_rate'] = pr_df['item_picks'] / ((pr_df['item_time'] / 3600) + 0.0001)
        pr_df = pr_df.merge(sku_median_rates, on='itemCode', how='left')
        pr_df['target_rate'] = pr_df['target_rate'].fillna(warehouse_median_rate)
        pr_df['Item_Pick_Rate_Score'] = (
            (pr_df['item_pick_rate'] / (pr_df['target_rate'] + 0.0001)) * 100
        ).clip(upper=100.0)

        # P2: Item Throughput Score
        pr_df['item_throughput'] = pr_df['item_picks'] / (pr_df['item_time'] + 0.001)
        zt_df = pr_df[['itemCode', 'month', 'storageSectionId', 'item_throughput']].copy()
        zt_df['Item_Throughput_Score'] = (
            (zt_df['item_throughput'] / throughput_95th_ceiling_global) * 100
        ).clip(upper=100.0)

        # P3: Item Picker Performance Score (Volume-Weighted)
        table1['picker_rate_raw']   = table1['pickQty'] / ((table1['totalTimeSeconds'] / 3600) + 0.001)
        table1['picker_prod_score'] = (table1['picker_rate_raw'] / warehouse_median_rate) * 100
        table1['picker_prod_score'] = table1['picker_prod_score'].clip(upper=100.0)
        table1['weighted_picker']   = table1['picker_prod_score'] * table1['pickQty']
        zone_picker_df = table1.groupby(['itemCode', 'month', 'storageSectionId']).agg(
            sum_weighted_picker=('weighted_picker', 'sum'),
            sum_pick_qty       =('pickQty', 'sum')
        ).reset_index()
        zone_picker_df['Item_Picker_Performance_Score'] = (
            zone_picker_df['sum_weighted_picker'] / (zone_picker_df['sum_pick_qty'] + 0.001)
        ).clip(upper=100.0)

        # P4: Space Score (Zone-level broadcast)
        space_raw = table2.dropna(subset=['st_sec_id', 'itemCode']).copy()
        space_raw.rename(columns={'st_sec_id': 'storageSectionId'}, inplace=True)
        zm_space = (
            space_raw.groupby(['storageSectionId', 'month'])['utilization']
            .mean().reset_index()
            .rename(columns={'utilization': 'Item_Space_Score'})
        )
        zm_space['Item_Space_Score'] = zm_space['Item_Space_Score'].clip(upper=100.0)
        space_df = space_raw[['itemCode', 'month', 'storageSectionId']].drop_duplicates()
        space_df = space_df.merge(
            zm_space[['storageSectionId', 'month', 'Item_Space_Score']],
            on=['storageSectionId', 'month'], how='left'
        )

        # P5: Item Equipment Score (Frozen Baselines)
        equip_stats = table1.groupby(['pickHeNo', 'month', 'storageSectionId']).agg(
            active_equip_hours  =('totalTimeSeconds', lambda x: x.sum() / 3600),
            total_machine_picks =('pickQty', 'sum')
        ).reset_index()
        equip_stats['Picks_per_Equipment_Hour'] = (
            equip_stats['total_machine_picks'] / (equip_stats['active_equip_hours'] + 0.0001)
        )
        equip_stats = equip_stats.merge(machine_baselines, on='pickHeNo', how='left')
        equip_stats['Specific_Machine_Baseline'] = equip_stats['Specific_Machine_Baseline'].fillna(
            equip_stats['Picks_per_Equipment_Hour'].median()
        )
        equip_stats['Rel_Equip_Eff'] = (
            (equip_stats['Picks_per_Equipment_Hour'] / (equip_stats['Specific_Machine_Baseline'] + 0.0001)) * 100
        ).clip(upper=100.0)
        t1_equip = table1[['itemCode', 'month', 'storageSectionId', 'pickHeNo', 'pickQty']].merge(
            equip_stats[['pickHeNo', 'month', 'storageSectionId', 'Rel_Equip_Eff']],
            on=['pickHeNo', 'month', 'storageSectionId'], how='left'
        )
        t1_equip['weighted_equip'] = t1_equip['Rel_Equip_Eff'] * t1_equip['pickQty']
        zone_equip_df = t1_equip.groupby(['itemCode', 'month', 'storageSectionId']).agg(
            sum_weighted_equip=('weighted_equip', 'sum'),
            sum_equip_qty     =('pickQty', 'sum')
        ).reset_index()
        zone_equip_df['Item_Equipment_Score'] = (
            zone_equip_df['sum_weighted_equip'] / (zone_equip_df['sum_equip_qty'] + 0.001)
        ).clip(upper=100.0)

        # P6: Item Putaway Throughput Score (units/sec)
        pt_df = table3.groupby(['itemCode', 'month', 'storageSectionId']).agg(
            qty =('putAwayQty', 'sum'),
            time=('totalTimeinSeconds', 'sum')
        ).reset_index()
        pt_df['putaway_rate'] = pt_df['qty'] / (pt_df['time'] + 0.001)
        pt_df['Item_Putaway_Throughput_Score'] = (
            (pt_df['putaway_rate'] / putaway_75th_ceiling_global) * 100
        ).clip(upper=100.0)

        # P7: Item Dock-to-Stock Score
        table3['time_qty_product'] = table3['totalTimeinSeconds'] * table3['putAwayQty']
        d2s_agg = table3.groupby(['itemCode', 'month', 'storageSectionId']).agg(
            sum_time_qty_product=('time_qty_product', 'sum'),
            total_putaway_volume=('putAwayQty', 'sum')
        ).reset_index()
        d2s_agg['weighted_d2s_seconds'] = (
            d2s_agg['sum_time_qty_product'] / (d2s_agg['total_putaway_volume'] + 0.001)
        )
        d2s_agg['Item_D2S_Score'] = (
            (d2s_baseline_seconds / (d2s_agg['weighted_d2s_seconds'] + 0.001)) * 100
        ).clip(upper=100.0)

        # P8: Item Putaway Location Accuracy Score
        table3['bin_match'] = (table3['storageBin'] == table3['proposed_st_bin']).astype(int)
        acc_df = table3.groupby(['itemCode', 'month', 'storageSectionId']).agg(
            matches=('bin_match', 'sum'),
            total  =('bin_match', 'count')
        ).reset_index()
        acc_df['Item_Putaway_Accuracy_Score'] = (acc_df['matches'] / acc_df['total']) * 100

        print("All 8 parameters computed.")

        # --- Section 4: Assemble master table --- DO NOT MODIFY ---
        KEYS = ['itemCode', 'month', 'storageSectionId']
        score_cols_map = [
            (pr_df,          'Item_Pick_Rate_Score'),
            (zt_df,          'Item_Throughput_Score'),
            (zone_picker_df, 'Item_Picker_Performance_Score'),
            (space_df,       'Item_Space_Score'),
            (zone_equip_df,  'Item_Equipment_Score'),
            (pt_df,          'Item_Putaway_Throughput_Score'),
            (d2s_agg,        'Item_D2S_Score'),
            (acc_df,         'Item_Putaway_Accuracy_Score'),
        ]

        master = pd.concat([df[KEYS] for df, _ in score_cols_map]).drop_duplicates().reset_index(drop=True)

        for df_param, col in score_cols_map:
            master = master.merge(df_param[KEYS + [col]], on=KEYS, how='left')

        PARAM_COLS = [col for _, col in score_cols_map]
        for col in PARAM_COLS:
            grp_median = master.groupby(['itemCode', 'storageSectionId'])[col].transform('median')
            master[col] = master[col].fillna(grp_median)
            master[col] = master[col].fillna(master[col].median())
            master[col] = master[col].clip(lower=0.0, upper=100.0)

        print(f"Master shape: {master.shape} | Nulls: {master[PARAM_COLS].isna().sum().sum()}")

        # --- Section 5: Scale & score --- DO NOT MODIFY ---
        X_raw    = master[features_to_cluster].values
        X_scaled = scaler.transform(X_raw)
        master   = master.reset_index(drop=True)

        for model, wts, score_col, tier_col in [
            (critic_model,  critic_weights,  'CRITIC_Efficiency_Score',  'CRITIC_Efficiency_Tier'),
            (ahp_model,     ahp_weights,     'AHP_Efficiency_Score',     'AHP_Efficiency_Tier'),
            (hybrid_model,  hybrid_weights,  'Hybrid_Efficiency_Score',  'Hybrid_Efficiency_Tier'),
        ]:
            master = calculate_track_scores(
                master, X_scaled, model, wts, score_col, tier_col
            )

        print("Multi-track scoring complete.")

        # --- Section 6: Build final output ---
        output_cols = (
            KEYS + features_to_cluster + [
                'CRITIC_Efficiency_Score',  'CRITIC_Efficiency_Tier',
                'AHP_Efficiency_Score',     'AHP_Efficiency_Tier',
                'Hybrid_Efficiency_Score',  'Hybrid_Efficiency_Tier'
            ]
        )

        final_df = master[output_cols].copy()
        final_df = final_df.sort_values(['month', 'storageSectionId', 'itemCode']).reset_index(drop=True)

        num_cols = final_df.select_dtypes(include=[np.number]).columns
        final_df[num_cols] = final_df[num_cols].round(2)

        max_dates = [
            d for d in [
                table1['pickDate'].max(),
                table2['createdDate'].max(),
                table3['putaway_Date'].max()
            ] if pd.notnull(d)
        ]
        last_date_str = max(max_dates).strftime('%Y-%m-%d') if max_dates else f"{SCORE_YEAR}-12-31"
        final_df['score_year']           = SCORE_YEAR
        final_df['last_calculated_date'] = last_date_str

        # --- Section 7: Write to MS SQL ---
        print(f"Writing {len(final_df):,} rows to tbl_item_efficiency_scores...")

        with engine.begin() as conn:
            final_df.to_sql(
                name      ='tbl_item_efficiency_scores',
                con       = conn,
                if_exists ='replace',
                index     = False,
                chunksize = 1000,
                method    ='multi'
            )

        print(f"{len(final_df):,} rows written successfully.")

        # Build weight summary
        weight_summary = {
            track: {
                feature: round(float(w) * 100, 2)
                for feature, w in zip(features_to_cluster, wts)
            }
            for track, wts in [
                ('CRITIC',  critic_weights),
                ('AHP',     ahp_weights),
                ('HYBRID',  hybrid_weights)
            ]
        }

        return {
            "status":               "success",
            "score_year":           SCORE_YEAR,
            "last_calculated_date": last_date_str,
            "rows_inserted":        len(final_df),
            "table":                "tbl_item_efficiency_scores",
            "weights":              weight_summary
        }

    except Exception as e:
        import traceback
        print(f"Pipeline Error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================================
# HEALTH CHECK — for Java to ping before triggering the pipeline
# =====================================================================
@app.get("/health")
def health_check():
    return {
        "status":       "ok",
        "model_loaded": len(model_artifacts) > 0,
        "score_year":   SCORE_YEAR
    }


# =====================================================================
# LOCAL DEV RUNNER
# =====================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)