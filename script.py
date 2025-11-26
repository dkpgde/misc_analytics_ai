# %%
import os
import subprocess
import sys
import time
import warnings

from chronos import BaseChronosPipeline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import torch

# %%
warnings.filterwarnings("ignore", module="chronos")

# %%
BATCH_MODE = True

BASE_PATH = '...'
INPUT_FILE = os.path.join(BASE_PATH, 'train.csv')
ONE_TRAIN = os.path.join(BASE_PATH, 'train_one.csv')
ONE_TEST = os.path.join(BASE_PATH, 'test_one.csv')

# %%
LEAD_TIME = 14
SERVICE_LEVEL = 0.97
HOLDING_COST = 0.05
STOCKOUT_COST = 45
ORDERING_COST = 20

# %%
def plot_adf_benchmark(adf_stat, critical_values, title='Stationarity Test Result'):
    labels = ['ADF Statistic'] + list(critical_values.keys())
    values = [adf_stat] + list(critical_values.values())
    colors = ['#1f77b4'] + ['#d62728' for _ in critical_values]

    plt.figure(figsize=(10, 6))
    
    bars = plt.barh(labels, values, color=colors, alpha=0.8)
    
    for bar in bars:
        plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, 
                 f'{bar.get_width():.4f}', 
                 va='center', ha='left' if bar.get_width() > 0 else 'right',
                 fontweight='bold')

    plt.axvline(critical_values['5%'], color='green', linestyle='--', linewidth=2, label='5% Confidence Threshold')
    
    plt.title(title, fontsize=14)
    plt.xlabel('Test Statistic Value (More Negative is Better)')
    plt.legend()
    plt.grid(axis='x', alpha=0.3)
    
    status = 'STATIONARY' if adf_stat < critical_values['5%'] else 'NON-STATIONARY'
    plt.figtext(0.5, -0.05, f'Result: {status}', 
                ha='center', fontsize=12, bbox={'facecolor':'orange', 'alpha':0.2, 'pad':5})

    plt.tight_layout()
    plt.show()

# %%
def calculate_policy(mean_forecast, forecast_error_std_dev):
    z_score = stats.norm.ppf(SERVICE_LEVEL)

    safety_stock = z_score * forecast_error_std_dev * np.sqrt(LEAD_TIME)

    reorder_point = (mean_forecast * LEAD_TIME) + safety_stock

    annual_demand = mean_forecast * 365
    if HOLDING_COST > 0:
        order_quantity = np.sqrt((2 * annual_demand * ORDERING_COST) / (HOLDING_COST * 365))
    else:
        order_quantity = mean_forecast * LEAD_TIME

    return {
        'safety_stock': round(safety_stock, 2),
        'reorder_point': round(reorder_point, 2),
        'order_quantity': round(order_quantity, 2)
    }

# %%
def run_dynamic_simulation(forecast_series, actual_demand, forecast_error_std_dev, verbose=False):
    n_days = len(actual_demand)
    
    z_score = stats.norm.ppf(SERVICE_LEVEL)
    
    safety_stock = z_score * forecast_error_std_dev * np.sqrt(LEAD_TIME)
    
    avg_daily_demand = np.mean(forecast_series)
    if HOLDING_COST > 0:
        order_quantity = np.sqrt((2 * avg_daily_demand * 365 * ORDERING_COST) / (HOLDING_COST * 365))
    else:
        order_quantity = avg_daily_demand * LEAD_TIME
    
    order_quantity = max(1.0, round(order_quantity, 0))

    initial_lookahead = sum(forecast_series[:LEAD_TIME])
    inventory = initial_lookahead + safety_stock
    
    pipeline_orders = []
    
    total_holding_cost = 0.0
    total_stockout_cost = 0.0
    total_lost_sales = 0.0

    if verbose:
        print(f'\nDynamic Simulation Start')
        print(f'SS (Dynamic): {safety_stock:.2f} | EOQ: {order_quantity}')
        print(f'{"Day":<5} | {"Fcst(LT)":<9} | {"DynROP":<8} | {"InvPos":<8} | {"NetInv":<8} | {"Demand":<8} | {"Lost":<6} | {"Cost":<8}')
        print('-' * 90)

    for day in range(n_days):
        
        arrived_qty = sum([qty for arr_day, qty in pipeline_orders if arr_day == day])
        inventory += arrived_qty
        pipeline_orders = [o for o in pipeline_orders if o[0] > day] 
        
        demand = actual_demand[day]
        if inventory >= demand:
            sales = demand
            inventory -= demand
            lost_sales = 0
        else:
            sales = inventory
            lost_sales = demand - inventory
            inventory = 0
        
        total_lost_sales += lost_sales

        daily_holding = inventory * HOLDING_COST
        daily_stockout = lost_sales * STOCKOUT_COST
        total_holding_cost += daily_holding
        total_stockout_cost += daily_stockout
        
        start_idx = day + 1
        end_idx = start_idx + LEAD_TIME
        
        if end_idx <= n_days:
            expected_demand_during_lt = sum(forecast_series[start_idx : end_idx])
        else:
            available_days = n_days - start_idx
            if available_days > 0:
                known = sum(forecast_series[start_idx:])
                padded = avg_daily_demand * (LEAD_TIME - available_days)
                expected_demand_during_lt = known + padded
            else:
                expected_demand_during_lt = avg_daily_demand * LEAD_TIME

        dynamic_rop = expected_demand_during_lt + safety_stock
        
        on_order_qty = sum([qty for _, qty in pipeline_orders])
        inventory_position = inventory + on_order_qty
        
        action = ''
        if inventory_position <= dynamic_rop:
            arrival_day = day + LEAD_TIME
            pipeline_orders.append((arrival_day, order_quantity))
            action = 'ORDER'

        if verbose:
             print(f'{day:<5} | {expected_demand_during_lt:<9.1f} | {dynamic_rop:<8.1f} | {inventory_position:<8.1f} | {inventory:<8.1f} | {demand:<8.1f} | {lost_sales:<6.1f} | {(daily_holding+daily_stockout):<8.1f}')

    total_cost = total_holding_cost + total_stockout_cost
    service_level = 1 - (total_lost_sales / (sum(actual_demand)+0.01))
    
    if verbose:
        print('\n')
        print(f'Total Holding:  ${total_holding_cost:,.2f}')
        print(f'Total Stockout: ${total_stockout_cost:,.2f}')
        print(f'TOTAL COST:     ${total_cost:,.2f}')
        print(f'Service Level:  {service_level:.2%}')
        print('\n')

    return round(total_cost, 2), service_level

# %%
def get_trend_adjusted_baseline(train_df, valid_df, verbose=True):
    #Define the split
    last_date = train_df['date'].max()
    split_date = last_date - pd.Timedelta(days=90)

    #Seasonality
    history_df = train_df[train_df['date'] <= split_date].copy()
    history_df['day_of_year'] = history_df['date'].dt.dayofyear
    history_df.loc[history_df['day_of_year'] == 366, 'day_of_year'] = 365
    seasonal_profile = history_df.groupby('day_of_year')['sales'].mean()

    #Trend: Calculated on "Recent" vs "Previous Year"
    recent_vol = train_df[train_df['date'] > split_date]['sales'].mean()
    
    start_prev = split_date - pd.Timedelta(days=365)
    end_prev = last_date - pd.Timedelta(days=365)
    
    prev_vol_mask = (train_df['date'] > start_prev) & \
                    (train_df['date'] <= end_prev)
    prev_vol = train_df[prev_vol_mask]['sales'].mean()
    
    trend_factor = recent_vol / prev_vol if prev_vol > 0 else 1.0
    trend_factor = min(max(trend_factor, 0.8), 1.2) 
    
    if verbose:
        print(f'Detected Trend Factor: {trend_factor:.2f}x')

    #Forecast
    valid_df['day_of_year'] = valid_df['date'].dt.dayofyear
    valid_df.loc[valid_df['day_of_year'] == 366, 'day_of_year'] = 365
    baseline_forecast = valid_df['day_of_year'].map(seasonal_profile).values * trend_factor
    
    #In-Sample Accuracy (RMSE)
    train_check = train_df.copy()
    train_check['day_of_year'] = train_check['date'].dt.dayofyear
    train_check.loc[train_check['day_of_year'] == 366, 'day_of_year'] = 365
    
    train_forecast = train_check['day_of_year'].map(seasonal_profile) * trend_factor
    residuals = train_check['sales'] - train_forecast
    rmse = np.sqrt(np.mean(residuals[-365:] ** 2))
    
    return baseline_forecast, rmse, trend_factor

# %%
def run_analysis_for_pair(train_df, test_df, pipeline, verbose=True, plot=True):
    results = {}
    
    # EDA and Stationarity
    if plot:
        df_plot = train_df.copy()
        df_plot.set_index('date', inplace=True)

        plt.figure(figsize=(15, 6))
        plt.plot(df_plot['sales'], label='Daily Sales', linewidth=1)
        plt.title('Daily Sales History')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        print('\nPerforming Seasonal Decomposition (Period=365)...')
        decomposition = seasonal_decompose(df_plot['sales'], model='additive', period=365)
        fig = decomposition.plot()
        fig.set_size_inches(15, 10)
        plt.suptitle('Seasonal Decomposition', y=1.02)
        plt.show()

    if verbose:
        print('\nRunning Augmented Dickey-Fuller Test...')
        adf_result = adfuller(train_df['sales'])
        print(f'ADF Statistic: {adf_result[0]:.4f}')
        print(f'p-value: {adf_result[1]:.4f}')
        if plot:
            plot_adf_benchmark(adf_result[0], adf_result[4])

    #Baseline Simulation
    if verbose:
        print('\n--- Starting Baseline Simulation ---')
    
    baseline_forecast, baseline_std, trend_factor = get_trend_adjusted_baseline(train_df, test_df, verbose=verbose)
    baseline_mean_forecast = np.mean(baseline_forecast)
    
    if verbose:
        print(f'\n[Baseline Stats] Mean Forecast: {baseline_mean_forecast:.2f}, RMSE: {baseline_std:.2f}, Trend Factor: {trend_factor:.2f}')

    if plot:
        plt.figure(figsize=(15, 6))
        plt.plot(test_df['date'], test_df['sales'], label='Actual Demand', color='black', alpha=0.3)
        plt.plot(test_df['date'], baseline_forecast, label='Baseline (Trend Adjusted)', color='green', linewidth=2)
        lower_bound = baseline_forecast - (1.96 * baseline_std)
        upper_bound = baseline_forecast + (1.96 * baseline_std)
        plt.fill_between(test_df['date'], lower_bound, upper_bound, color='green', alpha=0.1, label='95% Confidence Interval')
        plt.title('Baseline: Trend Adjusted Seasonal Naive Forecast')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    actual_demand = test_df['sales'].values
    baseline_cost, baseline_sl = run_dynamic_simulation(
        forecast_series=baseline_forecast,
        actual_demand=actual_demand,
        forecast_error_std_dev=baseline_std,
        verbose=verbose
    )

    if verbose:
        print('\n')
        print(f'BASELINE TOTAL COST (Scenario 0): ${baseline_cost:,.2f}')
        print('\n')

    results['baseline_cost'] = baseline_cost
    results['baseline_sl'] = baseline_sl

    #Chronos Forecasting
    context = torch.tensor(train_df['sales'].values)
    prediction_length = len(test_df)
    
    if verbose:
        print(f'Forecasting {prediction_length} days into the future with Chronos...')

    forecast_result = pipeline.predict(context, prediction_length)
    forecast_samples = forecast_result[0].numpy()
    
    daily_means = np.mean(forecast_samples, axis=0)
    chronos_scalar_mean = np.mean(daily_means)
    daily_std_devs = np.std(forecast_samples, axis=0)
    chronos_predicted_sigma = np.mean(daily_std_devs)

    if verbose:
        print('\n')
        print('CHRONOS BOLT RESULTS')
        print(f'Mean Forecast (Daily): {chronos_scalar_mean:.2f}')
        print(f'Predicted Volatility (Sigma): {chronos_predicted_sigma:.2f}')

    if plot:
        plt.figure(figsize=(15, 6))
        plt.plot(pd.to_datetime(test_df['date']), test_df['sales'], label='Actual Demand', color='black', alpha=0.3)
        plt.plot(pd.to_datetime(test_df['date']), daily_means, label='Chronos Mean', color='blue', linewidth=2)
        lower_bound = np.quantile(forecast_samples, 0.05, axis=0)
        upper_bound = np.quantile(forecast_samples, 0.95, axis=0)
        plt.fill_between(pd.to_datetime(test_df['date']), lower_bound, upper_bound, color='blue', alpha=0.1, label='90% Confidence Interval')
        plt.title('Chronos Bolt: Probabilistic Forecast')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    #Chronos Simulation
    if verbose:
        print(f'\n[CHRONOS] Calibrating model confidence...')

    train_sales = train_df['sales'].values
    calibration_window = 60 
    train_context = torch.tensor(train_sales[:-calibration_window])

    calibration_forecast = pipeline.predict(train_context, calibration_window)[0].numpy()
    calibration_mean = np.mean(calibration_forecast, axis=0)
    calibration_actuals = train_sales[-calibration_window:]

    chronos_residuals = calibration_actuals - calibration_mean
    chronos_calibrated_sigma = np.sqrt(np.mean(chronos_residuals ** 2))

    if verbose:
        print(f'Chronos Internal Sigma (Unsafe): {chronos_predicted_sigma:.2f}')
        print(f'Chronos Calibrated Sigma (Safe): {chronos_calibrated_sigma:.2f}')
        print(f'Running Dynamic Simulation with Calibrated Sigma...')

    chronos_cost, chronos_sl = run_dynamic_simulation(
        forecast_series=daily_means, 
        actual_demand=actual_demand, 
        forecast_error_std_dev=chronos_calibrated_sigma, 
        verbose=verbose
    )

    results['chronos_cost'] = chronos_cost
    results['chronos_sl'] = chronos_sl
    
    savings = baseline_cost - chronos_cost
    pct_savings = (savings / baseline_cost) * 100 if baseline_cost > 0 else 0

    if verbose:
        print('FINAL RESULTS')
        print('\n')
        print(f'Baseline Cost: ${baseline_cost:,.2f}')
        print(f'Chronos Cost:  ${chronos_cost:,.2f}')
        print('\n')
        print(f'Net Savings:   ${savings:,.2f}')
        print(f'Reduction:     {pct_savings:.2f}%')

    return results

# %%
print('Phase 1: Data Preparation')
print(f'Loading raw data from {INPUT_FILE}...')
df_all = pd.read_csv(INPUT_FILE)
df_all.columns = df_all.columns.str.strip()
if 'store' not in df_all.columns and 'strore' in df_all.columns:
    df_all.rename(columns={'strore': 'store'}, inplace=True)
df_all['date'] = pd.to_datetime(df_all['date'])

print('Initializing Chronos Pipeline...')
pipeline = BaseChronosPipeline.from_pretrained(
    'amazon/chronos-bolt-base',
    device_map='cpu',
    torch_dtype=torch.bfloat16,
)

if BATCH_MODE:
    print('\nBATCH MODE ENABLED')
    pairs = df_all[['store', 'item']].drop_duplicates().values
    print(f'Found {len(pairs)} store-item pairs.')
    
    results_list = []
            
    start_time = time.time()
    for i, (store, item) in enumerate(pairs):
        print(f'Processing {i+1}/{len(pairs)}: Store {store}, Item {item}...', end='\r')
                
        subset = df_all[(df_all['store'] == store) & (df_all['item'] == item)].sort_values('date').copy()
        train_subset = subset[subset['date'].dt.year < 2017].copy()
        test_subset = subset[subset['date'].dt.year == 2017].copy()
                
        try:
            res = run_analysis_for_pair(train_subset, test_subset, pipeline, verbose=False, plot=False)
            res['store'] = store
            res['item'] = item
            results_list.append(res)
        except Exception as e:
            print(f'\nError processing Store {store}, Item {item}: {e}')

    print(f'\nProcessing complete in {time.time() - start_time:.2f} seconds.')
            
    if results_list:
        results_df = pd.DataFrame(results_list)
                
        avg_baseline_cost = results_df['baseline_cost'].mean()
        avg_chronos_cost = results_df['chronos_cost'].mean()
        avg_baseline_sl = results_df['baseline_sl'].mean()
        avg_chronos_sl = results_df['chronos_sl'].mean()
                
        avg_savings = avg_baseline_cost - avg_chronos_cost
        avg_savings_pct = (avg_savings / avg_baseline_cost) if avg_baseline_cost != 0 else 0.0
                
        print('\nBATCH RESULTS')
        print(f'Processed Pairs: {len(results_df)}')
        print(f'Avg Baseline Cost: ${avg_baseline_cost:,.2f}')
        print(f'Avg Chronos Cost:  ${avg_chronos_cost:,.2f}')
        print(f'Avg Savings:       ${avg_savings:,.2f}')
        print(f'Avg Savings %:     {avg_savings_pct:.2%}')
        print(f'Avg Baseline SL:   {avg_baseline_sl:.2%}')
        print(f'Avg Chronos SL:    {avg_chronos_sl:.2%}')
                
        results_df.to_csv(os.path.join(BASE_PATH, 'batch_results.csv'), index=False)
    else:
        print('No valid results collected.')

else:
    print('\n--- SINGLE ITEM MODE ---')
    subset = df_all[(df_all['store'] == 6) & (df_all['item'] == 1)].sort_values('date').copy()
    train_subset = subset[subset['date'].dt.year < 2017].copy()
    test_subset = subset[subset['date'].dt.year == 2017].copy()

    train_subset.to_csv(ONE_TRAIN, index=False)
    test_subset.to_csv(ONE_TEST, index=False)
    print(f'Files saved: Train ({len(train_subset)} rows), Test ({len(test_subset)} rows).')
            
    run_analysis_for_pair(train_subset, test_subset, pipeline, verbose=True, plot=True)


