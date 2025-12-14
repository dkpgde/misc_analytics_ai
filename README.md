### EV Batteries: 
Moved to separate repo  

### RPT-1 playground:  
Simple test on a synthetic dataset.   
Time series forecasting: R2 of 93% with ±15% noise introduced.  
Label prediction accuracy 90%.  

### RPT-1 for feature imputation:  
Experiment in using SAP RPT-1 for feature imputation, compared to mean/mode, KNN, MICE.  
RPT-1 wins out in the MNAR scenario, as it is capable of uncovering the patterns.  
Advantage in the MCAR scenario is insignificant.  
Results for MNAR:  

Method    Time (s)  Rec_RMSE  Std_Dev_Ratio (reconstructed standard deviation)  
Mean/Mode    0.02    7.7050         0.6234     
KNN         93.61    7.7381         0.7152     
MICE         1.03    5.2592         0.8145     
SAP_RPT    188.49    4.3755         0.8785  

### Chronos:
Dataset: [Store Item Demand Forecasting Challenge (Kaggle)](https://www.kaggle.com/competitions/demand-forecasting-kernels-only/overview)   
Simulation of Chronos-2 vs XGBoost vs a Trend-Adjusted Seasonal Baseline Model.    
Measured by total holding + stockout costs.  
Chronos outperforms by 18% against baseline, 13% against XGBoost.     
Service levels: Chronos - 99.91%,  XGBoost - 99.53%, Baseline - 99.46%  

### BPM:
Process mining done on [BPI Challenge 2019 dataset](https://icpmconference.org/2019/icpm-2019/contests-challenges/bpi-challenge-2019/) (Purchase-to-Pay)    
20% speedup opportunity found.  