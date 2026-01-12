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
Results:  
                 Method Mechanism  Time (s)  Rec_RMSE  Std_Dev_Ratio 
0  Baseline (Mean/Mode)      MNAR      0.02    7.7175         0.6233   
1                   KNN      MNAR     95.71    7.7332         0.7150   
2                  MICE      MNAR      0.74    5.4459         0.8038   
3               SAP_RPT      MNAR    201.66    4.3507         0.8793   

Method Mechanism  Time (s)  Rec_RMSE  Std_Dev_Ratio 
0  Baseline (Mean/Mode)      MCAR      0.02    5.3893         0.8367   
1                   KNN      MCAR     77.95    4.3526         0.9015   
2                  MICE      MCAR      1.31    3.2538         0.9650   
3               SAP_RPT      MCAR    168.88    2.4460         0.9921

### Chronos:
Dataset: [Store Item Demand Forecasting Challenge (Kaggle)](https://www.kaggle.com/competitions/demand-forecasting-kernels-only/overview)   
Simulation of Chronos-2 vs XGBoost vs Trend-Adjusted Seasonal Baseline Model.    
Measured by total holding + stockout costs.  
Chronos outperforms by 18% against baseline, 13% against XGBoost.     
Service levels: Chronos - 99.75%,  XGBoost - 99.30%, Baseline - 99.17%  

### BPM:
Process mining done on [BPI Challenge 2019 dataset](https://icpmconference.org/2019/icpm-2019/contests-challenges/bpi-challenge-2019/) (Purchase-to-Pay)    
20% speedup opportunity found.  