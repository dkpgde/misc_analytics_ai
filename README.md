### EV Batteries: 
This is a modification of the original code from:

**EVBattery: A Large-Scale Electric Vehicle Dataset for Battery Health and Capacity Estimation**
> He et al. (2022). **arXiv:2201.12358**. [Link](https://arxiv.org/abs/2201.12358)  

This version only uses Dataset 3 due to limited compute (Dataset 3 is the smallest provided).

Added a more robust training approach to all models, using high epoch numbers and early stopping.
The transformers didn't add as much of a performance boost as I expected, and the higher computational cost is not justified.  
On the other hand, replacing the MSE with Huber loss during inference did provide a small performance boost. This loss is less likely to classify a car as faulty based on a couple outliers, which are normal during charging.  

**Baseline (original GRU DyAD)**:
AUROC: 0.698 ± 0.128
1. **Transformers instead of GRU**:  
AUROC: 0.8250 ± 0.0687
AUPRC: 0.9189 ± 0.0356  
Train time (seconds): 355.5470 ± 309.7916  
Test time (seconds): 14.1624 ± 0.7317  
2. **Original GRU with Huber loss during inference**:  
AUROC: 0.8104 ± 0.0863  
AUPRC: 0.9066 ± 0.0576  
Train time (seconds): 253.8557 ± 96.9742  
Test time (seconds): 18.1198 ± 0.9427  
3. **Original GRU with MSE during inference (improved by more robust training)**:  
AUROC: 0.7810 ± 0.0699 
AUPRC: 0.9017 ± 0.0389
Train time (seconds): 226.9936 ± 143.9765  
Test time (seconds): 18.0903 ± 0.9725  

Next steps would be to run the same on Dataset 1 and 1 and 3 combined, then to use Mamba SSM. But I don't have the compute and my curiosity is satisfied for now.   

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