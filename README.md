### EV Batteries: 
This is a modification of the original code from:

**EVBattery: A Large-Scale Electric Vehicle Dataset for Battery Health and Capacity Estimation**
> He et al. (2022). **arXiv:2201.12358**. [Link](https://arxiv.org/abs/2201.12358)  

Added a more robust training approach to all models, using high epoch numbers and early stopping.
0. Original GRU DyAD:
AUROC: 0.698 ± 0.128
1. Transformers instead of GRU:
AUROC: 0.8185 ± 0.0762
Train time (seconds): 596.4706 ± 381.4815
Test time (seconds): 20.3622 ± 1.5165
2. Original GRU with Huber loss during inference:
AUROC: 0.7973 ± 0.0604
Train time (seconds): 344.2512 ± 44.7582
Test time (seconds): 17.2531 ± 0.7793
3. Original GRU with MSE during inference (improved by more robust training):
AUROC: 0.7414 ± 0.1124
Train time (seconds): 270.0323 ± 108.7613
Test time (seconds): 17.7491 ± 2.2887

### RPT-1:
Simple test on a synthetic dataset.   
Time series forecasting: R2 of 93% with ±15% noise introduced.  
Label prediction accuracy 90%.  

### Chronos:
Dataset: [Store Item Demand Forecasting Challenge](https://www.kaggle.com/competitions/demand-forecasting-kernels-only/overview) (Kaggle)  
Simulation of Chronos Bolt vs a Trend-Adjusted Seasonal Baseline Model.    
Chronos outperforms by 50% across the entire dataset.     
Wins by more on items with higher volatility, loses on items with lower volatility.    
Bolt used despite not being recommended for such long horizons because of it's speed.  
Bolt is capable of processing the entire dataset on-device in about 25 minutes.  

### BPM:
Process mining done on [BPI Challenge 2019 dataset](https://icpmconference.org/2019/icpm-2019/contests-challenges/bpi-challenge-2019/) (Purchase-to-Pay)    
20% speedup opportunity found.  