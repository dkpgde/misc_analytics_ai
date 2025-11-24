### EV Batteries: 
This is a small modification of the original code from:

**EVBattery: A Large-Scale Electric Vehicle Dataset for Battery Health and Capacity Estimation**
> He et al. (2022). **arXiv:2201.12358**. [Link](https://arxiv.org/abs/2201.12358)  

Attempted to replace the GRU with Transformers. GRU is better.   
Transformers are 50% slower in training, 15% slower in inference. 
Transformers AUROC 0.76 ± 0.1, GRU AUROC 0.78 ± 0.07.   
Used Huber loss in inference unlike the paper, in order to minimize the effect of outliers on deciding whether the car is anomalous.  

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