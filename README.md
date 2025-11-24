EV Batteries:  
Recreation of the DyAD from arXiv:2201.12358.  
Attempted to replace the GRU with Transformers. GRU is better.   
Transformers are 50% slower in training, 15% slower in inference. 
Transformers AUROC 0.76, GRU AUROC 0.78.   
Used Huber loss in inference unlike the paper, in order to minimize the effect of outliers on deciidng whether the car is anomalous.  
