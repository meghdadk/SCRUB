### This is a Python implementation of "Machine Unlearning Unbound"

### The main experiments
- Use [small_scale_unlearning.ipynb](https://github.com/Meghdad92/SCRUBS/blob/main/small_scale_unlearning.ipynb) for:
  - small-scale experiemnts
  - large forget-set size experiemnts
- Use [large_scale_unlearning.ipynb](https://github.com/Meghdad92/SCRUBS/blob/main/large_scale_unlearning.ipynb) for:
  - large-scale experiments
- Use [small_scale_ictest.ipynb](https://github.com/Meghdad92/SCRUBS/blob/main/small_scale_ictest.ipynb) for:
  - Interclass Confusion Metric experiemnts from [pdf](https://arxiv.org/pdf/2201.06640.pdf)
- Use [Bad_T.ipynb](https://github.com/Meghdad92/SCRUBS/blob/main/Bad_T.ipynb) for:
  - Running the experiments for [Can Bad Teaching Induce Forgetting?](https://arxiv.org/pdf/2205.08096.pdf) baseline

### Models choices
- For small-scale experiments:
  - allcnn --filters = 1.0
  - resnet --filters = 0.4
- For large-scale experiments:
  - allcnn --filters = 1.0
  - resnet --filters = 1.0
  
### Datasets choices
- For small-scale datasets:
  - small_cifar5, small_cifar6
  - small_lacuna5, small_lacuna6
- For large-scale datasets:
  - cifar10
  - lacuna10

### References
We have used the codes from the following two repositories:

(Selective Forgetting)[https://github.com/AdityaGolatkar/SelectiveForgetting.git]

(RepDistiller)[https://github.com/HobbitLong/RepDistiller.git]
