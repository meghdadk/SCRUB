### This is a Python implementation of "Towards Unbounded Machine Unlearning"

### The main experiments
- Use small_scale_unlearning.ipynb for:
  - small-scale experiemnts
  - large forget-set size experiemnts
- Use large_scale_unlearning.ipynb for:
  - large-scale experiments
- Use small_scale_ictest.ipynb for:
  - Interclass Confusion Metric experiemnts from [pdf](https://arxiv.org/pdf/2201.06640.pdf)
- Use large_scale_ictest.ipynb for:
  - Interclass Confusion Metric experiemnts from [pdf](https://arxiv.org/pdf/2201.06640.pdf)
- Use MIA_experiments.ipynb for:
  - Membership Inference Attack based on the model's loss values

### Models choices
- For small-scale experiments:
  - allcnn --filters = 1.0
  - resnet --filters = 0.4
- For large-scale experiments:
  - allcnn --filters = 1.0
  - resnet --filters = 1.0
  
### Datasets choices
- For small-scale datasets:
  - small_cifar5
  - small_lacuna5
- For large-scale datasets:
  - cifar10
  - lacuna10

### References
We have used the code from the following two repositories:

(Selective Forgetting)[https://github.com/AdityaGolatkar/SelectiveForgetting.git]

(RepDistiller)[https://github.com/HobbitLong/RepDistiller.git]
