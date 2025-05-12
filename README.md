# CS4782-5782FinalDeliverable

### Introduction
This project attempts to re-implement "Deep Networks with Stochastic Depth" by Huang et al. The paper introduces a novel training technique where ResNet layers are randomly dropped out during training according to a survival probability schedule, improving generalization and addressing challenges like vanishing gradients.

### Chosen Result
We aim to reproduce the performance shown in Figure 3, where stochastic depth reduces test error and improves training efficiency on the CIFAR-10 dataset. The results highlights the technique’s effectiveness in enhancing ResNet performance compared to constant-depth networks.


### GitHub Contents

* `code/` contains the full PyTorch re-implementation of the stochastic depth training procedure, including:
  * `ResNet.py` - Base ResNet
  * `ResNetDrop.py` - ResNet with Stochastic Depth model
  * `train.py` - main training loop
* `data/` contains a `README.md` file with instructions on how to download and preprocess the CIFAR-10 dataset
* `results/` includes the plot and loss logs of our result
* `poster/` contains our in-class presentation poster
* `report/` has our final PDF report submitted

### Re-implementation Details 

ResNetDrop.py implements ResNet with stochastic depth. It has the same architecture as ResNet with the following differences: ResidualBlockDrop has an additional argument survival_rate. In the training part of the forward function of ResidualBlockDrop, we use survival_rate to sample a Bernoulli variable gate. If the gate is 0, the output is the residual connection only. If testing, we use the survival_rate to scale down the output of ResidualBlockDrop. 
The __init __ of ResNetDrop creates a list of 54 survival rates, determined using the linear decay rule. It splits this list into three lists, one for each block_layer. The block_layer function assigns the respective survival rates to each residual block. 

### Reproduction Steps
The project can be reproduced by running train.py - this script can be copied into a 
Google collab notebook or it can be also be run on a G2 instance. Using a G2 instance, 
our code ran for about 13 hours. 

### Results/Insights

Our code managed to reproduce extremely similar results to those in the paper (6.80% and 5.20% error for constant and stochastic in our results, respectively; 6.41% and 5.25% error for constant and stochastic in the paper, respectively). You can expect very similar results to the paper’s 110-layer ResNet model, with small variations given the randomness of the ResNet Stochastic model. You will probably also notice a shorter training time for the Stochastic model, but this varies depending on the hardware used.

### Conclusion 

Through our re-implementation of the stochastic depth method from “Deep Networks with Stochastic Depth”, we successfully reproduced the performance improvements reported in the original paper. Our results on CIFAR-10 showed that stochastic depth not only lowers test error compared to constant-depth ResNets, but also accelerates training by reducing the number of active layers during each iteration.

This project gave us hands-on insight into how stochastic regularization techniques can improve deep network training by addressing vanishing gradients and overfitting. We also gained experience translating theory from research papers into working code—carefully managing randomness, training schedules, and model architecture to match reported benchmarks.


### References
Huang, G., Sun, Y., Liu, Z., Sedra, D., Weinberger, K. Q.: Deep networks with stochastic depth (2016)

### Acknowledgements
This project was conducted as part of a class assignment. We would like to thank the original authors for their work and for making their methods publicly available. We would also like to thank the course instructors and teaching assistants for their guidance, support, and feedback throughout the project and class.
