# CS4782-5782FinalDeliverable

### Introduction
This project attempts to re-implement "Deep Networks with Stochastic Depth" by Huang et al. The paper introduces a novel training technique where ResNet layers are randomly dropped out during training according to a survival probability schedule, improving generalization and addressing challenges like vanishing gradients.

### Chosen Result
We aim to reproduce the performance shown in Figure 3, where stochastic depth reduces test error and improves training efficiency on the CIFAR-10 dataset. The results highlights the technique’s effectiveness in enhancing ResNet performance compared to constant-depth networks.


### GitHub Contents

* `README.md` Provides an overview of the project, implementation details, reproduction instructions, and results
* `code/` contains the full PyTorch re-implementation of the stochastic depth training procedure, including:
  * `ResNet.py` - Base ResNet
  * `ResNetDrop.py` - ResNet with Stochastic Depth model
  * `train.py` - main training loop
* `data/` contains a `README.md` file with instructions on how to download and preprocess the CIFAR-10 dataset
* `results/` includes the plot and loss logs of our result
* `poster/` contains our in-class presentation poster
* `report/` has our final PDF report submitted
* `LICENSE` Specifies the license under which your code is released (MIT License).
* `.gitignore` Specifies files or directories that should be ignored by Git.


### Re-implementation Details 
ResNet.py implements the standard ResNet baseline with fixed-depth residual block. The architecture follows the design proposed in the paper. The model consists of an initial convolutional layer, three main stages that consist of Residual Blocks, and after the final residual layer, a global average pooling layer is applied.

ResNetDrop.py implements ResNet with stochastic depth. It has the same architecture as ResNet with the following differences: ResidualBlockDrop has an additional argument survival_rate. In the training part of the forward function of ResidualBlockDrop, we use survival_rate to sample a Bernoulli variable gate. If the gate is 0, the output is the residual connection only. If testing, we use the survival_rate to scale down the output of ResidualBlockDrop. 
The __init __ of ResNetDrop creates a list of 54 survival rates, determined using the linear decay rule. It splits this list into three lists, one for each block_layer. The block_layer function assigns the respective survival rates to each residual block. 

train.py follows closely the training methods used by the authors of the paper for the 110-layer models. We trained two models, a baseline ResNet and a ResNet with stochastic depth, using SGD with Nesterov momentum, weight decay, and a MultiStepLR scheduler with milestones at 250 and 375 epochs, in order to have a fair comparison and stay true to paper. CIFAR-10 data is preprocessed with standard augmentation techniques, like random crop, horizontal flip, normalization, etc. Training, validation, and test splits are handled manually and allow for performance tracking.

### Reproduction Steps
In order to reimplement our code and reproduce the results:
* **Clone the GitHub repository**
```bash
git clone https://github.com/rachaelclose/CS4782-5782FinalDeliverable.git
cd CS4782-5782FinalDeliverable
```
* **Install the required libraries**
  * torch
  * torchvision
  * matplotlib
```bash
pip install torch torchvision matplotlib
```
* **Ensure that the following file structure exists:**
<pre>├── train.py 
├── ResNet.py
├── ResNetDrop.py 
└── checkpoints/ </pre>

* **Train the models**
```bash
python train.py
```

We run our code using the G2 cluster at Cornell. Using a G2 instance,  our code ran for about 13 hours. 

### Results/Insights
![image] (results/figure3.png)

Our code managed to reproduce extremely similar results to those in the paper (6.80% and 5.20% error for constant and stochastic in our results, respectively; 6.41% and 5.25% error for constant and stochastic in the paper, respectively). You can expect very similar results to the paper’s 110-layer ResNet model, with small variations given the randomness of the ResNet Stochastic model. You will probably also notice a shorter training time for the Stochastic model, but this varies depending on the hardware used.

### Conclusion 

Through our re-implementation of the stochastic depth method from “Deep Networks with Stochastic Depth”, we successfully reproduced the performance improvements reported in the original paper. Our results on CIFAR-10 showed that stochastic depth not only lowers test error compared to constant-depth ResNets, but also accelerates training by reducing the number of active layers during each iteration.

This project gave us hands-on insight into how stochastic regularization techniques can improve deep network training by addressing vanishing gradients and overfitting. We also gained experience translating theory from research papers into working code—carefully managing randomness, training schedules, and model architecture to match reported benchmarks.


### References
Huang, G., Sun, Y., Liu, Z., Sedra, D., Weinberger, K. Q.: Deep networks with stochastic depth (2016)

### Acknowledgements
This project was conducted as part of a class assignment. We would like to thank the original authors for their work and for making their methods publicly available. We would also like to thank the course instructors and teaching assistants for their guidance, support, and feedback throughout the project and class.
