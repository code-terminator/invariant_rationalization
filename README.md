# Invariant Rationalization on Beer Review
This branch contains the Tensorflow implementation of the beer review used in our paper.

The original [beer review dataset](http://snap.stanford.edu/data/web-BeerAdvocate.html) has been removed by the dataset’s original author, at the request of the data owner, BeerAdvocate.  To respect the wishes and legal rights of the data owner, we do not include it in our repo.  In order to run the code on beer review, we ask you to first obtain the dataset from the original authors who released the dataset. We will then be happy to provide our data and environment partitions to whoever is granted rights to the data.

Once you have access to the dataset and have prepared it in the desired format, simply run the `beer_demo.ipynb` to generate results like the one shown in the saved log in the notebook.
```console
----> [Final result] dev inv acc: 0.8225, dev enb acc: 0.8110, The best annotation result: sparsity: 0.1561, precision: 0.4962, recall: 0.4999, f1: 0.4981.
```
It is worth mentioning that the result would vary even when you use a fixed seed. This is likely due to the non-deterministic problem of Tensorflow, CuDNN, and the low-level implementation of the `cudnngru` module (at least for the version we used and tested on).  

In our experiment, we found under the same parameter setting, the results are sensitive to different initializations, because the proposed algorithm can easily be trapped in poor local optima, which is possibly due to the unstable convergence behavior of the adversarial training, as well as the straight-through gradient approximation error.  

To eliminate such failure cases, for a given parameter setting, it is important to run the model learning procedure multiple times with different initializations and select the best model based on the best dev accuracy of the environment agnostic predictor.  Specifically, in the demo, the F1 scores for five different runs are 0.1220, 0.2643, 0.4981, 0.1998, and 0.4360.  And the output performance is 0.4981 since it has the highest dev accuracy of 0.8225.  We welcome interested users to try other optimization techniques recently found to be useful in stabilizing the convergence behavior, such as Gumbel softmax with temperature annealing, and feeding the logits of the environment-agnostic predictor as another input to the environment-dependent predictor.  Due to the different initializations, you may see different results from run to run, but the general trend remains the same.

Important hyperparameters has been considered in our experiments:

1. The `diff_lambda` that trade-off the invariant loss.  A reasonable range would be `{0.5, 1., 2., 5., 10., 20., 50., 100.}`.

2. The number of consecutive stochastic gradient descent steps of each player within one iteration.  Our optimization algorithm updates the generator for N1 steps, then the environment-agnostic predictor for N2 steps, and finally the environment-aware predictor for N3 steps. Suggested options for (N1, N2, N3) include (1, 3, 3) and (1, 1, 1).   

3. The number of different initializations used for each parameter setting.   

4. Function form of h in equation (8). Valid choices include h(t)=t and h(t)=ReLU(t).

We also provide a simple wrapper `beer_model_selection.py` to help hyperparameter tuning.  
Please be reminded that the model selection is based on the best dev accuracy, and should not be based on any test set performance metrics.  
