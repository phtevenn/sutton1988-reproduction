# **Sutton 1988 TD Lambda Reproduction**


The goal of this project is to reproduce figures out Richard Sutton's 1988 paper, [Learning to Predict by Methods of Temporal Differences](http://incompleteideas.net/papers/sutton-88-with-erratum.pdf)


## TD ($\lambda$) Overview

In TD($\lambda$), predictions and weight vectors are updated by considering the changes in prediction at time t $P_t$ and the next iteration $P_{t+1}$. The most recent predictions are weighted more heavily, and earlier predictions are exponentially discounted by a factor of $\lambda$. The weight update rule for this algorith is as follows:
$$
\begin{align}
    \triangle w_t = \alpha (P_{t+1} - P{t})\sum^t_{k=1}\lambda^{t-k}\nabla_wP_k
\end{align}
$$

Where $\triangle w_t$ is the weight update and $\alpha$ is the learning rate hyperparameter. By utilizing the exponentially decaying lambda, we are also able to compute these updates incrementally using the eligibility trace, which is updated as follows:
$$
\begin{align}
    e_{t+1} = \nabla P_{t+1} + \lambda e_t
\end{align}
$$

This algorithm, in effect, is also the weighted combination of all n-step predictors in a n-step Markov decision process.

In this experiment, figures 3, 4, and 5 of the original paper were reproduced using the bounded random walk experiment presented in Sutton's original paper.

## Figure 3 - Repeated Representation Training

## Figure 4 - Optimal Learning Rates

## Figure 5 - Best $\lambda$ with Fixed Optimal$\alpha$
