# rateRNN_Mujoco
![](doc/HalfCheetah.gif)
![](doc/Walker2d.gif)
- Tiny network
- Faster than most existing algorithms
- Work well on making periodic movement such as "Inverted Pendulum", "Swimmer", "Hopper", "Ant", "Half Cheetah" and "Walker2D". Work bad on controlling problems such as "Reacher", "Inverted Double Pendulum" and "Humanoid".
## Neuron model
![](doc/rateRNN.jpg)
$$ \tau \dot{\bm{x}} = -\bm{x}+W\bm{r}+\mathrm{Input} \\ 
\bm{r} = \frac{1}{1+e^{-\bm{x}}}$$

## Algorithm
for $g=1,2,...,G$ generations:  
&emsp; if $g==1$:  
&emsp;&emsp; $P^1 = \phi(N_parent)$ {initialize random weights in parants}  
&emsp; $C^g = P^g + noise$  
&emsp; $C^g_{N_children+1,...,N_children+N_parent} = P^g$  
&emsp;Evaluate $F_i = F(C^g_i)$  
&emsp;Sort $C^g_i$ with descending order by $F_i$  
&emsp;Set $P^{g+1}=C^g_{1,2,...,N_parent}$  

- Initialization:
Set a new environment $env$  
for $t=1,2,...,initiation\_length$:  
&emsp; if $t==1$:  
&emsp;&emsp; Reset the environment $obs_1 = Reset(env)$  
&emsp; Choose a random action $act_t$  
&emsp; $obs_{t+1}=step(env,act_t)$

$obs\_mean = mean(obs)$  
$obs\_std = std(obs)$  

- Evaluation function $F$:
Set a new environment $env$  
Set new random network state $x$  
$total/_reward=0
for $step=1,2,...,warmup\_steps$:  
&emsp; $\tau \dot{\bm{x}} = -\bm{x}+W\bm{r}+\mathrm{baseline\_input}$  
&emsp; $\bm{r} = \frac{1}{1+e^{-\bm{x}}}$  
while not $done$:  
&emsp; if $t==1$:  
&emsp;&emsp; Reset the environment $obs_1 = Reset(env)$  
&emsp; for $step=1,2,...,warmup\_steps$:  
&emsp;&emsp; $\tau \dot{\bm{x}} = -\bm{x}+W\bm{r}+\mathrm{baseline\_input} + \frac{obs_t-obs\_mean}{obs\_std}$  
&emsp;&emsp; $\bm{r} = \frac{1}{1+e^{-\bm{x}}}$  
&emsp; $act_t=\psi(r)$  
&emsp; $obs_{t+1}, reward_t = step(env,action)$  
&emsp; $total\_reward = total\_reward+reward_t$  
return $total\_reward$


## parameters
- Parameters can be set in `get_hyperparams.py`  

| Name | Value |
| ------ | ------ |
| $\tau$ | 0.01 |
| baseline_input | 0.2 |
| $N_{neuron}$ | 30 |
| $N_{parent}$ | 16 |
| $N_{children}$ | 256 |
| initiation_length | 100000 |
| weights_clip | 5 |
| warmup_steps | 100 |

