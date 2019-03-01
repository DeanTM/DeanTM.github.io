---
layout: post
date: 2019-02-16
title: AB Testing and Multi-Armed Bandits
permalink: /:title/
description: a look at standard and Bayesian AB testing methods compared with multi armed bandits
---

#### Another incomplete post!

Wikipedia describes A/B testing as
> a way to compare two versions of a single variable, typically by testing a subject's response to variant A against variant B, and determining which of the two variants is more effective.

In effect, it's the process of randomly splitting your experiment subjects into two groups:  

* group A: the control group, exposed usually to the old version of a product (variant A)
* group B: the group exposed to the new alternative (variant B)

A comparison is then done of the respones of each group to see if a property, usually the mean, is better for one group or another. A statistical test can, and should, also be done to see if the difference is due to chance.

A/B testing can be extended or adapted, depending on the intended goals, into the Bayesian framework or into the domain of bandit optimization.

Here, I'll create a dummy example to compare A/B testing with these adaptations.

The code for the bodies of the functions are moved to the end, so as to not interrupt the reading.

## Our Example

I racked my brain to come up with a more interesting example, but failed. I needed an example that:  

* allows for optimising a non-trivial property (for Bayesian A/B testing)
* exemplifies a usecase for the adaptive bandit optimisation  (for bandit optimisation)

We manage an online store, let's say for example [ThisIsWhyImBroke](https://www.thisiswhyimbroke.com/), for a customers who pay us (I make no assumption that this is how ThisIsWhyImBroke is financed): 

* a fixed amount per click through $U_{CTR}$
* a fixed (larger) amount for each sale made via click-throughs from our site i.e. every time there is a _conversion_ $U_{conv}$ (it may be worth considering the complexities of extending this to a _percentage_ of sale price)

As new web-surfers arrive at our page, we present them with one of two versions of the site, the old version $A$ and the new version $B$. Let's say we randomly assign customers to each version with equal probability.

Unbeknownst to us, our customers come in 3 flavours, of equal proportions, each with their own purchasing and click-through properties for each version of the site:

1. The group which loves the new site, and has their CTR (click-through rate) and conversion rates shifted up by variant $B$.  
2. The conservative group who wish that CSS never came along. They like plain, simple, $A$. However, it turns out they're more likely to buy products if they're presented with variant $B$.
3. The browsing group who are enjoying the ease of use of the new variant $B$, but actually won't buy anything more frequently than before.

...and we have a mixture model! We'll use this to generate our data.

We _could_ use the Beta distribution to describe the distributions of CTR and buying probabilities for each group. This means we need 4 alphas, and 4 betas, for each group: 2 (for each variant) times 2 (for CTR, and buying probability). A total of 24 parameters!

For simplicity though, I chose a fixed value for each CTR and conversion rates. Our generative model for our experiment results then works as follows:  
While we're running the test (until we've seen $N$ potential customers), customers come sampled with equal probability from each group. They are presented, with equal probability, a variant of the site. They then sample from a Bernoulli (with a parameter depending on the group and the site variant) whether they click-through, and whether they buy. We will generate data for buying _even if they don't click through_, but in analysis the buying probability defaults to 0 if they don't click through.


```python
print_params()
```

    group: 1
      variant: A
    	 ctr :  0.3
    	 conv :  0.1
      variant: B
    	 ctr :  0.6
    	 conv :  0.2
    group: 2
      variant: A
    	 ctr :  0.4
    	 conv :  0.2
      variant: B
    	 ctr :  0.1
    	 conv :  0.4
    group: 3
      variant: A
    	 ctr :  0.3
    	 conv :  0.15
      variant: B
    	 ctr :  0.7
    	 conv :  0.15



```python
# our data
N = 2000
groups, variants, click_throughs, conversions = run_experiment(N)
print_data(variants, click_throughs, conversions)
```

    Variant A:
    Number of click-throughs:  321 out of 987
    Number of potential purchases:  142 out of 987
    Number of actual purchases:  58 out of 321
    
    Variant B:
    Number of click-throughs:  482 out of 1013
    Number of potential purchases:  264 out of 1013
    Number of actual purchases:  96 out of 482
    


We see that variant $B$ has higher click through rates and nearly _double_ the number of actual purchases!

Because we have a lot at stake, we should run statistical tests to see how confident we can be with this result.

---

## Standard A/B Test

First, we may want to look at the actual utility of these two outcomes. If we simply want to maximise our income, we can do so by maximising the expected total utility, where total utility is:

$$U_{total} = U_{CTR} \cdot \text{CTR} + U_{buy} \cdot \text{conversion rate}$$

I call this the _linear utility_ because it is a linear function of the click-through and conversion rates.

For each variant we compute this value, then we can use the variant yielding the highest value. We can try and infer confidence intervals from the sample size, or we could use [bootstrapping](https://en.wikipedia.org/wiki/Bootstrapping_%28statistics%29) or [jackknifing](https://en.wikipedia.org/wiki/Jackknife_resampling) to infer the variance of our estimate.

Since these are Bernoulli variables (did/did not click/convert), which are effectively just Binomial variables with one trial, an appropriate test to assess the significance of our result would be the [chi-squared test](https://en.wikipedia.org/wiki/Chi-squared_test).

For more information on computing confidence intervals, see [here](https://arxiv.org/pdf/1501.07768.pdf).


```python
print_linear_utility(
    u_ctr, u_conv, variants,
    click_throughs, conversions
)
```

    Variant A
    Total utility:   19.044219716122676
    
    Variant B
    Total utility:   21.344455686039986
    


I we want to look at the confidence intervals using standard tests, we run into a [problem](https://conversionxl.com/blog/testing-statistics-mistakes/#8): our data is not normally distributed. The _value_ of each customer $c$ which comes to our site can be seen rather untidily as:
$$value(c) = \begin{cases}
    & U_{ctr} + \left(\begin{cases}
        & U_{buy} \text{ if customer buys something}\\
        & 0 \text{ otherwise}\\
    \end{cases}\right) \text{ if customer clicks}\\
    & 0 \text{ otherwise }
\end{cases}$$

A significant portion of our customers have a value of $0$, none have a negative value, and the distribution is discrete.  

Hence, I would suggest bootstrapping. But let's move on.

---

## Bayesian A/B Test

The standard frequentist A/B test has a few shortcomings which the Bayesian methodology can overcome:

* The Bayesian methodology allows one to use **prior estimates** of the CTR and buying probability of the old variant $A$, allowing us to funnel newcomers more frequently to the new variant. The Bayesian methodology does not require near equal sample sizes. However, this comes at a cost: the dataset becomes _incomplete_ without the same prior information, and other people looking at the data will not be able to obtain the same result.  
* The Bayesian methodology **yields posterior probabilities** which we can use to gain confidence in our results without having to perform obscure (imho) tests.  
* The Bayesian methodology allows for **complicated cost/utility functions**. Where before we wished to maximise both CTR and purchases, and require more complicated tests and limitations on the results to do so, in the Bayesian framework we can use our posterior distributions to evaluate complex utility functions. For example, a competitor may boast a high CTR but makes no claims about conversion rate. To outdo them by having high conversion rate, but without compromising CTR much, we may with to maximise this (thumbsucked) value:
$$U_{total} = \begin{cases}
    &\text{CTR }\cdot U_{CTR} + \left(\text{conversion rate}\right)^2\cdot U_{conv} \quad\text{ if } \text{CTR } \leq 0.25\\
    &\log\left(\text{CTR }\cdot U_{CTR}\right) + \text{conversion rate}\cdot U_{conv} \quad\text{ if } \text{CTR } \gt 0.25\\
\end{cases}$$
where the first case accounts for erratic nature of the Law of Large Numbers with small sample sizes (conversion rate is poorly estimated if CTR is low, so we wouldn't want too much weight on it, so we reduce it by squaring it).

I'll only demonstrate the second and third cases here, because I'm ~~lazy~~ avoiding the math around choosing sample sizes and thumbsucking some confidence values.

### Available Posterior

Let's take a look at what happens when we fit a Bayesian model (using PyMC3) to evaluate a posterior for CTR and conversion rates for each variant. Note that these posteriors are in some sense approximations of the three groups' preferences combined, since we do not know that there are three "true" groups.

So let's create and look at our model:


```python
model = fit_model(variants, click_throughs, conversions)
print_model(model)
```




![model dag](/assets/plots/ab_testing/output_10_0.svg)



With this model, we can sample potential conversion rates and CTRs, giving us a distribution of outcomes. We can use these samples to compute more complex functions. We can also look at the distributions of the linear utility from before.


```python
# sample a trace
trace = get_posterior(model)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [convrate_B, convrate_A, ctr_B, ctr_A]
    Sampling 2 chains: 100%|██████████| 21000/21000 [00:26<00:00, 778.57draws/s]



```python
plot_parameter_trace(trace)
```


![rate posteriors](/assets/plots/ab_testing/output_13_0.png)


The standard approach would have just yielded the means, as ticked along the x axis, but now we have full _distributional information_ and it wasn't that hard! We can see that while variant $B$ _significantly_ improves the CTR, we see a large overlap for the covnersion rate - and variant $B$ is acutally _worse_ here!

The "convrate" variables we are observing are the posteriors for the _true_ conversion rate, which includes the fact that it's 0 if a potential customer doesn't even click.

Now let's look at the distribution of our linear utility.


```python
plot_linear_utility(trace)
```


![linear utility posteriors](/assets/plots/ab_testing/output_15_0.png)


We can readily see how (not) significant our improvement is. In fact, we can even subtract the samples for variant $B$ from those for $A$ and directly see how probable it is that matters will _worsen_ with the second variant:


```python
plot_linear_utility_difference(trace)
```


![differences in utility](/assets/plots/ab_testing/output_17_0.png)


There seems to be a fair probability that the outcome we observed can arise with variant $B$ actually being _worse_ (in terms of linear utility).

### Complicated Cost/Utility Functions

It's also possible to use a different utility function. Let's see how it rates our two variants against each other with the thumbsuscked utility from before; we can replace the threshold $0.25$ with a parameter $\theta$ to see how it changes for different theta.


```python
plot_ridiculous_utility(trace, [0.25, 0.5])
```


![ridiculous utility posteriors](/assets/plots/ab_testing/output_20_0.png)


Here variant $B$ wins again, for both thresholds. But notice for the threshold $0.5$ the second mode to the right: this kind of thing warns us of potentially odd outcomes (in this case, in our favour, if we continue with variant $B$).

---

## Bandit Optimisation

Bandit optimisation puts a whole new spin on the affair: while we are running the experiment, we are losing utility (money) every time we present a newcomer with the worse variant. Bandit optimisation allows one an estimate of the best variant to site visitors such that: 

* we still learn what the true best is
* we reduce _regret_: the difference between the optimal utility and the achieved utility

In this framework, we do not need to wait until the end of the test before capitalising on results (we do not need to _set_ an end to the test). The _downside_ of all this is that the data we collect is biased as all hell and cannot really be used for other analyses. 


```python
# TODO
```

## Functions Appendix

Here's the code used to generate the outputs in the cells above.


```python
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
from scipy import stats
import pymc3 as pm
import theano.tensor as tt
import seaborn as sns
```


```python
preferences = {
    1: {
        'A': {
            'ctr': 0.3,
            'conv': 0.1
        },
        'B': {
            'ctr': 0.6,
            'conv': 0.2
        }
    },
    2: {
        'A': {
            'ctr': 0.4,
            'conv': 0.2
        },
        'B': {
            'ctr': 0.1,
            'conv': 0.4
        }
    },
    3: {
        'A': {
            'ctr': 0.3,
            'conv': 0.15
        },
        'B': {
            'ctr': 0.7,
            'conv': 0.15
        }
    }
}
u_ctr = 3
u_conv = 100
```


```python
def print_params():
    for group in preferences:
        print(f'group: {group}')
        for variant in preferences[group]:
            print(f'  variant: {variant}')
            for k, v in preferences[group][variant].items():
                print('\t', k, ': ', v)
```


```python
def run_experiment(N=1000):
    groups = np.empty(shape=N, dtype=int)
    variants = np.empty(shape=N, dtype=str)
    click_throughs = np.empty(shape=N, dtype=int)
    conversions = np.empty(shape=N, dtype=int)
    for c in range(N):
        group = np.random.randint(1, 4)
        variant = np.random.choice(['A', 'B'])
        ctr = preferences[group][variant]['ctr']
        buy_prob = preferences[group][variant]['conv']
        click_through = stats.binom.rvs(n=1, p=ctr)
        conversion = stats.binom.rvs(n=1, p=buy_prob)
        
        groups[c] = group
        variants[c] = variant
        click_throughs[c] = click_through
        conversions[c] = conversion
    return groups, variants, click_throughs, conversions
```


```python
def print_data(variants, click_throughs, conversions):
    for variant in ['A', 'B']:
        exposed = (variants == variant).sum()
        clicks = click_throughs[variants == variant].sum()
        converts = conversions[variants == variant].sum()
        true_converts = conversions[(variants == variant) & (click_throughs == 1)].sum()
        print(f'Variant {variant}:')
        print(f'Number of click-throughs:  {clicks} out of {exposed}')
        print(f'Number of potential purchases:  {converts} out of {exposed}')
        print(f'Number of actual purchases:  {true_converts} out of {clicks}')
        print()
```


```python
def get_linear_utility(
    u_ctr, u_conv, variant,
    variants, click_throughs, conversions
):
    ctr = click_throughs[variants == variant].mean()
    conv = conversions[(variants == variant) & (click_throughs==1)].mean()
    return u_ctr * ctr + u_conv * conv

def print_linear_utility(
    u_ctr, u_conv,
    variants, click_throughs, conversions
):
    for variant in ['A', 'B']:
        print(f'Variant {variant}')
        u_total = get_linear_utility(
            u_ctr, u_conv, variant,
            variants, click_throughs, conversions
        )
        print('Total utility:  ', u_total)
        print()
```


```python
def fit_model(
    variants, click_throughs, conversions
):
    # it's important we don't use the "unknown"
    # conversion values, as this would be unobserved
    # in reality
    true_conversions = conversions.copy()
    true_conversions[click_throughs == 0] = 0
    with pm.Model() as model:
        mask_A = variants=='A'
        mask_B = variants=='B'
        # this can probably be improved with a shape=2 param
        ctr_A = pm.Beta('ctr_A', alpha=1, beta=1)
        ctr_B = pm.Beta('ctr_B', alpha=1, beta=1)
        convrate_A = pm.Beta('convrate_A', alpha=1, beta=1)
        convrate_B = pm.Beta('convrate_B', alpha=1, beta=1)
        click_A = pm.Bernoulli(
            'click_A', p=ctr_A,
            observed=click_throughs[mask_A]
        )
        click_B = pm.Bernoulli(
            'click_B', p=ctr_B,
            observed=click_throughs[mask_B]
        )
        
        convert_A = pm.Bernoulli(
            'convert_A', p=convrate_A * click_A,
            observed=true_conversions[mask_A]
        )
        convert_B = pm.Bernoulli(
            'convert_B', p=convrate_B * click_B,
            observed=true_conversions[mask_B]
        )
    return model

def print_model(model):
    # has python-graphviz dependency
    return pm.model_to_graphviz(model)

def get_posterior(model, trace_size=10000, burnin=10000):
    with model:
        trace = pm.sample(trace_size)
        burned_trace = trace[burnin:]
    return trace
```


```python
def plot_parameter_trace(trace):
    figsize(12.5, 5)
    colours = 'bmgr'
    x_ticks = []
    for i, variable in enumerate(['ctr_A', 'ctr_B', 'convrate_A', 'convrate_B']):
        sns.distplot(trace[variable], label=variable, color=colours[i])
        mean = trace[variable].mean()
        plt.vlines(
            mean,
            0, 10,
            linestyles='--',
            color=colours[i],
    #         label='_'.join([variable, 'mean'])
        )
        x_ticks.append(mean)
    plt.xticks(x_ticks)
    plt.legend(title='parameter')
    plt.grid(alpha=0.7, ls=':')
    plt.title("Posteriors of CTRs and Conversion Rates")
    plt.xlabel('rate')
```


```python
def linear_util_vectorized(u_ctr, ctr, u_conv, conv):
     return u_ctr * ctr + u_conv * conv

def plot_linear_utility(trace):
    figsize(12.5, 5)
    colours = 'bmgr'
    x_ticks = []
    for i, (variant, CTR_conv) in enumerate(zip(['A', 'B'], [('ctr_A', 'convrate_A'), ('ctr_B', 'convrate_B')])):
        linear_utils = linear_util_vectorized(
            u_ctr, trace[CTR_conv[0]], u_conv, trace[CTR_conv[1]]
        )
        sns.distplot(
            linear_utils,
            color=colours[i],
            label=variant
        )
        mean = np.mean(linear_utils)
        plt.vlines(
            mean,
            0, 0.1,
            linestyles='--',
            color=colours[i],
        )
        x_ticks.append(mean)
    plt.xticks(x_ticks)
    plt.grid(alpha=0.7, ls=':')
    plt.title("Posteriors of Linear Utility")
    plt.xlabel("Linear Utility,\n average value of each page visitor")
    plt.legend(title='variant')

def plot_linear_utility_difference(trace):
    figsize(12.5, 5)
    linear_utils_A = linear_utils = linear_util_vectorized(
            u_ctr, trace['ctr_A'], u_conv, trace['convrate_A']
        )
    linear_utils_B = linear_utils = linear_util_vectorized(
            u_ctr, trace['ctr_B'], u_conv, trace['convrate_B']
        )
    a_minus_b = linear_utils_A - linear_utils_B
    ax = sns.kdeplot(a_minus_b, color='b')
    n, bins, patches = ax.hist(
        a_minus_b,
        color='b',
        bins=40,
        alpha=0.6,
        density=True
    #         label=variant
    )
    for i in range(len(patches)):
        if bins[i] >= 0:
            patches[i].set_facecolor('k')

    from matplotlib.patches import Patch
    handles = [
        Patch(color='k', label='loss', alpha=0.6),
        Patch(color='b', label='win', alpha=0.6)
    ]
    plt.legend(handles=handles)

    mean = np.mean(a_minus_b)
    plt.vlines(
        mean,
        0, 0.1,
        linestyles='--',
        color='b',
    )
    plt.grid(alpha=0.7, ls=':')
    plt.title("Posteriors of A minus B")
    plt.xlabel("$utility_A - utility_B$,\n average value lost for each page visitor")
```


```python
def ridiculous_utility(u_ctr, ctr, u_conv, conv, threshold=0.25):
    utility = np.zeros_like(ctr)
    low_ctr_mask = ctr <= threshold
    case_1 = u_ctr * ctr[low_ctr_mask] + (conv[low_ctr_mask]**2) * u_conv
    case_2 = np.log(u_ctr * ctr[~low_ctr_mask]) + conv[~low_ctr_mask] * u_conv
    utility[low_ctr_mask] = case_1
    utility[~low_ctr_mask] = case_2
    return utility

def plot_ridiculous_utility(trace, thetas=[0.25]):
    figsize(12.5, 5)
    colours = 'bmgr'
    x_ticks = []
    i = 0
    for theta in thetas:
        for variant, CTR_conv in zip(
            ['A', 'B'],
            [('ctr_A', 'convrate_A'), ('ctr_B', 'convrate_B')]
        ):
            utils = ridiculous_utility(
                u_ctr, trace[CTR_conv[0]], u_conv, trace[CTR_conv[1]],
                theta
            )
            sns.distplot(
                utils,
                color=colours[i],
                label=variant + f' | {theta}'
            )
            mean = np.mean(utils)
            plt.vlines(
                mean,
                0, 0.1,
                linestyles='--',
                color=colours[i],
        #         label='_'.join([variable, 'mean'])
            )
            x_ticks.append(mean)
            i += 1
    plt.xticks(x_ticks)
    plt.grid(alpha=0.7, ls=':')
    plt.title("Posteriors of Ridiculous Utility")
    plt.xlabel("Ridiculous Utility,\n average value of each page visitor")
    plt.legend(title='variant | $\\theta$')
```
