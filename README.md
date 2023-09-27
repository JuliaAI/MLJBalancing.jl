# MLJBalancing
A package providing composite models wrapping class imbalance algorithms from [Imbalance.jl](https://github.com/JuliaAI/Imbalance.jl) with classifiers from [MLJ](https://github.com/alan-turing-institute/MLJ.jl). 

## ⏬ Instalattion
```julia
import Pkg;
Pkg.add("MLJBalancing")
```

## 🚅 Sequential Resampling

This package allows chaining of resampling methods from Imbalance.jl with classification models from MLJ. Simply construct a `BalancedModel` object while specifying the model (classifier) and an arbitrary number of resamplers (also called *balancers* - typically oversamplers and/or under samplers).

### 📖 Example

#### Construct the resamplers and the model
```julia
SMOTENC = @load SMOTENC pkg=Imbalance verbosity=0
TomekUndersampler = @load TomekUndersampler pkg=Imbalance verbosity=0

oversampler = SMOTENC(k=5, ratios=1.0, rng=42)
undersampler = TomekUndersampler(min_ratios=0.5, rng=42)

logistic_model = LogisticClassifier()
```

#### Wrap them all in BalancedModel
```julia
balanced_model = BalancedModel(model=logistic_model, balancer1=oversampler, balancer2=undersampler)
```
Here training data will be passed to `balancer1` then `balancer2`, whose output is used to train the classifier `model`.  In prediction, the resamplers `balancer1` and `blancer2` are bypassed. 

In general, there can be any number of balancers, and the user can give the balancers arbitrary names. 

#### At this point, they behave like one single model
You can fit, predict, cross-validate and finetune it like any other MLJ model. Here is an example for finetuning
```julia
r1 = range(balanced_model, :(balancer1.k), lower=3, upper=10)
r2 = range(balanced_model, :(balancer2.min_ratios), lower=0.1, upper=0.9)

tuned_balanced_model = TunedModel(
    model=balanced_model,
    tuning=Grid(goal=4),
    resampling=CV(nfolds=4),
    range=[r1, r2],
    measure=cross_entropy
);

mach = machine(tuned_balanced_model, X, y);
fit!(mach, verbosity=0);
fitted_params(mach).best_model
```

## 🚆🚆 Parallel Resampling with Balanced Bagging

The package also offers an implementation of bagging over probabilistic classifiers where the majority class is repeatedly undersampled `T` times down to the size of the minority class. This undersampling scheme was proposed in the *EasyEnsemble* algorithm found in the paper *Exploratory Undersampling for Class-Imbalance Learning.* by *Xu-Ying Liu, Jianxin Wu, & Zhi-Hua Zhou* where an Adaboost model was used and the output scores were averaged.


#### Construct a BalancedBaggingClassifier
In this you must specify the model, and optionally specify the number of bags `T` and the random number generator `rng`. If `T` is not specified it is set as the ratio between the majority and minority counts. If `rng` isn't specified then `default_rng()` is used.

```julia
LogisticClassifier = @load LogisticClassifier pkg=MLJLinearModels verbosity=0
logistic_model = LogisticClassifier()
bagging_model = BalancedBaggingClassifier(model=logistic_model, T=10, rng=Random.Xoshiro(42))
```

#### Now it behaves like one single model
You can fit, predict, cross-validate and finetune it like any other probabilistic MLJ model where `X` must be a table input (e.g., a dataframe).
```julia
mach = machine(bagging_model, X, y)
fit!(mach)
pred = predict(mach, X)
```