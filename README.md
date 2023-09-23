# MLJBalancing
A package providing composite models wrapping class imbalance algorithms from [Imbalance.jl](https://github.com/JuliaAI/Imbalance.jl).

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
Here data will be passed to balancer1 then balancer2 and then the model. In general, there can be any number of balancers.

#### At this point, they behave like one single model
You can fit, predict, cross-validate and finetune it like any other MLJ model. Here is an example for finetuning
```julia
r1 = range(balanced_model, :(balancer1.k), lower=3, upper=10)
r2 = range(balanced_model, :(balancer2.min_ratios), lower=0.1, upper=0.9)

tuned_balanced_model = TunedModel(model=balanced_model,
									  tuning=Grid(goal=4),
									  resampling=CV(nfolds=4),
									  range=[r1, r2],
									  measure=cross_entropy);

mach = machine(tuned_balanced_model, X, y);
fit!(mach, verbosity=0);
fitted_params(mach).best_model
```

## 🚆🚆 Parallel Resampling with EasyEnsemble

Coming soon...