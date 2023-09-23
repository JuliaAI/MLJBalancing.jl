module MLJBalancing

using MLJBase
using MLJModelInterface
using MLUtils
using OrderedCollections
using Random
using Random: AbstractRNG, Xoshiro, rand
using StatsBase: sample

MMI = MLJModelInterface

include("balanced_bagging.jl")
export BalancedBaggingClassifier

end
