module MLJBalancing

using MLJBase
using MLJModelInterface
using MLUtils
using OrderedCollections
using Random
using Random: AbstractRNG, Xoshiro, rand
using StatsBase: sample

MMI = MLJModelInterface

include("BalancedBagging.jl")
export BalancedBaggingClassifier

end
