module MLJBalancing

using MLJBase
using MLJModelInterface
using OrderedCollections
MMI = MLJModelInterface

include("balanced_model.jl")
export BalancedModel

end