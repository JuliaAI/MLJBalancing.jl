module MLJBalancing

using MLJBase
using MLJModelInterface
using OrderedCollections
MMI = MLJModelInterface

include("BalancedModel.jl")
export BalancedModel

end