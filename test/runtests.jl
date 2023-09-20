using Test
using MLJBalancing
using Imbalance
import MLJBase
using MLJModelInterface
MMI = MLJModelInterface
using Random
using DataFrames
using Tables
using CategoricalArrays
using MLJ

include("utils.jl")
include("BalancedBagging.jl")
