#=
This is how the struct and the constructor for the model balancer
would look if it were to support only the probabilistic model type:

struct BalancedModel <:ProbabilisticNetworkComposite
    balancer                     # oversampler or undersampler
    model::Probabilistic         # get rid of abstract types
end

BalancedModel(;model=nothing, balancer=nothing) = BalancedModel(model, balancer) 
BalancedModel(model; kwargs...) = BalancedModel(; model, kwargs...)    

In the following, we use macros to automate code generation of these for all model
types
=#

### 1. Define model structs

# Supported Model Types
const SUPPORTED_MODEL_TYPES = (:Probabilistic, :Deterministic, :Interval)

# A dictionary to convert e.g., from Probabilistic to BalancedModelProbabilistic
const MODELTYPE_TO_COMPOSITETYPE = Dict(atom => Symbol("BalancedModel$atom") for atom in SUPPORTED_MODEL_TYPES)
# A dictionary to convert e.g., form Probabilistic to ProbabilisticNetworkComposite
const MODELTYPE_TO_SUPERTYPE = Dict(atom => Symbol("$(atom)NetworkComposite") for atom in SUPPORTED_MODEL_TYPES)

# Define a struct for each model type (corresponds to a composite type and supertype used in struct)
for model_type in SUPPORTED_MODEL_TYPES
    struct_name = MODELTYPE_TO_COMPOSITETYPE[model_type]
    super_type  = MODELTYPE_TO_SUPERTYPE[model_type]
    ex = quote
        mutable struct $struct_name{balancernames, M <: $model_type} <: $super_type
            balancers
            model::M
            function $struct_name(balancernames, balancers, model::M) where M <: $model_type
                # generate an instance and use balancernames as type parameter
                return new{balancernames, M}(balancers, model)
            end
        end
    end
    eval(ex)
end

### 2. Define one keyword constructor for model structs

# A version of MODELTYPE_TO_COMPOSITETYPE with evaluated keys and values (used in keyword constructor)
const MODELTYPE_TO_COMPOSITETYPE_EVAL = Dict()
for MODELTYPE in SUPPORTED_MODEL_TYPES
    type = MODELTYPE_TO_COMPOSITETYPE[MODELTYPE]
    @eval(MODELTYPE_TO_COMPOSITETYPE_EVAL[$MODELTYPE] = $type)
end

# To represent any model type (to check input model type is one of them in keyword constructor)
const UNION_MODEL_TYPES = Union{keys(MODELTYPE_TO_COMPOSITETYPE_EVAL)...}


# Possible Errors (for the constructor as well)
const ERR_MODEL_UNSPECIFIED = ErrorException("Expected an atomic model as argument. None specified. ")

const WRN_BALANCER_UNSPECIFIED = "No balancer was provided. Data will be directly passed to the model. "

const PRETTY_SUPPORTED_MODEL_TYPES = join([string("`", opt, "`") for opt in SUPPORTED_MODEL_TYPES], ", ",", and ")

const ERR_UNSUPPORTED_MODEL(model) = ErrorException(
    "Only these model supertypes support wrapping: "*
    "$PRETTY_SUPPORTED_MODEL_TYPES.\n"*
    "Model provided has type `$(typeof(model))`. "
)
const ERR_NUM_ARGS_BM = "`BalancedModel` can at most have one non-keyword argument where the model is passed."                


"""
    BalancedModel(; model=nothing, balancer1=balancer_model1, balancer2=balancer_model2, ...)
    BalancedModel(model;  balancer1=balancer_model1, balancer2=balancer_model2, ...)

Wraps a classification model with balancers that resample the data before passing it to the model.

# Arguments
- `balancers::AbstractVector=[]`: A vector of balancers (i.e., resampling models). 
    Data passed to the model will be first passed to the balancers sequentially.
- `model=nothing`: The classification model which must be provided.
"""
function BalancedModel(args...; model=nothing, named_balancers...)
    # check model and balancer are given
    length(args) <= 1 || throw(ERR_NUM_ARGS_BM)
    if length(args) === 1
        atom = first(args)
        model === nothing ||
            @warn WRN_MODEL_GIVEN
        model = atom
    else
        model === nothing && throw(ERR_MODEL_UNSPECIFIED)
    end
    # check model is supported
    model isa UNION_MODEL_TYPES  || throw(ERR_UNSUPPORTED_MODEL(model))

    nt = NamedTuple(named_balancers)
    balancernames = keys(nt)
    balancers = collect(nt)
    # warn if balancer is not given
    isempty(balancers) && @warn WRN_BALANCER_UNSPECIFIED
    # call the appropriate constructor
    return MODELTYPE_TO_COMPOSITETYPE_EVAL[MMI.abstract_type(model)](balancernames, balancers, model)
end


### 3. Make a property for each balancer given via keyword arguments

# set the property names to include the keyword arguments
Base.propertynames(::BalancedModelProbabilistic{balancernames}) where balancernames =
    tuple(:model, balancernames...)

# overload getproperty to return the balancer form the vector in the struct
for model_type in SUPPORTED_MODEL_TYPES
    struct_name = MODELTYPE_TO_COMPOSITETYPE[model_type]
    ex = quote
        function Base.getproperty(b::$struct_name{balancernames}, name::Symbol) where balancernames
            balancers = getfield(b, :balancers)
            for j in eachindex(balancernames)
                name === balancernames[j] && return balancers[j]
            end
            return getfield(b, name)
        end
    end
    eval(ex)
end


const ERR_NO_PROP = ArgumentError("trying to access property $name which does not exist")
# overload set property to set the property from the vector in the struct
for model_type in SUPPORTED_MODEL_TYPES
    struct_name = MODELTYPE_TO_COMPOSITETYPE[model_type]
    ex = quote
        function Base.setproperty!(b::$struct_name{balancernames}, name::Symbol, val) where balancernames
            # find the balancer model with given balancer names
            idx = findfirst(==(name), balancernames)
            # get it from the vector in the struct and set it with the value
            !isnothing(idx) && return getfield(b, :balancers)[idx] = val
            # the other only option is model
            name === :model && return setfield(b, :model, val)
            throw(ERR_NO_PROP)
        end
    end
    eval(ex)
end



### 4. Define the prefit method
# used below, represents any composite model type offered by our package (e.g., BalancedProbabilisitcMode)
const UNION_COMPOSITE_TYPES{balancernames} = Union{[type{balancernames} for type in values(MODELTYPE_TO_COMPOSITETYPE_EVAL)]...}

"""
Overload the prefit method to export a learning network composed of a sequential pipeline of balancers
    followed by a final model.
"""
function MLJBase.prefit(balanced_model::UNION_COMPOSITE_TYPES{balancernames}, verbosity, _X, _y) where balancernames
    # the learning network:
    X = source(_X)
    y = source(_y)
    X_over, y_over = X, y
    # Let's transform the data through :balancer1, :balancer2,...
    for symbolic_balancer in balancernames
        balancer = getproperty(balanced_model, symbolic_balancer)
        mach1 = machine(balancer)
        data =  MLJBase.transform(mach1, X_over, y_over)
        X_over, y_over= first(data), last(data)
    end
    # we use the oversampled data for training:
    mach2 = machine(:model, X_over, y_over)     # wrap with the data to be trained
    # but consume new prodution data from the source:
    yhat = MLJBase.predict(mach2, X)
    # return the learning network interface:
    return (; predict=yhat)
end


### 5. Provide package information and pass up model traits
MMI.package_name(::Type{<:UNION_COMPOSITE_TYPES}) = "MLJBalancing"
MMI.package_license(::Type{<:UNION_COMPOSITE_TYPES}) = "MIT"
MMI.package_uuid(::Type{<:UNION_COMPOSITE_TYPES}) = "45f359ea-796d-4f51-95a5-deb1a414c586"
MMI.is_wrapper(::Type{<:UNION_COMPOSITE_TYPES}) = true
MMI.package_url(::Type{<:UNION_COMPOSITE_TYPES}) ="https://github.com/JuliaAI/MLJBalancing.jl"

# All the composite types BalancedModelProbabilistic, BalancedModelDeterministic, etc.
const COMPOSITE_TYPES = values(MODELTYPE_TO_COMPOSITETYPE)
for composite_type in COMPOSITE_TYPES
    quote
        MMI.iteration_parameter(::Type{<:$composite_type{balancernames, M}}) where {balancernames, M} =
            MLJBase.prepend(:model, iteration_parameter(M))
    end |> eval
    for trait in [
        :input_scitype,
        :output_scitype,
        :target_scitype,
        :fit_data_scitype,
        :predict_scitype,
        :transform_scitype,
        :inverse_transform_scitype,
        :is_pure_julia,
        :supports_weights,
        :supports_class_weights,
        :supports_online,
        :supports_training_losses,
        :is_supervised,
        :prediction_type]
        quote
            MMI.$trait(::Type{<:$composite_type{balancernames, M}}) where {balancernames, M} = MMI.$trait(M)
        end |> eval
    end
end
