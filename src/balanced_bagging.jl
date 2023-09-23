
"""
Return a dictionary `result` mapping each unique value in a given abstract vector `y`
    to the vector of indices where that value occurs.
"""
function group_inds(y::AbstractVector{T}) where {T}
    result = LittleDict{T,AbstractVector{Int}}()
    for (i, v) in enumerate(y)
        # Make a new entry in the dict if it doesn't exist
        if !haskey(result, v)
            result[v] = []
        end
        # It exists, so push the index belonging to the class
        push!(result[v], i)
    end
    return freeze(result)
end

const ERR_MULTICLASS_UNSUPP(num_classes) =
    "Only binary classification supported by BalancedBaggingClassifier. Got $num_classes classes"

"""
Given an abstract vector `y` where any element takes one of two values, return the indices of the
    most frequent of them, the indices of the least frequent of them, and the counts of each.
"""
function get_majority_minority_inds_counts(y)
    # a tuple mapping each class to its indices
    labels_inds = collect(group_inds(y))
    num_classes = length(labels_inds)
    num_classes == 2 || throw(ArgumentError(ERR_MULTICLASS_UNSUPP(num_classes)))
    # get the length of each class
    first_class_count = length(labels_inds[1][2])
    second_class_count = length(labels_inds[2][2])
    # get the majority and minority inds by comparing lengths
    if first_class_count > second_class_count
        majority_inds, minority_inds = labels_inds[1][2], labels_inds[2][2]
        return majority_inds, minority_inds, first_class_count, second_class_count
    else
        majority_inds, minority_inds = labels_inds[2][2], labels_inds[1][2]
        return majority_inds, minority_inds, second_class_count, first_class_count
    end
end

"""
Given data `X`, `y` where `X` is a table and `y` is an abstract vector (which may be wrapped in nodes), 
    the indices and counts of the majority and minority classes and abstract rng,
    return `X_sub`, `y_sub`, in the form of nodes, which are the result of randomly undersampling 
    the majority class data in `X`, `y` so that both classes occur equally frequently.
"""
function get_some_balanced_subset(
    X,
    y,
    majority_inds,
    minority_inds,
    majority_count,
    minority_count,
    rng::AbstractRNG,
)
    # randomly sample a subset of size minority_count indices from those belonging to majority class
    random_inds = sample(rng, 1:majority_count, minority_count, replace = true)
    majority_inds_undersampled = majority_inds[random_inds]
    # find the corresponding subset of data which includes all minority and majority subset
    balanced_subset_inds = vcat(minority_inds, majority_inds_undersampled)
    X_sub = node(X -> getobs(X, balanced_subset_inds), X)
    y_sub = node(y -> y[balanced_subset_inds], y)
    return X_sub, y_sub
end


"""
Construct an BalancedBaggingClassifier model.
"""
mutable struct BalancedBaggingClassifier{RI<:Union{AbstractRNG, Integer},I<:Integer,P<:Probabilistic} <:
               ProbabilisticNetworkComposite
    model::P
    T::I
    rng::RI
end

rng_handler(rng::Integer) = Random.Xoshiro(rng)
rng_handler(rng::AbstractRNG) = rng
const ERR_MISSING_CLF = "No model specified. Please specify a probabilistic classifier using the `model` keyword argument."
const ERR_BAD_T = "The number of ensemble models `T` cannot be negative."
const INFO_DEF_T(T_def) = "The number of ensemble models was not given and was thus, automatically set to $T_def"*
                          " which is the ratio of the frequency of the majority class to that of the minority class"
function BalancedBaggingClassifier(;
    model = nothing,
    T = 0,
    rng = Random.default_rng(),
)
    model === nothing && error(ERR_MISSING_CLF)
    T < 0 && error(ERR_BAD_T)      
    rng = rng_handler(rng)    
    return BalancedBaggingClassifier(model, T, rng)
end

function MLJBase.prefit(composite_model::BalancedBaggingClassifier, verbosity, X, y)
    Xs, ys = source(X), source(y)
    majority_inds, minority_inds, majority_count, minority_count =
        get_majority_minority_inds_counts(y)
    T = composite_model.T
    if composite_model.T == 0
        T_def = round(Int, majority_count/minority_count)
        T = T_def
        @info  INFO_DEF_T(T_def)
    end
    # get as much balanced subsets as needed
    X_y_list_s = [
        get_some_balanced_subset(
            Xs,
            ys,
            majority_inds,
            minority_inds,
            majority_count,
            minority_count,
            composite_model.rng,
        ) for i in 1:T
    ]
    # Make a machine for each
    machines = (machine(:model, Xsub, ysub) for (Xsub, ysub) in X_y_list_s)
    # Average the predictions from nodes
    all_preds = [MLJBase.predict(mach, Xs) for (mach, (X, _)) in zip(machines, X_y_list_s)]
    yhat = mean(all_preds)
    return (; predict=yhat )
end

### To register with MLJ
MMI.metadata_pkg(
    BalancedBaggingClassifier,
    name = "MLJBalancing",
    package_uuid = "45f359ea-796d-4f51-95a5-deb1a414c586",
    package_url = "https://github.com/JuliaAI/MLJBalancing.jl",
    is_pure_julia = true,
)

MMI.metadata_model(
    BalancedBaggingClassifier,
    input_scitype = Union{Union{Infinite,Finite}},
    output_scitype = Union{Union{Infinite,Finite}},
    target_scitype = AbstractVector,
    load_path = "MLJBalancing." * string(BalancedBaggingClassifier),
)

MMI.iteration_parameter(::Type{<:BalancedBaggingClassifier{P}}) where {P} =
    MLJBase.prepend(:model, iteration_parameter(P))
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
    :prediction_type,
]
    quote
        MMI.$trait(::Type{<:BalancedBaggingClassifier{P}}) where {P} = MMI.$trait(P)
    end |> eval
end

"""
    BalancedBaggingClassifier
    A model type for constructing a balanced bagging classifier, based on [MLJBalancing.jl](https://github.com/JuliaAI/MLJBalancing).

    From MLJ, the type can be imported using

    `BalancedBaggingClassifier = @load BalancedBaggingClassifier pkg=MLJBalancing``

    Construct an instance with default hyper-parameters using the syntax `bagging_model = BalancedBaggingClassifier(model=...)`

    Given a probablistic classifier.`BalancedBaggingClassifier` performs bagging by undersampling
    only majority data in each bag so that its includes as much samples as in the minority data.
    This is proposed with an Adaboost classifier where the output scores are averaged in the paper
    Xu-Ying Liu, Jianxin Wu, & Zhi-Hua Zhou. (2009). Exploratory Undersampling for Class-Imbalance Learning. 
    IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics), 39 (2), 539–5501 


    # Training data

    In MLJ or MLJBase, bind an instance `model` to data with
    
        mach = machine(model, X, y)
    
    where
    
    - `X`: input features of a form supported by the `model` being wrapped (typically a table, e.g., `DataFrame`,
        with `Continuous` columns will be supported, as a minimum)
    
    - `y`: the binary target, which can be any `AbstractVector` where `length(unique(y)) == 2`


    Train the machine with `fit!(mach, rows=...)`.


    # Hyperparameters

    - `model<:Probabilistic`: The classifier to use to train on each bag.

    - `T::integer=0`: The number of bags to be used in the ensemble. If not given, will be set as
        the ratio between the frequency of the majority and minority classes.

    - `rng::Union{AbstractRNG, Integer}=default_rng()`: Either an `AbstractRNG` object or an `Integer` 
    seed to be used with `Xoshiro`

    # Operations

    - `predict(mach, Xnew)`: return predictions of the target given
    features `Xnew` having the same scitype as `X` above. Predictions
    are probabilistic, but uncalibrated.

    - `predict_mode(mach, Xnew)`: return the mode of each prediction above



    # Example

    ```julia
    using MLJ

    # Load base classifier and BalancedBaggingClassifier
    BalancedBaggingClassifier = @load BalancedBaggingClassifier pkg=MLJBalancing
    LogisticClassifier = @load LogisticClassifier pkg=MLJLinearModels verbosity=0

    # Construct the base classifier and use it to construct a BalancedBaggingClassifier
    logistic_model = LogisticClassifier()
    model = BalancedBaggingClassifier(model=logistic_model, T=50)

    # Load the data and train the BalancedBaggingClassifier
    X, y = @load_iris
    mach = machine(model, X, y) |> fit!
    
    # Predict using the trained model
    Xnew = (sepal_length = [6.4, 7.2, 7.4],
            sepal_width = [2.8, 3.0, 2.8],
            petal_length = [5.6, 5.8, 6.1],
            petal_width = [2.1, 1.6, 1.9],)

    yhat = predict(mach, Xnew) # probabilistic predictions
    predict_mode(mach, Xnew)   # point predictions
    ```
"""
BalancedBaggingClassifier
