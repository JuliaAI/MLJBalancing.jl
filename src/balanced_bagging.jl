
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
Given data `X`, `y` where `X` is a table and `y` is an abstract vector, the indices and counts
    of the majority and minority classes and an integer for the seed, return `X_sub`, `y_sum`
    which are the result of randomly undersampling the majority class data in `X`, `y` so that
    both classes occur equally frequently.
"""
function get_some_balanced_subset(
    X,
    y,
    majority_inds,
    minority_inds,
    majority_count,
    minority_count,
    rng::Integer,
)
    rng = Xoshiro(rng)
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
mutable struct BalancedBaggingClassifier{R<:AbstractRNG,I<:Integer,P<:Probabilistic} <:
               ProbabilisticNetworkComposite
    model::P
    T::I
    rng::R
end

const ERR_MISSING_CLF = "No classifier specified. Please specify a probabilistic classifier using the `classifier` keyword argument."
const ERR_MISSING_T = "The number of ensemble models must be specified. Please specify the number of models using the `T` keyword argument."
function BalancedBaggingClassifier(;
    classifier = nothing,
    T = nothing,
    rng = Random.default_rng(),
)
    classifier === nothing && error(ERR_MISSING_CLF)
    T === nothing && error(ERR_MISSING_T)           # probably want to do better.
    return BalancedBaggingClassifier(classifier, T, rng)
end

function MLJBase.prefit(model::BalancedBaggingClassifier, verbosity, X, y)
    rngs = rand(model.rng, 1:model.T*10, model.T)
    println(rngs)
    Xs, ys = source(X), source(y)
    majority_inds, minority_inds, majority_count, minority_count =
        get_majority_minority_inds_counts(y)
    # get as much balanced subsets as needed
    X_y_list_s = [
        get_some_balanced_subset(
            Xs,
            ys,
            majority_inds,
            minority_inds,
            majority_count,
            minority_count,
            rng,
        ) for rng in rngs
    ]
    # Make a machine for each
    machines = (machine(:model, Xsub, ysub) for (Xsub, ysub) in X_y_list_s)
    # Average the predictions from nodes
    all_preds = [MLJBase.predict(mach, Xs) for (mach, (X, _)) in zip(machines, X_y_list_s)]
    println(all_preds)
    yhat = mean(all_preds)
    return (; predict = yhat)
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
$(MMI.doc_header(BalancedBaggingClassifier))

    Given a probablistic classifier.`BalancedBaggingClassifier` performs bagging by only undersampling
    majority data in each bag so that its includes as much samples as in the minority data.
    When the classifier is Adaboost, it is tantamount to EasyEnsemble as presented in Xu-Ying Liu, 
    Jianxin Wu, & Zhi-Hua Zhou. (2009). Exploratory Undersampling for Class-Imbalance Learning. 
    IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics), 39 (2), 539–5501 


    # Training data

    In MLJ or MLJBase, bind an instance `model` to data with
    
        mach = machine(model, X, y)
    
    where
    
    - `X`: any table of input features (e.g., a `DataFrame`) so long as elements in each column
        are subtypes of either the `Finite` or `Infinite` scientific types.
    
    - `y`: the binary target, which can be any `AbstractVector` where `length(unique(y)) == 2`


    Train the machine with `fit!(mach, rows=...)`.


    # Hyperparameters

    - `classifier<:Probabilistic`: The classifier to use to train on each bag.

    - `T::integer`: The number of bags 

    - `rng::AbstractRNG`: The random number generator to use. 

    # Operations

    - `predict(mach, Xnew)`: return predictions of the target given
    features `Xnew` having the same scitype as `X` above. Predictions
    are probabilistic, but uncalibrated.

    - `predict_mode(mach, Xnew)`: instead return the mode of each
    prediction above.


    # Example

    ```julia
    using MLJ

    # Load base classifier and BalancedBaggingClassifier
    BalancedBaggingClassifier = @load BalancedBaggingClassifier pkg=MLJBalancing
    LogisticClassifier = @load LogisticClassifier pkg=MLJLinearModels verbosity=0

    # Construct the base classifier and use it to construct a BalancedBaggingClassifier
    logistic_model = LogisticClassifier()
    model = BalancedBaggingClassifier(classifier=logistic_model, T=50)

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
