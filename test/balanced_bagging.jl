
@testset "group_inds and get_majority_minority_inds_counts" begin
    y = [0, 0, 0, 0, 1, 1, 1, 0]
    @test MLJBalancing.group_inds(y) == Dict(0 => [1, 2, 3, 4, 8], 1 => [5, 6, 7])
    @test MLJBalancing.get_majority_minority_inds_counts(y) ==
          ([1, 2, 3, 4, 8], [5, 6, 7], 5, 3)
    y = [0, 0, 0, 0, 1, 1, 1, 0, 2, 2, 2]
    @test_throws MLJBalancing.ERR_MULTICLASS_UNSUPP(3) MLJBalancing.get_majority_minority_inds_counts(
        y,
    )
end

@testset "BalancedBaggingClassifier" begin
    X, y = generate_imbalanced_data(
        100,
        5;
        cat_feats_num_vals = [3, 2, 1, 2],
        probs = [0.9, 0.1],
        type = "ColTable",
        rng = 42,
    )
    majority_inds, minority_inds, majority_count, minority_count =
        MLJBalancing.get_majority_minority_inds_counts(y)
    Xs, ys = MLJBase.source(X), MLJBase.source(y)
    X_sub, y_sub = MLJBalancing.get_some_balanced_subset(
        Xs,
        ys,
        majority_inds,
        minority_inds,
        majority_count,
        minority_count,
        Random.Xoshiro(42)
    )
    X_sub, y_sub = X_sub(rows = 1:100), y_sub(rows = 1:100)
    majority_inds_sub, minority_inds_sub, _, _ =
        MLJBalancing.get_majority_minority_inds_counts(y_sub)

    X_sub = Tables.matrix(X_sub)
    X = Tables.matrix(X)
    # minority untouched
    @test sum(X_sub[minority_inds_sub, :]) == sum(X[minority_inds, :])
    # majority undersampled
    @test issubset(
        Set(eachrow(X_sub[majority_inds_sub, :])),
        Set(eachrow(X[majority_inds, :])),
    )
    # balances the data
    @test length(y_sub[minority_inds_sub]) === length(y_sub[majority_inds_sub])
end

@testset "End-to-end Test" begin
    ## setup parameters
    R = Random.Xoshiro(42)
    T = 2
    LogisticClassifier = @load LogisticClassifier pkg = MLJLinearModels verbosity = 0
    model = LogisticClassifier()

    ## setup data
    # training
    X, y = generate_imbalanced_data(
        100,
        5;
        cat_feats_num_vals = [3, 2, 1, 2],
        probs = [0.9, 0.1],
        type = "ColTable",
        rng = 42,
    )
    # testing
    Xt, yt = generate_imbalanced_data(
        5,
        5;
        cat_feats_num_vals = [3, 2, 1, 2],
        probs = [0.9, 0.1],
        type = "ColTable",
        rng = 42,
    )

    ## prepare subsets
    majority_inds, minority_inds, majority_count, minority_count =
        MLJBalancing.get_majority_minority_inds_counts(y)
    Xs, ys = MLJBase.source(X), MLJBase.source(y)
    X_sub1, y_sub1 = MLJBalancing.get_some_balanced_subset(
        Xs,
        ys,
        majority_inds,
        minority_inds,
        majority_count,
        minority_count,
        R,
    )
    X_sub1, y_sub1 = X_sub1(rows = 1:100), y_sub1(rows = 1:100)
    X_sub2, y_sub2 = MLJBalancing.get_some_balanced_subset(
        Xs,
        ys,
        majority_inds,
        minority_inds,
        majority_count,
        minority_count,
        R,
    )
    X_sub2, y_sub2 = X_sub2(rows = 1:100), y_sub2(rows = 1:100)

    # training manually
    mach1 = machine(model, X_sub1, y_sub1)
    fit!(mach1)
    mach2 = machine(model, X_sub2, y_sub2)
    fit!(mach2)
    pred1 = MLJBase.predict(mach1, Xt)
    pred2 = MLJBase.predict(mach2, Xt)
    pred_manual = mean([pred1, pred2])

    ## using BalancedBagging
    modelo = BalancedBaggingClassifier(model = model, T = 2, rng = Random.Xoshiro(42))
    mach = machine(modelo, X, y)
    fit!(mach)
    pred_auto = MLJBase.predict(mach, Xt)
    @test sum(pred_manual) ≈ sum(pred_auto)
end
