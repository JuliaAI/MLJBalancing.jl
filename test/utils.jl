# A function to generate imbalanced data
function generate_imbalanced_data(
    num_rows,
    num_continuous_feats;
    cat_feats_num_vals = [],
    probs = [0.5, 0.5],
    type = "ColTable",
    insert_y = nothing,
    rng = default_rng(),
)
    rng = Random.Xoshiro(rng)
    # Generate y as a categorical array with classes 0, 1, 2, ..., k-1
    cum_probs = cumsum(probs)
    rands = rand(rng, num_rows)
    y = CategoricalArray([findfirst(x -> rands[i] <= x, cum_probs) - 1 for i = 1:num_rows])

    if num_continuous_feats > 0
        Xc = rand(rng, Float64, num_rows, num_continuous_feats)
    else
        Xc = Matrix{Int64}(undef, num_rows, 0)
    end

    for num_levels in cat_feats_num_vals
        Xc = hcat(Xc, rand(rng, 1:num_levels, num_rows))
    end

    if !isnothing(insert_y)
        Xc = hcat(Xc[:, 1:insert_y-1], y, Xc[:, insert_y:end])
    end

    DXc = Tables.table(Xc)

    if type == "Matrix"
        X = Xc
    elseif type == "RowTable"
        X = Tables.rowtable(DXc)
    elseif type == "ColTable"
        X = Tables.columntable(DXc)
    elseif type == "MatrixTable"
        X = Tables.table(Xc)
    elseif type == "DictRowTable"
        X = Tables.dictrowtable(DXc)
    elseif type == "DictColTable"
        X = Tables.dictcolumntable(DXc)
    else
        error("Invalid type")
    end

    return X, y
end
