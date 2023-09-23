@testset "BalancedModel" begin
	### end-to-end test
	# Create and split data
	X, y = generate_imbalanced_data(100, 5; probs = [0.2, 0.3, 0.5])
	X = DataFrame(X)
	train_inds, test_inds =
		partition(eachindex(y), 0.8, shuffle = true, stratify = y, rng = Random.Xoshiro(42))
	X_train, X_test = X[train_inds, :], X[test_inds, :]
	y_train, y_test = y[train_inds], y[test_inds]

	# Load models and balancers
	using LIBSVM
	SVC = @load SVC
	using MLJLinearModels
	LogisticClassifier = @load LogisticClassifier
	
	# Here are a probabilistic and a deterministic model
	model_prob = LogisticClassifier()
	model_det = SVC(gamma = 0.01)
	# And here are three resamplers from Imbalance. 
	# The package should actually work with any `Static` transformer of the form  `(X, y) -> (Xout, yout)`
	# provided that it implements the MLJ interface. Here, the balancer is the transformer
	balancer1 = Imbalance.MLJ.RandomOversampler(ratios = 1.0, rng = 42)
	balancer2 = Imbalance.MLJ.SMOTENC(k = 10, ratios = 1.2, rng = 42)
	balancer3 = Imbalance.MLJ.ROSE(ratios = 1.3, rng = 42)

	### 1. Make a pipeline of the three balancers and a probablistic model
	## ordinary way
	mach = machine(balancer1)
	Xover, yover = MLJBase.transform(mach, X_train, y_train)
	mach = machine(balancer2)
	Xover, yover = MLJBase.transform(mach, Xover, yover)
	mach = machine(balancer3)
	Xover, yover = MLJBase.transform(mach, Xover, yover)

	mach = machine(model_prob, Xover, yover)
	fit!(mach)
	y_pred = MLJBase.predict(mach, X_test)

	# with MLJ balancing
	@test_throws MLJBalancing.ERR_MODEL_UNSPECIFIED begin
		BalancedModel(b1 = balancer1, b2 = balancer2, b3 = balancer3)
	end
	@test_throws "ArgumentError: Only these model supertypes support wrapping: `Probabilistic`, `Deterministic`, and `Interval`.\nModel provided has type `Int64`." begin
		BalancedModel(model = 1, b1 = balancer1, b2 = balancer2, b3 = balancer3)
	end
	@test_logs (:warn, MLJBalancing.WRN_BALANCER_UNSPECIFIED) begin
		BalancedModel(model = model_prob)
	end

	balanced_model =
		BalancedModel(model = model_prob, b1 = balancer1, b2 = balancer2, b3 = balancer3)
	mach = machine(balanced_model, X_train, y_train)
	fit!(mach)
	y_pred2 = MLJBase.predict(mach, X_test)

	@test y_pred ≈ y_pred2

	### 2. Make a pipeline of the three balancers and a deterministic model
	## ordinary way
	mach = machine(balancer1)
	Xover, yover = MLJBase.transform(mach, X_train, y_train)
	mach = machine(balancer2)
	Xover, yover = MLJBase.transform(mach, Xover, yover)
	mach = machine(balancer3)
	Xover, yover = MLJBase.transform(mach, Xover, yover)

	mach = machine(model_det, Xover, yover)
	fit!(mach)
	y_pred = MLJBase.predict(mach, X_test)

	# with MLJ balancing
	balanced_model =
		BalancedModel(model = model_det, b1 = balancer1, b2 = balancer2, b3 = balancer3)
	mach = machine(balanced_model, X_train, y_train)
	fit!(mach)
	y_pred2 = MLJBase.predict(mach, X_test)

	@test y_pred == y_pred2

	### check that setpropertyname and getpropertyname work
	Base.getproperty(balanced_model, :b1) == balancer1
	Base.setproperty!(balanced_model, :b1, balancer2)
	Base.getproperty(balanced_model, :b1) == balancer2
	@test_throws MLJBalancing.ERR_NO_PROP begin
		Base.setproperty!(balanced_model, :name11, balancer2)
	end
end
