"""
Tests for Part 1 - the foundations.

These are not "does it run" tests. Each one asserts on a property that would be
FALSE if the algorithm were wrong:

    gradient descent finds the mean
    linear regression recovers a slope of 2
    a too-large learning rate diverges while a sane one converges
    cross-entropy's gradient is much larger than squared error's when the model
    is confidently wrong
    a degree-15 polynomial overfits and ridge regularisation fixes it
    k-means recovers centres it was never shown
    the top 2 principal components carry almost all the variance
    a hidden layer beats a linear model on a problem a line cannot solve

If you change an algorithm and a test here fails, the algorithm changed
behaviour - which is exactly what a test is for.
"""

from __future__ import annotations

import numpy as np

from mlconcepts.foundations import (
    gradient_descent,
    kmeans,
    linear_regression,
    logistic_regression,
    loss_functions,
    neural_network,
    overfitting,
    pca,
    what_is_ml,
)


# ---------------------------------------------------------------- 1. the loop


def test_gradient_descent_finds_the_mean():
    """The one-parameter model should land on the mean of the data."""
    y = np.array([1.0, 3.0, 5.0, 7.0])
    w, history = what_is_ml.fit(y, learning_rate=0.1, steps=500)

    assert abs(w - y.mean()) < 0.01
    assert history[-1] < history[0]  # the loss actually fell


# ------------------------------------------------------- 2. linear regression


def test_linear_regression_recovers_the_true_line():
    x, y = what_is_ml.data.make_linear(n=200, noise=0.4, seed=1)
    w, b, history = linear_regression.fit(x, y)

    assert abs(w - 2.0) < 0.15, f"slope was {w}"
    assert abs(b - 1.0) < 0.15, f"intercept was {b}"
    assert linear_regression.r_squared(y, w * x + b) > 0.9
    assert history[-1] < history[0]


# ---------------------------------------------------------- 3. learning rate


def test_learning_rate_controls_convergence():
    x, y = what_is_ml.data.make_linear(n=120, seed=0)

    small = gradient_descent.fit(x, y, learning_rate=0.005)
    good = gradient_descent.fit(x, y, learning_rate=0.05)
    huge = gradient_descent.fit(x, y, learning_rate=1.0)

    assert good[-1] < small[-1], "the sane rate should beat the tiny one"
    assert not np.isfinite(huge[-1]) or huge[-1] > huge[0], "a huge rate should diverge"


def test_mini_batch_still_converges():
    x, y = what_is_ml.data.make_linear(n=120, seed=0)
    history = gradient_descent.fit_mini_batch(x, y, steps=200)
    assert history[-1] < history[0] * 0.5


# ----------------------------------------------------- 4. logistic regression


def test_logistic_regression_classifies():
    x, y = what_is_ml.data.make_classification(n=300, seed=0)
    x = (x - x.mean(axis=0)) / x.std(axis=0)

    w, b, history = logistic_regression.fit(x, y)
    accuracy = logistic_regression.accuracy(logistic_regression.sigmoid(x @ w + b), y)

    assert accuracy > 0.9, f"accuracy was {accuracy}"
    assert history[-1] < history[0]


def test_sigmoid_is_stable_at_the_extremes():
    """A naive implementation overflows here; this one should not."""
    z = np.array([-1000.0, -50.0, 0.0, 50.0, 1000.0])
    p = logistic_regression.sigmoid(z)

    assert np.all(np.isfinite(p))
    assert np.all((p >= 0.0) & (p <= 1.0))
    assert p[2] == 0.5


# -------------------------------------------------------------- 5. the losses


def test_cross_entropy_pushes_harder_when_confidently_wrong():
    """
    This is the whole point of the page. With the true label 1 and the model
    predicting 0.01, the cross-entropy gradient must dwarf the squared-error one.
    """
    p, y = 0.01, 1.0
    ce = abs(loss_functions.cross_entropy_grad_z(p, y))
    mse = abs(loss_functions.squared_error_grad_z(p, y))

    assert ce > 0.9
    assert ce > 20 * mse, f"expected a large gap, got ce={ce} mse={mse}"


def test_both_gradients_agree_when_the_model_is_unsure():
    """At p = 0.5 the two losses disagree by a small, known factor."""
    p, y = 0.5, 1.0
    ce = loss_functions.cross_entropy_grad_z(p, y)
    mse = loss_functions.squared_error_grad_z(p, y)
    assert abs(ce) == 0.5
    assert abs(mse / ce - 0.5) < 1e-9  # mse = 2*(p-y)*p*(1-p) = 0.5 * ce


# -------------------------------------------------------------- 6. overfitting


def test_high_degree_overfits_and_ridge_fixes_it():
    x, y, x_test, y_test = overfitting.make_data()

    deg3 = overfitting.fit_poly(overfitting.poly_features(x, 3), y)
    deg15 = overfitting.fit_poly(overfitting.poly_features(x, 15), y)

    train3 = overfitting.mse(overfitting.poly_features(x, 3) @ deg3, y)
    train15 = overfitting.mse(overfitting.poly_features(x, 15) @ deg15, y)
    test3 = overfitting.mse(overfitting.poly_features(x_test, 3) @ deg3, y_test)
    test15 = overfitting.mse(overfitting.poly_features(x_test, 15) @ deg15, y_test)

    # Fits the training data better...
    assert train15 < train3
    # ...and is far worse on data it has never seen.
    assert test15 > 10 * test3, f"test15={test15}, test3={test3}"

    # Regularisation pulls it back.
    regularised = overfitting.fit_poly(overfitting.poly_features(x, 15), y, lam=1e-4)
    test15_ridge = overfitting.mse(overfitting.poly_features(x_test, 15) @ regularised, y_test)
    assert test15_ridge < test15 / 10


def test_polynomial_basis_has_the_expected_shape():
    """A degree-3 fit has 4 coefficients: the bias, x, x^2 and x^3."""
    x = np.linspace(0, 1, 5)
    assert overfitting.poly_features(x, 3).shape == (5, 4)


# ---------------------------------------------------------------- 7. k-means


def test_kmeans_recovers_centres_it_never_saw():
    points, true_centres = what_is_ml.data.make_clusters(k=3, per_cluster=60, seed=0)
    centres, labels, history = kmeans.fit(points, k=3, seed=0)

    found_order = np.lexsort((centres[:, 1], centres[:, 0]))
    true_order = np.lexsort((true_centres[:, 1], true_centres[:, 0]))
    error = np.max(np.abs(centres[found_order] - true_centres[true_order]))

    assert error < 0.5, f"centres were off by {error}"
    assert len(set(labels.tolist())) == 3  # every cluster got used
    assert history[-1] < history[0]


def test_more_clusters_never_means_more_inertia():
    points, _ = what_is_ml.data.make_clusters(k=3, per_cluster=40, seed=0)
    inertias = [kmeans.fit(points, k=k, seed=0)[2][-1] for k in (1, 2, 3, 4)]
    assert inertias == sorted(inertias, reverse=True)


def test_kmeans_plus_plus_spreads_the_starting_centres():
    points, _ = what_is_ml.data.make_clusters(k=4, per_cluster=30, seed=0)
    rng = np.random.default_rng(0)
    centres = kmeans.kmeans_plus_plus(points, 4, rng)
    assert len({tuple(np.round(c, 3)) for c in centres}) == 4


# --------------------------------------------------------------------- 8. PCA


def test_pca_finds_the_two_real_dimensions():
    x = what_is_ml.data.make_correlated(n=300, seed=0)
    model = pca.pca_fit(x, n_components=2)

    assert model["explained"].shape == (5,)
    assert model["explained"][:2].sum() > 0.9, "two components should carry most of it"
    assert abs(model["explained"].sum() - 1.0) < 1e-9, "the ratios must add to 1"

    # And the compression is faithful.
    z = pca.transform(x, model)
    assert z.shape == (300, 2)
    back = pca.inverse_transform(z, model)
    assert np.mean((x - back) ** 2) < 0.1


# --------------------------------------------------------- 9. neural network


def test_a_hidden_layer_beats_a_linear_model_on_a_ring():
    x, y = what_is_ml.data.make_ring(n=400, seed=0)
    x = (x - x.mean(axis=0)) / x.std(axis=0)

    # Linear model: cannot wrap around, so it is stuck near chance.
    w, b, _ = logistic_regression.fit(x, y, learning_rate=0.5, steps=2000)
    linear_accuracy = logistic_regression.accuracy(
        logistic_regression.sigmoid(x @ w + b), y
    )

    # One hidden layer: bends the boundary around the ring.
    params, loss_history, _ = neural_network.train(x, y, n_hidden=8, steps=3000)
    net_accuracy = neural_network.accuracy(neural_network.forward(x, params)["A2"], y)

    assert linear_accuracy < 0.8, f"the baseline should struggle, got {linear_accuracy}"
    assert net_accuracy > 0.9, f"the network should not, got {net_accuracy}"
    assert net_accuracy > linear_accuracy + 0.2
    assert loss_history[-1] < loss_history[0]


def test_backward_pass_produces_the_right_shapes():
    x, y = what_is_ml.data.make_ring(n=50, seed=0)
    params = neural_network.init_params(2, 6)
    cache = neural_network.forward(x, params)
    grads = neural_network.backward(x, y, params, cache)

    for name, param in params.items():
        assert grads[name].shape == param.shape, f"gradient shape wrong for {name}"
        assert np.all(np.isfinite(grads[name]))


def test_hidden_units_are_not_initialised_identically():
    """
    Zero init makes every hidden unit compute the same thing and receive the
    same gradient, so they stay identical forever - a classic silent bug.

    W1 has shape (n_inputs, n_hidden), so the hidden units are the COLUMNS.
    """
    params = neural_network.init_params(2, 8, seed=0)
    assert np.any(params["W1"] != 0), "weights must not all be zero"

    columns = {tuple(np.round(column, 6)) for column in params["W1"].T}
    assert len(columns) == 8, "every hidden unit needs its own starting weights"

    # And the same seed must reproduce exactly the same initialisation.
    again = neural_network.init_params(2, 8, seed=0)
    assert np.allclose(params["W1"], again["W1"])
