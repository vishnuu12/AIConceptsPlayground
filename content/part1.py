"""
Part 1 content: the nine foundations pages.

Every page has the same shape:

    number, slug, title, part     identity
    blurb                         one line, for the index card
    why                           the "why it matters" panel
    sections                      the explanation, with code
    results                       the REAL numbers the demo printed
    demo                          the command to run it yourself
    plot                          the image the demo generated
    interview                     questions with answers you can say out loud

`results` is deliberately copied from an actual run, not invented. If you change
a demo and the numbers move, update them here - a page that quotes numbers the
code no longer produces is worse than no numbers at all.
"""

PART_1: list[dict] = [
    {
        "number": 1,
        "slug": "what-is-ml",
        "part": 1,
        "title": "What machine learning actually is",
        "blurb": "One loop - guess, measure, nudge - with a single parameter so you can check it by hand.",
        "why": [
            "Machine learning is usually taught as a catalogue of algorithms, which makes it look like "
            "a pile of unrelated tricks. It is one loop: the model guesses, a loss measures how wrong "
            "the guess was, a gradient says which way to move, and the parameters move that way. Repeat.",
            "Everything later in this repo - logistic regression, k-means, a neural network, an LLM - is "
            "a variation on step one and step three. Get the loop in your head and the rest is detail.",
        ],
        "sections": [
            {
                "heading": "The smallest possible model",
                "paragraphs": [
                    "Rather than start with features, start with a single number. The model has one "
                    "parameter, w, and always predicts w. Gradient descent has to discover the one value "
                    "that minimises squared error - which turns out to be the mean of the data.",
                    "That is a real training run: a parameter, a loss, a gradient, an update loop. It is "
                    "just small enough to verify by hand, which is the point.",
                ],
                "code": (
                    "loss = mean((w - y)^2)\n"
                    "d loss / d w = mean(2 * (w - y))\n\n"
                    "w = w - learning_rate * gradient   # and repeat"
                ),
            },
            {
                "heading": "The vocabulary, once",
                "bullets": [
                    "Features (x) - the inputs. Labels (y) - the answer you want to predict.",
                    "Parameters (w, b) - the numbers the model learns. Not the same as hyperparameters.",
                    "Hyperparameters - the numbers YOU choose: learning rate, number of clusters, degree.",
                    "Loss - a single number saying how wrong the model is. Training means making it smaller.",
                    "Gradient - the direction and steepness of the loss with respect to each parameter.",
                    "Supervised - you have labels. Unsupervised - you do not. Reinforcement - you get a reward.",
                ],
            },
        ],
        "results": [
            "true mean of y = 1.4219",
            "learned w = 1.4219  (discovered, never told)",
            "loss fell from 15.1453 to 13.1235 over 200 steps",
        ],
        "demo": "python -m mlconcepts.foundations.what_is_ml",
        "plot": "01-what-is-ml-loss.png",
        "interview": [
            {
                "q": "What is machine learning, in one sentence?",
                "a": "It is fitting the parameters of a model to data by repeatedly measuring how wrong the "
                "model is and nudging the parameters in the direction that reduces that error.",
            },
            {
                "q": "What is the difference between a parameter and a hyperparameter?",
                "a": "Parameters are learned from data - the weights and biases. Hyperparameters are chosen "
                "by you before training - the learning rate, the number of layers, the degree of the "
                "polynomial. The model finds the first; you find the second by experiment.",
            },
            {
                "q": "Why squared error rather than absolute error?",
                "a": "Squaring is differentiable everywhere, which makes the gradient easy, and it punishes "
                "large errors much more than small ones. The cost is that a single outlier dominates the "
                "loss, which is why absolute error or Huber loss is used when outliers matter.",
            },
            {
                "q": "What does 'training' actually change?",
                "a": "Only the parameters. The architecture, the loss function and the data are all fixed "
                "before training starts - training is just a loop that moves numbers downhill.",
            },
        ],
    },
    {
        "number": 2,
        "slug": "linear-regression",
        "part": 1,
        "title": "Linear regression from scratch",
        "blurb": "Two parameters, two gradients, 600 steps - and a slope of 2 emerges from noisy data.",
        "why": [
            "Linear regression is the hello world of supervised learning, and it is worth writing by hand "
            "once because every later model is the same code with a fancier guess function.",
            "If you skip this and jump to a framework, you never see that training is: compute a couple of "
            "derivatives, subtract them, repeat. When a real model fails to converge you have no mental "
            "model for why.",
        ],
        "sections": [
            {
                "heading": "The whole algorithm",
                "paragraphs": [
                    "The model is a line. The loss is mean squared error. The gradients are two partial "
                    "derivatives, one per parameter. Both parameters are updated from the SAME prediction - "
                    "updating w first and then recomputing for b is a subtle and very common bug.",
                ],
                "code": (
                    "prediction = w * x + b\n"
                    "error      = prediction - y\n"
                    "loss       = mean(error ** 2)\n\n"
                    "grad_w = 2/n * sum(error * x)\n"
                    "grad_b = 2/n * sum(error)\n\n"
                    "w -= lr * grad_w\n"
                    "b -= lr * grad_b"
                ),
            },
            {
                "heading": "Many features is the same maths",
                "paragraphs": [
                    "Real data has many features, so the model becomes X @ w + b with X of shape "
                    "(n_samples, n_features). The gradient becomes a vector instead of two scalars, and "
                    "nothing else changes. That is why the single-feature version is the right place to start.",
                ],
            },
            {
                "heading": "You can also solve it exactly",
                "paragraphs": [
                    "Linear regression has a closed-form solution - the normal equations - so you do not have "
                    "to iterate at all. Gradient descent exists for the models where no closed form exists, "
                    "which is every model after this page. The overfitting demo uses the closed form because "
                    "it is exact and lets us focus on something else.",
                ],
            },
        ],
        "results": [
            "the data was generated as y = 2.0x + 1.0 + noise",
            "learned w = 1.978, learned b = 0.961",
            "loss (MSE) = 0.3726, R^2 = 0.9728",
        ],
        "demo": "python -m mlconcepts.foundations.linear_regression",
        "plot": "02-linear-regression.png",
        "interview": [
            {
                "q": "Write the gradient update for linear regression.",
                "a": "prediction = w*x + b; error = prediction - y; grad_w = 2/n * sum(error * x); "
                "grad_b = 2/n * sum(error); then subtract learning_rate times each from the parameter.",
            },
            {
                "q": "What does R squared tell you?",
                "a": "The fraction of the variance in y that the model explains. 1.0 is perfect, 0.0 is no "
                "better than always predicting the mean, and negative means worse than the mean. It is "
                "scale-free, so it is comparable across datasets - unlike MSE.",
            },
            {
                "q": "Why standardise features before gradient descent?",
                "a": "Gradient descent takes one step size for all parameters. If one feature ranges over "
                "thousands and another over fractions, the loss is a long thin valley and a single learning "
                "rate is either too big for one direction or too small for the other. Scaling makes it round.",
            },
            {
                "q": "When would you not use gradient descent for this?",
                "a": "When a closed-form solution exists and the data fits in memory. For ordinary linear "
                "regression the normal equations are exact and need no learning rate. Gradient descent wins "
                "when the model has no closed form, or the data is too large to invert a matrix over.",
            },
        ],
    },
    {
        "number": 3,
        "slug": "gradient-descent",
        "part": 1,
        "title": "Gradient descent and the learning rate",
        "blurb": "Three runs, one dial: crawling, converging, and exploding - plus why SGD is noisy on purpose.",
        "why": [
            "The learning rate is the one hyperparameter every practitioner has to reason about, and it is "
            "easiest to understand by watching it fail. Too small and the run crawls. About right and the "
            "loss falls smoothly. Too large and it overshoots, bounces harder each step, and diverges.",
            "Understanding WHY it diverges is the part interviews ask about, and it is one inequality.",
        ],
        "sections": [
            {
                "heading": "Why a big step explodes",
                "paragraphs": [
                    "Gradient descent assumes the loss is roughly a smooth bowl. Take a step wider than the "
                    "bowl and you land higher up the far side; the next gradient is larger, so the next step "
                    "overshoots further. For mean squared error the update is stable while the learning rate "
                    "is below 2 divided by the largest curvature of the loss, and diverges above it.",
                ],
                "code": (
                    "# stable while:   learning_rate < 2 / (largest curvature)\n"
                    "# for MSE on this data the curvature is about 2 * mean(x^2) ~ 6,\n"
                    "# so the ceiling is around 0.33 - and 1.0 blows straight past it."
                ),
            },
            {
                "heading": "Batch versus stochastic",
                "paragraphs": [
                    "Full-batch descent uses every example per step: smooth, but expensive, and it can settle "
                    "into a flat region. Stochastic descent uses one example or a mini-batch: the curve is "
                    "noisy but each step is cheap, and the noise itself helps escape bad regions. Real "
                    "training almost always uses mini-batches.",
                ],
            },
            {
                "heading": "The fixes people actually use",
                "bullets": [
                    "Schedule it: start larger, decay over time, so you take big steps early and settle later.",
                    "Momentum: keep a running average of past gradients so a consistent direction accelerates.",
                    "Adam / RMSProp: per-parameter step sizes, adapted from the size of recent gradients.",
                    "Warmup: start tiny for a few hundred steps, then rise - standard for transformers.",
                    "Gradient clipping: cap the gradient norm so one bad batch cannot blow the weights up.",
                ],
            },
        ],
        "results": [
            "lr 0.005 (too small): 15.737 -> 1.5712   (still crawling)",
            "lr 0.05  (about right): 15.737 -> 0.3728",
            "lr 1.0   (too large): diverged",
            "mini-batch (size 8): 12.052 -> 0.3897   (noisy but arrives)",
        ],
        "demo": "python -m mlconcepts.foundations.gradient_descent",
        "plot": "03-learning-rate.png",
        "interview": [
            {
                "q": "What happens if the learning rate is too large?",
                "a": "Each step overshoots the minimum. The next gradient is bigger, so the next step "
                "overshoots further, and the loss grows until it becomes infinite or NaN. For MSE the "
                "threshold is 2 divided by the largest curvature of the loss.",
            },
            {
                "q": "Batch, mini-batch or stochastic?",
                "a": "Mini-batch, almost always. Full batch gives an exact gradient but costs a full pass per "
                "step; single-example SGD is cheap but very noisy. Mini-batches give a good enough gradient "
                "with hardware-friendly matrix operations, and the noise helps escape sharp minima.",
            },
            {
                "q": "How would you diagnose a run that is not converging?",
                "a": "Plot the loss. Flat and high means the learning rate is too small or the features are "
                "unscaled. Rising or NaN means too large. Oscillating means too large relative to the "
                "curvature. Falling then plateauing means you may need a schedule or more capacity.",
            },
            {
                "q": "Why does momentum help?",
                "a": "It accumulates a moving average of past gradients, so a consistent direction builds "
                "speed and a jittery one cancels out. It also lets you use a larger effective step in "
                "directions that are consistently downhill, which speeds up long thin valleys.",
            },
        ],
    },
    {
        "number": 4,
        "slug": "logistic-regression",
        "part": 1,
        "title": "Logistic regression and the sigmoid",
        "blurb": "Squash a line into a probability, train it with cross-entropy, and the gradient stays simple.",
        "why": [
            "Linear regression predicts a number; classification needs a probability, and a probability has "
            "to live between 0 and 1. The sigmoid does the squashing, and the loss changes from squared "
            "error to cross-entropy.",
            "The payoff is that the gradient stays beautifully simple: prediction minus truth, exactly as "
            "in linear regression. That is why logistic regression is the natural next step - the loss "
            "changed, the shape of the update did not.",
        ],
        "sections": [
            {
                "heading": "The model and the decision boundary",
                "paragraphs": [
                    "z = w.x + b is any real number; p = sigmoid(z) is a probability. The decision boundary "
                    "is where p crosses 0.5, which is where z = 0, which is a straight line. Everything the "
                    "model learns is that line's position and angle.",
                ],
                "code": (
                    "p = 1 / (1 + exp(-z))\n"
                    "loss = -(y * log(p) + (1 - y) * log(1 - p))\n"
                    "d loss / d z = (p - y)          # that is the whole gradient"
                ),
            },
            {
                "heading": "The vanishing gradient, in miniature",
                "paragraphs": [
                    "The sigmoid's derivative is p(1-p), which goes to zero at both extremes. A unit that is "
                    "saturated stops learning. This is the vanishing gradient problem in miniature, and it is "
                    "why hidden layers in modern networks use ReLU instead.",
                ],
            },
            {
                "heading": "Why not just use a threshold on linear regression?",
                "paragraphs": [
                    "You could, and it would be worse. Squared error on a 0/1 target is not a proper "
                    "probability model, its gradient shrinks exactly when the model is confidently wrong, and "
                    "it gives you no calibrated probability - only a number you have to threshold by hand.",
                ],
            },
        ],
        "results": [
            "loss fell from 0.6931 to 0.1054 over 400 steps",
            "training accuracy 0.955 on two overlapping blobs",
            "0.6931 is log(2) - the loss of a model that knows nothing",
        ],
        "demo": "python -m mlconcepts.foundations.logistic_regression",
        "plot": "04-logistic-regression.png",
        "interview": [
            {
                "q": "Why does logistic regression use cross-entropy instead of squared error?",
                "a": "Because squared error's gradient with respect to z is multiplied by the sigmoid "
                "derivative p(1-p), which vanishes when the model is confident. A confidently wrong model "
                "then gets an almost zero gradient. Cross-entropy has no such factor, so being wrong hard "
                "produces a large correction.",
            },
            {
                "q": "What is the decision boundary of logistic regression?",
                "a": "A straight line (a hyperplane in higher dimensions) - the set of points where w.x + b "
                "= 0, which is where the predicted probability is exactly 0.5.",
            },
            {
                "q": "How do you read the coefficients?",
                "a": "A one-unit increase in feature i multiplies the odds by exp(w_i). So a positive weight "
                "increases the odds, and exp(w_i) is the odds ratio. That interpretability is why logistic "
                "regression is still used in regulated settings where you must justify a decision.",
            },
            {
                "q": "What is the vanishing gradient problem?",
                "a": "When the derivative of the activation is near zero, gradients shrink as they flow "
                "backwards and early layers learn very slowly. Sigmoid saturates at both ends; ReLU does "
                "not saturate for positive inputs, which is why it became the default.",
            },
        ],
    },
    {
        "number": 5,
        "slug": "loss-functions",
        "part": 1,
        "title": "Loss functions: why cross-entropy",
        "blurb": "The same model trained with two different gradients - and a table showing why one of them stalls.",
        "why": [
            "People use cross-entropy for classification because that is what you do. This page shows the "
            "actual reason with numbers: squared error stops caring exactly when the model is most wrong.",
            "That is not a philosophical difference. It is a factor of p(1-p) in the gradient, and it is why "
            "training a classifier with MSE is slow and sometimes simply stalls.",
        ],
        "sections": [
            {
                "heading": "The one-line difference",
                "paragraphs": [
                    "Both losses are functions of the predicted probability p, given the true label y. "
                    "Differentiate each with respect to the pre-sigmoid value z - because that is what drives "
                    "the weights - and the difference appears immediately.",
                ],
                "code": (
                    "dL/dz  (squared error) = 2(p - y) * p * (1 - p)\n"
                    "dL/dz  (cross-entropy) =     (p - y)\n\n"
                    "# the extra p(1-p) is the sigmoid derivative, and it goes to ZERO\n"
                    "# when the model is confident."
                ),
            },
            {
                "heading": "What the table shows",
                "paragraphs": [
                    "With the true label 1 and the model predicting 0.01, the cross-entropy gradient is "
                    "-0.99 and the squared-error gradient is -0.0196 - about fifty times smaller. The model "
                    "that is most wrong gets the weakest correction.",
                ],
            },
            {
                "heading": "Choosing a loss",
                "bullets": [
                    "Regression: MSE for well-behaved errors, MAE or Huber when outliers should not dominate.",
                    "Binary classification: binary cross-entropy.",
                    "Multi-class: categorical cross-entropy, after a softmax.",
                    "Ranking: pairwise or listwise losses, because absolute values do not matter, order does.",
                    "Imbalanced classes: weighted cross-entropy or focal loss, so the rare class still counts.",
                    "When in doubt: the loss must match how you will be judged. Optimising the wrong thing "
                    "gives you a model that is excellent at something nobody cares about.",
                ],
            },
        ],
        "results": [
            "at p=0.01, true label 1:  CE gradient -0.9900, MSE gradient -0.0196  (50x)",
            "at p=0.99, true label 1:  CE gradient -0.0100, MSE gradient -0.0002  (50x)",
            "same model, 300 steps: CE final loss 0.1068, MSE final loss 0.1321",
        ],
        "demo": "python -m mlconcepts.foundations.loss_functions",
        "plot": "05-loss-functions.png",
        "interview": [
            {
                "q": "Derive why cross-entropy has a cleaner gradient than MSE for classification.",
                "a": "With p = sigmoid(z), dp/dz = p(1-p). The chain rule puts that factor into the MSE "
                "gradient, giving 2(p-y)p(1-p), which vanishes when p is near 0 or 1. Cross-entropy is "
                "defined so that the p(1-p) cancels, leaving (p-y).",
            },
            {
                "q": "When would you prefer MAE over MSE?",
                "a": "When the target has outliers you do not want to dominate training. MSE squares the "
                "error, so one point far off can outweigh hundreds of good ones. MAE is linear in the error "
                "and is more robust - at the cost of a non-smooth gradient at zero, which is why Huber "
                "exists as a compromise.",
            },
            {
                "q": "What is focal loss for?",
                "a": "Class imbalance. It multiplies cross-entropy by a factor that down-weights examples "
                "the model already classifies confidently, so training focuses on the hard, usually rare, "
                "cases. It was introduced for dense object detection where background dominates.",
            },
            {
                "q": "Can you train a classifier with MSE anyway?",
                "a": "It will often converge on easy problems - and this demo shows the gap is small when "
                "the model does not start confidently wrong. It becomes a real problem with bad "
                "initialisation, imbalanced data or deeper networks, which is exactly when you least want "
                "a loss that stops caring.",
            },
        ],
    },
    {
        "number": 6,
        "slug": "overfitting",
        "part": 1,
        "title": "Overfitting, splits and regularisation",
        "blurb": "A degree-15 polynomial scores 0.008 on training data and 480 on test data. Then ridge fixes it.",
        "why": [
            "A model that is perfect on the data it trained on is worthless if it is wrong on everything "
            "else. This page makes that concrete, and the U-shape it produces is the most important picture "
            "in machine learning.",
            "Underfitting is bias: the model's assumptions are too rigid. Overfitting is variance: the "
            "answer moves too much when the data moves. Regularisation trades a little bias for a lot less "
            "variance.",
        ],
        "sections": [
            {
                "heading": "The experiment",
                "paragraphs": [
                    "Twenty noisy points from a smooth curve, and polynomials of increasing degree. Degree 1 "
                    "cannot capture the shape. Degree 3 captures the signal and ignores the noise. Degree 15 "
                    "memorises the noise - its weights reach 1.3e10 and its test error explodes.",
                ],
                "code": (
                    "degree   train MSE    test MSE\n"
                    "     1      0.3022      0.2856   underfit\n"
                    "     3      0.0263      0.0900   good\n"
                    "     9      0.0137      0.1001   OVERFIT\n"
                    "    15      0.0084    479.8545   OVERFIT"
                ),
            },
            {
                "heading": "The two fixes",
                "paragraphs": [
                    "Split the data, and never judge a model on data it has seen - keep a third untouched set "
                    "for the final report. Then regularise: add a penalty on the size of the weights so a "
                    "large weight has to earn its place.",
                    "Ridge (L2) shrinks weights smoothly. Lasso (L1) can drive them exactly to zero, so it "
                    "doubles as feature selection. Elastic net mixes the two.",
                ],
                "code": (
                    "lambda   train MSE    test MSE    max |w|\n"
                    " 0e+00      0.0084    479.8545   13435906759\n"
                    " 1e-04      0.0196      0.0736         17.14\n"
                    " 1e-02      0.0816      0.1145          5.96"
                ),
            },
            {
                "heading": "The other regularisers",
                "bullets": [
                    "More data. The most reliable fix, and the one nobody wants to hear.",
                    "Dropout - randomly zero units during training so no unit becomes a crutch.",
                    "Early stopping - stop when validation error stops improving.",
                    "Data augmentation - more effective data without collecting more.",
                    "Simpler model - fewer parameters is still a legitimate answer.",
                ],
            },
        ],
        "results": [
            "degree 15: train MSE 0.0084, test MSE 479.8545",
            "degree 15 weights reached 1.3e10",
            "ridge lambda=1e-4 cut test MSE from 479.85 to 0.0736",
        ],
        "demo": "python -m mlconcepts.foundations.overfitting",
        "plot": "06-overfitting.png",
        "interview": [
            {
                "q": "How do you detect overfitting?",
                "a": "Compare training error with error on a held-out set. A large gap - low training, high "
                "validation - is overfitting. If both are high, you are underfitting instead, and more "
                "capacity is what you need, not less.",
            },
            {
                "q": "Why not just use the test set to tune?",
                "a": "Because then it stops being a test set. Tuning against it leaks information and you "
                "will report an optimistic number. Split three ways: train to fit, validation to choose, "
                "test once at the end.",
            },
            {
                "q": "How does L2 regularisation work?",
                "a": "It adds lambda times the sum of squared weights to the loss. The gradient therefore "
                "includes 2*lambda*w, which pulls every weight toward zero a little on each step. Large "
                "weights - which are what a model uses to fit noise - get pulled hardest.",
            },
            {
                "q": "Bias and variance, briefly?",
                "a": "Bias is error from wrong assumptions: the model cannot represent the truth, so it "
                "underfits everywhere. Variance is error from sensitivity to the particular sample: the "
                "model fits noise, so it changes a lot if you resample. Regularisation buys lower variance "
                "at the cost of a little bias.",
            },
        ],
    },
    {
        "number": 7,
        "slug": "kmeans",
        "part": 1,
        "title": "k-means clustering",
        "blurb": "Four lines of algorithm, k-means++ so it starts sensibly, and an elbow plot to choose k.",
        "why": [
            "Everything before this was supervised - we had labels. Most data does not, and k-means is the "
            "simplest useful thing you can do without them.",
            "It is also a good lesson in local minima: assign and update each strictly reduce the objective, "
            "so it always converges - but only to a nearby answer, which is why the initialisation matters "
            "so much.",
        ],
        "sections": [
            {
                "heading": "The algorithm",
                "paragraphs": [
                    "Guess k centres, assign every point to its nearest, move each centre to the mean of its "
                    "members, repeat until nothing moves. That is all of it.",
                ],
                "code": (
                    "for _ in range(max_steps):\n"
                    "    labels  = argmin(distance(points, centres))   # assign\n"
                    "    centres = mean(points[labels == j])           # update\n"
                    "    if labels did not change: stop"
                ),
            },
            {
                "heading": "Why k-means++ exists",
                "paragraphs": [
                    "Random initialisation can put two centres in the same blob and leave a real cluster "
                    "empty. k-means++ picks the first centre at random and each next one with probability "
                    "proportional to its squared distance from the centres so far, so the starts spread out. "
                    "A few lines of code, dramatically fewer bad runs.",
                ],
            },
            {
                "heading": "Where k-means fails",
                "bullets": [
                    "It assumes clusters are round and similar in size - give it two crescents and it cuts "
                    "them down the middle. Use DBSCAN or a Gaussian mixture for shapes.",
                    "It needs k in advance. The elbow plot is a heuristic, not a proof.",
                    "It is sensitive to scale, so standardise first.",
                    "Every point is assigned to a cluster, so outliers drag centres toward themselves.",
                    "It is not deterministic: different seeds give different answers, so always report the "
                    "inertia and try several runs.",
                ],
            },
        ],
        "results": [
            "recovered all 3 true centres, max error 0.159 - without ever seeing them",
            "converged in 1 step from a k-means++ start",
            "elbow: k=1 6704, k=2 2076, k=3 181, k=4 158",
        ],
        "demo": "python -m mlconcepts.foundations.kmeans",
        "plot": "07-kmeans.png",
        "interview": [
            {
                "q": "Why does k-means converge at all?",
                "a": "The assign step cannot increase the total squared distance (each point moves to its "
                "nearest centre), and the update step cannot increase it either (the mean minimises squared "
                "distance for a fixed assignment). A non-increasing bounded sequence must converge - but to "
                "a local minimum.",
            },
            {
                "q": "How do you choose k?",
                "a": "The elbow plot looks for the bend in inertia as k grows. The silhouette score measures "
                "how much better each point fits its own cluster than the next best. Domain knowledge beats "
                "both when you have it.",
            },
            {
                "q": "k-means versus DBSCAN?",
                "a": "k-means needs k, assumes round equal clusters, and assigns every point. DBSCAN finds "
                "clusters of arbitrary shape from density, needs no k, and labels sparse points as noise - "
                "but it is sensitive to its epsilon and struggles when clusters have very different densities.",
            },
            {
                "q": "Why is k-means++ better than random initialisation?",
                "a": "Because random starts can be clustered together, leaving some true clusters with no "
                "centre - and once two centres share a blob, the update keeps them there. Spreading the "
                "starts by squared distance makes that much less likely.",
            },
        ],
    },
    {
        "number": 8,
        "slug": "pca",
        "part": 1,
        "title": "PCA: finding the directions that matter",
        "blurb": "Five features, two hidden factors: one eigen-decomposition finds them and keeps 99.8% of the variance.",
        "why": [
            "Data with many correlated features is painful to plot and wasteful to model. PCA finds the "
            "directions along which the data actually varies, so you can throw the rest away.",
            "It is also the cleanest example of a method with no iteration and no learning rate: centre the "
            "data, take the covariance, eigen-decompose it, keep the top components. That is the whole thing.",
        ],
        "sections": [
            {
                "heading": "Four steps",
                "bullets": [
                    "Centre and standardise, so spread is measured in comparable units.",
                    "Compute the covariance matrix - which features move together.",
                    "Take its eigenvectors (the directions of pure variance) and eigenvalues (how much).",
                    "Project onto the top few components, and keep the rest as the reconstruction error.",
                ],
                "code": (
                    "covariance = np.cov(standardised, rowvar=False)\n"
                    "eigenvalues, eigenvectors = np.linalg.eigh(covariance)\n"
                    "order = np.argsort(eigenvalues)[::-1]        # eigh returns ASCENDING\n"
                    "components = eigenvectors[:, order][:, :n_components]\n"
                    "projected  = standardised @ components"
                ),
            },
            {
                "heading": "What you get",
                "paragraphs": [
                    "A scree plot so you can say 'two of five components carry 99.8% of the signal'. "
                    "Compression from five numbers per row to two. A projection you can actually look at. "
                    "And the reconstruction error, which is the honest measure of what you threw away.",
                ],
            },
            {
                "heading": "When not to use it",
                "bullets": [
                    "Variance is not usefulness. If the label depends on a small-variance direction, PCA will "
                    "delete exactly the thing you needed. Try supervised alternatives (LDA) or feature "
                    "selection instead.",
                    "It only finds LINEAR structure. A curved manifold needs something else (t-SNE, UMAP).",
                    "It is sensitive to scale, so standardise unless the units are genuinely comparable.",
                    "The components are combinations of your original features, so you lose interpretability - "
                    "which is often the whole reason you wanted fewer features.",
                ],
            },
        ],
        "results": [
            "PC1 55.4%, PC2 44.4% - cumulative 99.8%",
            "5 numbers per row reduced to 2",
            "reconstruction error 0.0081 of 5.3281 total variance",
        ],
        "demo": "python -m mlconcepts.foundations.pca",
        "plot": "08-pca.png",
        "interview": [
            {
                "q": "Why are the principal components the eigenvectors of the covariance matrix?",
                "a": "Maximising the projected variance subject to a unit-norm direction is a constrained "
                "optimisation whose solution is the eigenvector of the covariance matrix with the largest "
                "eigenvalue. The eigenvalue IS the variance along that direction, so the components come out "
                "already ranked by how much they explain.",
            },
            {
                "q": "How do you decide how many components to keep?",
                "a": "Cumulative explained variance is the usual rule - keep enough to reach 90-95%. Or keep "
                "the components whose eigenvalue is above 1 (the Kaiser rule), or use the elbow in the scree "
                "plot. All three are heuristics; the real question is what the downstream task needs.",
            },
            {
                "q": "PCA versus feature selection?",
                "a": "PCA builds NEW features as combinations of the old ones, which usually maximises "
                "compression but destroys interpretability. Feature selection keeps the original columns, so "
                "you can still explain the model - at the cost of keeping some redundancy.",
            },
            {
                "q": "Why standardise before PCA?",
                "a": "PCA maximises variance, and variance depends on units. A feature measured in rupees "
                "has a far bigger variance than one measured as a ratio, so it would dominate the components "
                "for no reason other than its scale.",
            },
        ],
    },
    {
        "number": 9,
        "slug": "neural-network",
        "part": 1,
        "title": "A neural network from scratch",
        "blurb": "Forward pass, backpropagation by hand, 33 parameters - and 60% becomes 99.8% on a problem a line cannot solve.",
        "why": [
            "This is the centrepiece. Everything before it was a model whose gradient you could write down "
            "by hand; here the gradient is computed by the chain rule applied backwards through a network.",
            "That is exactly what PyTorch does for you, and exactly what you should be able to explain. If "
            "you can read the backward pass on this page, you understand backpropagation.",
        ],
        "sections": [
            {
                "heading": "The network",
                "code": (
                    "Z1 = X @ W1 + b1\n"
                    "A1 = tanh(Z1)\n"
                    "Z2 = A1 @ W2 + b2\n"
                    "A2 = sigmoid(Z2)\n"
                    "loss = binary cross-entropy"
                ),
            },
            {
                "heading": "The backward pass is the forward pass in reverse",
                "paragraphs": [
                    "Read it slowly. Each line takes the gradient arriving from the layer above and multiplies "
                    "it by the local derivative of the function that layer applied. That is the chain rule, "
                    "and it is the whole of backpropagation.",
                ],
                "code": (
                    "dZ2 = (A2 - y) / n            # same (prediction - truth) as before\n"
                    "dW2 = A1.T @ dZ2\n"
                    "db2 = sum(dZ2)\n"
                    "dA1 = dZ2 @ W2.T\n"
                    "dZ1 = dA1 * (1 - A1**2)       # the tanh derivative\n"
                    "dW1 = X.T @ dZ1\n"
                    "db1 = sum(dZ1)"
                ),
            },
            {
                "heading": "Why the hidden layer matters",
                "paragraphs": [
                    "The data is a ring: a blob surrounded by a circle. No straight line separates them, so "
                    "logistic regression is stuck near chance however long you train it. One hidden layer "
                    "with a non-linear activation bends the boundary into a shape that wraps around.",
                    "That gap - 60% to 99.8% - is the reason deep learning exists, and it is one layer.",
                ],
            },
            {
                "heading": "The details that are easy to get wrong",
                "bullets": [
                    "Zero initialisation is a bug: every hidden unit computes the same thing and gets the "
                    "same gradient, so they stay identical forever.",
                    "Scale the initial weights by 1/sqrt(fan_in) so the pre-activations stay where tanh still "
                    "has a gradient.",
                    "The loss must be divided by the batch size, or the effective learning rate depends on "
                    "the batch size.",
                    "You need the tanh derivative (1 - a^2), not the tanh itself, in the backward pass.",
                ],
            },
        ],
        "results": [
            "architecture 2 -> 8 (tanh) -> 1 (sigmoid), 33 parameters",
            "logistic regression: 60.5% accuracy",
            "this network: 99.8% accuracy",
            "loss 0.6876 -> 0.0096",
        ],
        "demo": "python -m mlconcepts.foundations.neural_network",
        "plot": "09-neural-network.png",
        "interview": [
            {
                "q": "Explain backpropagation in two sentences.",
                "a": "It is the chain rule applied to a computation graph, computed from the output backwards. "
                "At each step you take the gradient arriving from the layer above and multiply it by the "
                "local derivative of the function that layer applied, which gives you the gradient with "
                "respect to that layer's inputs and parameters.",
            },
            {
                "q": "Why is zero initialisation a bad idea?",
                "a": "Every hidden unit in a layer would compute the same output and receive the same "
                "gradient, so they would update identically and stay identical forever. The layer would "
                "behave like a single unit no matter how wide it is. Random init breaks the symmetry.",
            },
            {
                "q": "Why tanh here and ReLU in practice?",
                "a": "tanh is smooth and bounded, which makes it easy to reason about and to differentiate by "
                "hand - good for a demo. ReLU does not saturate for positive inputs, so gradients do not "
                "vanish, and it is much cheaper to compute. That is why ReLU and its variants are the "
                "default in deep networks.",
            },
            {
                "q": "What does the hidden layer actually do?",
                "a": "Each hidden unit learns a direction in input space and fires when the input is on one "
                "side of it. The output layer then combines those into a decision boundary that is a union "
                "of half-spaces - which can wrap around a ring, where a single half-space cannot.",
            },
            {
                "q": "Why divide the gradient by the batch size?",
                "a": "Because the loss is defined as the MEAN over the batch. Without the division, a larger "
                "batch would produce a proportionally larger gradient, so the effective learning rate would "
                "silently depend on the batch size.",
            },
        ],
    },
]
