/* ============================================================================
   math-foundations.js - Part 1's algorithms, as pure functions.

   NO DOM, NO GLOBALS, NO RENDERING. Everything here takes plain arrays and
   returns plain objects, which is what lets it be run and asserted under Node
   (see tests/js/math.test.js) against the same properties the Python demos
   assert in tests/test_foundations.py.

   It is a second implementation of mlconcepts/foundations/, deliberately: the
   browser cannot run NumPy, and re-writing the maths is the only way the page
   can respond to a slider. Two implementations of the same algorithm is a
   liability, so the tests keep them honest.
   ========================================================================== */

(function () {
  'use strict';

  /* ------------------------------------------------------------------ random */

  /** mulberry32 - a small, fast, seeded PRNG. Same seed, same numbers, always. */
  function rng(seed) {
    var a = (seed >>> 0) || 1;
    return function () {
      a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /** Box-Muller, so the toy datasets have the same shape as the Python ones. */
  function normal(r) {
    var u = 1 - r();
    var v = r();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  /* -------------------------------------------------------------------- data */

  function makeLinear(n, noise, seed) {
    n = n || 120; noise = noise == null ? 0.6 : noise;
    var r = rng(seed || 0), x = [], y = [];
    for (var i = 0; i < n; i++) {
      var xi = -3 + r() * 6;
      x.push(xi);
      y.push(2 * xi + 1 + normal(r) * noise);
    }
    return { x: x, y: y };
  }

  function makeClassification(n, seed) {
    n = n || 200;
    var r = rng(seed || 0), half = Math.floor(n / 2);
    var X = [], y = [], i;
    for (i = 0; i < half; i++) {
      X.push([-1.2 + normal(r) * 0.85, -1.0 + normal(r) * 0.85]); y.push(0);
    }
    for (i = half; i < n; i++) {
      X.push([1.2 + normal(r) * 0.85, 1.0 + normal(r) * 0.85]); y.push(1);
    }
    return { X: X, y: y };
  }

  /** An inner blob inside a ring - the problem a straight line cannot solve. */
  function makeRing(n, seed) {
    n = n || 400;
    var r = rng(seed || 0), inner = Math.floor(n / 2);
    var X = [], y = [], i;
    for (i = 0; i < inner; i++) {
      var ri = normal(r) * 0.35, ai = r() * 2 * Math.PI;
      X.push([ri * Math.cos(ai), ri * Math.sin(ai)]); y.push(0);
    }
    for (i = inner; i < n; i++) {
      var ro = 1.6 + normal(r) * 0.25, ao = r() * 2 * Math.PI;
      X.push([ro * Math.cos(ao), ro * Math.sin(ao)]); y.push(1);
    }
    return { X: X, y: y };
  }

  function makeClusters(k, perCluster, seed) {
    k = k || 3; perCluster = perCluster || 60;
    var r = rng(seed || 0);
    var centres = [], X = [];
    for (var c = 0; c < k; c++) { centres.push([-6 + r() * 12, -6 + r() * 12]); }
    for (var i = 0; i < k; i++) {
      for (var j = 0; j < perCluster; j++) {
        X.push([centres[i][0] + normal(r) * 0.7, centres[i][1] + normal(r) * 0.7]);
      }
    }
    return { X: X, centres: centres };
  }

  function makeCorrelated(n, seed) {
    n = n || 300;
    var r = rng(seed || 0);
    var mixing = [[1.0, 0.6], [0.9, -0.4], [0.2, 1.1], [-0.7, 0.9], [0.5, 0.5]];
    var X = [];
    for (var i = 0; i < n; i++) {
      var a = normal(r), b = normal(r);
      var row = [];
      for (var f = 0; f < 5; f++) {
        row.push(a * mixing[f][0] + b * mixing[f][1] + normal(r) * 0.05);
      }
      X.push(row);
    }
    return X;
  }

  /* ------------------------------------------------------------------ 1. loop */

  /** The one-parameter model: predict a constant, minimise MSE. Lands on the mean. */
  function fitMean(y, learningRate, steps) {
    learningRate = learningRate == null ? 0.1 : learningRate;
    steps = steps || 200;
    var w = 0, history = [];
    for (var s = 0; s < steps; s++) {
      var sum = 0, i;
      for (i = 0; i < y.length; i++) { sum += (w - y[i]) * (w - y[i]); }
      history.push(sum / y.length);
      var grad = 0;
      for (i = 0; i < y.length; i++) { grad += 2 * (w - y[i]); }
      grad /= y.length;
      w -= learningRate * grad;
    }
    return { w: w, history: history };
  }

  function mean(a) {
    var s = 0;
    for (var i = 0; i < a.length; i++) { s += a[i]; }
    return s / a.length;
  }

  /* ------------------------------------------------------- 2. linear regression */

  /**
   * Gradient descent for y = w*x + b.
   *
   * `initial` lets the caller resume from a previous state, which is what the
   * interactive demo uses to train a few steps per animation frame.
   */
  function fitLinear(x, y, learningRate, steps, recordEvery, initial) {
    learningRate = learningRate == null ? 0.05 : learningRate;
    steps = steps || 600;
    recordEvery = recordEvery || 1;
    var w = initial ? initial.w : 0;
    var b = initial ? initial.b : 0;
    var history = [], n = x.length;
    for (var s = 0; s < steps; s++) {
      var gw = 0, gb = 0, loss = 0;
      for (var i = 0; i < n; i++) {
        var err = (w * x[i] + b) - y[i];
        loss += err * err;
        gw += err * x[i];
        gb += err;
      }
      loss /= n;
      gw = 2 * gw / n;
      gb = 2 * gb / n;
      w -= learningRate * gw;
      b -= learningRate * gb;
      if (s % recordEvery === 0) { history.push(loss); }
    }
    return { w: w, b: b, history: history, loss: history[history.length - 1] };
  }

  function rSquared(y, prediction) {
    var m = mean(y), ssRes = 0, ssTot = 0;
    for (var i = 0; i < y.length; i++) {
      ssRes += (y[i] - prediction[i]) * (y[i] - prediction[i]);
      ssTot += (y[i] - m) * (y[i] - m);
    }
    return 1 - ssRes / ssTot;
  }

  /* ---------------------------------------------------------- 3. learning rate */

  function gdHistory(x, y, learningRate, steps) {
    steps = steps || 40;
    var w = 0, b = 0, n = x.length, history = [];
    for (var s = 0; s < steps; s++) {
      var gw = 0, gb = 0, loss = 0, i;
      for (i = 0; i < n; i++) {
        var err = (w * x[i] + b) - y[i];
        loss += err * err; gw += err * x[i]; gb += err;
      }
      loss /= n;
      history.push(isFinite(loss) ? loss : Infinity);
      if (!isFinite(loss)) { continue; }
      w -= learningRate * (2 * gw / n);
      b -= learningRate * (2 * gb / n);
    }
    return history;
  }

  function miniBatchHistory(x, y, learningRate, batchSize, steps, seed) {
    learningRate = learningRate == null ? 0.05 : learningRate;
    batchSize = batchSize || 8; steps = steps || 40;
    var r = rng(seed || 0), w = 0, b = 0, n = x.length, history = [];
    for (var s = 0; s < steps; s++) {
      var gw = 0, gb = 0, count = 0, i, idx;
      for (i = 0; i < batchSize; i++) {
        idx = Math.floor(r() * n);
        var err = (w * x[idx] + b) - y[idx];
        gw += err * x[idx]; gb += err; count++;
      }
      w -= learningRate * (2 * gw / count);
      b -= learningRate * (2 * gb / count);
      var full = 0;
      for (i = 0; i < n; i++) {
        var e2 = (w * x[i] + b) - y[i];
        full += e2 * e2;
      }
      history.push(full / n);
    }
    return history;
  }

  /* ------------------------------------------------------ 4. logistic regression */

  function sigmoid(z) {
    if (z >= 0) { return 1 / (1 + Math.exp(-z)); }
    var e = Math.exp(z);
    return e / (1 + e);
  }

  /** Logistic regression by gradient descent on cross-entropy. */
  function fitLogistic(X, y, learningRate, steps, recordEvery, initial) {
    learningRate = learningRate == null ? 0.5 : learningRate;
    steps = steps || 400;
    recordEvery = recordEvery || 10;
    var n = X.length, d = X[0].length;
    var w = initial ? initial.w.slice() : new Array(d).fill(0);
    var b = initial ? initial.b : 0;
    var history = [];
    for (var s = 0; s < steps; s++) {
      var gw = new Array(d).fill(0), gb = 0, loss = 0;
      for (var i = 0; i < n; i++) {
        var z = b, k;
        for (k = 0; k < d; k++) { z += w[k] * X[i][k]; }
        var p = sigmoid(z);
        var eps = 1e-12;
        var pc = Math.min(1 - eps, Math.max(eps, p));
        loss += -(y[i] * Math.log(pc) + (1 - y[i]) * Math.log(1 - pc));
        var err = p - y[i];
        for (k = 0; k < d; k++) { gw[k] += err * X[i][k]; }
        gb += err;
      }
      loss /= n;
      for (var j = 0; j < d; j++) { w[j] -= learningRate * gw[j] / n; }
      b -= learningRate * gb / n;
      if (s % recordEvery === 0) { history.push(loss); }
    }
    return { w: w, b: b, history: history, loss: history[history.length - 1] };
  }

  function predictLogistic(model, X) {
    var out = [];
    for (var i = 0; i < X.length; i++) {
      var z = model.b;
      for (var k = 0; k < X[i].length; k++) { z += model.w[k] * X[i][k]; }
      out.push(sigmoid(z));
    }
    return out;
  }

  function accuracy(probs, y) {
    var hit = 0;
    for (var i = 0; i < y.length; i++) {
      if ((probs[i] >= 0.5 ? 1 : 0) === y[i]) { hit++; }
    }
    return hit / y.length;
  }

  /** Standardise columns - gradient descent on un-scaled inputs is slow. */
  function standardise(X) {
    var n = X.length, d = X[0].length, out = [], i, k;
    var mu = new Array(d).fill(0), sd = new Array(d).fill(0);
    for (i = 0; i < n; i++) { for (k = 0; k < d; k++) { mu[k] += X[i][k]; } }
    for (k = 0; k < d; k++) { mu[k] /= n; }
    for (i = 0; i < n; i++) {
      for (k = 0; k < d; k++) { sd[k] += (X[i][k] - mu[k]) * (X[i][k] - mu[k]); }
    }
    for (k = 0; k < d; k++) { sd[k] = Math.sqrt(sd[k] / n) || 1; }
    for (i = 0; i < n; i++) {
      var row = [];
      for (k = 0; k < d; k++) { row.push((X[i][k] - mu[k]) / sd[k]); }
      out.push(row);
    }
    return out;
  }

  /* ------------------------------------------------------------ 5. loss functions */

  function ceGradZ(p, y) { return p - y; }

  function mseGradZ(p, y) { return 2 * (p - y) * p * (1 - p); }

  /* -------------------------------------------------------------- 6. overfitting */

  function makeOverfitData(n, noise, seed) {
    n = n || 20; noise = noise == null ? 0.25 : noise;
    var r = rng(seed || 0), x = [], y = [], xt = [], yt = [];
    for (var i = 0; i < n; i++) {
      var xi = r();
      x.push(xi);
      y.push(Math.sin(2 * Math.PI * xi) + normal(r) * noise);
    }
    for (var j = 0; j < 400; j++) {
      var xj = r();
      xt.push(xj);
      yt.push(Math.sin(2 * Math.PI * xj) + normal(r) * noise);
    }
    return { x: x, y: y, xTest: xt, yTest: yt };
  }

  function polyFeatures(x, degree) {
    var out = [];
    for (var i = 0; i < x.length; i++) {
      var row = [];
      for (var d = 0; d <= degree; d++) { row.push(Math.pow(x[i], d)); }
      out.push(row);
    }
    return out;
  }

  /** Gaussian elimination with partial pivoting. Small systems only. */
  function solve(A, b) {
    var n = b.length, M = [], i, j, k;
    for (i = 0; i < n; i++) { M.push(A[i].slice().concat([b[i]])); }
    for (k = 0; k < n; k++) {
      var pivot = k;
      for (i = k + 1; i < n; i++) {
        if (Math.abs(M[i][k]) > Math.abs(M[pivot][k])) { pivot = i; }
      }
      var tmp = M[k]; M[k] = M[pivot]; M[pivot] = tmp;
      var pv = M[k][k];
      if (Math.abs(pv) < 1e-14) { pv = pv >= 0 ? 1e-14 : -1e-14; }
      for (j = k; j <= n; j++) { M[k][j] /= pv; }
      for (i = 0; i < n; i++) {
        if (i === k) { continue; }
        var factor = M[i][k];
        if (factor === 0) { continue; }
        for (j = k; j <= n; j++) { M[i][j] -= factor * M[k][j]; }
      }
    }
    return M.map(function (row) { return row[n]; });
  }

  /**
   * Least squares on polynomial features, with optional ridge.
   *
   * The bias column is never penalised - shrinking the intercept would just
   * shift the whole curve, which is not what regularisation is for.
   */
  function fitPoly(X, y, lambda) {
    lambda = lambda || 0;
    var n = X.length, p = X[0].length, i, j, k;
    var XtX = [], Xty = new Array(p).fill(0);
    for (i = 0; i < p; i++) { XtX.push(new Array(p).fill(0)); }
    for (i = 0; i < n; i++) {
      for (j = 0; j < p; j++) {
        Xty[j] += X[i][j] * y[i];
        for (k = 0; k < p; k++) { XtX[j][k] += X[i][j] * X[i][k]; }
      }
    }
    for (j = 0; j < p; j++) {
      XtX[j][j] += (j === 0 ? 0 : lambda) + 1e-9; // a whisker of stability at lambda=0
    }
    return solve(XtX, Xty);
  }

  function predictPoly(coef, X) {
    var out = [];
    for (var i = 0; i < X.length; i++) {
      var s = 0;
      for (var j = 0; j < coef.length; j++) { s += coef[j] * X[i][j]; }
      out.push(s);
    }
    return out;
  }

  function mse(a, b) {
    var s = 0;
    for (var i = 0; i < a.length; i++) { s += (a[i] - b[i]) * (a[i] - b[i]); }
    return s / a.length;
  }

  /* ------------------------------------------------------------------ 7. k-means */

  function kmeansPlusPlus(X, k, r) {
    var centres = [X[Math.floor(r() * X.length)].slice()];
    while (centres.length < k) {
      var dists = X.map(function (p) {
        var best = Infinity;
        for (var c = 0; c < centres.length; c++) {
          var d = dist2(p, centres[c]);
          if (d < best) { best = d; }
        }
        return best;
      });
      var total = dists.reduce(function (a, b) { return a + b; }, 0);
      if (total === 0) { centres.push(X[Math.floor(r() * X.length)].slice()); continue; }
      var target = r() * total, acc = 0, pick = X.length - 1;
      for (var i = 0; i < dists.length; i++) {
        acc += dists[i];
        if (acc >= target) { pick = i; break; }
      }
      centres.push(X[pick].slice());
    }
    return centres;
  }

  function dist2(a, b) {
    var s = 0;
    for (var i = 0; i < a.length; i++) { s += (a[i] - b[i]) * (a[i] - b[i]); }
    return s;
  }

  function assignClusters(X, centres) {
    return X.map(function (p) {
      var best = 0, bestD = Infinity;
      for (var c = 0; c < centres.length; c++) {
        var d = dist2(p, centres[c]);
        if (d < bestD) { bestD = d; best = c; }
      }
      return best;
    });
  }

  function updateCentres(X, labels, k) {
    var sums = [], counts = new Array(k).fill(0), i;
    for (i = 0; i < k; i++) { sums.push(new Array(X[0].length).fill(0)); }
    for (i = 0; i < X.length; i++) {
      var c = labels[i];
      counts[c]++;
      for (var d = 0; d < X[i].length; d++) { sums[c][d] += X[i][d]; }
    }
    for (i = 0; i < k; i++) {
      if (counts[i] === 0) { continue; }
      for (var d2 = 0; d2 < sums[i].length; d2++) { sums[i][d2] /= counts[i]; }
    }
    return sums;
  }

  function inertia(X, labels, centres) {
    var s = 0;
    for (var i = 0; i < X.length; i++) { s += dist2(X[i], centres[labels[i]]); }
    return s;
  }

  /** One step: assign, then update. Returns the new state - the UI calls this. */
  function kmeansStep(X, centres) {
    var labels = assignClusters(X, centres);
    var next = updateCentres(X, labels, centres.length);
    return { centres: next, labels: labels, inertia: inertia(X, labels, next) };
  }

  function fitKMeans(X, k, seed, maxSteps) {
    maxSteps = maxSteps || 100;
    var r = rng(seed || 0);
    var centres = kmeansPlusPlus(X, k, r);
    var labels = assignClusters(X, centres);
    var history = [inertia(X, labels, centres)];
    for (var s = 0; s < maxSteps; s++) {
      var step = kmeansStep(X, centres);
      centres = step.centres;
      history.push(step.inertia);
      var same = true;
      for (var i = 0; i < labels.length; i++) {
        if (labels[i] !== step.labels[i]) { same = false; break; }
      }
      labels = step.labels;
      if (same) { break; }
    }
    return { centres: centres, labels: labels, history: history, inertia: history[history.length - 1] };
  }

  /* ---------------------------------------------------------------------- 8. PCA */

  function identity(n) {
    var out = [];
    for (var i = 0; i < n; i++) {
      var row = new Array(n).fill(0);
      row[i] = 1;
      out.push(row);
    }
    return out;
  }

  /** Cyclic Jacobi. For a symmetric matrix this is short, stable and enough. */
  function jacobiEigen(A, maxSweeps) {
    var n = A.length, a = A.map(function (row) { return row.slice(); });
    var v = identity(n);
    var sweeps = maxSweeps || 60;

    for (var s = 0; s < sweeps; s++) {
      var off = 0;
      for (var i = 0; i < n; i++) {
        for (var j = i + 1; j < n; j++) { off += a[i][j] * a[i][j]; }
      }
      if (off < 1e-20) { break; }

      for (var p = 0; p < n; p++) {
        for (var q = p + 1; q < n; q++) {
          if (Math.abs(a[p][q]) < 1e-18) { continue; }
          var theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
          var t = theta === 0 ? 1
            : (theta > 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
          var c = 1 / Math.sqrt(t * t + 1), sn = t * c, k;

          for (k = 0; k < n; k++) {
            var akp = a[k][p], akq = a[k][q];
            a[k][p] = c * akp - sn * akq;
            a[k][q] = sn * akp + c * akq;
          }
          for (k = 0; k < n; k++) {
            var apk = a[p][k], aqk = a[q][k];
            a[p][k] = c * apk - sn * aqk;
            a[q][k] = sn * apk + c * aqk;
          }
          for (k = 0; k < n; k++) {
            var vkp = v[k][p], vkq = v[k][q];
            v[k][p] = c * vkp - sn * vkq;
            v[k][q] = sn * vkp + c * vkq;
          }
        }
      }
    }

    var values = [];
    for (var d = 0; d < n; d++) { values.push(a[d][d]); }
    return { values: values, vectors: v };
  }

  function pcaFit(X, nComponents) {
    var n = X.length, d = X[0].length, i, j, k;
    var mu = new Array(d).fill(0), sd = new Array(d).fill(0);
    for (i = 0; i < n; i++) { for (k = 0; k < d; k++) { mu[k] += X[i][k]; } }
    for (k = 0; k < d; k++) { mu[k] /= n; }
    for (i = 0; i < n; i++) {
      for (k = 0; k < d; k++) { sd[k] += (X[i][k] - mu[k]) * (X[i][k] - mu[k]); }
    }
    for (k = 0; k < d; k++) { sd[k] = Math.sqrt(sd[k] / n) || 1; }

    var Z = X.map(function (row) {
      var out = [];
      for (var f = 0; f < d; f++) { out.push((row[f] - mu[f]) / sd[f]); }
      return out;
    });

    // covariance (d x d), sample-normalised the same way numpy's cov is
    var cov = [];
    for (i = 0; i < d; i++) { cov.push(new Array(d).fill(0)); }
    for (i = 0; i < n; i++) {
      for (j = 0; j < d; j++) {
        for (k = 0; k < d; k++) { cov[j][k] += Z[i][j] * Z[i][k]; }
      }
    }
    var denom = Math.max(1, n - 1);
    for (j = 0; j < d; j++) { for (k = 0; k < d; k++) { cov[j][k] /= denom; } }

    var eig = jacobiEigen(cov);
    var idx = eig.values.map(function (v, ii) { return ii; })
      .sort(function (a, b) { return eig.values[b] - eig.values[a]; });

    var total = eig.values.reduce(function (a, b) { return a + b; }, 0);
    var explained = idx.map(function (ii) { return eig.values[ii] / total; });

    var count = nComponents == null ? d : nComponents;
    var components = [];
    for (var c = 0; c < d; c++) { components.push(new Array(count).fill(0)); }
    for (var comp = 0; comp < count; comp++) {
      for (var row2 = 0; row2 < d; row2++) {
        components[row2][comp] = eig.vectors[row2][idx[comp]];
      }
    }

    return { mean: mu, std: sd, components: components, explained: explained, standardised: Z };
  }

  function pcaTransform(model, X) {
    var d = X[0].length, count = model.components[0].length, out = [];
    for (var i = 0; i < X.length; i++) {
      var z = [];
      for (var f = 0; f < d; f++) { z.push((X[i][f] - model.mean[f]) / model.std[f]); }
      var row = [];
      for (var c = 0; c < count; c++) {
        var s = 0;
        for (var f2 = 0; f2 < d; f2++) { s += z[f2] * model.components[f2][c]; }
        row.push(s);
      }
      out.push(row);
    }
    return out;
  }

  function pcaInverse(model, Z) {
    var d = model.mean.length, out = [];
    for (var i = 0; i < Z.length; i++) {
      var row = [];
      for (var f = 0; f < d; f++) {
        var s = 0;
        for (var c = 0; c < Z[i].length; c++) { s += Z[i][c] * model.components[f][c]; }
        row.push(s * model.std[f] + model.mean[f]);
      }
      out.push(row);
    }
    return out;
  }

  /* --------------------------------------------------------- 9. neural network */

  function initParams(nIn, nHidden, seed) {
    var r = rng(seed || 0);
    var W1 = [], b1 = new Array(nHidden).fill(0), i, j;
    for (i = 0; i < nIn; i++) {
      var row = [];
      for (j = 0; j < nHidden; j++) { row.push(normal(r) / Math.sqrt(nIn)); }
      W1.push(row);
    }
    var W2 = [], b2 = [0];
    for (j = 0; j < nHidden; j++) { W2.push([normal(r) / Math.sqrt(nHidden)]); }
    return { W1: W1, b1: b1, W2: W2, b2: b2 };
  }

  function forward(X, params) {
    var n = X.length, h = params.b1.length, i, j;
    var Z1 = [], A1 = [];
    for (i = 0; i < n; i++) {
      var z1 = [], a1 = [];
      for (j = 0; j < h; j++) {
        var s = params.b1[j];
        for (var k = 0; k < X[i].length; k++) { s += X[i][k] * params.W1[k][j]; }
        z1.push(s);
        a1.push(Math.tanh(s));
      }
      Z1.push(z1); A1.push(a1);
    }
    var Z2 = [], A2 = [];
    for (i = 0; i < n; i++) {
      var z = params.b2[0];
      for (j = 0; j < h; j++) { z += A1[i][j] * params.W2[j][0]; }
      Z2.push([z]);
      A2.push([sigmoid(z)]);
    }
    return { Z1: Z1, A1: A1, Z2: Z2, A2: A2 };
  }

  function backward(X, y, params, cache) {
    var n = X.length, h = params.b1.length, i, j, k;
    var dW1 = [], db1 = new Array(h).fill(0), dW2 = [], db2 = [0];
    for (i = 0; i < X[0].length; i++) { dW1.push(new Array(h).fill(0)); }
    for (j = 0; j < h; j++) { dW2.push([0]); }

    var dZ2 = [];
    for (i = 0; i < n; i++) { dZ2.push([(cache.A2[i][0] - y[i]) / n]); }

    for (i = 0; i < n; i++) {
      for (j = 0; j < h; j++) { dW2[j][0] += cache.A1[i][j] * dZ2[i][0]; }
      db2[0] += dZ2[i][0];
    }

    var dZ1 = [];
    for (i = 0; i < n; i++) {
      var row = [];
      for (j = 0; j < h; j++) {
        var dA1 = dZ2[i][0] * params.W2[j][0];
        row.push(dA1 * (1 - cache.A1[i][j] * cache.A1[i][j]));
      }
      dZ1.push(row);
    }

    for (i = 0; i < n; i++) {
      for (k = 0; k < X[i].length; k++) {
        for (j = 0; j < h; j++) { dW1[k][j] += X[i][k] * dZ1[i][j]; }
      }
      for (j = 0; j < h; j++) { db1[j] += dZ1[i][j]; }
    }

    return { W1: dW1, b1: db1, W2: dW2, b2: db2 };
  }

  function mlpLoss(A2, y) {
    var s = 0, eps = 1e-12;
    for (var i = 0; i < y.length; i++) {
      var p = Math.min(1 - eps, Math.max(eps, A2[i][0]));
      s += -(y[i] * Math.log(p) + (1 - y[i]) * Math.log(1 - p));
    }
    return s / y.length;
  }

  function mlpAccuracy(A2, y) {
    var hit = 0;
    for (var i = 0; i < y.length; i++) {
      if ((A2[i][0] >= 0.5 ? 1 : 0) === y[i]) { hit++; }
    }
    return hit / y.length;
  }

  function mlpTrainStep(X, y, params, learningRate) {
    var cache = forward(X, params);
    var grads = backward(X, y, params, cache);
    for (var k = 0; k < params.W1.length; k++) {
      for (var j = 0; j < params.W1[k].length; j++) {
        params.W1[k][j] -= learningRate * grads.W1[k][j];
      }
    }
    for (var j2 = 0; j2 < params.b1.length; j2++) { params.b1[j2] -= learningRate * grads.b1[j2]; }
    for (var j3 = 0; j3 < params.W2.length; j3++) {
      params.W2[j3][0] -= learningRate * grads.W2[j3][0];
    }
    params.b2[0] -= learningRate * grads.b2[0];
    return cache;
  }

  function trainMLP(X, y, nHidden, learningRate, steps, seed) {
    var params = initParams(X[0].length, nHidden, seed || 0);
    var lossHistory = [], accHistory = [], step;
    for (step = 0; step < steps; step++) {
      var cache = mlpTrainStep(X, y, params, learningRate);
      if (step % 20 === 0 || step === steps - 1) {
        lossHistory.push(mlpLoss(cache.A2, y));
        accHistory.push(mlpAccuracy(cache.A2, y));
      }
    }
    return { params: params, lossHistory: lossHistory, accHistory: accHistory };
  }

  function mlpProbability(params, x, y) {
    return forward([[x, y]], params).A2[0][0];
  }

  /* ------------------------------------------------------------------ export */

  var MATH = {
    rng: rng, normal: normal,
    makeLinear: makeLinear, makeClassification: makeClassification,
    makeRing: makeRing, makeClusters: makeClusters, makeCorrelated: makeCorrelated,
    fitMean: fitMean, mean: mean,
    fitLinear: fitLinear, rSquared: rSquared,
    gdHistory: gdHistory, miniBatchHistory: miniBatchHistory,
    sigmoid: sigmoid, fitLogistic: fitLogistic, predictLogistic: predictLogistic,
    accuracy: accuracy, standardise: standardise,
    ceGradZ: ceGradZ, mseGradZ: mseGradZ,
    makeOverfitData: makeOverfitData, polyFeatures: polyFeatures, fitPoly: fitPoly,
    predictPoly: predictPoly, mse: mse, solve: solve,
    kmeansPlusPlus: kmeansPlusPlus, kmeansStep: kmeansStep, fitKMeans: fitKMeans,
    assignClusters: assignClusters, inertia: inertia, dist2: dist2,
    jacobiEigen: jacobiEigen, pcaFit: pcaFit, pcaTransform: pcaTransform, pcaInverse: pcaInverse,
    initParams: initParams, forward: forward, backward: backward,
    mlpLoss: mlpLoss, mlpAccuracy: mlpAccuracy, mlpTrainStep: mlpTrainStep,
    trainMLP: trainMLP, mlpProbability: mlpProbability
  };

  if (typeof window !== 'undefined') { window.ML1 = MATH; }
  if (typeof module !== 'undefined' && module.exports) { module.exports = MATH; }
})();
