/* ============================================================================
   demos-foundations.js - the nine interactive Part 1 demos.

   Each one is registered by slug and installs itself into the element that the
   site build emits on the matching concept page. All the maths comes from
   window.ML1 (math-foundations.js); this file only builds controls, draws and
   animates.

   Nothing here runs under Node - the file returns immediately when there is no
   document, so the maths can be imported and tested on its own.
   ========================================================================== */

(function () {
  'use strict';

  if (typeof document === 'undefined') { return; }

  var K = window.DemoKit;
  var M = window.ML1;
  var C = K.COLORS;

  function stepAxis(plot, count, every, label) {
    plot.frame({ xTicks: 5, yTicks: 4, xLabel: label || 'step' });
  }

  /* ============================================================ 1. what-is-ml */

  K.registerDemo('what-is-ml', function (root) {
    var y = M.makeLinear(200, 0.6, 0).y;
    var truth = M.mean(y);
    var yMax = M.mean(y.map(function (v) { return v * v; })) * 1.05;

    var controls = K.group(root);
    var plots = K.row(root);
    var out = K.readout(root);

    var state = { lr: 0.1, steps: 200 };

    var lossSurface = K.canvasIn(plots, 300, 220);
    var lossPlot = K.makePlot(lossSurface, { xMin: 0, xMax: 200, yMin: 0, yMax: yMax, padB: 26, padL: 40 });

    var stripSurface = K.canvasIn(plots, 300, 220);
    var yMin = Math.min.apply(null, y), yMaxData = Math.max.apply(null, y);
    var stripPlot = K.makePlot(stripSurface, { xMin: yMin, xMax: yMaxData, yMin: 0, yMax: 1, padB: 26, padL: 14 });

    function render() {
      var result = M.fitMean(y, state.lr, state.steps);

      lossPlot.frame({ xTicks: 5, yTicks: 4, xLabel: 'step', yLabel: 'MSE', yDigits: 0 });
      lossPlot.line(result.history.map(function (v, i) { return [i, v]; }), { color: C.accent, width: 2 });
      lossPlot.marker(result.history.length - 1, result.history[result.history.length - 1],
        { color: C.ok, size: 3.5 });

      // The parameters as a strip: every data point, the model's guess, the answer.
      stripPlot.frame({ xTicks: 4, yTicks: 0, xLabel: 'value of y', yDigits: 0 });
      stripPlot.scatter(y.map(function (v, i) { return [v, 0.5 + ((i % 7) - 3) * 0.035]; }),
        { color: C.info, r: 2, alpha: 0.4 });
      stripPlot.line([[truth, 0], [truth, 1]], { color: C.ok, width: 2, dash: [4, 3] });
      stripPlot.line([[result.w, 0], [result.w, 1]], { color: C.accent, width: 2.5 });
      stripPlot.text(truth, 0.96, 'true mean', { color: C.ok, align: 'center', font: '10px ui-sans-serif' });
      stripPlot.text(result.w, 0.04, 'learned w', { color: C.accent, align: 'center', font: '10px ui-sans-serif' });

      out.set(
        '<div><span class="ro__k">learned w</span><span class="ro__v">' + result.w.toFixed(4) + '</span></div>' +
        '<div><span class="ro__k">true mean</span><span class="ro__v">' + truth.toFixed(4) + '</span></div>' +
        '<div><span class="ro__k">loss, step 0</span><span class="ro__v">' + result.history[0].toFixed(3) + '</span></div>' +
        '<div><span class="ro__k">loss, final</span><span class="ro__v">' + result.history[result.history.length - 1].toFixed(4) + '</span></div>'
      );
    }

    K.slider(controls, {
      label: 'learning rate', min: 0.01, max: 0.5, step: 0.01, value: state.lr,
      format: function (v) { return v.toFixed(2); },
      onInput: function (v) { state.lr = v; render(); }
    });
    K.slider(controls, {
      label: 'steps', min: 10, max: 400, step: 10, value: state.steps,
      format: function (v) { return String(v); },
      onInput: function (v) { state.steps = v; render(); }
    });

    K.caption(root, 'The model is one number and it always predicts it. Nothing here knows about ' +
      'features, or about a line. It is the smallest possible training run: measure the error, ' +
      'work out which way is downhill, take a step. Drag the learning rate down and it still ' +
      'arrives - just slower.');

    render();
  });

  /* ===================================================== 2. linear regression */

  K.registerDemo('linear-regression', function (root) {
    var controls = K.group(root);
    var host = K.group(root);
    var out = K.readout(root);

    var state = { noise: 0.6, lr: 0.05 };
    var data = M.makeLinear(120, state.noise, 0);
    var model = { w: 0, b: 0 };
    var step = 0;
    var history = [];

    var surface = K.canvasIn(host, 420, 250);
    var plot = K.makePlot(surface, { xMin: -3.4, xMax: 3.4, yMin: -8, yMax: 8, padB: 26, padL: 36 });

    function draw() {
      plot.frame({ xTicks: 5, yTicks: 4, xLabel: 'x', yLabel: 'y', yDigits: 0 });
      plot.scatter(data.x.map(function (v, i) { return [v, data.y[i]]; }), { color: C.info, r: 2.4, alpha: 0.6 });
      // The fit, sampled across the full x range so it reads as a line.
      var grid = [];
      for (var g = -3.4; g <= 3.4; g += 0.1) { grid.push([g, model.w * g + model.b]); }
      plot.line(grid, { color: C.accent, width: 2.4 });

      var pred = data.x.map(function (v) { return model.w * v + model.b; });
      out.set(
        '<div><span class="ro__k">w</span><span class="ro__v">' + model.w.toFixed(3) + '</span></div>' +
        '<div><span class="ro__k">b</span><span class="ro__v">' + model.b.toFixed(3) + '</span></div>' +
        '<div><span class="ro__k">R²</span><span class="ro__v">' + (step > 5 ? M.rSquared(data.y, pred).toFixed(4) : '—') + '</span></div>' +
        '<div><span class="ro__k">step</span><span class="ro__v">' + step + '</span></div>'
      );
    }

    var loop = K.animator(function () {
      var chunk = M.fitLinear(data.x, data.y, state.lr, 12, 1, model);
      model.w = chunk.w; model.b = chunk.b;
      step += 12;
      draw();
      return step < 2400;
    });

    K.button(controls, 'train', function () { loop.start(); });
    K.button(controls, 'reset', function () {
      loop.stop(); model = { w: 0, b: 0 }; step = 0; draw();
    });

    K.slider(controls, {
      label: 'noise', min: 0.05, max: 1.6, step: 0.05, value: state.noise,
      format: function (v) { return v.toFixed(2); },
      onInput: function (v) {
        loop.stop(); state.noise = v; data = M.makeLinear(120, v, 0);
        model = { w: 0, b: 0 }; step = 0; draw();
      }
    });
    K.slider(controls, {
      label: 'learning rate', min: 0.01, max: 0.3, step: 0.01, value: state.lr,
      format: function (v) { return v.toFixed(2); },
      onInput: function (v) { state.lr = v; }
    });

    K.caption(root, 'Press train and watch the line climb into the data. The data really is ' +
      'y = 2x + 1 plus noise - nothing told the model that. Two parameters, two gradients, and ' +
      'it finds the slope on its own.');

    draw();
  });

  /* ====================================================== 3. learning rate */

  K.registerDemo('gradient-descent', function (root) {
    var data = M.makeLinear(120, 0.6, 0);
    var controls = K.group(root);
    var host = K.group(root);
    var out = K.readout(root);

    var REFERENCE = [
      { label: '0.005 too small', lr: 0.005, color: C.warn },
      { label: '0.05 about right', lr: 0.05, color: C.ok },
      { label: '1.0 too large', lr: 1.0, color: C.bad }
    ];

    var state = { lr: 0.05 };

    var surface = K.canvasIn(host, 420, 250);
    var plot = K.makePlot(surface, { xMin: 0, xMax: 40, yMin: 0.05, yMax: 100, padB: 26, padL: 40 });

    function render() {
      plot.frame({ xTicks: 5, yTicks: 4, xLabel: 'step', yLabel: 'MSE (log scale)', yDigits: 0 });

      REFERENCE.forEach(function (ref) {
        var history = M.gdHistory(data.x, data.y, ref.lr, 40);
        var pts = [];
        for (var i = 0; i < history.length; i++) {
          if (isFinite(history[i]) && history[i] > 0) { pts.push([i, Math.min(history[i], 100)]); }
        }
        plot.line(pts, { color: ref.color, width: 1.6, dash: [4, 3] });
      });

      var mine = M.gdHistory(data.x, data.y, state.lr, 40);
      var myPts = [];
      for (var j = 0; j < mine.length; j++) {
        if (isFinite(mine[j]) && mine[j] > 0) { myPts.push([j, Math.min(mine[j], 100)]); }
      }
      plot.line(myPts, { color: C.accent, width: 2.6 });

      var final = mine[mine.length - 1];
      var verdict = !isFinite(final) ? 'diverged' : final.toFixed(4);

      out.set(
        REFERENCE.map(function (ref) {
          var h = M.gdHistory(data.x, data.y, ref.lr, 40);
          var v = h[h.length - 1];
          return '<div><span class="ro__k">' + ref.label + '</span><span class="ro__v">' +
            (isFinite(v) ? v.toFixed(3) : 'diverged') + '</span></div>';
        }).join('') +
        '<div class="ro__hi"><span class="ro__k">your rate ' + state.lr.toFixed(3) + '</span>' +
        '<span class="ro__v">' + verdict + '</span></div>'
      );
    }

    K.slider(controls, {
      label: 'learning rate', min: 0.002, max: 1.2, step: 0.002, value: state.lr,
      format: function (v) { return v.toFixed(3); },
      onInput: function (v) { state.lr = v; render(); }
    });

    K.caption(root, 'The three dashed curves are fixed reference rates. The bold line is yours. ' +
      'Push it past about 0.33 and the loss stops falling - it climbs. That threshold is not ' +
      'arbitrary: the update is stable while the rate is below 2 divided by the curvature of the loss.');

    render();
  });

  /* ================================================= 4. logistic regression */

  K.registerDemo('logistic-regression', function (root) {
    var raw = M.makeClassification(200, 0);
    var X = M.standardise(raw.X);
    var controls = K.group(root);
    var host = K.group(root);
    var out = K.readout(root);

    var state = { lr: 0.5 };
    var model = { w: [0, 0], b: 0 };

    var surface = K.canvasIn(host, 420, 260);
    var plot = K.makePlot(surface, { xMin: -3, xMax: 3, yMin: -3, yMax: 3, padB: 26, padL: 30 });

    function draw() {
      plot.heat(function (x, y) {
        var z = model.b + model.w[0] * x + model.w[1] * y;
        return M.sigmoid(z);
      }, K.RAMP_DIVERGING, 2);

      plot.frame({ xTicks: 4, yTicks: 4, xLabel: 'feature 1', yLabel: 'feature 2', xDigits: 0, yDigits: 0 });

      var zeros = [], ones = [];
      for (var i = 0; i < X.length; i++) {
        (raw.y[i] ? ones : zeros).push([X[i][0], X[i][1]]);
      }
      plot.scatter(zeros, { color: '#ffffff', r: 2.6, alpha: 0.95 });
      plot.scatter(ones, { color: '#111827', r: 2.6, alpha: 0.95 });

      var probs = M.predictLogistic(model, X);
      out.set(
        '<div><span class="ro__k">accuracy</span><span class="ro__v">' + M.accuracy(probs, raw.y).toFixed(3) + '</span></div>' +
        '<div><span class="ro__k">w</span><span class="ro__v">[' + model.w.map(function (v) { return v.toFixed(2); }).join(', ') + ']</span></div>' +
        '<div><span class="ro__k">b</span><span class="ro__v">' + model.b.toFixed(2) + '</span></div>'
      );
    }

    var loop = K.animator(function () {
      var chunk = M.fitLogistic(X, raw.y, state.lr, 10, 1, model);
      model = { w: chunk.w, b: chunk.b };
      draw();
      return true;
    });

    K.button(controls, 'train', function () { loop.start(); });
    K.button(controls, 'stop', function () { loop.stop(); });
    K.button(controls, 'reset', function () { loop.stop(); model = { w: [0, 0], b: 0 }; draw(); });

    K.slider(controls, {
      label: 'learning rate', min: 0.05, max: 3, step: 0.05, value: state.lr,
      format: function (v) { return v.toFixed(2); },
      onInput: function (v) { state.lr = v; }
    });

    K.caption(root, 'Blue is one class, red the other; the white band is where the model is ' +
      'unsure. The decision boundary is where the colour crosses the middle - and it is always a ' +
      'straight line, which is the honest limitation of logistic regression.');

    draw();
  });

  /* ====================================================== 5. loss functions */

  K.registerDemo('loss-functions', function (root) {
    var controls = K.group(root);
    var host = K.group(root);
    var out = K.readout(root);

    var state = { p: 0.01, y: 1 };

    var surface = K.canvasIn(host, 420, 240);
    var plot = K.makePlot(surface, { xMin: 0, xMax: 1, yMin: -1.05, yMax: 0.05, padB: 26, padL: 40 });

    function render() {
      plot.frame({ xTicks: 5, yTicks: 4, xLabel: 'predicted probability p', yLabel: 'gradient', xDigits: 1, yDigits: 2 });

      var ce = [], mse = [];
      for (var i = 0; i <= 100; i++) {
        var p = i / 100;
        ce.push([p, M.ceGradZ(p, state.y)]);
        mse.push([p, M.mseGradZ(p, state.y)]);
      }
      plot.line(mse, { color: C.warn, width: 2 });
      plot.line(ce, { color: C.accent, width: 2 });

      var myCe = M.ceGradZ(state.p, state.y), myMse = M.mseGradZ(state.p, state.y);
      plot.marker(state.p, myCe, { color: C.accent, size: 5, ring: '#fff' });
      plot.marker(state.p, myMse, { color: C.warn, size: 5, ring: '#fff' });

      var ratio = myMse === 0 ? Infinity : Math.abs(myCe / myMse);

      out.set(
        '<div><span class="ro__k">cross-entropy</span><span class="ro__v">' + myCe.toFixed(4) + '</span></div>' +
        '<div><span class="ro__k">squared error</span><span class="ro__v">' + myMse.toFixed(4) + '</span></div>' +
        '<div class="ro__hi"><span class="ro__k">CE pushes</span><span class="ro__v">' +
        (isFinite(ratio) ? ratio.toFixed(1) + 'x harder' : '—') + '</span></div>'
      );
    }

    K.slider(controls, {
      label: 'predicted probability p', min: 0.001, max: 0.999, step: 0.001, value: state.p,
      format: function (v) { return v.toFixed(3); },
      onInput: function (v) { state.p = v; render(); }
    });
    K.select(controls, {
      label: 'true label', value: '1',
      options: [{ value: '1', label: 'y = 1' }, { value: '0', label: 'y = 0' }],
      onChange: function (v) { state.y = parseInt(v, 10); render(); }
    });

    K.caption(root, 'Both curves are the gradient of the loss with respect to the pre-sigmoid ' +
      'value z, which is what actually moves the weights. Squared error (orange) is multiplied by ' +
      'p(1-p), so it collapses exactly where the model is most confident. Drag p to 0.01 with the ' +
      'label set to 1 and watch the gap open.');

    render();
  });

  /* ======================================================= 6. overfitting */

  K.registerDemo('overfitting', function (root) {
    var controls = K.group(root);
    var plots = K.row(root);
    var out = K.readout(root);

    var state = { degree: 3, lambda: 0, noise: 0.25 };
    var data = M.makeOverfitData(20, state.noise, 0);

    var fitSurface = K.canvasIn(plots, 300, 230);
    var fitPlot = K.makePlot(fitSurface, { xMin: 0, xMax: 1, yMin: -2.4, yMax: 2.4, padB: 26, padL: 34 });

    var errSurface = K.canvasIn(plots, 300, 230);
    var errPlot = K.makePlot(errSurface, { xMin: 1, xMax: 15, yMin: 0.005, yMax: 1.2, padB: 26, padL: 40 });

    function errorCurve() {
      var rows = [];
      for (var d = 1; d <= 15; d++) {
        var coef = M.fitPoly(M.polyFeatures(data.x, d), data.y, state.lambda);
        var train = M.mse(M.predictPoly(coef, M.polyFeatures(data.x, d)), data.y);
        var test = M.mse(M.predictPoly(coef, M.polyFeatures(data.xTest, d)), data.yTest);
        rows.push({ degree: d, train: train, test: test });
      }
      return rows;
    }

    function render() {
      var coef = M.fitPoly(M.polyFeatures(data.x, state.degree), data.y, state.lambda);

      fitPlot.frame({ xTicks: 4, yTicks: 4, xLabel: 'x', yLabel: 'y', xDigits: 1, yDigits: 1 });
      var truth = [], fitted = [];
      for (var g = 0; g <= 1.001; g += 0.01) {
        truth.push([g, Math.sin(2 * Math.PI * g)]);
        var row = [];
        for (var d = 0; d <= state.degree; d++) { row.push(Math.pow(g, d)); }
        fitted.push([g, M.predictPoly(coef, [row])[0]]);
      }
      fitPlot.line(truth, { color: C.ok, width: 2, dash: [5, 3] });
      fitPlot.line(fitted, { color: C.bad, width: 2.2 });
      fitPlot.scatter(data.x.map(function (v, i) { return [v, data.y[i]]; }), { color: '#111827', r: 2.8 });

      var rows = errorCurve();
      errPlot.frame({ xTicks: 5, yTicks: 4, xLabel: 'polynomial degree', yLabel: 'MSE (log)', xDigits: 0, yDigits: 2 });
      errPlot.line(rows.map(function (r) { return [r.degree, Math.max(0.005, r.train)]; }), { color: C.info, width: 2 });
      errPlot.line(rows.map(function (r) { return [r.degree, Math.max(0.005, r.test)]; }), { color: C.bad, width: 2 });
      errPlot.marker(state.degree, Math.max(0.005, rows[state.degree - 1].train), { color: C.info, size: 4 });
      errPlot.marker(state.degree, Math.max(0.005, rows[state.degree - 1].test), { color: C.bad, size: 4 });

      var mine = rows[state.degree - 1];
      out.set(
        '<div><span class="ro__k">degree</span><span class="ro__v">' + state.degree + '</span></div>' +
        '<div><span class="ro__k">train MSE</span><span class="ro__v">' + mine.train.toFixed(4) + '</span></div>' +
        '<div class="ro__hi"><span class="ro__k">test MSE</span><span class="ro__v">' + mine.test.toFixed(4) + '</span></div>' +
        '<div><span class="ro__k">lambda</span><span class="ro__v">' + (state.lambda === 0 ? '0 (off)' : state.lambda.toExponential(0)) + '</span></div>'
      );
    }

    K.slider(controls, {
      label: 'polynomial degree', min: 1, max: 15, step: 1, value: state.degree,
      format: function (v) { return String(v); },
      onInput: function (v) { state.degree = v; render(); }
    });
    K.slider(controls, {
      label: 'ridge lambda (log)', min: 0, max: 4, step: 1, value: 0,
      format: function (v) { return v === 0 ? '0 (off)' : Math.pow(10, -v).toExponential(0); },
      onInput: function (v) { state.lambda = v === 0 ? 0 : Math.pow(10, -v); render(); }
    });
    K.slider(controls, {
      label: 'noise', min: 0.05, max: 0.5, step: 0.05, value: state.noise,
      format: function (v) { return v.toFixed(2); },
      onInput: function (v) { state.noise = v; data = M.makeOverfitData(20, v, 0); render(); }
    });

    K.caption(root, 'Twenty noisy points. Turn the degree up and the red curve stops following ' +
      'the shape and starts threading every point - training error keeps falling while test error ' +
      'climbs. Then turn lambda up: the curve smooths out and the test error comes back down.');

    render();
  });

  /* ============================================================= 7. k-means */

  K.registerDemo('kmeans', function (root) {
    var data = M.makeClusters(3, 60, 0);
    var controls = K.group(root);
    var host = K.group(root);
    var out = K.readout(root);

    var state = { k: 3, seed: 0 };
    var centres = [], labels = [], inertia = 0, steps = 0;

    var surface = K.canvasIn(host, 420, 260);
    var plot = K.makePlot(surface, { xMin: -9, xMax: 9, yMin: -9, yMax: 9, padB: 26, padL: 30 });

    var PALETTE = ['#6d28d9', '#0ea5e9', '#16a34a', '#f59e0b', '#dc2626', '#db2777'];

    function reset() {
      var r = M.rng(state.seed);
      centres = M.kmeansPlusPlus(data.X, state.k, r);
      labels = M.assignClusters(data.X, centres);
      inertia = M.inertia(data.X, labels, centres);
      steps = 0;
      draw();
    }

    function oneStep() {
      var next = M.kmeansStep(data.X, centres);
      centres = next.centres; labels = next.labels; inertia = next.inertia;
      steps += 1;
      draw();
    }

    function draw() {
      plot.frame({ xTicks: 4, yTicks: 4, xLabel: '', yLabel: '', xDigits: 0, yDigits: 0 });

      for (var c = 0; c < state.k; c++) {
        var pts = [];
        for (var i = 0; i < data.X.length; i++) {
          if (labels[i] === c) { pts.push(data.X[i]); }
        }
        plot.scatter(pts, { color: PALETTE[c % PALETTE.length], r: 2.2, alpha: 0.75 });
      }
      for (var j = 0; j < centres.length; j++) {
        plot.marker(centres[j][0], centres[j][1], { color: '#111827', size: 7, ring: '#ffffff', cross: true });
      }
      for (var t = 0; t < data.centres.length; t++) {
        plot.marker(data.centres[t][0], data.centres[t][1], { color: '#9ca3af', size: 5, ring: '#ffffff' });
      }

      out.set(
        '<div><span class="ro__k">k</span><span class="ro__v">' + state.k + '</span></div>' +
        '<div><span class="ro__k">inertia</span><span class="ro__v">' + inertia.toFixed(1) + '</span></div>' +
        '<div><span class="ro__k">steps</span><span class="ro__v">' + steps + '</span></div>' +
        '<div><span class="ro__k">legend</span><span class="ro__v">✕ found · ● true</span></div>'
      );
    }

    K.button(controls, 'step', oneStep);
    K.button(controls, 'run to convergence', function () {
      for (var i = 0; i < 30; i++) {
        var before = labels.join(',');
        oneStep();
        if (labels.join(',') === before) { break; }
      }
    });
    K.button(controls, 'new start', function () { state.seed += 1; reset(); });

    K.slider(controls, {
      label: 'k (clusters)', min: 1, max: 6, step: 1, value: state.k,
      format: function (v) { return String(v); },
      onInput: function (v) { state.k = v; reset(); }
    });

    K.caption(root, 'Grey dots are the centres the data was generated from - k-means never sees ' +
      'them. Press step and watch the crosses drift onto the blobs. Press new start to see that a ' +
      'different initialisation can land somewhere slightly different: it converges, but only locally.');

    reset();
  });

  /* ================================================================= 8. PCA */

  K.registerDemo('pca', function (root) {
    var X = M.makeCorrelated(300, 0);
    var controls = K.group(root);
    var plots = K.row(root);
    var out = K.readout(root);

    var state = { components: 2 };

    var screeSurface = K.canvasIn(plots, 300, 230);
    var screePlot = K.makePlot(screeSurface, { xMin: 0, xMax: 6, yMin: 0, yMax: 60, padB: 26, padL: 34 });

    var projSurface = K.canvasIn(plots, 300, 230);
    var projPlot = K.makePlot(projSurface, { xMin: -5, xMax: 5, yMin: -5, yMax: 5, padB: 26, padL: 34 });

    function render() {
      var model = M.pcaFit(X, Math.max(2, state.components));
      var kept = model.explained.slice(0, state.components)
        .reduce(function (a, b) { return a + b; }, 0);

      screePlot.frame({ xTicks: 5, yTicks: 4, xLabel: 'component', yLabel: '% variance', xDigits: 0, yDigits: 0 });
      var bars = model.explained.map(function (e) { return e * 100; });
      screePlot.bars(bars, {
        colours: bars.map(function (_, i) { return i < state.components ? C.accent : '#cbd5e1'; }),
        width: 0.7
      });
      for (var i = 1; i <= 5; i++) {
        screePlot.text(i - 0.5, 1.5, 'PC' + i, { align: 'center', color: C.text, font: '9px ui-sans-serif' });
      }

      var Z = M.pcaTransform(model, X);
      projPlot.frame({ xTicks: 4, yTicks: 4, xLabel: 'PC1', yLabel: 'PC2', xDigits: 0, yDigits: 0 });
      projPlot.scatter(Z.map(function (r) { return [r[0], r[1]]; }), { color: C.info, r: 2, alpha: 0.6 });

      var back = M.pcaInverse(model, Z);
      var residual = 0, total = 0;
      for (var n = 0; n < X.length; n++) {
        for (var f = 0; f < X[n].length; f++) {
          residual += (X[n][f] - back[n][f]) * (X[n][f] - back[n][f]);
        }
      }
      for (var n2 = 0; n2 < X.length; n2++) {
        var m = M.mean(X.map(function (r) { return r[0]; }));
        total += (X[n2][0] - m) * (X[n2][0] - m);
      }

      out.set(
        '<div><span class="ro__k">components kept</span><span class="ro__v">' + state.components + ' of 5</span></div>' +
        '<div><span class="ro__k">variance kept</span><span class="ro__v">' + (kept * 100).toFixed(1) + '%</span></div>' +
        '<div><span class="ro__k">residual</span><span class="ro__v">' + (residual / X.length).toFixed(4) + '</span></div>'
      );
    }

    K.slider(controls, {
      label: 'components kept', min: 1, max: 5, step: 1, value: state.components,
      format: function (v) { return String(v); },
      onInput: function (v) { state.components = v; render(); }
    });

    K.caption(root, 'Five features were generated from two hidden factors. The bars are the ' +
      'variance each principal component carries - the first two take almost all of it, which is ' +
      'why the scatter on the right still looks like the data. Keep only PC1 and watch the ' +
      'residual climb.');

    render();
  });

  /* ====================================================== 9. neural network */

  K.registerDemo('neural-network', function (root) {
    var raw = M.makeRing(400, 0);
    var X = M.standardise(raw.X);

    var controls = K.group(root);
    var plots = K.row(root);
    var out = K.readout(root);

    var state = { hidden: 8, lr: 1.0, linear: false };
    var params = null;
    var step = 0;
    var lossHistory = [];
    var accHistory = [];

    var boundSurface = K.canvasIn(plots, 300, 260);
    var boundPlot = K.makePlot(boundSurface, { xMin: -3, xMax: 3, yMin: -3, yMax: 3, padB: 26, padL: 30 });

    var lossSurface = K.canvasIn(plots, 300, 260);
    var lossPlot = K.makePlot(lossSurface, { xMin: 0, xMax: 200, yMin: 0, yMax: 0.75, padB: 26, padL: 38 });

    var linearModel = null;

    function reset() {
      if (state.linear) {
        linearModel = M.fitLogistic(X, raw.y, 0.5, 2000);
        params = null;
      } else {
        params = M.initParams(2, state.hidden, 0);
        linearModel = null;
      }
      step = 0;
      lossHistory = [];
      accHistory = [];
      draw();
    }

    function currentProbs() {
      if (state.linear) { return M.predictLogistic(linearModel, X); }
      var A2 = M.forward(X, params).A2;
      return A2.map(function (r) { return r[0]; });
    }

    function draw() {
      var probs = currentProbs();

      boundPlot.heat(function (x, y) {
        if (state.linear) {
          return M.sigmoid(linearModel.b + linearModel.w[0] * x + linearModel.w[1] * y);
        }
        return M.mlpProbability(params, x, y);
      }, K.RAMP_DIVERGING, state.linear ? 2 : 3);

      boundPlot.frame({ xTicks: 4, yTicks: 4, xLabel: '', yLabel: '', xDigits: 0, yDigits: 0 });
      var zeros = [], ones = [];
      for (var i = 0; i < X.length; i++) {
        (raw.y[i] ? ones : zeros).push([X[i][0], X[i][1]]);
      }
      boundPlot.scatter(zeros, { color: '#ffffff', r: 2, alpha: 0.9 });
      boundPlot.scatter(ones, { color: '#111827', r: 2, alpha: 0.9 });

      lossPlot.frame({ xTicks: 5, yTicks: 4, xLabel: 'step', yLabel: 'loss', xDigits: 0, yDigits: 2 });
      if (lossHistory.length > 1) {
        lossPlot.line(lossHistory.map(function (v, i) { return [i * 20, v]; }), { color: C.accent, width: 2 });
      }

      var acc = M.accuracy(probs, raw.y);
      out.set(
        '<div><span class="ro__k">model</span><span class="ro__v">' +
        (state.linear ? 'no hidden layer' : '2 → ' + state.hidden + ' → 1') + '</span></div>' +
        '<div class="ro__hi"><span class="ro__k">accuracy</span><span class="ro__v">' + acc.toFixed(3) + '</span></div>' +
        '<div><span class="ro__k">step</span><span class="ro__v">' + step + '</span></div>' +
        '<div><span class="ro__k">loss</span><span class="ro__v">' +
        (lossHistory.length ? lossHistory[lossHistory.length - 1].toFixed(4) : '—') + '</span></div>'
      );
    }

    var loop = K.animator(function () {
      if (state.linear) { return false; }
      for (var i = 0; i < 40; i++) {
        var cache = M.mlpTrainStep(X, raw.y, params, state.lr);
        step += 1;
        if (step % 20 === 0) {
          lossHistory.push(M.mlpLoss(cache.A2, raw.y));
          accHistory.push(M.mlpAccuracy(cache.A2, raw.y));
        }
      }
      draw();
      return step < 3000;
    });

    K.button(controls, 'train', function () { if (!state.linear) { loop.start(); } });
    K.button(controls, 'stop', function () { loop.stop(); });
    K.button(controls, 'reset', function () { loop.stop(); reset(); });

    K.slider(controls, {
      label: 'hidden units', min: 2, max: 16, step: 1, value: state.hidden,
      format: function (v) { return String(v); },
      onInput: function (v) { loop.stop(); state.hidden = v; reset(); }
    });
    K.slider(controls, {
      label: 'learning rate', min: 0.2, max: 2, step: 0.1, value: state.lr,
      format: function (v) { return v.toFixed(1); },
      onInput: function (v) { state.lr = v; }
    });
    K.toggle(controls, {
      label: 'no hidden layer (the baseline)', value: state.linear,
      onChange: function (v) { loop.stop(); state.linear = v; reset(); }
    });

    K.caption(root, 'The data is a ring: a blob inside a circle. With no hidden layer the boundary ' +
      'is a straight line and accuracy stalls around 60%, however long you train. Tick the hidden ' +
      'layer off and on, then press train - the boundary bends around the ring and the accuracy ' +
      'jumps to almost perfect. That difference is the reason deep learning exists, and it is one layer.');

    reset();
  });
})();
