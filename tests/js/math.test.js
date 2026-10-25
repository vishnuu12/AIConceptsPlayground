/* ============================================================================
   math.test.js - the browser demos, tested.

   Two jobs:

   1. Assert the JavaScript maths behaves like the Python maths. It is a second
      implementation of the same algorithms (the browser cannot run NumPy), and a
      second implementation is a liability unless something keeps it honest.
      These assertions mirror tests/test_foundations.py and tests/test_llm.py.

   2. Smoke-test the demo UIs against a minimal fake DOM. A canvas demo has no
      output to assert on, but "does it throw" is exactly the class of bug that
      would otherwise only show up in a browser console.

   Run:  node tests/js/math.test.js [path/to/knowledge.js]
   ========================================================================== */

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..');
const STATIC = path.join(ROOT, 'site', 'static');

let checks = 0;
let failures = [];

function check(name, fn) {
  checks += 1;
  try {
    fn();
  } catch (error) {
    failures.push(name + ': ' + error.message);
  }
}

/* ======================================================================
   Part A - the maths
   ====================================================================== */

const M1 = require(path.join(STATIC, 'math-foundations.js'));
const M2 = require(path.join(STATIC, 'math-llm.js'));

check('gradient descent finds the mean', () => {
  const y = [1, 3, 5, 7];
  const { w, history } = M1.fitMean(y, 0.1, 500);
  assert.ok(Math.abs(w - M1.mean(y)) < 0.01, 'learned ' + w);
  assert.ok(history[history.length - 1] < history[0], 'loss should fall');
});

check('linear regression recovers the true line', () => {
  const { x, y } = M1.makeLinear(200, 0.4, 1);
  const fit = M1.fitLinear(x, y, 0.05, 600);
  assert.ok(Math.abs(fit.w - 2) < 0.2, 'slope ' + fit.w);
  assert.ok(Math.abs(fit.b - 1) < 0.25, 'intercept ' + fit.b);
  assert.ok(M1.rSquared(y, x.map((v) => fit.w * v + fit.b)) > 0.9);
});

check('the learning rate decides whether the run converges', () => {
  const { x, y } = M1.makeLinear(120, 0.6, 0);
  const small = M1.gdHistory(x, y, 0.005, 40);
  const good = M1.gdHistory(x, y, 0.05, 40);
  const huge = M1.gdHistory(x, y, 1.0, 40);
  assert.ok(good[39] < small[39], 'a sane rate should beat a tiny one');
  assert.ok(!isFinite(huge[39]) || huge[39] > huge[0], 'a huge rate should diverge');
});

check('logistic regression classifies', () => {
  const { X, y } = M1.makeClassification(300, 0);
  const scaled = M1.standardise(X);
  const model = M1.fitLogistic(scaled, y, 0.5, 400);
  const accuracy = M1.accuracy(M1.predictLogistic(model, scaled), y);
  assert.ok(accuracy > 0.9, 'accuracy ' + accuracy);
});

check('cross-entropy pushes much harder when confidently wrong', () => {
  const ce = Math.abs(M1.ceGradZ(0.01, 1));
  const mse = Math.abs(M1.mseGradZ(0.01, 1));
  assert.ok(ce > 0.9, 'ce gradient ' + ce);
  assert.ok(ce > 20 * mse, 'expected a large gap, got ' + ce + ' vs ' + mse);
});

check('a high degree overfits and ridge fixes it', () => {
  const d = M1.makeOverfitData(20, 0.25, 0);
  const coef3 = M1.fitPoly(M1.polyFeatures(d.x, 3), d.y, 0);
  const coef15 = M1.fitPoly(M1.polyFeatures(d.x, 15), d.y, 0);

  const train3 = M1.mse(M1.predictPoly(coef3, M1.polyFeatures(d.x, 3)), d.y);
  const train15 = M1.mse(M1.predictPoly(coef15, M1.polyFeatures(d.x, 15)), d.y);
  const test3 = M1.mse(M1.predictPoly(coef3, M1.polyFeatures(d.xTest, 3)), d.yTest);
  const test15 = M1.mse(M1.predictPoly(coef15, M1.polyFeatures(d.xTest, 15)), d.yTest);

  assert.ok(train15 < train3, 'degree 15 should fit training better');
  assert.ok(test15 > 3 * test3, 'test15=' + test15 + ' test3=' + test3);

  const ridged = M1.fitPoly(M1.polyFeatures(d.x, 15), d.y, 1e-4);
  const test15r = M1.mse(M1.predictPoly(ridged, M1.polyFeatures(d.xTest, 15)), d.yTest);
  assert.ok(test15r < test15 / 3, 'ridge should help: ' + test15r + ' vs ' + test15);
});

check('k-means recovers centres it was never shown', () => {
  const { X, centres: truth } = M1.makeClusters(3, 60, 0);
  const found = M1.fitKMeans(X, 3, 0);

  const order = (list) => list.map((c) => c.slice()).sort((a, b) => a[0] - b[0]);
  const a = order(found.centres), b = order(truth);
  let worst = 0;
  for (let i = 0; i < a.length; i++) {
    worst = Math.max(worst, M1.dist2(a[i], b[i]));
  }
  assert.ok(Math.sqrt(worst) < 0.5, 'centre error ' + Math.sqrt(worst));
  assert.ok(found.history[found.history.length - 1] < found.history[0]);
});

check('PCA finds the two real dimensions', () => {
  const X = M1.makeCorrelated(300, 0);
  const model = M1.pcaFit(X, 2);
  const top2 = model.explained[0] + model.explained[1];
  assert.ok(top2 > 0.9, 'top two components carry ' + top2);
  assert.ok(Math.abs(model.explained.reduce((x, y) => x + y, 0) - 1) < 1e-9);

  const Z = M1.pcaTransform(model, X);
  assert.strictEqual(Z[0].length, 2);
  const back = M1.pcaInverse(model, Z);
  let residual = 0;
  for (let i = 0; i < X.length; i++) {
    for (let f = 0; f < X[i].length; f++) { residual += Math.pow(X[i][f] - back[i][f], 2); }
  }
  assert.ok(residual / X.length < 0.1, 'residual ' + residual / X.length);
});

check('a hidden layer beats a linear model on a ring', () => {
  const { X, y } = M1.makeRing(400, 0);
  const scaled = M1.standardise(X);

  const linear = M1.fitLogistic(scaled, y, 0.5, 2000);
  const linearAccuracy = M1.accuracy(M1.predictLogistic(linear, scaled), y);

  const net = M1.trainMLP(scaled, y, 8, 1.0, 3000, 0);
  const netAccuracy = M1.mlpAccuracy(M1.forward(scaled, net.params).A2, y);

  assert.ok(linearAccuracy < 0.8, 'baseline should struggle, got ' + linearAccuracy);
  assert.ok(netAccuracy > 0.9, 'network should not, got ' + netAccuracy);
  assert.ok(net.lossHistory[net.lossHistory.length - 1] < net.lossHistory[0]);
});

check('hidden units are not initialised identically', () => {
  const params = M1.initParams(2, 8, 0);
  const columns = new Set(params.W1[0].map((_, j) => params.W1.map((row) => row[j]).join(',')));
  assert.strictEqual(columns.size, 8);
});

check('backward pass produces the right shapes', () => {
  const { X, y } = M1.makeRing(50, 0);
  const params = M1.initParams(2, 6, 0);
  const grads = M1.backward(X, y, params, M1.forward(X, params));
  assert.strictEqual(grads.W1.length, 2);
  assert.strictEqual(grads.W1[0].length, 6);
  assert.strictEqual(grads.W2.length, 6);
  assert.strictEqual(grads.b1.length, 6);
  assert.ok(grads.W1.every((row) => row.every((v) => isFinite(v))));
});

check('cosine similarity is a similarity', () => {
  const a = M2.embed('gradient descent and the learning rate');
  const b = M2.embed('the learning rate controls the step size');
  const c = M2.embed('k-means groups unlabelled points');
  assert.ok(Math.abs(M2.cosine(a, a) - 1) < 1e-9);
  assert.ok(M2.cosine(a, b) > M2.cosine(a, c));
  assert.ok(Math.abs(M2.cosine(M2.embed('car'), M2.embed('automobile'))) < 1e-9,
    'shared-word matching cannot see synonyms - that is the point');
});

check('embeddings are deterministic across calls', () => {
  assert.deepStrictEqual(M2.embed('the model learns from the data'),
    M2.embed('the model learns from the data'));
});

check('BPE learns whole common words', () => {
  const merges = M2.trainBPE('the model learns. the model predicts. the model is trained.', 40);
  assert.deepStrictEqual(M2.encodeWord('model', merges), ['model</w>']);
  assert.ok(M2.encodeWord('model', merges).length < M2.encodeWord('models', merges).length);
});

check('temperature changes the distribution, top-k filters it', () => {
  const logits = [3, 2, 1, 0];
  const cold = M2.softmax(logits, 0.1);
  const warm = M2.softmax(logits, 1.0);
  const hot = M2.softmax(logits, 5.0);

  assert.ok(cold[0] > 0.99, 'near-zero temperature should be almost deterministic');
  assert.ok(M2.entropy(cold) < M2.entropy(warm));
  assert.ok(M2.entropy(warm) < M2.entropy(hot));
  assert.ok(Math.abs(warm.reduce((a, b) => a + b, 0) - 1) < 1e-9);

  const top1 = M2.softmax(M2.topK(logits, 1), 1.0);
  assert.strictEqual(top1[0], 1);
  assert.strictEqual(top1.filter((p) => p > 0).length, 1);
});

check('the validator rejects incomplete payloads', () => {
  assert.deepStrictEqual(M2.validate({ name: 'A', email: 'a@b.com', city: 'Pune' }), []);
  assert.ok(M2.validate({ name: 'A', email: 'a@b.com' }).length > 0);
  assert.ok(M2.validate({ name: 'A', email: '', city: 'Pune' }).length > 0);
  assert.ok(M2.validate('nonsense').length > 0);
});

check('extraction succeeds, and the repair loop gives up', () => {
  const good = M2.extractWithRepair("Hi, I'm Rohith from Chennai - r.k@example.com works.", 3);
  assert.ok(good.payload);
  assert.strictEqual(good.payload.name, 'Rohith');
  assert.strictEqual(good.payload.email, 'r.k@example.com');
  assert.strictEqual(good.log.length, 1, 'a clean extraction should take one attempt');

  const bad = M2.extractWithRepair('I am Vinith, somewhere in Bangalore.', 3);
  assert.strictEqual(bad.payload, null);
  assert.strictEqual(bad.log.length, 3, 'the attempt limit must be respected');
});

check('chunking overlaps, and rejects an impossible overlap', () => {
  const words = Array.from({ length: 100 }, (_, i) => 'w' + i).join(' ');
  const chunks = M2.chunkText(words, 30, 10);
  assert.strictEqual(chunks.length, 5);
  assert.deepStrictEqual(chunks[0].split(' ').slice(-10), chunks[1].split(' ').slice(0, 10));
  assert.throws(() => M2.chunkText('a b c d', 2, 2));
});

check('safe_eval does arithmetic and refuses everything else', () => {
  assert.strictEqual(M2.safeEval('2400 * 0.15'), 360);
  assert.strictEqual(M2.safeEval('(2 + 3) * 4'), 20);
  assert.strictEqual(M2.safeEval('-5 + 2'), -3);

  ['print(1)', '(1).__class__', 'x + 1', '1 if True else 2', '[1, 2]', '1; 2'].forEach((hostile) => {
    assert.throws(() => M2.safeEval(hostile), 'should have refused: ' + hostile);
  });
});

/* ======================================================================
   Part B - the knowledge base, generated from the Python
   ====================================================================== */

const knowledgePath = process.argv[2];
let DOCS = [];
let GOLDEN = [];

if (knowledgePath && fs.existsSync(knowledgePath)) {
  const fakeWindow = {};
  // eslint-disable-next-line no-new-func
  new Function('window', fs.readFileSync(knowledgePath, 'utf8'))(fakeWindow);
  DOCS = fakeWindow.KB_DOCUMENTS || [];
  GOLDEN = fakeWindow.KB_GOLDEN || [];
}

check('the generated knowledge base is present', () => {
  assert.ok(DOCS.length >= 6, 'expected documents, got ' + DOCS.length);
  assert.ok(GOLDEN.length >= 4, 'expected golden cases, got ' + GOLDEN.length);
  assert.ok(DOCS.every((d) => d.id && d.title && d.text));
});

check('retrieval ranks the right document first', () => {
  const index = M2.buildIndex(DOCS, 45, 12);
  const top = M2.retrieve('What does ridge regularisation do?', index, 1)[0];
  assert.strictEqual(top[1].docId, 'overfitting', 'got ' + top[1].docId);
});

check('grounded answers cite, ungrounded ones do not', () => {
  const index = M2.buildIndex(DOCS, 45, 12);
  const passages = M2.retrieve('Why is cross-entropy used for classification?', index, 3);
  const grounded = M2.answerFromContext(M2.buildContext(passages.map((p) => p[1])),
    'Why is cross-entropy used for classification?');
  const ungrounded = M2.answerWithoutContext('Why is cross-entropy used for classification?', true);
  assert.ok(/\(source:/.test(grounded), 'grounded answer should cite: ' + grounded);
  assert.ok(!/\(source:/.test(ungrounded));
});

check('the agent loop runs several tools and terminates', () => {
  const result = M2.runAgent('What is 2400 * 0.15, and what is the weather in Chennai?', DOCS, 6);
  const tools = result.trace.filter((t) => t.kind === 'tool_call').map((t) => t.tool);
  assert.ok(tools.indexOf('calculator') !== -1, 'calculator should run');
  assert.ok(tools.indexOf('get_weather') !== -1, 'get_weather should run');
  assert.strictEqual(result.trace[result.trace.length - 1].kind, 'answer');
  assert.ok(result.answer.indexOf('360') !== -1, 'answer: ' + result.answer);
  assert.ok(result.answer.indexOf('Chennai') !== -1);
});

check('a failing tool does not crash the loop', () => {
  const result = M2.runAgent('What is 5 apples plus 3 oranges?', DOCS, 6);
  const calls = result.trace.filter((t) => t.kind === 'tool_call');
  assert.ok(calls.length > 0, 'the calculator should have been called');
  assert.ok('error' in calls[0].result);
  assert.ok(result.answer.length > 0);
});

check('the agent loop respects its step limit', () => {
  const result = M2.runAgent('search the docs and search the docs', DOCS, 2);
  assert.ok(result.trace.length <= 3, 'trace was ' + result.trace.length);
});

check('the eval harness finds a real difference', () => {
  const report = M2.runEvaluation(GOLDEN, DOCS, 45, 12, 3);
  assert.ok(report.withRag.passed > report.withoutRetrieval.passed,
    'RAG ' + report.withRag.passed + ' vs none ' + report.withoutRetrieval.passed);
  assert.ok(report.withRag.cited > report.withoutRetrieval.cited);
  assert.strictEqual(report.rows.length, GOLDEN.length);
  assert.deepStrictEqual(report.regressions, []);
});

/* ======================================================================
   Part C - the demo UIs, against a fake DOM
   ====================================================================== */

function makeContext2D() {
  const noop = () => {};
  return {
    canvas: null,
    fillStyle: '', strokeStyle: '', lineWidth: 1, font: '',
    textAlign: '', textBaseline: '', globalAlpha: 1,
    lineJoin: '', lineCap: '', imageSmoothingEnabled: true,
    scale: noop, save: noop, restore: noop, translate: noop, rotate: noop,
    beginPath: noop, closePath: noop, moveTo: noop, lineTo: noop, arc: noop,
    fill: noop, stroke: noop, fillRect: noop, strokeRect: noop, clearRect: noop,
    fillText: noop, setLineDash: noop, drawImage: noop, putImageData: noop,
    createImageData: (w, h) => ({ width: w, height: h, data: new Uint8ClampedArray(w * h * 4) }),
    getImageData: (x, y, w, h) => ({ width: w, height: h, data: new Uint8ClampedArray(w * h * 4) })
  };
}

function makeElement(tag) {
  const node = {
    tagName: String(tag).toUpperCase(),
    className: '', textContent: '', innerHTML: '', value: '',
    type: '', checked: false, placeholder: '', hidden: false,
    style: {}, children: [], parentNode: null,
    _attributes: {}, _listeners: {},
    appendChild(child) { this.children.push(child); child.parentNode = this; return child; },
    setAttribute(key, value) { this._attributes[key] = String(value); },
    getAttribute(key) { return Object.prototype.hasOwnProperty.call(this._attributes, key) ? this._attributes[key] : null; },
    addEventListener(type, fn) { (this._listeners[type] = this._listeners[type] || []).push(fn); },
    removeEventListener() {},
    getContext() { return makeContext2D(); },
    querySelectorAll() { return []; },
    classList: { add() {}, remove() {}, contains() { return false; } }
  };
  return node;
}

const SLUGS = [
  'what-is-ml', 'linear-regression', 'gradient-descent', 'logistic-regression',
  'loss-functions', 'overfitting', 'kmeans', 'pca', 'neural-network',
  'embeddings', 'tokenization', 'prompting', 'rag', 'tools-agent', 'evaluation'
];

const mounts = SLUGS.map((slug) => {
  const mount = makeElement('div');
  mount.setAttribute('data-demo', slug);
  mount._slug = slug;
  return mount;
});

const pendingFrames = [];
const domReadyListeners = [];

global.window = {
  devicePixelRatio: 2,
  requestAnimationFrame(fn) { pendingFrames.push(fn); return pendingFrames.length; },
  cancelAnimationFrame() {},
  addEventListener() {}
};
global.document = {
  readyState: 'loading',            // so demos.js defers boot to DOMContentLoaded
  createElement: makeElement,
  querySelectorAll(selector) { return selector === '[data-demo]' ? mounts : []; },
  getElementById() { return null; },
  addEventListener(type, fn) { if (type === 'DOMContentLoaded') { domReadyListeners.push(fn); } }
};

// The math files were required before the globals existed, so hand their exports
// to the window the demos read from.
global.window.ML1 = M1;
global.window.ML2 = M2;
global.window.KB_DOCUMENTS = DOCS;
global.window.KB_GOLDEN = GOLDEN;

require(path.join(STATIC, 'demos.js'));
require(path.join(STATIC, 'math-foundations.js'));
require(path.join(STATIC, 'demos-foundations.js'));
require(path.join(STATIC, 'math-llm.js'));
require(path.join(STATIC, 'demos-llm.js'));

check('every demo slug is registered', () => {
  const registry = global.window.DemoKit;
  assert.ok(registry, 'DemoKit should be on window');
  domReadyListeners.forEach((fn) => fn());   // boot
  mounts.forEach((mount) => {
    const errored = mount.children.some((child) =>
      String(child.className).indexOf('demo__error') !== -1);
    assert.ok(!errored, mount._slug + ' failed to install: ' +
      (mount.children.find((c) => String(c.className).indexOf('demo__error') !== -1) || {}).textContent);
  });
});

check('one animation frame of every demo runs without throwing', () => {
  // The animators never fire in this fake DOM (requestAnimationFrame just
  // records), so drive a few rounds by hand. Bounded, so an animator that never
  // reports itself finished cannot hang the test.
  for (let round = 0; round < 3; round++) {
    const batch = pendingFrames.splice(0, pendingFrames.length);
    if (batch.length === 0) { break; }
    batch.forEach((fn) => fn());
  }
});

/* ======================================================================
   Report
   ====================================================================== */

if (failures.length) {
  console.error('\nJS demo tests: ' + (checks - failures.length) + '/' + checks + ' passed\n');
  failures.forEach((f) => console.error('  FAIL  ' + f));
  process.exit(1);
}

console.log('JS demo tests: ' + checks + '/' + checks + ' passed '
  + '(math + knowledge + ' + SLUGS.length + ' demo UIs)');
