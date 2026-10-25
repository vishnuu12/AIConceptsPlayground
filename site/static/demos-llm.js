/* ============================================================================
   demos-llm.js - the six interactive Part 2 demos.

   The knowledge base and the golden set are generated from the PYTHON source
   (mlconcepts/llm/knowledge.py and evaluation.py) by tools/build_site.py and
   served as static/knowledge.js. That matters: there is one copy of the
   documents, so the page and the test suite cannot drift apart.
   ========================================================================== */

(function () {
  'use strict';

  if (typeof document === 'undefined') { return; }

  var K = window.DemoKit;
  var M = window.ML2;
  var C = K.COLORS;

  var DOCS = window.KB_DOCUMENTS || [];
  var GOLDEN = window.KB_GOLDEN || [];

  function esc(text) {
    return String(text)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function barRow(label, score, max, colour) {
    var pct = max > 0 ? Math.max(2, (score / max) * 100) : 0;
    return '<div class="bar">' +
      '<span class="bar__label">' + esc(label) + '</span>' +
      '<span class="bar__track"><span class="bar__fill" style="width:' + pct.toFixed(1) +
      '%;background:' + (colour || C.accent) + '"></span></span>' +
      '<span class="bar__value">' + score.toFixed(3) + '</span>' +
      '</div>';
  }

  /* ======================================================== 10. embeddings */

  K.registerDemo('embeddings', function (root) {
    var controls = K.group(root);
    var list = K.group(root);
    var out = K.readout(root);

    var corpus = [
      'Gradient descent moves the parameters downhill along the loss.',
      'The learning rate decides how big each step of gradient descent is.',
      'Logistic regression predicts a probability using the sigmoid function.',
      'Cross-entropy is the loss function for classification problems.',
      'k-means groups unlabelled points by repeatedly moving cluster centres.',
      'PCA finds the directions in which the data varies the most.',
      'An embedding turns text into a vector so similarity becomes geometry.',
      'Retrieval augmented generation grounds an answer in retrieved documents.',
      'Tokenization splits text into the subword units a model actually reads.'
    ];

    var state = { query: 'how big should the learning rate be?' };

    function render() {
      var q = M.embed(state.query);
      var scored = corpus.map(function (text) { return [M.cosine(q, M.embed(text)), text]; });
      scored.sort(function (a, b) { return b[0] - a[0]; });

      var max = scored[0][0];
      list.innerHTML = scored.map(function (pair, i) {
        return barRow(pair[1], pair[0], max, i === 0 ? C.ok : C.accent);
      }).join('');

      out.set(
        '<div class="ro__hi"><span class="ro__k">best match</span>' +
        '<span class="ro__v">' + scored[0][0].toFixed(3) + '</span></div>' +
        '<div><span class="ro__k">self-similarity</span><span class="ro__v">' +
        M.cosine(q, q).toFixed(4) + '</span></div>' +
        '<div><span class="ro__k">dimension</span><span class="ro__v">256</span></div>'
      );
    }

    K.textInput(controls, {
      label: 'query', value: state.query,
      placeholder: 'ask something about these sentences',
      onInput: function (v) { state.query = v; render(); }
    });

    var presets = K.group(controls, 'demo__presets');
    ['how big should the learning rate be?',
      'what loss do you use for classification?',
      'how do you find similar text?'].forEach(function (q) {
      K.button(presets, q.slice(0, 34) + (q.length > 34 ? '…' : ''), function () {
        state.query = q;
        render();
      });
    });

    K.caption(root, 'Type a question and watch the ranking move. This uses feature hashing, not a ' +
      'trained model: it matches SHARED WORDS, which is why "car" and "automobile" score zero here ' +
      'and a real embedding would put them together. Everything else - the geometry, the ranking - ' +
      'is exactly what a real vector search does.');

    render();
  });

  /* ====================================================== 11. tokenization */

  K.registerDemo('tokenization', function (root) {
    var CORPUS = 'the model learns from the data. the model predicts the next token from the ' +
      'previous tokens. training the model changes the parameters of the model. the loss measures ' +
      'how wrong the model is, and the model is trained to lower the loss. a large language model ' +
      'is trained on a large amount of text.';

    var merges = M.trainBPE(CORPUS, 60);
    var VOCAB = ['the', 'a', 'model', 'data', 'loss', 'token', 'cat', 'bicycle'];
    var LOGITS = [4.2, 3.1, 2.6, 2.2, 1.4, 0.9, 0.2, -0.6];

    var controls = K.group(root);
    var chipHost = K.group(root);
    var host = K.group(root);
    var out = K.readout(root);

    var state = { text: 'the model learns from the data', temperature: 1.0, topK: 8 };

    var surface = K.canvasIn(host, 420, 210);
    var plot = K.makePlot(surface, { xMin: 0, xMax: 8, yMin: 0, yMax: 1.02, padB: 30, padL: 36 });

    function renderChips() {
      var words = state.text.split(/\s+/).filter(Boolean);
      var chips = [];
      words.forEach(function (w) {
        M.encodeWord(w, merges).forEach(function (token) {
          chips.push('<span class="tok">' + esc(token.replace('</w>', '')) + '</span>');
        });
      });
      chipHost.innerHTML = '<div class="tokrow">' + chips.join('') + '</div>';
    }

    function renderDistribution() {
      var masked = M.topK(LOGITS, state.topK);
      var probs = M.softmax(masked, state.temperature);
      var shown = probs.map(function (p) { return isFinite(p) ? p : 0; });

      plot.frame({ xTicks: 0, yTicks: 4, yLabel: 'probability', yDigits: 1 });
      plot.bars(shown, { colours: shown.map(function (p) { return p > 0 ? C.accent : '#e5e7eb'; }), width: 0.62 });
      for (var i = 0; i < VOCAB.length; i++) {
        plot.text(i + 0.5, 0.02, VOCAB[i], { align: 'center', color: C.text, font: '10px ui-sans-serif' });
      }

      out.set(
        '<div><span class="ro__k">words</span><span class="ro__v">' + state.text.split(/\s+/).filter(Boolean).length + '</span></div>' +
        '<div><span class="ro__k">bpe tokens</span><span class="ro__v">' + M.tokenizeBPE(state.text, merges).length + '</span></div>' +
        '<div><span class="ro__k">entropy</span><span class="ro__v">' + M.entropy(shown).toFixed(3) + '</span></div>' +
        '<div><span class="ro__k">top token</span><span class="ro__v">' + VOCAB[shown.indexOf(Math.max.apply(null, shown))] + '</span></div>'
      );
    }

    K.textInput(controls, {
      label: 'text', value: state.text,
      onInput: function (v) { state.text = v; renderChips(); renderDistribution(); }
    });
    K.slider(controls, {
      label: 'temperature', min: 0.1, max: 2.5, step: 0.1, value: state.temperature,
      format: function (v) { return v.toFixed(1); },
      onInput: function (v) { state.temperature = v; renderDistribution(); }
    });
    K.slider(controls, {
      label: 'top-k', min: 1, max: 8, step: 1, value: state.topK,
      format: function (v) { return String(v); },
      onInput: function (v) { state.topK = v; renderDistribution(); }
    });

    K.caption(root, 'The chips are what a model actually reads: the BPE merges were learned from ' +
      'the small corpus at the top of this file, so "model" is one token and "unbelievable" would ' +
      'be many. The bars are the real sampling rule - temperature reshapes the distribution, ' +
      'top-k deletes the tail. Set top-k to 1 and the model becomes deterministic.');

    renderChips();
    renderDistribution();
  });

  /* ========================================================= 12. prompting */

  K.registerDemo('prompting', function (root) {
    var controls = K.group(root);
    var trace = K.group(root);
    var out = K.readout(root);

    var state = { text: "Hi, I'm Rohith from Chennai - rohith.k@example.com works best." };

    function render() {
      var result = M.extractWithRepair(state.text, 3);
      var messages = [
        { role: 'system', content: 'You are a data extraction service. Return ONLY a JSON object with exactly these keys: name, email, city. Use null for anything you cannot find. No prose, no markdown fences.' },
        { role: 'user', content: 'My name is Asha and you can reach me at asha@example.com. I am in Pune.' },
        { role: 'assistant', content: '{"name":"Asha","email":"asha@example.com","city":"Pune"}' },
        { role: 'user', content: state.text }
      ];

      var promptTokens = messages.reduce(function (a, m) { return a + M.countTokens(m.content); }, 0);
      var zeroShot = promptTokens -
        M.countTokens(messages[1].content) - M.countTokens(messages[2].content);

      trace.innerHTML =
        messages.map(function (m) {
          return '<div class="msg"><span class="msg__role msg__role--' + m.role + '">' + m.role +
            '</span><span class="msg__text">' + esc(m.content.slice(0, 96)) +
            (m.content.length > 96 ? '…' : '') + '</span></div>';
        }).join('') +
        '<div class="attempts">' + result.log.map(function (entry) {
          var ok = entry.problems.length === 0;
          return '<div class="attempt ' + (ok ? 'attempt--ok' : 'attempt--bad') + '">' +
            'attempt ' + entry.attempt + ': ' + (ok ? 'valid' : esc(entry.problems.join('; '))) + '</div>';
        }).join('') + '</div>';

      out.set(
        '<div><span class="ro__k">prompt tokens (few-shot)</span><span class="ro__v">' + promptTokens + '</span></div>' +
        '<div><span class="ro__k">without examples</span><span class="ro__v">' + zeroShot + '</span></div>' +
        '<div class="ro__hi"><span class="ro__k">result</span><span class="ro__v">' +
        (result.payload ? esc(JSON.stringify(result.payload)) : 'gave up (invalid)') + '</span></div>'
      );
    }

    K.textInput(controls, {
      label: 'user message', value: state.text,
      onInput: function (v) { state.text = v; render(); }
    });

    var presets = K.group(controls, 'demo__presets');
    K.button(presets, 'a clean request', function () {
      state.text = "Hi, I'm Rohith from Chennai - rohith.k@example.com works best."; render();
    });
    K.button(presets, 'one with no email', function () {
      state.text = 'I am Vinith, no contact details, somewhere in Bangalore.'; render();
    });

    K.caption(root, 'The message list is the program: a system contract, two few-shot examples, ' +
      'then your text. Delete the email from the input and watch the validator reject every attempt ' +
      '- the repair loop runs, and correctly gives up. No amount of retrying invents an address.');

    render();
  });

  /* =============================================================== 13. RAG */

  K.registerDemo('rag', function (root) {
    var controls = K.group(root);
    var host = K.group(root);
    var chunksHost = K.group(root);
    var out = K.readout(root);

    var state = { question: 'What happens if the learning rate is too large?', k: 3, grounded: true };
    var index = M.buildIndex(DOCS, 45, 12);

    var surface = K.canvasIn(host, 420, 220);
    var plot = K.makePlot(surface, { xMin: 0, xMax: index.length, yMin: 0, yMax: 1.02, padB: 26, padL: 36 });

    function render() {
      var ranked = M.retrieve(state.question, index, index.length);
      var top = ranked.slice(0, state.k);
      var max = ranked[0][0] || 1;

      plot.frame({ xTicks: 0, yTicks: 4, yLabel: 'cosine similarity', yDigits: 1 });
      plot.bars(ranked.map(function (p) { return p[0]; }), {
        colours: ranked.map(function (p, i) { return i < state.k ? C.accent : '#e5e7eb'; }),
        width: 0.8
      });
      plot.text(index.length / 2, 1.06, 'all ' + index.length + ' chunks, ranked', { align: 'center', pixel: false, color: C.text, font: '10px ui-sans-serif' });

      chunksHost.innerHTML = top.map(function (pair, i) {
        return '<div class="chunk"><div class="chunk__head"><span class="chunk__rank">' + (i + 1) +
          '</span><span class="chunk__label">' + esc(pair[1].label) + '</span>' +
          '<span class="chunk__score">' + pair[0].toFixed(3) + '</span></div>' +
          '<p>' + esc(pair[1].text) + '</p></div>';
      }).join('');

      var answer;
      if (state.grounded) {
        answer = M.answerFromContext(M.buildContext(top.map(function (p) { return p[1]; })), state.question);
      } else {
        answer = M.answerWithoutContext(state.question, true);
      }

      out.set(
        '<div><span class="ro__k">top score</span><span class="ro__v">' + ranked[0][0].toFixed(3) + '</span></div>' +
        '<div><span class="ro__k">top chunk</span><span class="ro__v">' + ranked[0][1].label + '</span></div>' +
        '<div><span class="ro__k">answer</span><span class="ro__v ro__v--wide">' + esc(answer) + '</span></div>'
      );
    }

    K.textInput(controls, {
      label: 'question', value: state.question,
      onInput: function (v) { state.question = v; render(); }
    });
    K.slider(controls, {
      label: 'chunks retrieved (k)', min: 1, max: 6, step: 1, value: state.k,
      format: function (v) { return String(v); },
      onInput: function (v) { state.k = v; render(); }
    });
    K.toggle(controls, {
      label: 'turn retrieval OFF (answer from memory)', value: false,
      onChange: function (v) { state.grounded = !v; render(); }
    });

    var presets = K.group(controls, 'demo__presets');
    ['What happens if the learning rate is too large?',
      'Why is cross-entropy better than squared error?',
      'Why is there an overlap between chunks?'].forEach(function (q) {
      K.button(presets, q.slice(0, 32) + '…', function () { state.question = q; render(); });
    });

    K.caption(root, 'Every bar is one chunk of the knowledge base, ranked by cosine similarity. ' +
      'The dark bars are the ones that made it into the prompt. Now tick "turn retrieval OFF": the ' +
      'same question, the same model, and a confident answer with nothing behind it. That is the ' +
      'difference grounding makes.');

    render();
  });

  /* ====================================================== 14. tool calling */

  K.registerDemo('tools-agent', function (root) {
    var controls = K.group(root);
    var traceHost = K.group(root);
    var out = K.readout(root);

    var state = { question: 'What is 2400 * 0.15, and what is the weather in Chennai?' };

    function render() {
      var result = M.runAgent(state.question, DOCS, 6);
      var toolCalls = result.trace.filter(function (t) { return t.kind === 'tool_call'; });

      traceHost.innerHTML = result.trace.map(function (entry) {
        if (entry.kind === 'tool_call') {
          var failed = 'error' in entry.result;
          return '<div class="step ' + (failed ? 'step--bad' : '') + '">' +
            '<span class="step__n">step ' + entry.step + '</span>' +
            '<span class="step__act">→ ' + esc(entry.tool) + '(' + esc(JSON.stringify(entry.arguments)) + ')</span>' +
            '<span class="step__res">' + esc(JSON.stringify(entry.result)) + '</span></div>';
        }
        if (entry.kind === 'answer') {
          return '<div class="step step--ok"><span class="step__n">step ' + entry.step +
            '</span><span class="step__act">← answer</span></div>';
        }
        return '<div class="step step--bad"><span class="step__n">step ' + entry.step +
          '</span><span class="step__act">gave up: step limit reached</span></div>';
      }).join('');

      out.set(
        '<div><span class="ro__k">tools available</span><span class="ro__v">' +
        M.TOOL_SCHEMAS.map(function (t) { return t.name; }).join(', ') + '</span></div>' +
        '<div><span class="ro__k">tool calls</span><span class="ro__v">' + toolCalls.length + '</span></div>' +
        '<div class="ro__hi"><span class="ro__k">final answer</span><span class="ro__v ro__v--wide">' +
        esc(result.answer) + '</span></div>'
      );
    }

    K.textInput(controls, {
      label: 'question', value: state.question,
      onInput: function (v) { state.question = v; render(); }
    });

    var presets = K.group(controls, 'demo__presets');
    K.button(presets, 'two tools at once', function () {
      state.question = 'What is 2400 * 0.15, and what is the weather in Chennai?'; render();
    });
    K.button(presets, 'search the docs', function () {
      state.question = 'Search the docs for chunking.'; render();
    });
    K.button(presets, 'a tool that fails', function () {
      state.question = 'What is 5 apples plus 3 oranges?'; render();
    });

    K.caption(root, 'The loop is a while loop with a budget. The model asks, the code runs the ' +
      'tool, the result goes back in. Try the third preset: the calculator gets an expression it ' +
      'cannot parse, and the error is handed back as a result instead of crashing anything. The ' +
      'model never runs code - your page does.');

    render();
  });

  /* ========================================================= 15. evaluation */

  K.registerDemo('evaluation', function (root) {
    var controls = K.group(root);
    var tableHost = K.group(root);
    var out = K.readout(root);

    var state = { done: false };
    var report = null;

    function render() {
      if (!report) { return; }

      var verdict = function (ok) {
        return '<span class="mark ' + (ok ? 'mark--ok' : 'mark--bad') + '">' +
          (ok ? 'PASS' : 'fail') + '</span>';
      };

      tableHost.innerHTML = '<table class="evaltable"><thead><tr>' +
        '<th>question</th><th>retr</th><th>A</th><th>B</th></tr></thead><tbody>' +
        report.rows.map(function (row) {
          return '<tr><td>' + esc(row.question) + '</td>' +
            '<td>' + verdict(row.retrievalHit) + '</td>' +
            '<td>' + verdict(row.withoutRetrieval.containsFact) + '</td>' +
            '<td>' + verdict(row.withRag.containsFact) + '</td></tr>';
        }).join('') + '</tbody></table>';

      var a = report.withoutRetrieval, b = report.withRag;
      out.set(
        '<div><span class="ro__k">retrieval hit rate</span><span class="ro__v">' +
        (report.retrievalHitRate * 100).toFixed(0) + '%</span></div>' +
        '<div><span class="ro__k">A · no retrieval</span><span class="ro__v">' + a.passed + '/' + a.total +
        ' facts, ' + (a.cited / a.total * 100).toFixed(0) + '% cited</span></div>' +
        '<div class="ro__hi"><span class="ro__k">B · RAG</span><span class="ro__v">' + b.passed + '/' + b.total +
        ' facts, ' + (b.cited / b.total * 100).toFixed(0) + '% cited</span></div>' +
        '<div><span class="ro__k">regressions</span><span class="ro__v">' +
        (report.regressions.length ? report.regressions.length : 'none') + '</span></div>'
      );
    }

    K.button(controls, 'run the evaluation', function () {
      report = M.runEvaluation(GOLDEN, DOCS, 45, 12, 3);
      state.done = true;
      render();
    });

    K.caption(root, 'Six questions, each with a fact the right answer must contain. Column A ' +
      'answers from memory, column B retrieves first. The interesting column is "retr": retrieval ' +
      'finds the right document far more often than the generated answer contains the fact - which ' +
      'is exactly why you score the two stages separately, and why "RAG is bad" is not a diagnosis.');

    if (!GOLDEN.length) {
      tableHost.innerHTML = '<p class="demo__error">The golden set was not generated. ' +
        'Run tools/build_site.py to produce static/knowledge.js.</p>';
    }
  });
})();
