/* ============================================================================
   math-llm.js - Part 2's building blocks, as pure functions.

   Same rules as math-foundations.js: no DOM, no globals, no rendering. The
   knowledge base is passed IN rather than read from a global, so every function
   here is testable on its own under Node.

   This mirrors mlconcepts/llm/. The model is a simulator there and it is a
   simulator here - what is real, in both, is the plumbing: chunking, embedding,
   ranking, prompt assembly, the tool loop and the scoring.
   ========================================================================== */

(function () {
  'use strict';

  /* ------------------------------------------------------------------ hashing */

  /**
   * FNV-1a, 32-bit.
   *
   * A STABLE hash is the whole point: the demo must produce the same vectors on
   * every page load, and a real index must produce the same vectors after a
   * restart. (This is the JS cousin of the crc32 the Python side uses - they do
   * not need to agree, only to be deterministic.)
   */
  function hash32(str) {
    var h = 0x811c9dc5;
    for (var i = 0; i < str.length; i++) {
      h ^= str.charCodeAt(i);
      h = Math.imul(h, 0x01000193);
    }
    return h >>> 0;
  }

  var TOKEN_RE = /[a-z0-9']+/g;

  // Same filtering as the Python side: short words and stopwords are dropped,
  // because they appear in every question and every document and so rank at
  // random. Keeping the two implementations aligned is the whole point.
  var MIN_TOKEN_LENGTH = 3;
  var STOPWORDS = {};
  ('the and for are was with that this from you not but has can its all does what how why when ' +
    'which into than they them then there these some such only also more most other over same ' +
    'very will just about after before being both each few had have here him his may our out she ' +
    'should their under upon were who would your').split(' ').forEach(function (w) {
    STOPWORDS[w] = true;
  });

  function tokens(text) {
    var all = String(text).toLowerCase().match(TOKEN_RE) || [];
    var out = [];
    for (var i = 0; i < all.length; i++) {
      if (all[i].length >= MIN_TOKEN_LENGTH && !STOPWORDS[all[i]]) { out.push(all[i]); }
    }
    return out;
  }

  function countTokens(text) {
    var m = String(text).match(/\w+|[^\w\s]/g);
    return m ? m.length : 0;
  }

  /**
   * A DIFFERENT rule from `tokens`, on purpose.
   *
   * `tokens` (used for embeddings) drops stopwords, because a shared "the"
   * carries no meaning. This one keeps them and only drops two-letter words,
   * because it mirrors MockLLM._answer_from_context on the Python side exactly -
   * and the two implementations have to agree, or the page and the test suite
   * would describe different systems.
   */
  function answerTokens(text) {
    var all = String(text).toLowerCase().match(TOKEN_RE) || [];
    var out = [];
    for (var i = 0; i < all.length; i++) {
      if (all[i].length > 2) { out.push(all[i]); }
    }
    return out;
  }

  /* --------------------------------------------------------------- 10. embeddings */

  function embed(text, dim) {
    dim = dim || 256;
    var vec = new Array(dim).fill(0);
    var words = tokens(text);
    var i;

    for (i = 0; i < words.length; i++) {
      vec[hash32(words[i]) % dim] += 1;
    }
    for (i = 0; i + 1 < words.length; i++) {
      vec[hash32(words[i] + '_' + words[i + 1]) % dim] += 0.5;
    }

    var sum = 0;
    for (i = 0; i < dim; i++) { sum += vec[i] * vec[i]; }
    var norm = Math.sqrt(sum);
    if (norm === 0) { return vec; }
    for (i = 0; i < dim; i++) { vec[i] /= norm; }
    return vec;
  }

  function cosine(a, b) {
    var dot = 0, na = 0, nb = 0;
    for (var i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      na += a[i] * a[i];
      nb += b[i] * b[i];
    }
    var denom = Math.sqrt(na) * Math.sqrt(nb);
    return denom === 0 ? 0 : dot / denom;
  }

  function nearest(query, corpus, k) {
    var q = embed(query);
    var scored = corpus.map(function (text) { return [cosine(q, embed(text)), text]; });
    scored.sort(function (a, b) { return b[0] - a[0]; });
    return scored.slice(0, k || 3);
  }

  /* ------------------------------------------------------------ 11. tokenization */

  var END = '</w>';
  var SEP = '\u0001';   // between symbols in a word
  var PAIR = '\u0002';  // between the two halves of a pair key

  function mergeSymbols(symbols, pair) {
    var out = [], i = 0;
    while (i < symbols.length) {
      if (i + 1 < symbols.length && symbols[i] === pair[0] && symbols[i + 1] === pair[1]) {
        out.push(symbols[i] + symbols[i + 1]);
        i += 2;
      } else {
        out.push(symbols[i]);
        i += 1;
      }
    }
    return out;
  }

  /** Byte pair encoding, learned from the corpus you pass in. */
  function trainBPE(text, numMerges) {
    numMerges = numMerges || 60;
    var counts = new Map();

    String(text).toLowerCase().split(/\s+/).forEach(function (word) {
      if (!word) { return; }
      var symbols = word.split('').concat([END]);
      var key = symbols.join(SEP);
      counts.set(key, (counts.get(key) || 0) + 1);
    });

    var merges = [];

    for (var round = 0; round < numMerges; round++) {
      var pairs = new Map();
      counts.forEach(function (frequency, key) {
        var symbols = key.split(SEP);
        for (var i = 0; i + 1 < symbols.length; i++) {
          var p = symbols[i] + PAIR + symbols[i + 1];
          pairs.set(p, (pairs.get(p) || 0) + frequency);
        }
      });
      if (pairs.size === 0) { break; }

      var best = null, bestCount = -1;
      pairs.forEach(function (count, p) {
        if (count > bestCount || (count === bestCount && p < best)) {
          bestCount = count; best = p;
        }
      });

      var halves = best.split(PAIR);
      merges.push([halves[0], halves[1]]);

      var next = new Map();
      counts.forEach(function (frequency, key) {
        var merged = mergeSymbols(key.split(SEP), halves).join(SEP);
        next.set(merged, (next.get(merged) || 0) + frequency);
      });
      counts = next;
    }

    return merges;
  }

  function encodeWord(word, merges) {
    var symbols = String(word).toLowerCase().split('').concat([END]);
    for (var i = 0; i < merges.length; i++) {
      symbols = mergeSymbols(symbols, merges[i]);
    }
    return symbols;
  }

  function tokenizeBPE(text, merges) {
    var out = [];
    String(text).split(/\s+/).forEach(function (word) {
      if (!word) { return; }
      encodeWord(word, merges).forEach(function (t) { out.push(t); });
    });
    return out;
  }

  /* -------------------------------------------------------------- sampling */

  function softmax(logits, temperature) {
    var n = logits.length, i;
    if (temperature <= 0) {
      var out = new Array(n).fill(0);
      var best = 0;
      for (i = 1; i < n; i++) { if (logits[i] > logits[best]) { best = i; } }
      out[best] = 1;
      return out;
    }
    var scaled = logits.map(function (v) { return v / temperature; });
    var max = Math.max.apply(null, scaled);
    var exps = scaled.map(function (v) { return Math.exp(v - max); });
    var total = exps.reduce(function (a, b) { return a + b; }, 0);
    return exps.map(function (v) { return v / total; });
  }

  function entropy(probabilities) {
    var s = 0;
    for (var i = 0; i < probabilities.length; i++) {
      var p = Math.max(1e-12, probabilities[i]);
      s -= p * Math.log(p);
    }
    return s;
  }

  function topK(logits, k) {
    var idx = logits.map(function (v, i) { return i; })
      .sort(function (a, b) { return logits[b] - logits[a]; });
    var keep = idx.slice(0, Math.max(0, k));
    var out = logits.map(function () { return -Infinity; });
    keep.forEach(function (i) { out[i] = logits[i]; });
    return out;
  }

  /* ------------------------------------------------------- 12. prompting */

  function extractFields(text) {
    var email = String(text).match(/[\w.+-]+@[\w-]+\.[\w.]+/);
    var name = String(text).match(/\b(?:my name is|I am|I'm)\s+([A-Z][a-z]+)/);
    var city = String(text).match(/\b(?:in|from)\s+([A-Z][a-z]+)\b/);
    return {
      name: name ? name[1] : null,
      email: email ? email[0] : null,
      city: city ? city[1] : null
    };
  }

  var REQUIRED_FIELDS = ['name', 'email', 'city'];

  function validate(payload) {
    var problems = [];
    if (!payload || typeof payload !== 'object') { return ['output was not a JSON object']; }

    REQUIRED_FIELDS.forEach(function (field) {
      if (!(field in payload)) { problems.push("missing field '" + field + "'"); }
      else if (payload[field] == null || payload[field] === '') {
        problems.push("field '" + field + "' is empty");
      }
    });
    if (typeof payload.email === 'string' && payload.email.indexOf('@') === -1) {
      problems.push('email does not look like an email');
    }
    return problems;
  }

  /** Ask, parse, validate, repair - with a hard attempt limit. */
  function extractWithRepair(text, maxAttempts) {
    maxAttempts = maxAttempts || 3;
    var log = [];
    for (var attempt = 1; attempt <= maxAttempts; attempt++) {
      var payload = extractFields(text);
      var problems = validate(payload);
      log.push({ attempt: attempt, problems: problems });
      if (problems.length === 0) { return { payload: payload, log: log }; }
    }
    return { payload: null, log: log };
  }

  /* ------------------------------------------------------------ 13. RAG */

  function chunkText(text, size, overlap) {
    size = size || 45; overlap = overlap == null ? 12 : overlap;
    if (overlap >= size) { throw new Error('overlap must be smaller than size'); }

    var words = String(text).split(/\s+/).filter(Boolean);
    var chunks = [], start = 0;

    while (start < words.length) {
      var end = start + size;
      chunks.push(words.slice(start, end).join(' '));
      if (end >= words.length) { break; }
      start = end - overlap;
    }
    return chunks;
  }

  function buildIndex(documents, size, overlap) {
    var index = [];
    documents.forEach(function (doc) {
      chunkText(doc.text, size, overlap).forEach(function (text, i) {
        index.push({
          docId: doc.id, title: doc.title, index: i,
          label: doc.id + '#' + i,
          text: text,
          vector: embed(text)
        });
      });
    });
    return index;
  }

  function retrieve(question, index, k) {
    var q = embed(question);
    var scored = index.map(function (chunk) { return [cosine(q, chunk.vector), chunk]; });
    scored.sort(function (a, b) { return b[0] - a[0]; });
    return scored.slice(0, k || 3);
  }

  function buildContext(passages) {
    return passages.map(function (p, i) {
      return '[' + (i + 1) + '] (' + p.label + ') ' + p.text;
    }).join('\n');
  }

  /**
   * The simulator's answer: pick the sentence that overlaps the question most,
   * and cite the passage it came from. Mirrors MockLLM._answer_from_context.
   */
  function answerFromContext(context, question) {
    var marker = /\[(\d+)\]\s*\(([^)]+)\)/g;
    var passages = [], match, lastIndex = 0, pending = null;

    while ((match = marker.exec(context)) !== null) {
      if (pending) { pending.text = context.slice(lastIndex, match.index).trim(); passages.push(pending); }
      pending = { label: match[2], text: '' };
      lastIndex = marker.lastIndex;
    }
    if (pending) { pending.text = context.slice(lastIndex).trim(); passages.push(pending); }
    if (passages.length === 0) { passages = [{ label: '1', text: context.trim() }]; }

    var qWords = {};
    answerTokens(question).forEach(function (w) { qWords[w] = 1; });
    var qCount = Object.keys(qWords).length || 1;

    // Quote the best sentence from each of the top two passages - mirroring
    // MockLLM._answer_from_context. One sentence is not enough: the sentence
    // that repeats the question's words is often not the one holding the answer.
    var quotes = [];
    for (var p = 0; p < passages.length && quotes.length < 2; p++) {
      var passage = passages[p];
      var bestScore = 0, bestSentence = null;

      // Split on sentence-ending punctuation WITHOUT a lookbehind regex, which
      // older Safari does not support.
      (passage.text.match(/[^.!?]+[.!?]*/g) || []).forEach(function (sentence) {
        sentence = sentence.trim();
        if (!sentence) { return; }
        var hits = 0, seen = {};
        answerTokens(sentence).forEach(function (w) {
          if (qWords[w] && !seen[w]) { seen[w] = 1; hits += 1; }
        });
        var score = hits / qCount;
        if (score > bestScore) { bestScore = score; bestSentence = sentence; }
      });

      if (bestSentence) { quotes.push(bestSentence + ' (source: ' + passage.label + ')'); }
    }

    if (quotes.length === 0) { return 'The context does not appear to answer that question.'; }
    return quotes.join(' ');
  }

  function answerWithoutContext(question, hallucinate) {
    if (hallucinate) {
      return 'That is covered in section 4.2 of the handbook: the value is 42, and it applies ' +
        'from the next release. (No source - I was not given one.)';
    }
    return 'Got it - here is my take. You asked about: ' + String(question).slice(0, 120);
  }

  /* --------------------------------------------------------- 14. tool calling */

  /**
   * Arithmetic only, parsed by hand.
   *
   * Never eval() model output. The Python side does this with the ast module and
   * a whitelist of node types; here it is a recursive-descent parser that can
   * produce a number and nothing else.
   */
  function safeEval(expression) {
    // Whitespace is SKIPPED between tokens, never stripped. Stripping it turns
    // "5   3" into "53", which parses perfectly - a silent wrong answer, which
    // is exactly the class of bug this function exists to prevent.
    var s = String(expression);
    var i = 0;

    function skipWhitespace() {
      while (i < s.length && /\s/.test(s.charAt(i))) { i++; }
    }
    function peek() { skipWhitespace(); return s.charAt(i); }
    function number() {
      skipWhitespace();
      var start = i;
      while (i < s.length && /[0-9.]/.test(s.charAt(i))) { i++; }
      if (start === i) { throw new Error('expected a number'); }
      var value = parseFloat(s.slice(start, i));
      if (!isFinite(value)) { throw new Error('not a finite number'); }
      return value;
    }
    function factor() {
      var c = peek();
      if (c === '+') { i++; return factor(); }
      if (c === '-') { i++; return -factor(); }
      if (c === '(') {
        i++;
        var value = expression_();
        if (peek() !== ')') { throw new Error('expected a closing bracket'); }
        i++;
        return value;
      }
      return number();
    }
    function term() {
      var value = factor();
      var c = peek();
      while (c === '*' || c === '/') {
        i++;
        var right = factor();
        value = c === '*' ? value * right : value / right;
        c = peek();
      }
      return value;
    }
    function expression_() {
      var value = term();
      var c = peek();
      while (c === '+' || c === '-') {
        i++;
        var right = term();
        value = c === '+' ? value + right : value - right;
        c = peek();
      }
      return value;
    }

    var result = expression_();
    skipWhitespace();
    if (i !== s.length) { throw new Error('unsupported expression element'); }
    if (!isFinite(result)) { throw new Error('not a finite number'); }
    return result;
  }

  var WEATHER = {
    chennai: { summary: 'Chennai is 34C and humid, light breeze.' },
    bangalore: { summary: 'Bangalore is 27C and cloudy.' },
    mumbai: { summary: 'Mumbai is 31C and rainy.' },
    delhi: { summary: 'Delhi is 38C and hazy.' }
  };

  function calculator(expression) {
    try {
      return { expression: expression, result: safeEval(expression) };
    } catch (error) {
      return { expression: expression, error: error.message };
    }
  }

  function getWeather(city) {
    var key = String(city).trim().toLowerCase();
    if (!WEATHER[key]) { return { city: city, error: "no data for '" + city + "'" }; }
    return { city: city, summary: WEATHER[key].summary };
  }

  function searchDocs(query, documents) {
    var q = embed(query);
    var scored = documents.map(function (doc) { return [cosine(q, embed(doc.text)), doc]; });
    scored.sort(function (a, b) { return b[0] - a[0]; });
    return {
      query: query,
      results: scored.slice(0, 3).map(function (pair) {
        return pair[1].title + ' (' + pair[0].toFixed(2) + ')';
      })
    };
  }

  var TOOL_SCHEMAS = [
    { name: 'get_weather', description: 'Current weather for a city.' },
    { name: 'calculator', description: 'Evaluate an arithmetic expression.' },
    { name: 'search_docs', description: 'Search the documentation for a phrase.' }
  ];

  var TRIGGERS = {
    get_weather: ['weather', 'raining', 'forecast'],
    calculator: ['calculate', 'what is', 'how much is', 'compute'],
    search_docs: ['search', 'look up', 'find the', 'documentation']
  };

  function planToolCall(question, alreadyCalled) {
    var lowered = String(question).toLowerCase();
    var names = Object.keys(TRIGGERS);
    for (var i = 0; i < names.length; i++) {
      var name = names[i];
      if (alreadyCalled.indexOf(name) !== -1) { continue; }
      var hit = TRIGGERS[name].some(function (t) { return lowered.indexOf(t) !== -1; });
      if (!hit) { continue; }

      if (name === 'get_weather') {
        var m = String(question).match(/\bin\s+([A-Za-z]+)/) ||
                String(question).match(/\b([A-Z][a-zA-Z]+)\b/);
        return { name: name, arguments: { city: m ? m[1].charAt(0).toUpperCase() + m[1].slice(1) : 'Chennai' } };
      }
      if (name === 'calculator') {
        return { name: name, arguments: { expression: String(question).replace(/[^0-9+\-*/(). ]/g, ' ').trim() } };
      }
      return { name: name, arguments: { query: String(question).trim().slice(0, 80) } };
    }
    return null;
  }

  function executeTool(call, documents) {
    if (call.name === 'calculator') { return calculator(call.arguments.expression); }
    if (call.name === 'get_weather') { return getWeather(call.arguments.city); }
    if (call.name === 'search_docs') { return searchDocs(call.arguments.query, documents); }
    return { error: "unknown tool '" + call.name + "'" };
  }

  function summariseToolResult(call, result) {
    if (call.name === 'calculator') {
      if ('result' in result) { return result.expression + ' = ' + result.result; }
      return 'calculator failed: ' + result.error;
    }
    if (call.name === 'get_weather') {
      return 'summary' in result ? result.summary : "get_weather failed: " + result.error;
    }
    var n = (result.results || []).length;
    return n ? 'Found ' + n + ' matching passage(s); the top one is: ' + result.results[0]
             : 'No matches.';
  }

  /** The agent loop: bounded, and a tool error is a result rather than a crash. */
  function runAgent(question, documents, maxSteps) {
    maxSteps = maxSteps || 6;
    var trace = [], called = [], parts = [];

    for (var step = 1; step <= maxSteps; step++) {
      var call = planToolCall(question, called);
      if (!call) {
        var answer = parts.length ? parts.join(' ') : answerWithoutContext(question, false);
        trace.push({ step: step, kind: 'answer', text: answer });
        return { answer: answer, trace: trace };
      }
      called.push(call.name);
      var result = executeTool(call, documents);
      trace.push({ step: step, kind: 'tool_call', tool: call.name, arguments: call.arguments, result: result });
      parts.push(summariseToolResult(call, result));
    }

    trace.push({ step: maxSteps, kind: 'gave_up' });
    return { answer: 'I could not finish that within the step limit.', trace: trace };
  }

  /* ----------------------------------------------------------- 15. evaluation */

  var CITATION_RE = /\(source:\s*[^)]+\)/;

  function scoreAnswer(answer, expectedFact) {
    return {
      containsFact: String(answer).toLowerCase().indexOf(String(expectedFact).toLowerCase()) !== -1,
      citesSource: CITATION_RE.test(String(answer))
    };
  }

  function retrievalHit(question, docId, index, k) {
    return retrieve(question, index, k || 3).some(function (pair) { return pair[1].docId === docId; });
  }

  function runEvaluation(cases, documents, size, overlap, k) {
    var index = buildIndex(documents, size, overlap);
    var rows = cases.map(function (testCase) {
      var passages = retrieve(testCase.question, index, k || 3);
      var context = buildContext(passages.map(function (p) { return p[1]; }));
      var grounded = answerFromContext(context, testCase.question);
      var ungrounded = answerWithoutContext(testCase.question, true);

      return {
        question: testCase.question,
        expectedFact: testCase.expectedFact,
        docId: testCase.docId,
        retrievalHit: passages.some(function (p) { return p[1].docId === testCase.docId; }),
        withoutRetrieval: scoreAnswer(ungrounded, testCase.expectedFact),
        withRag: scoreAnswer(grounded, testCase.expectedFact),
        grounded: grounded
      };
    });

    function tally(key) {
      var passed = rows.filter(function (r) { return r[key].containsFact; }).length;
      var cited = rows.filter(function (r) { return r[key].citesSource; }).length;
      return { passed: passed, cited: cited, total: rows.length };
    }

    return {
      index: index,
      rows: rows,
      retrievalHitRate: rows.filter(function (r) { return r.retrievalHit; }).length / rows.length,
      withoutRetrieval: tally('withoutRetrieval'),
      withRag: tally('withRag'),
      regressions: rows.filter(function (r) {
        return r.withoutRetrieval.containsFact && !r.withRag.containsFact;
      }).map(function (r) { return r.question; })
    };
  }

  /* ------------------------------------------------------------------ export */

  var MATH = {
    hash32: hash32, tokens: tokens, answerTokens: answerTokens, countTokens: countTokens,
    embed: embed, cosine: cosine, nearest: nearest,
    trainBPE: trainBPE, encodeWord: encodeWord, tokenizeBPE: tokenizeBPE,
    softmax: softmax, entropy: entropy, topK: topK,
    extractFields: extractFields, validate: validate, extractWithRepair: extractWithRepair,
    REQUIRED_FIELDS: REQUIRED_FIELDS,
    chunkText: chunkText, buildIndex: buildIndex, retrieve: retrieve, buildContext: buildContext,
    answerFromContext: answerFromContext, answerWithoutContext: answerWithoutContext,
    safeEval: safeEval, calculator: calculator, getWeather: getWeather, searchDocs: searchDocs,
    TOOL_SCHEMAS: TOOL_SCHEMAS, planToolCall: planToolCall, executeTool: executeTool,
    summariseToolResult: summariseToolResult, runAgent: runAgent,
    scoreAnswer: scoreAnswer, retrievalHit: retrievalHit, runEvaluation: runEvaluation
  };

  if (typeof window !== 'undefined') { window.ML2 = MATH; }
  if (typeof module !== 'undefined' && module.exports) { module.exports = MATH; }
})();
