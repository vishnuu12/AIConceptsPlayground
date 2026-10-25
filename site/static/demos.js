/* ============================================================================
   demos.js - the shared toolkit for the interactive concept demos.

   No framework, no build step. Three things:

     1. a small canvas plotter (frame / line / scatter / heat / bars / marker)
     2. form controls (slider, button, toggle, select, text input, readout)
     3. a registry so a page says `data-demo="gradient-descent"` and the right
        demo installs itself.

   The maths deliberately does NOT live here - it lives in demos-foundations.js
   and demos-llm.js as pure functions with no DOM access, so it can be tested
   under Node against the Python results in mlconcepts/. This file only draws
   and wires up.
   ========================================================================== */

(function () {
  'use strict';

  var COLORS = {
    grid: '#e9edf3',
    axis: '#a6b2c4',
    text: '#5b6b82',
    accent: '#6d28d9',
    info: '#0369a1',
    ok: '#15803d',
    warn: '#b45309',
    bad: '#b91c1c',
    ink: '#0b1220',
    faint: '#cbd5e1'
  };

  /* ---------------------------------------------------------------- helpers */

  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) { node.className = cls; }
    if (text != null) { node.textContent = text; }
    return node;
  }

  function clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }

  function round(v, digits) {
    var f = Math.pow(10, digits == null ? 2 : digits);
    return Math.round(v * f) / f;
  }

  function num(v, digits) {
    if (!isFinite(v)) { return 'diverged'; }
    var d = digits == null ? 3 : digits;
    return v.toFixed(d);
  }

  function colour(hex, alpha) {
    if (alpha == null || alpha >= 1) { return hex; }
    var r = parseInt(hex.slice(1, 3), 16);
    var g = parseInt(hex.slice(3, 5), 16);
    var b = parseInt(hex.slice(5, 7), 16);
    return 'rgba(' + r + ',' + g + ',' + b + ',' + alpha + ')';
  }

  /* ------------------------------------------------------------ the canvas */

  function canvasIn(parent, width, height) {
    var canvas = el('canvas', 'demo__canvas');
    // The backing store is fixed and the CSS scales it, so the drawing is crisp
    // on a retina screen without any resize bookkeeping.
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    canvas.style.aspectRatio = width + ' / ' + height;
    parent.appendChild(canvas);

    var ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    return { canvas: canvas, ctx: ctx, w: width, h: height };
  }

  /**
   * A tiny 2D plotter over a data rectangle.
   *
   * Every method takes DATA coordinates, not pixels - that is the whole point
   * of the wrapper, so a demo reads like the maths rather than like geometry.
   */
  function makePlot(surface, opts) {
    var o = opts || {};
    var ctx = surface.ctx;
    var padL = o.padL == null ? 34 : o.padL;
    var padR = o.padR == null ? 8 : o.padR;
    var padT = o.padT == null ? 10 : o.padT;
    var padB = o.padB == null ? 24 : o.padB;

    var plotW = surface.w - padL - padR;
    var plotH = surface.h - padT - padB;

    var xMin = o.xMin, xMax = o.xMax, yMin = o.yMin, yMax = o.yMax;

    function px(x) { return padL + (x - xMin) / (xMax - xMin) * plotW; }
    function py(y) { return padT + (yMax - y) / (yMax - yMin) * plotH; }
    function xOf(i) { return xMin + (i / Math.max(1, plotW)) * (xMax - xMin); }
    function yOf(j) { return yMax - (j / Math.max(1, plotH)) * (yMax - yMin); }

    var plot = {
      ctx: ctx, surface: surface,
      plotW: plotW, plotH: plotH, padL: padL, padT: padT,
      px: px, py: py, xOf: xOf, yOf: yOf,

      ranges: function () { return { xMin: xMin, xMax: xMax, yMin: yMin, yMax: yMax }; },
      setY: function (lo, hi) { yMin = lo; yMax = hi; },

      clear: function () {
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, surface.w, surface.h);
      },

      frame: function (opts2) {
        var f = opts2 || {};
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, surface.w, surface.h);

        // grid
        ctx.strokeStyle = COLORS.grid;
        ctx.lineWidth = 1;
        var gx = f.xTicks || 5, gy = f.yTicks || 5;
        var i, t;
        for (i = 0; i <= gx; i++) {
          t = px(xMin + (xMax - xMin) * i / gx);
          ctx.beginPath(); ctx.moveTo(t, padT); ctx.lineTo(t, padT + plotH); ctx.stroke();
        }
        for (i = 0; i <= gy; i++) {
          t = py(yMin + (yMax - yMin) * i / gy);
          ctx.beginPath(); ctx.moveTo(padL, t); ctx.lineTo(padL + plotW, t); ctx.stroke();
        }

        // zero lines
        ctx.strokeStyle = COLORS.faint;
        if (yMin < 0 && yMax > 0) {
          ctx.beginPath(); ctx.moveTo(padL, py(0)); ctx.lineTo(padL + plotW, py(0)); ctx.stroke();
        }
        if (xMin < 0 && xMax > 0) {
          ctx.beginPath(); ctx.moveTo(px(0), padT); ctx.lineTo(px(0), padT + plotH); ctx.stroke();
        }

        // axis labels
        ctx.fillStyle = COLORS.text;
        ctx.font = '10px ui-monospace, monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        for (i = 0; i <= gx; i++) {
          var xv = xMin + (xMax - xMin) * i / gx;
          ctx.fillText(num(xv, f.xDigits == null ? 1 : f.xDigits), px(xv), padT + plotH + 5);
        }
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        for (i = 0; i <= gy; i++) {
          var yv = yMin + (yMax - yMin) * i / gy;
          ctx.fillText(num(yv, f.yDigits == null ? 1 : f.yDigits), padL - 5, py(yv));
        }

        if (f.xLabel) {
          ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
          ctx.fillText(f.xLabel, padL + plotW / 2, surface.h - 1);
        }
        if (f.yLabel) {
          ctx.save();
          ctx.translate(9, padT + plotH / 2);
          ctx.rotate(-Math.PI / 2);
          ctx.textAlign = 'center'; ctx.textBaseline = 'top';
          ctx.fillText(f.yLabel, 0, 0);
          ctx.restore();
        }
        return plot;
      },

      line: function (points, style) {
        var s = style || {};
        if (!points.length) { return plot; }
        ctx.strokeStyle = s.color || COLORS.accent;
        ctx.lineWidth = s.width || 2;
        ctx.setLineDash(s.dash || []);
        ctx.beginPath();
        for (var i = 0; i < points.length; i++) {
          var X = px(points[i][0]), Y = py(points[i][1]);
          if (i === 0) { ctx.moveTo(X, Y); } else { ctx.lineTo(X, Y); }
        }
        ctx.stroke();
        ctx.setLineDash([]);
        return plot;
      },

      scatter: function (points, style) {
        var s = style || {};
        var r = s.r || 2.6;
        ctx.fillStyle = s.color || COLORS.info;
        ctx.globalAlpha = s.alpha == null ? 0.85 : s.alpha;
        for (var i = 0; i < points.length; i++) {
          var X = px(points[i][0]), Y = py(points[i][1]);
          if (X < padL - 4 || X > padL + plotW + 4) { continue; }
          ctx.beginPath();
          ctx.arc(X, Y, r, 0, Math.PI * 2);
          ctx.fill();
        }
        ctx.globalAlpha = 1;
        return plot;
      },

      marker: function (x, y, style) {
        var s = style || {};
        var X = px(x), Y = py(y);
        ctx.fillStyle = s.color || COLORS.accent;
        if (s.cross) {
          ctx.strokeStyle = s.color || COLORS.accent;
          ctx.lineWidth = 2.4;
          var d = s.size || 6;
          ctx.beginPath();
          ctx.moveTo(X - d, Y - d); ctx.lineTo(X + d, Y + d);
          ctx.moveTo(X + d, Y - d); ctx.lineTo(X - d, Y + d);
          ctx.stroke();
        } else {
          ctx.beginPath();
          ctx.arc(X, Y, s.size || 4, 0, Math.PI * 2);
          ctx.fill();
          if (s.ring) {
            ctx.strokeStyle = s.ring; ctx.lineWidth = 1.5; ctx.stroke();
          }
        }
        return plot;
      },

      /**
       * Paint a scalar field over the plot area.
       *
       * `fn(x, y) -> [0, 1]` is evaluated per pixel. Used for the decision
       * boundaries: the probability is the field and the colour is the answer.
       */
      heat: function (fn, ramp, scale) {
        // `scale` > 1 samples the field coarsely and lets the canvas interpolate.
        // A decision boundary costs one forward pass per pixel, so at full
        // resolution it is too slow to redraw while the network is training.
        var step = Math.max(1, Math.round(scale || 1));
        var pw = Math.max(1, Math.round(plotW / step));
        var ph = Math.max(1, Math.round(plotH / step));
        var off = document.createElement('canvas');
        off.width = pw; off.height = ph;
        var octx = off.getContext('2d');
        var img = octx.createImageData(pw, ph);
        var data = img.data;
        var stops = ramp || RAMP_DIVERGING;

        for (var j = 0; j < ph; j++) {
          var y = yOf(j * step);
          for (var i = 0; i < pw; i++) {
            var t = clamp(fn(xOf(i * step), y), 0, 1);
            var c = sampleRamp(stops, t);
            var k = (j * pw + i) * 4;
            data[k] = c[0]; data[k + 1] = c[1]; data[k + 2] = c[2]; data[k + 3] = 255;
          }
        }
        octx.putImageData(img, 0, 0);
        ctx.imageSmoothingEnabled = true;
        ctx.drawImage(off, padL, padT, plotW, plotH);
        return plot;
      },

      /** Vertical bars, one per value. */
      bars: function (values, style) {
        var s = style || {};
        var n = values.length;
        if (!n) { return plot; }
        var slot = plotW / n;
        var barW = slot * (s.width || 0.66);
        var base = py(0);
        for (var i = 0; i < n; i++) {
          var cx = padL + slot * (i + 0.5);
          var top = py(values[i]);
          ctx.fillStyle = (s.colours && s.colours[i]) || s.color || COLORS.accent;
          ctx.globalAlpha = s.alpha == null ? 1 : s.alpha;
          ctx.fillRect(cx - barW / 2, Math.min(top, base), barW, Math.abs(base - top) || 1);
        }
        ctx.globalAlpha = 1;
        return plot;
      },

      text: function (x, y, str, style) {
        var s = style || {};
        ctx.fillStyle = s.color || COLORS.ink;
        ctx.font = s.font || '11px ui-sans-serif, system-ui, sans-serif';
        ctx.textAlign = s.align || 'left';
        ctx.textBaseline = s.baseline || 'top';
        ctx.fillText(str, s.pixel ? x : px(x), s.pixel ? y : py(y));
        return plot;
      }
    };

    return plot;
  }

  /* ------------------------------------------------------------- colour ramps */

  var RAMP_DIVERGING = [[59, 130, 246], [248, 250, 252], [220, 38, 38]];
  var RAMP_SEQ = [[237, 233, 254], [109, 40, 217], [15, 23, 42]];

  function sampleRamp(stops, t) {
    var n = stops.length - 1;
    var scaled = clamp(t, 0, 1) * n;
    var i = Math.min(n - 1, Math.floor(scaled));
    var f = scaled - i;
    var a = stops[i], b = stops[i + 1];
    return [
      Math.round(a[0] + (b[0] - a[0]) * f),
      Math.round(a[1] + (b[1] - a[1]) * f),
      Math.round(a[2] + (b[2] - a[2]) * f)
    ];
  }

  /* ---------------------------------------------------------------- controls */

  function slider(parent, spec) {
    var row = el('div', 'ctl');
    var head = el('div', 'ctl__head');
    head.appendChild(el('span', 'ctl__label', spec.label));
    var value = el('span', 'ctl__value', spec.format ? spec.format(spec.value) : num(spec.value));
    head.appendChild(value);

    var input = el('input', 'ctl__range');
    input.type = 'range';
    input.min = spec.min; input.max = spec.max;
    input.step = spec.step == null ? 1 : spec.step;
    input.value = spec.value;
    input.setAttribute('aria-label', spec.label);

    input.addEventListener('input', function () {
      var v = parseFloat(input.value);
      value.textContent = spec.format ? spec.format(v) : num(v);
      spec.onInput(v);
    });

    row.appendChild(head);
    row.appendChild(input);
    parent.appendChild(row);

    return {
      set: function (v) {
        input.value = v;
        value.textContent = spec.format ? spec.format(v) : num(v);
      },
      value: function () { return parseFloat(input.value); }
    };
  }

  function button(parent, text, onClick) {
    var b = el('button', 'ctl__btn', text);
    b.type = 'button';
    b.addEventListener('click', onClick);
    parent.appendChild(b);
    return b;
  }

  function toggle(parent, spec) {
    var wrap = el('label', 'ctl__toggle');
    var input = el('input');
    input.type = 'checkbox';
    input.checked = !!spec.value;
    input.addEventListener('change', function () { spec.onChange(input.checked); });
    wrap.appendChild(input);
    wrap.appendChild(el('span', null, spec.label));
    parent.appendChild(wrap);
    return { set: function (v) { input.checked = v; } };
  }

  function select(parent, spec) {
    var row = el('div', 'ctl');
    var head = el('div', 'ctl__head');
    head.appendChild(el('span', 'ctl__label', spec.label));
    row.appendChild(head);

    var sel = el('select', 'ctl__select');
    sel.setAttribute('aria-label', spec.label);
    spec.options.forEach(function (opt) {
      var o = el('option', null, opt.label);
      o.value = opt.value;
      if (opt.value === spec.value) { o.selected = true; }
      sel.appendChild(o);
    });
    sel.addEventListener('change', function () { spec.onChange(sel.value); });
    row.appendChild(sel);
    parent.appendChild(row);
    return sel;
  }

  function textInput(parent, spec) {
    var row = el('div', 'ctl');
    var head = el('div', 'ctl__head');
    head.appendChild(el('span', 'ctl__label', spec.label));
    row.appendChild(head);

    var input = el('input', 'ctl__text');
    input.type = 'text';
    input.value = spec.value || '';
    if (spec.placeholder) { input.placeholder = spec.placeholder; }
    input.setAttribute('aria-label', spec.label);
    input.addEventListener('input', function () { spec.onInput(input.value); });
    row.appendChild(input);
    parent.appendChild(row);
    return input;
  }

  function readout(parent) {
    var box = el('div', 'demo__readout');
    parent.appendChild(box);
    return {
      node: box,
      set: function (html) { box.innerHTML = html; }
    };
  }

  function caption(parent, text) {
    parent.appendChild(el('p', 'demo__caption', text));
  }

  /** A row of controls laid out side by side. */
  function group(parent, cls) {
    var g = el('div', 'demo__group' + (cls ? ' ' + cls : ''));
    parent.appendChild(g);
    return g;
  }

  /**
   * A frame-driven loop, so a demo can train visibly instead of blocking.
   *
   * `onFrame()` does a slice of the work and returns true to keep going.
   */
  function animator(onFrame) {
    var handle = null;
    var state = { running: false };

    function tick() {
      if (!state.running) { return; }
      if (onFrame() === false) { state.running = false; return; }
      handle = window.requestAnimationFrame(tick);
    }

    state.start = function () {
      if (state.running) { return; }
      state.running = true;
      handle = window.requestAnimationFrame(tick);
    };
    state.stop = function () {
      state.running = false;
      if (handle) { window.cancelAnimationFrame(handle); handle = null; }
    };
    return state;
  }

  /** A flex row, for demos that show two plots side by side. */
  function row(parent) {
    var r = el('div', 'demo__row');
    parent.appendChild(r);
    return r;
  }

  /* ---------------------------------------------------------------- registry */

  var registry = {};
  var order = [];

  function registerDemo(slug, init) {
    registry[slug] = init;
    order.push(slug);
  }

  function boot() {
    var mounts = document.querySelectorAll('[data-demo]');
    Array.prototype.forEach.call(mounts, function (mount) {
      var slug = mount.getAttribute('data-demo');
      var init = registry[slug];
      if (!init) {
        mount.appendChild(el('p', 'demo__error',
          'This demo did not load. Check the browser console.'));
        return;
      }
      try {
        init(mount);
      } catch (error) {
        mount.appendChild(el('p', 'demo__error', 'This demo failed to run: ' + error.message));
        if (window.console && window.console.error) { window.console.error(error); }
      }
    });
  }

  if (typeof document !== 'undefined') {
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', boot);
    } else {
      boot();
    }
  }

  /* ------------------------------------------------------------------ export */

  var DemoKit = {
    COLORS: COLORS,
    RAMP_DIVERGING: RAMP_DIVERGING,
    RAMP_SEQ: RAMP_SEQ,
    el: el, clamp: clamp, num: num, round: round, colour: colour,
    canvasIn: canvasIn, makePlot: makePlot, sampleRamp: sampleRamp,
    slider: slider, button: button, toggle: toggle, select: select,
    textInput: textInput, readout: readout, caption: caption, group: group,
    animator: animator, row: row,
    registerDemo: registerDemo
  };

  if (typeof window !== 'undefined') { window.DemoKit = DemoKit; }
  if (typeof module !== 'undefined' && module.exports) { module.exports = DemoKit; }
})();
