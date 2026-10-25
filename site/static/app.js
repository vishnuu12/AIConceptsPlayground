/* ============================================================================
   The only JavaScript on the site: a filter for the interview page.

   Deliberately tiny. A content site does not need a framework, and keeping this
   to twenty lines means there is nothing here that can break the pages.
   ========================================================================== */

(function () {
  'use strict';

  var input = document.getElementById('qa-filter');
  var count = document.getElementById('qa-count');
  var empty = document.getElementById('qa-empty');
  var body = document.getElementById('qa-body');

  if (!input || !body) {
    return; // not the interview page
  }

  var panels = Array.prototype.slice.call(body.querySelectorAll('[data-topic]'));
  var headings = Array.prototype.slice.call(body.querySelectorAll('.group'));
  var total = body.querySelectorAll('details.qa').length;

  function apply() {
    var needle = input.value.trim().toLowerCase();
    var shown = 0;

    panels.forEach(function (panel) {
      var visible = 0;

      Array.prototype.slice.call(panel.querySelectorAll('details.qa')).forEach(function (item) {
        var text = item.textContent.toLowerCase();
        var match = needle === '' || text.indexOf(needle) !== -1;
        item.hidden = !match;
        if (match) { visible += 1; }
      });

      panel.hidden = visible === 0;
      shown += visible;
    });

    // A group heading with no visible panels underneath is just noise.
    headings.forEach(function (heading) {
      var next = heading.nextElementSibling;
      var anyVisible = false;
      while (next && !next.classList.contains('group')) {
        if (next.hasAttribute('data-topic') && !next.hidden) { anyVisible = true; }
        next = next.nextElementSibling;
      }
      heading.hidden = !anyVisible;
    });

    if (count) {
      count.textContent = needle === ''
        ? total + ' questions shown'
        : shown + ' of ' + total + ' questions match';
    }
    if (empty) { empty.hidden = shown !== 0; }
  }

  input.addEventListener('input', apply);
})();
