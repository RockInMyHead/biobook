/* Lab-Journal AI chat: KaTeX render + SVG + interactive E-field widgets */
(function (global) {
  'use strict';

  var K = 8.99e9;
  var EPS0 = 8.85e-12;
  var DEFAULTS = {
    point: { q: 2e-9, r: 0.1 },
    dipole: { q: 2e-9, d: 1e-3, r: 0.1 },
    plane: { sigma: 5e-9, r: 0.1 },
    sphere: { Q: 1e-8, R: 0.05, r: 0.1 },
    cylinder: { lambda: 3e-9, r: 0.1 }
  };
  var MODE_LABELS = {
    point: 'Точечный заряд',
    dipole: 'Диполь',
    plane: 'Заряженная плоскость',
    sphere: 'Сфера',
    cylinder: 'Цилиндр'
  };

  function esc(s) {
    return String(s || '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function sci(x) {
    if (!isFinite(x)) return '—';
    var ax = Math.abs(x);
    if (ax !== 0 && (ax < 1e-3 || ax >= 1e4)) return x.toExponential(3);
    return (Math.round(x * 1e4) / 1e4).toString();
  }

  function katexSafe(tex, display) {
    if (typeof katex === 'undefined') return esc(tex);
    try {
      return katex.renderToString(tex, {
        throwOnError: false,
        displayMode: !!display,
        strict: 'ignore'
      });
    } catch (e) {
      return esc(tex);
    }
  }

  function sanitizeSvg(svg) {
    var s = String(svg || '').trim();
    if (!/^<svg[\s>]/i.test(s)) return '';
    if (/<script|on\w+\s*=|javascript:/i.test(s)) return '';
    return s;
  }

  function compute(mode, p) {
    var r = Math.max(1e-6, +p.r || 0.1);
    if (mode === 'point') {
      var q = +p.q || 0;
      var E = K * Math.abs(q) / (r * r);
      var V = K * q / r;
      return { E: E, V: V, note: 'E = k|q|/r², φ = kq/r' };
    }
    if (mode === 'dipole') {
      var qd = +p.q || 0;
      var d = Math.max(1e-6, +p.d || 1e-3);
      var mom = qd * d;
      var E = (1 / (4 * Math.PI * EPS0)) * (2 * mom) / (r * r * r);
      var V = (1 / (4 * Math.PI * EPS0)) * mom / (r * r);
      return { E: E, V: V, p: mom, note: 'ось диполя: E ≈ (1/4πε₀)·2p/r³, φ ≈ (1/4πε₀)·p/r²' };
    }
    if (mode === 'plane') {
      var sigma = +p.sigma || 0;
      var Ep = Math.abs(sigma) / (2 * EPS0);
      return { E: Ep, V: Ep * r, note: 'бесконечная плоскость: E = σ/(2ε₀), Δφ ≈ E·r' };
    }
    if (mode === 'sphere') {
      var Q = +p.Q || 0;
      var R = Math.max(1e-6, +p.R || 0.05);
      var Es, Vs;
      if (r >= R) {
        Es = K * Math.abs(Q) / (r * r);
        Vs = K * Q / r;
      } else {
        Es = K * Math.abs(Q) * r / (R * R * R);
        Vs = K * Q * (3 * R * R - r * r) / (2 * R * R * R);
      }
      return { E: Es, V: Vs, note: r >= R ? 'снаружи как точечный заряд' : 'внутри равномерно заряженной сферы' };
    }
    if (mode === 'cylinder') {
      var lam = +p.lambda || 0;
      var Ec = Math.abs(lam) / (2 * Math.PI * EPS0 * r);
      var Vc = (lam / (2 * Math.PI * EPS0)) * Math.log(1 / r);
      return { E: Ec, V: Vc, note: 'бесконечный цилиндр: E = λ/(2πε₀ r), φ ~ −(λ/2πε₀)ln r' };
    }
    return { E: 0, V: 0, note: '' };
  }

  function drawField(canvas, mode, p) {
    var ctx = canvas.getContext('2d');
    var w = canvas.width;
    var h = canvas.height;
    var cx = w / 2;
    var cy = h / 2;
    var themeDark = document.documentElement.getAttribute('data-theme') === 'dark';
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = themeDark ? '#1a1a1a' : '#fafafa';
    ctx.fillRect(0, 0, w, h);

    function arrow(x1, y1, x2, y2, color) {
      var ang = Math.atan2(y2 - y1, x2 - x1);
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(x2, y2);
      ctx.lineTo(x2 - 7 * Math.cos(ang - 0.4), y2 - 7 * Math.sin(ang - 0.4));
      ctx.lineTo(x2 - 7 * Math.cos(ang + 0.4), y2 - 7 * Math.sin(ang + 0.4));
      ctx.closePath();
      ctx.fill();
    }

    var accent = '#2563eb';
    var neg = '#ef4444';
    var mute = themeDark ? '#666' : '#bbb';

    if (mode === 'point') {
      ctx.beginPath();
      ctx.arc(cx, cy, 10, 0, Math.PI * 2);
      ctx.fillStyle = accent;
      ctx.fill();
      ctx.fillStyle = '#fff';
      ctx.font = 'bold 12px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('+', cx, cy + 1);
      for (var i = 0; i < 12; i++) {
        var a = (i / 12) * Math.PI * 2;
        arrow(cx + 18 * Math.cos(a), cy + 18 * Math.sin(a),
              cx + 70 * Math.cos(a), cy + 70 * Math.sin(a), accent);
      }
      var pr = 55 + Math.min(80, (p.r || 0.1) * 400);
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = mute;
      ctx.beginPath();
      ctx.arc(cx, cy, pr, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = themeDark ? '#ccc' : '#444';
      ctx.font = '11px sans-serif';
      ctx.fillText('r', cx + pr * 0.7, cy - 8);
    } else if (mode === 'dipole') {
      var gap = 28;
      ctx.beginPath();
      ctx.arc(cx - gap, cy, 9, 0, Math.PI * 2);
      ctx.fillStyle = neg;
      ctx.fill();
      ctx.fillStyle = '#fff';
      ctx.font = 'bold 12px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('−', cx - gap, cy);
      ctx.beginPath();
      ctx.arc(cx + gap, cy, 9, 0, Math.PI * 2);
      ctx.fillStyle = accent;
      ctx.fill();
      ctx.fillStyle = '#fff';
      ctx.fillText('+', cx + gap, cy);
      ctx.strokeStyle = mute;
      ctx.beginPath();
      ctx.moveTo(cx - gap + 10, cy);
      ctx.lineTo(cx + gap - 10, cy);
      ctx.stroke();
      arrow(cx + gap + 14, cy, cx + 90, cy, accent);
      arrow(cx - gap - 14, cy, cx - 90, cy, neg);
      for (var j = 1; j <= 3; j++) {
        var yy = j * 22;
        arrow(cx + gap, cy - 12, cx + 50, cy - yy, accent);
        arrow(cx + gap, cy + 12, cx + 50, cy + yy, accent);
        arrow(cx - gap, cy - 12, cx - 50, cy - yy, neg);
        arrow(cx - gap, cy + 12, cx - 50, cy + yy, neg);
      }
    } else if (mode === 'plane') {
      ctx.fillStyle = accent;
      ctx.globalAlpha = 0.25;
      ctx.fillRect(cx - 8, 20, 16, h - 40);
      ctx.globalAlpha = 1;
      ctx.fillRect(cx - 3, 20, 6, h - 40);
      for (var k = 0; k < 7; k++) {
        var y = 40 + k * 28;
        arrow(cx + 12, y, cx + 95, y, accent);
        arrow(cx - 12, y, cx - 95, y, accent);
      }
    } else if (mode === 'sphere') {
      var Rr = 42;
      ctx.beginPath();
      ctx.arc(cx, cy, Rr, 0, Math.PI * 2);
      ctx.strokeStyle = accent;
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.fillStyle = 'rgba(37,99,235,.12)';
      ctx.fill();
      for (var n = 0; n < 10; n++) {
        var an = (n / 10) * Math.PI * 2;
        arrow(cx + (Rr + 6) * Math.cos(an), cy + (Rr + 6) * Math.sin(an),
              cx + (Rr + 55) * Math.cos(an), cy + (Rr + 55) * Math.sin(an), accent);
      }
      ctx.fillStyle = themeDark ? '#ccc' : '#444';
      ctx.font = '11px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('R', cx, cy - Rr - 6);
    } else if (mode === 'cylinder') {
      ctx.beginPath();
      ctx.ellipse(cx, cy, 28, 70, 0, 0, Math.PI * 2);
      ctx.strokeStyle = accent;
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.fillStyle = 'rgba(37,99,235,.1)';
      ctx.fill();
      for (var m = 0; m < 8; m++) {
        var am = -Math.PI / 2 + (m / 7) * Math.PI;
        arrow(cx + 32 * Math.cos(am * 0.3), cy + 55 * Math.sin(am),
              cx + 95 * Math.cos(am * 0.35), cy + 55 * Math.sin(am), accent);
      }
    }

    ctx.fillStyle = themeDark ? '#aaa' : '#666';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(MODE_LABELS[mode] || mode, 12, 18);
  }

  function mountWidget(el, cfg) {
    var mode = (cfg && cfg.mode) || 'point';
    if (!DEFAULTS[mode]) mode = 'point';
    var params = Object.assign({}, DEFAULTS[mode], cfg || {});
    delete params.mode;

    el.className = 'gpt-widget';
    el.innerHTML =
      '<div class="gpt-widget-head">' +
        '<div class="gpt-widget-tabs"></div>' +
      '</div>' +
      '<div class="gpt-widget-body">' +
        '<canvas width="360" height="220" class="gpt-widget-canvas"></canvas>' +
        '<div class="gpt-widget-controls"></div>' +
      '</div>' +
      '<div class="gpt-widget-out"></div>';

    var tabs = el.querySelector('.gpt-widget-tabs');
    var controls = el.querySelector('.gpt-widget-controls');
    var canvas = el.querySelector('canvas');
    var out = el.querySelector('.gpt-widget-out');
    var state = { mode: mode, params: params };

    Object.keys(MODE_LABELS).forEach(function (m) {
      var b = document.createElement('button');
      b.type = 'button';
      b.className = 'gpt-widget-tab' + (m === state.mode ? ' active' : '');
      b.textContent = MODE_LABELS[m];
      b.addEventListener('click', function () {
        state.mode = m;
        state.params = Object.assign({}, DEFAULTS[m], { r: state.params.r || DEFAULTS[m].r });
        Array.prototype.forEach.call(tabs.children, function (c) { c.classList.remove('active'); });
        b.classList.add('active');
        buildControls();
        refresh();
      });
      tabs.appendChild(b);
    });

    function slider(key, label, min, max, step, scale) {
      scale = scale || 1;
      var wrap = document.createElement('label');
      wrap.className = 'gpt-widget-sl';
      var val = (state.params[key] != null ? state.params[key] : 0) / scale;
      wrap.innerHTML =
        '<span>' + esc(label) + '</span>' +
        '<input type="range" min="' + min + '" max="' + max + '" step="' + step + '" value="' + val + '">' +
        '<em></em>';
      var input = wrap.querySelector('input');
      var em = wrap.querySelector('em');
      function sync() {
        state.params[key] = (+input.value) * scale;
        em.textContent = sci(state.params[key]);
        refresh();
      }
      input.addEventListener('input', sync);
      em.textContent = sci(state.params[key]);
      controls.appendChild(wrap);
    }

    function buildControls() {
      controls.innerHTML = '';
      var m = state.mode;
      if (m === 'point') {
        slider('q', 'q, Кл', 0.2, 10, 0.1, 1e-9);
        slider('r', 'r, м', 0.02, 0.5, 0.01, 1);
      } else if (m === 'dipole') {
        slider('q', 'q, Кл', 0.2, 10, 0.1, 1e-9);
        slider('d', 'd, м', 0.2, 5, 0.1, 1e-3);
        slider('r', 'r, м', 0.02, 0.5, 0.01, 1);
      } else if (m === 'plane') {
        slider('sigma', 'σ, Кл/м²', 0.5, 20, 0.5, 1e-9);
        slider('r', 'Δx, м', 0.02, 0.5, 0.01, 1);
      } else if (m === 'sphere') {
        slider('Q', 'Q, Кл', 1, 50, 1, 1e-9);
        slider('R', 'R, м', 0.01, 0.2, 0.005, 1);
        slider('r', 'r, м', 0.01, 0.4, 0.005, 1);
      } else if (m === 'cylinder') {
        slider('lambda', 'λ, Кл/м', 0.5, 20, 0.5, 1e-9);
        slider('r', 'r, м', 0.02, 0.5, 0.01, 1);
      }
    }

    function refresh() {
      drawField(canvas, state.mode, state.params);
      var res = compute(state.mode, state.params);
      var texE = 'E \\approx ' + sci(res.E).replace(/e([+-]\\d+)/, '\\times 10^{$1}') + '\\,\\mathrm{Н/Кл}';
      var texV = '\\varphi \\approx ' + sci(res.V).replace(/e([+-]\\d+)/, '\\times 10^{$1}') + '\\,\\mathrm{В}';
      // simpler plain + katex for units line
      out.innerHTML =
        '<div class="gpt-widget-math">' + katexSafe('|\\vec{E}| \\approx ' + sci(res.E) + '\\,\\mathrm{Н/Кл}', false) + '</div>' +
        '<div class="gpt-widget-math">' + katexSafe('\\varphi \\approx ' + sci(res.V) + '\\,\\mathrm{В}', false) + '</div>' +
        '<div class="gpt-widget-note">' + esc(res.note) +
          (res.p != null ? ' · p = ' + sci(res.p) + ' Кл·м' : '') + '</div>';
    }

    buildControls();
    refresh();
  }

  function escapeScriptText(code) {
    // Prevent HTML parser from closing the script element early
    return String(code || '').replace(/<\/script/gi, '<\\/script');
  }

  // LLMs often emit over-escaped TeX: \\mathrm / \\tfrac instead of \mathrm / \tfrac
  function undoubleTexInHtml(s) {
    var out = String(s || '');
    var prev;
    do {
      prev = out;
      out = out.replace(/\\\\([a-zA-Z]+|[,;!])/g, '\\$1');
    } while (out !== prev);
    return out;
  }

  function undoubleTexInJs(s) {
    // In JS string literals models write \\\\mathrm → should be \\mathrm in source
    var out = String(s || '');
    var prev;
    do {
      prev = out;
      out = out.replace(/\\\\\\\\([a-zA-Z]+|[,;!])/g, '\\\\$1');
    } while (out !== prev);
    return out;
  }

  function wrapArtifactDoc(inner, id) {
    var raw = String(inner || '').trim();
    var headExtra = '';
    var body = raw;

    if (/<!DOCTYPE html|<html[\s>]/i.test(raw)) {
      var headMatch = raw.match(/<head[^>]*>([\s\S]*?)<\/head>/i);
      if (headMatch) {
        headExtra = headMatch[1].replace(/<script[\s\S]*?<\/script>/gi, '');
      }
      var bodyMatch = raw.match(/<body[^>]*>([\s\S]*)<\/body>/i);
      if (bodyMatch) body = bodyMatch[1];
      else {
        body = raw
          .replace(/<!DOCTYPE[^>]*>/i, '')
          .replace(/<\/?html[^>]*>/gi, '')
          .replace(/<head[\s\S]*?<\/head>/i, '')
          .replace(/<\/?body[^>]*>/gi, '');
      }
    }

    // Pull scripts out, then re-inject after DOM so they actually run against the markup
    var scripts = [];
    var htmlOnly = body.replace(/<script(\s[^>]*)?>([\s\S]*?)<\/script>/gi, function (_, attrs, code) {
      scripts.push({ attrs: attrs || '', code: undoubleTexInJs(code) });
      return '';
    });
    htmlOnly = undoubleTexInHtml(htmlOnly);

    var dark = document.documentElement.getAttribute('data-theme') === 'dark';
    var textColor = dark ? '#ececec' : '#111';
    var cardBg = dark ? '#242424' : '#fff';
    var cardBorder = dark ? 'rgba(255,255,255,.1)' : 'rgba(0,0,0,.08)';

    var parts = [];
    parts.push('<!DOCTYPE html><html><head><meta charset="utf-8">');
    parts.push('<meta name="viewport" content="width=device-width,initial-scale=1">');
    parts.push('<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css" crossorigin="anonymous">');
    parts.push('<style>');
    parts.push('*{box-sizing:border-box}');
    parts.push('html,body{margin:0;padding:0;height:auto!important;overflow:visible!important;');
    parts.push('background:transparent;color:' + textColor + ';');
    parts.push('font:14px/1.5 "Segoe UI",system-ui,-apple-system,sans-serif}');
    parts.push('#lab-root{display:block;padding:8px 4px 10px;overflow:visible;width:100%}');
    parts.push('canvas,svg,img{max-width:100%;height:auto;display:block}');
    parts.push('button,input,select{font:inherit}');
    parts.push('.card{border:1px solid ' + cardBorder + ';border-radius:14px;padding:14px 16px;background:' + cardBg + ';width:100%;max-width:100%;box-sizing:border-box}');
    parts.push('.row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}');
    parts.push('.muted{color:#6b7280;font-size:12px}');
    parts.push('.val{font-variant-numeric:tabular-nums;font-weight:600}');
    parts.push('input[type=range]{width:100%;accent-color:#2563eb}');
    parts.push('h1,h2,h3{margin:0 0 .5rem;font-weight:650;letter-spacing:-.02em}');
    parts.push('.katex{font-size:1.05em;color:inherit}');
    parts.push('.katex-display{margin:.65rem 0;overflow-x:auto}');
    parts.push('</style>');
    if (headExtra) parts.push(headExtra);
    parts.push('<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js" crossorigin="anonymous"><\/script>');
    parts.push('<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js" crossorigin="anonymous"><\/script>');
    parts.push('</head><body>');
    parts.push('<div id="lab-root">' + htmlOnly + '</div>');

    scripts.forEach(function (s) {
      parts.push('<script' + s.attrs + '>' + escapeScriptText(s.code) + '</script>');
    });

    // KaTeX helper + auto-run calc hooks + resize to parent
    // Built as one string to avoid per-line backslash escaping bugs
    var bootJs = [
      '(function(){',
      'var BS=String.fromCharCode(92);',
      'function undouble(s){',
      '  var t=String(s==null?"":s), p, re=new RegExp(BS+BS+"([a-zA-Z]+|[,;!])","g");',
      '  do{p=t;t=t.replace(re,BS+"$1");}while(t!==p);',
      '  return t;',
      '}',
      'function normTex(s){ return undouble(s).replace(/\\s+/g," ").trim(); }',
      'window.labMath=function(el,tex,display){',
      '  if(!el) return;',
      '  var t=normTex(tex);',
      '  if(!window.katex){ el.textContent=t; return; }',
      '  try{ el.innerHTML=katex.renderToString(t,{throwOnError:false,displayMode:!!display,strict:"ignore"}); }',
      '  catch(e){ el.textContent=t; }',
      '};',
      'function fixTextMath(root){',
      '  if(!root) return;',
      '  var w=document.createTreeWalker(root,NodeFilter.SHOW_TEXT,null);',
      '  var nodes=[]; while(w.nextNode()) nodes.push(w.currentNode);',
      '  nodes.forEach(function(n){',
      '    var v=n.nodeValue; if(!v||v.indexOf(BS+BS)===-1) return;',
      '    n.nodeValue=undouble(v);',
      '  });',
      '}',
      'function renderMath(root){',
      '  if(!root) return;',
      '  fixTextMath(root);',
      '  if(!window.renderMathInElement) return;',
      '  try{',
      '    renderMathInElement(root,{',
      '      delimiters:[',
      '        {left:"$$",right:"$$",display:true},',
      '        {left:"\\\\[",right:"\\\\]",display:true},',
      '        {left:"$",right:"$",display:false},',
      '        {left:"\\\\(",right:"\\\\)",display:false}',
      '      ],',
      '      throwOnError:false,strict:"ignore"',
      '    });',
      '  }catch(e){}',
      '}',
      'function kick(){',
      '  ["update","recalc","calculate","render","draw","refresh","compute","onInput","calc","init"].forEach(function(n){',
      '    try{ if(typeof window[n]==="function") window[n](); }catch(e){}',
      '  });',
      '  document.querySelectorAll("input,select,textarea").forEach(function(el){',
      '    try{ el.dispatchEvent(new Event("input",{bubbles:true})); }catch(e){}',
      '    try{ el.dispatchEvent(new Event("change",{bubbles:true})); }catch(e){}',
      '  });',
      '  renderMath(document.getElementById("lab-root"));',
      '}',
      'var last=0;',
      'function contentH(root){',
      '  var rect=root.getBoundingClientRect();',
      '  var top=rect.top, bottom=top;',
      '  var nodes=root.querySelectorAll("*");',
      '  for(var i=0;i<nodes.length;i++){',
      '    var r=nodes[i].getBoundingClientRect();',
      '    if(r.height<1&&r.width<1) continue;',
      '    if(r.bottom>bottom) bottom=r.bottom;',
      '  }',
      '  var h=Math.ceil(bottom-top+12);',
      '  if(!(h>0)) h=Math.ceil(rect.height||120);',
      '  return h;',
      '}',
      'function report(){',
      '  var root=document.getElementById("lab-root"); if(!root) return;',
      '  var h=Math.max(120,Math.min(contentH(root),520));',
      '  if(Math.abs(h-last)<2) return; last=h;',
      '  parent.postMessage({type:"lab-artifact-h",h:h,id:"' + id + '"},"*");',
      '}',
      'var root=document.getElementById("lab-root");',
      'if(window.ResizeObserver&&root){',
      '  var roPending=false;',
      '  new ResizeObserver(function(){',
      '    if(roPending) return; roPending=true;',
      '    requestAnimationFrame(function(){ roPending=false; report(); });',
      '  }).observe(root);',
      '}',
      '["input","change"].forEach(function(ev){',
      '  document.addEventListener(ev,function(){ requestAnimationFrame(function(){ kick(); report(); }); },true);',
      '});',
      'function boot(){ kick(); report(); }',
      'if(window.katex&&window.renderMathInElement) boot(); else {',
      '  var t=0,iv=setInterval(function(){',
      '    t++; if((window.katex&&window.renderMathInElement)||t>50){ clearInterval(iv); boot(); }',
      '  },50);',
      '}',
      '[80,200,500].forEach(function(t){ setTimeout(boot,t); });',
      '})();'
    ].join('\n');
    parts.push('<script>' + bootJs + '</script>');
    parts.push('</body></html>');
    return parts.join('');
  }

  // Store srcdocs separately — attribute size limits / escaping
  var _artifactDocs = {};

  function formatRich(text) {
    var raw = String(text || '');
    var blocks = [];
    function stash(html) {
      var i = blocks.length;
      blocks.push(html);
      return '%%BLK' + i + '%%';
    }

    // [[HTML:Title]] ... [[/HTML]]  or  [[HTML]] ... [[/HTML]]
    raw = raw.replace(/\[\[HTML(?::([^\]]+))?\]\]([\s\S]*?)\[\[\/HTML\]\]/gi, function (_, title, body) {
      var id = 'a' + Math.random().toString(36).slice(2, 10);
      _artifactDocs[id] = wrapArtifactDoc(body, id);
      return stash(
        '<div class="gpt-artifact-mount" data-id="' + esc(id) + '" data-title="' + esc((title || 'Интерактив').trim()) + '"></div>'
      );
    });

    // Also accept ```html ... ``` fences as artifacts
    raw = raw.replace(/```html\s*([\s\S]*?)```/gi, function (_, body) {
      var id = 'a' + Math.random().toString(36).slice(2, 10);
      _artifactDocs[id] = wrapArtifactDoc(body, id);
      return stash(
        '<div class="gpt-artifact-mount" data-id="' + esc(id) + '" data-title="Интерактив"></div>'
      );
    });

    raw = raw.replace(/\[\[WIDGET:([a-zA-Z0-9_-]+)\]\]([\s\S]*?)\[\[\/WIDGET\]\]/g, function (_, kind, body) {
      var cfg = {};
      try { cfg = JSON.parse(String(body).trim() || '{}'); } catch (e) { cfg = {}; }
      if (kind === 'efield') {
        if (!cfg.mode) cfg.mode = 'point';
        return stash('<div class="gpt-widget-mount" data-cfg="' + esc(JSON.stringify(cfg)) + '"></div>');
      }
      return stash('<div class="gpt-muted">Виджет «' + esc(kind) + '»</div>');
    });

    raw = raw.replace(/\[\[SVG\]\]([\s\S]*?)\[\[\/SVG\]\]/g, function (_, svg) {
      var clean = sanitizeSvg(svg);
      if (!clean) return '';
      return stash('<div class="gpt-diagram">' + clean + '</div>');
    });

    // Protect math
    raw = raw.replace(/\$\$([\s\S]+?)\$\$/g, function (_, tex) {
      return stash('<div class="gpt-math-block">' + katexSafe(tex.trim(), true) + '</div>');
    });
    raw = raw.replace(/\\\[([\s\S]+?)\\\]/g, function (_, tex) {
      return stash('<div class="gpt-math-block">' + katexSafe(tex.trim(), true) + '</div>');
    });
    raw = raw.replace(/\$([^\$\n]+?)\$/g, function (_, tex) {
      return stash('<span class="gpt-math-inline">' + katexSafe(tex.trim(), false) + '</span>');
    });
    raw = raw.replace(/\\\(([^\)]+?)\\\)/g, function (_, tex) {
      return stash('<span class="gpt-math-inline">' + katexSafe(tex.trim(), false) + '</span>');
    });

    // Non-html code fences → styled pre
    raw = raw.replace(/```([a-zA-Z0-9_-]*)\s*([\s\S]*?)```/g, function (_, lang, code) {
      return stash(
        '<pre class="gpt-md-pre"><code>' + esc(code.replace(/^\n|\n$/g, '')) + '</code></pre>'
      );
    });

    function inlineMd(s) {
      return s
        .replace(/\*\*\*([^*]+)\*\*\*/g, '<strong><em>$1</em></strong>')
        .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
        .replace(/(^|[^\*])\*([^*]+)\*(?!\*)/g, '$1<em>$2</em>')
        .replace(/`([^`]+)`/g, '<code>$1</code>');
    }

    function isMdTableRow(line) {
      var t = String(line || '').trim();
      return t.indexOf('|') !== -1 && /^\|?.+\|/.test(t) && (t.match(/\|/g) || []).length >= 2;
    }
    function isMdTableSep(line) {
      var t = String(line || '').trim();
      if (!t || t.indexOf('|') === -1) return false;
      // |---|:---:|---:|  (allow spaces)
      var core = t.replace(/^\|/, '').replace(/\|$/, '');
      var parts = core.split('|');
      if (parts.length < 1) return false;
      return parts.every(function (p) {
        return /^\s*:?-{1,}:?\s*$/.test(p);
      });
    }
    function splitMdTableCells(line) {
      var s = String(line || '').trim();
      if (s.charAt(0) === '|') s = s.slice(1);
      if (s.charAt(s.length - 1) === '|') s = s.slice(0, -1);
      return s.split('|').map(function (c) { return c.trim(); });
    }

    function mdToHtml(src) {
      var lines = String(src || '').split('\n');
      var out = [];
      var inUl = false;
      var inOl = false;
      var para = [];

      function closeLists() {
        if (inUl) { out.push('</ul>'); inUl = false; }
        if (inOl) { out.push('</ol>'); inOl = false; }
      }
      function flushPara() {
        if (!para.length) return;
        out.push('<p class="gpt-md-p">' + inlineMd(para.join(' ')) + '</p>');
        para = [];
      }

      for (var i = 0; i < lines.length; i++) {
        var line = lines[i];
        var trimmed = line.trim();

        if (/^%%BLK\d+%%$/.test(trimmed)) {
          flushPara();
          closeLists();
          out.push(trimmed);
          continue;
        }
        if (!trimmed) {
          flushPara();
          closeLists();
          continue;
        }

        // GFM pipe tables: header + separator + body rows
        if (isMdTableRow(trimmed) && i + 1 < lines.length && isMdTableSep(lines[i + 1].trim())) {
          flushPara();
          closeLists();
          var headers = splitMdTableCells(trimmed);
          i += 1; // skip separator
          var rows = [];
          while (i + 1 < lines.length && isMdTableRow(lines[i + 1].trim()) && !isMdTableSep(lines[i + 1].trim())) {
            i += 1;
            rows.push(splitMdTableCells(lines[i].trim()));
          }
          var htmlTbl = ['<div class="gpt-md-table-wrap"><table class="gpt-md-table"><thead><tr>'];
          headers.forEach(function (h) {
            htmlTbl.push('<th>' + inlineMd(h) + '</th>');
          });
          htmlTbl.push('</tr></thead><tbody>');
          rows.forEach(function (cells) {
            htmlTbl.push('<tr>');
            for (var c = 0; c < headers.length; c++) {
              htmlTbl.push('<td>' + inlineMd(cells[c] || '') + '</td>');
            }
            htmlTbl.push('</tr>');
          });
          htmlTbl.push('</tbody></table></div>');
          out.push(htmlTbl.join(''));
          continue;
        }

        var hm = trimmed.match(/^(#{1,4})\s+(.+)$/);
        if (hm) {
          flushPara();
          closeLists();
          var lvl = hm[1].length;
          out.push('<h' + lvl + ' class="gpt-md-h gpt-md-h' + lvl + '">' + inlineMd(hm[2]) + '</h' + lvl + '>');
          continue;
        }
        if (/^(-{3,}|\*{3,}|_{3,})$/.test(trimmed)) {
          flushPara();
          closeLists();
          out.push('<hr class="gpt-md-hr">');
          continue;
        }
        // after esc, > becomes &gt;
        var bq2 = trimmed.match(/^&gt;\s?(.*)$/);
        if (bq2) {
          flushPara();
          closeLists();
          out.push('<blockquote class="gpt-md-quote">' + inlineMd(bq2[1]) + '</blockquote>');
          continue;
        }
        var ul = trimmed.match(/^([\-\*•]|&bull;)\s+(.+)$/);
        if (ul) {
          flushPara();
          if (inOl) { out.push('</ol>'); inOl = false; }
          if (!inUl) { out.push('<ul class="gpt-md-ul">'); inUl = true; }
          out.push('<li>' + inlineMd(ul[2]) + '</li>');
          continue;
        }
        var ol = trimmed.match(/^(\d+)[\.\)]\s+(.+)$/);
        if (ol) {
          flushPara();
          if (inUl) { out.push('</ul>'); inUl = false; }
          if (!inOl) { out.push('<ol class="gpt-md-ol">'); inOl = true; }
          out.push('<li>' + inlineMd(ol[2]) + '</li>');
          continue;
        }

        closeLists();
        para.push(trimmed);
      }
      flushPara();
      closeLists();
      return out.join('');
    }

    var html = mdToHtml(esc(raw));
    html = html.replace(/%%BLK(\d+)%%/g, function (_, i) {
      return blocks[+i] || '';
    });
    return html;
  }

  function mountArtifact(el) {
    var id = el.getAttribute('data-id');
    var title = el.getAttribute('data-title') || 'Интерактив';
    var doc = _artifactDocs[id];
    if (!doc) {
      el.innerHTML = '<div class="gpt-muted">Артефакт недоступен (обновите ответ).</div>';
      return;
    }
    el.innerHTML =
      '<div class="gpt-artifact" data-artifact-id="' + esc(id) + '">' +
        '<div class="gpt-artifact-bar">' +
          '<span class="gpt-artifact-dot"></span>' +
          '<span class="gpt-artifact-title">' + esc(title) + '</span>' +
        '</div>' +
        '<iframe class="gpt-artifact-frame" title="' + esc(title) + '" ' +
          'sandbox="allow-scripts" scrolling="no" loading="lazy" referrerpolicy="no-referrer"></iframe>' +
      '</div>';
    var iframe = el.querySelector('iframe');
    iframe.setAttribute('scrolling', 'no');
    iframe.style.height = '180px';
    iframe.style.maxHeight = '520px';
    // Blob URL: scripts execute reliably (srcdoc + broken </script> escaping was killing JS)
    try {
      var blob = new Blob([doc], { type: 'text/html;charset=utf-8' });
      var url = URL.createObjectURL(blob);
      iframe.onload = function () {
        try { URL.revokeObjectURL(url); } catch (e) {}
      };
      iframe.src = url;
    } catch (e) {
      iframe.srcdoc = doc;
    }
  }

  function hydrate(root) {
    if (!root) return;
    root.querySelectorAll('.gpt-widget-mount').forEach(function (el) {
      if (el.getAttribute('data-hydrated') === '1') return;
      var cfg = {};
      try { cfg = JSON.parse(el.getAttribute('data-cfg') || '{}'); } catch (e) {}
      el.setAttribute('data-hydrated', '1');
      mountWidget(el, cfg);
    });
    root.querySelectorAll('.gpt-artifact-mount').forEach(function (el) {
      if (el.getAttribute('data-hydrated') === '1') return;
      el.setAttribute('data-hydrated', '1');
      mountArtifact(el);
    });
  }

  // Auto-resize sandboxed iframes to full content height (no inner scrollbar)
  if (!global.__labArtifactResizeBound) {
    global.__labArtifactResizeBound = true;
    window.addEventListener('message', function (ev) {
      var d = ev.data;
      if (!d || d.type !== 'lab-artifact-h' || !d.id) return;
      var frame = document.querySelector('.gpt-artifact[data-artifact-id="' + d.id + '"] iframe');
      if (!frame) return;
      var cap = window.innerWidth <= 820 ? 480 : 640;
      var h = Math.max(120, Math.min(+d.h || 120, cap));
      if (Math.abs((parseInt(frame.style.height, 10) || 0) - h) < 2) return;
      frame.style.height = h + 'px';
      frame.style.maxHeight = cap + 'px';
    });
  }

  global.LabAIChatRich = {
    formatRich: formatRich,
    hydrate: hydrate,
    mountWidget: mountWidget,
    defaults: DEFAULTS
  };
})(window);
