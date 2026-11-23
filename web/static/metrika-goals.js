/* Lab-Journal Metrika goals (counter 114213599). Safe no-op if ym missing. */
(function () {
  if (window.__bnMetrikaGoalsInit) return;
  window.__bnMetrikaGoalsInit = true;

  var COUNTER = 114213599;
  function hit(name, params) {
    try {
      if (typeof ym === 'function') ym(COUNTER, 'reachGoal', name, params || {});
    } catch (e) {}
  }
  window.bnGoal = hit;

  function onReady(fn) {
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', fn);
    else fn();
  }

  onReady(function () {
    // CTA → registration
    document.querySelectorAll('a[href*="mode=register"]').forEach(function (a) {
      a.addEventListener('click', function () {
        var href = a.getAttribute('href') || '';
        var plan = /plan=pro/i.test(href) ? 'pro' : 'free';
        hit('bn_cta_register', { plan: plan });
      });
    });

    // Yandex ID button (login markup uses #btn-yandex / #btn-yandex-quick)
    document.querySelectorAll(
      '[data-yandex-login], #btn-yandex, #btn-yandex-quick, a[href*="auth/yandex"], button[data-provider="yandex"]'
    ).forEach(function (el) {
      el.addEventListener('click', function () { hit('bn_yandex_id_click'); });
    });

    // Registration form submit — only when mode=register (shared #auth-form on /login)
    var regForm =
      document.getElementById('registerForm') ||
      document.getElementById('auth-form') ||
      document.querySelector('form[action*="register"]');
    if (regForm) {
      regForm.addEventListener('submit', function () {
        var isRegister = false;
        try {
          var sp = new URLSearchParams(location.search || '');
          if ((sp.get('mode') || '').toLowerCase() === 'register') isRegister = true;
        } catch (e) {}
        var action = (regForm.getAttribute('action') || '') + '';
        if (/register/i.test(action)) isRegister = true;
        var modeInput = regForm.querySelector('input[name="mode"]');
        if (modeInput && String(modeInput.value || '').toLowerCase() === 'register') isRegister = true;
        if (document.body && document.body.classList.contains('register-mode')) isRegister = true;
        var card = document.querySelector('.card.register-mode');
        if (card) isRegister = true;
        if (!isRegister) return;
        hit('bn_auth_submit', { mode: 'register' });
      });
    }

    // Prices in view ≥ 2s (timer; cancel on leave)
    var ceny = document.getElementById('ceny');
    if (ceny && 'IntersectionObserver' in window) {
      var seen = false;
      var timer = null;
      var io = new IntersectionObserver(function (entries) {
        entries.forEach(function (en) {
          if (!en.isIntersecting) {
            if (timer) { clearTimeout(timer); timer = null; }
            return;
          }
          if (seen || timer) return;
          timer = setTimeout(function () {
            timer = null;
            if (seen) return;
            seen = true;
            hit('bn_view_prices');
            io.disconnect();
          }, 2000);
        });
      }, { threshold: 0.35 });
      io.observe(ceny);
    }
    document.querySelectorAll('a[href*="#ceny"]').forEach(function (a) {
      a.addEventListener('click', function () { hit('bn_view_prices', { via: 'click' }); });
    });

    // FAQ open — toggle only when opening
    document.querySelectorAll('#faq details, [data-faq-item]').forEach(function (el) {
      el.addEventListener('toggle', function () {
        if (el.open) hit('bn_faq_open');
      });
    });

    // Guide scroll depth
    if (/\/guides\//.test(location.pathname)) {
      var fired = false;
      window.addEventListener('scroll', function () {
        if (fired) return;
        var h = document.documentElement.scrollHeight - window.innerHeight;
        if (h > 0 && window.scrollY / h >= 0.6) {
          fired = true;
          hit('bn_guide_read');
        }
      }, { passive: true });
    }
  });
})();
