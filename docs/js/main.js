/* ============================================================
   Video Content Gap Analyzer — Interactions
   ============================================================ */

(function () {
  'use strict';

  /* ---- Theme Toggle ---- */
  var THEME_KEY = 'vcga-theme';
  var html = document.documentElement;

  function getPreferred() {
    var stored = localStorage.getItem(THEME_KEY);
    if (stored) return stored;
    return 'light'; // default to light
  }

  function applyTheme(theme) {
    if (theme === 'dark') {
      html.setAttribute('data-theme', 'dark');
    } else {
      html.removeAttribute('data-theme');
    }
    localStorage.setItem(THEME_KEY, theme);
  }

  // Apply immediately (before DOMContentLoaded to prevent flash)
  applyTheme(getPreferred());

  /* ---- Scroll Reveal (IntersectionObserver) ---- */
  var STAGGER_MS = 100;

  var observer = new IntersectionObserver(
    function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;

        var el = entry.target;
        el.classList.add('visible');

        if (el.classList.contains('reveal-group')) {
          var children = el.querySelectorAll('.reveal-child');
          children.forEach(function (child, i) {
            child.style.setProperty('--delay', i * STAGGER_MS + 'ms');
          });
        }

        observer.unobserve(el);
      });
    },
    { threshold: 0.15 }
  );

  /* ---- DOMContentLoaded ---- */
  document.addEventListener('DOMContentLoaded', function () {

    // Observe reveal elements
    var targets = document.querySelectorAll('.reveal, .reveal-group');
    targets.forEach(function (el) {
      observer.observe(el);
    });

    // Theme toggle button
    var toggle = document.getElementById('theme-toggle');
    if (toggle) {
      toggle.addEventListener('click', function () {
        var current = html.hasAttribute('data-theme') ? 'dark' : 'light';
        applyTheme(current === 'dark' ? 'light' : 'dark');
      });
    }

    // Navbar scroll effect
    var nav = document.querySelector('.nav');
    var backToTop = document.querySelector('.back-to-top');
    var scrollThreshold = 80;
    var topThreshold = 400;

    function onScroll() {
      var y = window.scrollY;

      // Nav background
      if (y > scrollThreshold) {
        nav.classList.add('scrolled');
      } else {
        nav.classList.remove('scrolled');
      }

      // Back to top visibility
      if (backToTop) {
        if (y > topThreshold) {
          backToTop.classList.add('visible');
        } else {
          backToTop.classList.remove('visible');
        }
      }
    }

    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll(); // initial check

    // Back to top click
    if (backToTop) {
      backToTop.addEventListener('click', function () {
        window.scrollTo({ top: 0, behavior: 'smooth' });
      });
    }
  });
})();
