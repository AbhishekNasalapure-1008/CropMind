/* ═══════════════════════════════════════════════════════════════
   CropMind — main.js
   Handles: navbar, hero canvas particles, IntersectionObserver
            reveals, mobile menu, smooth scroll, active nav links
   ═══════════════════════════════════════════════════════════════ */

'use strict';

/* ── Navbar scroll effect ─────────────────────────────────────── */
const navbar = document.getElementById('navbar');
let lastScrollY = 0;

function handleNavbarScroll() {
  const y = window.scrollY;
  navbar.classList.toggle('scrolled', y > 40);
  lastScrollY = y;
}
window.addEventListener('scroll', handleNavbarScroll, { passive: true });
handleNavbarScroll();

/* ── Mobile hamburger menu ─────────────────────────────────────── */
const hamburger   = document.getElementById('hamburger');
const mobileMenu  = document.getElementById('nav-mobile');

hamburger?.addEventListener('click', () => {
  const open = hamburger.classList.toggle('open');
  mobileMenu.classList.toggle('open', open);
  document.body.style.overflow = open ? 'hidden' : '';
});

mobileMenu?.querySelectorAll('a').forEach(link => {
  link.addEventListener('click', () => {
    hamburger.classList.remove('open');
    mobileMenu.classList.remove('open');
    document.body.style.overflow = '';
  });
});

/* ── Smooth scroll for nav links ─────────────────────────────── */
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
  anchor.addEventListener('click', function (e) {
    const id = this.getAttribute('href');
    if (id === '#') return;
    const target = document.querySelector(id);
    if (!target) return;
    e.preventDefault();
    const navH = parseInt(getComputedStyle(document.documentElement)
      .getPropertyValue('--nav-height'), 10) || 68;
    window.scrollTo({
      top: target.getBoundingClientRect().top + window.pageYOffset - navH,
      behavior: 'smooth',
    });
  });
});

/* ── Active nav link on scroll ─────────────────────────────────── */
const sections  = document.querySelectorAll('section[id]');
const navAnchors = document.querySelectorAll('.nav-links a');

function updateActiveNav() {
  const navH = 80;
  let current = '';
  sections.forEach(sec => {
    const top = sec.getBoundingClientRect().top;
    if (top <= navH + 20) current = sec.id;
  });
  navAnchors.forEach(a => {
    a.classList.toggle('active', a.getAttribute('href') === `#${current}`);
  });
}
window.addEventListener('scroll', updateActiveNav, { passive: true });

/* ── IntersectionObserver: .reveal elements ──────────────────── */
const revealObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.classList.add('visible');
      revealObserver.unobserve(entry.target);
    }
  });
}, { threshold: 0.12, rootMargin: '0px 0px -60px 0px' });

document.querySelectorAll('.reveal').forEach(el => revealObserver.observe(el));

/* ── IntersectionObserver: pipeline steps ─────────────────────── */
const pipeObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (!entry.isIntersecting) return;
    const steps = entry.target.querySelectorAll('.pipe-step');
    steps.forEach((step, i) => {
      setTimeout(() => step.classList.add('visible'), i * 130);
    });
    pipeObserver.unobserve(entry.target);
  });
}, { threshold: 0.15 });

const pipeline = document.querySelector('.pipeline');
if (pipeline) pipeObserver.observe(pipeline);

/* ── HERO CANVAS — floating leaf particles ────────────────────── */
(function initHeroCanvas() {
  const canvas = document.getElementById('hero-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  function resize() {
    canvas.width  = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
  }
  resize();

  let resizeTimer;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(resize, 150);
  }, { passive: true });

  const LEAF_COUNT = 10;
  const COLORS = ['#4ADE80', '#A3E635', '#86EFAC', '#6EE7B7', '#D9F99D'];

  function randomBetween(a, b) { return a + Math.random() * (b - a); }

  // Draw a simple leaf shape
  function drawLeaf(ctx, x, y, size, rotation, opacity, color) {
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(rotation);
    ctx.globalAlpha = opacity;
    ctx.fillStyle = color;
    ctx.beginPath();
    // Leaf using bezier curves
    ctx.moveTo(0, -size);
    ctx.bezierCurveTo(size * 0.6, -size * 0.5, size * 0.9, size * 0.2, 0, size);
    ctx.bezierCurveTo(-size * 0.9, size * 0.2, -size * 0.6, -size * 0.5, 0, -size);
    ctx.fill();
    // Midrib
    ctx.strokeStyle = 'rgba(10,15,10,0.4)';
    ctx.lineWidth = 0.6;
    ctx.beginPath();
    ctx.moveTo(0, -size * 0.8);
    ctx.lineTo(0, size * 0.8);
    ctx.stroke();
    ctx.restore();
  }

  const leaves = Array.from({ length: LEAF_COUNT }, () => ({
    x: randomBetween(0, 1),   // normalized 0-1
    y: randomBetween(0, 1),
    size: randomBetween(8, 22),
    vx: randomBetween(-0.12, 0.12),
    vy: randomBetween(-0.15, -0.05),
    rotation: randomBetween(0, Math.PI * 2),
    rotSpeed: randomBetween(-0.008, 0.008),
    opacity: randomBetween(0.15, 0.55),
    color: COLORS[Math.floor(Math.random() * COLORS.length)],
    phaseOffset: randomBetween(0, Math.PI * 2),
  }));

  let animFrame;
  let tick = 0;

  function animate() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    tick += 0.012;

    const W = canvas.width;
    const H = canvas.height;

    leaves.forEach(leaf => {
      const sway = Math.sin(tick + leaf.phaseOffset) * 0.001;
      leaf.x += leaf.vx * 0.002 + sway;
      leaf.y += leaf.vy * 0.002;
      leaf.rotation += leaf.rotSpeed;

      // Wrap around
      if (leaf.y < -0.05) leaf.y = 1.05;
      if (leaf.y > 1.05)  leaf.y = -0.05;
      if (leaf.x < -0.05) leaf.x = 1.05;
      if (leaf.x > 1.05)  leaf.x = -0.05;

      drawLeaf(ctx, leaf.x * W, leaf.y * H, leaf.size, leaf.rotation, leaf.opacity, leaf.color);
    });

    animFrame = requestAnimationFrame(animate);
  }

  animate();

  // Pause animation when tab is hidden (performance)
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      cancelAnimationFrame(animFrame);
    } else {
      animate();
    }
  });
})();

/* ── Toast notification helper ────────────────────────────────── */
window.showToast = function(message, type = 'info', duration = 4000) {
  const container = document.getElementById('toast-container');
  if (!container) return;

  const icons = { error: '⚠️', success: '✅', info: '💡' };
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.innerHTML = `
    <span class="toast-icon">${icons[type] || '💡'}</span>
    <span class="toast-msg">${message}</span>
  `;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateX(20px)';
    toast.style.transition = 'all 0.3s ease';
    setTimeout(() => toast.remove(), 320);
  }, duration);
};
