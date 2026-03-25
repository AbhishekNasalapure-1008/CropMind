/* ═══════════════════════════════════════════════════════════════
   CropMind — upload.js
   Handles: drag & drop, file validation, preview, form submission,
            loading overlay animation, fetch to /api/analyze
   ═══════════════════════════════════════════════════════════════ */

'use strict';

/* ── DOM refs ─────────────────────────────────────────────────── */
const uploadZone = document.getElementById('upload-zone');
const fileInput = document.getElementById('file-input');
const uploadPlaceholder = document.getElementById('upload-placeholder');
const uploadPreview = document.getElementById('upload-preview');
const previewImg = document.getElementById('preview-img');
const progressWrap = document.getElementById('upload-progress-wrap');
const progressFill = document.getElementById('upload-progress-fill');
const analyzeBtn = document.getElementById('analyze-btn');
const analyzeForm = document.getElementById('analyze-form');
const loadingOverlay = document.getElementById('loading-overlay');
const loadingProgress = document.getElementById('loading-progress-fill');

let selectedFile = null;

/* ── Loading overlay steps ─────────────────────────────────────── */
const LOADING_STEPS = [
    'Preprocessing image...',
    'Segmenting leaf...',
    'Running AI model...',
    'Generating insights...',
];

function updateLoadingStep(index, pct) {
    const stepEls = document.querySelectorAll('.loading-step');
    stepEls.forEach((el, i) => {
        el.classList.remove('active', 'done');
        if (i < index) el.classList.add('done');
        else if (i === index) el.classList.add('active');
    });
    if (loadingProgress) loadingProgress.style.width = `${pct}%`;
}

function showLoading() {
    loadingOverlay.classList.add('show');
    document.body.style.overflow = 'hidden';
    // Animate through steps
    const stepDurations = [800, 1200, 1600, 600];
    let elapsed = 0;
    const pcts = [18, 45, 78, 95];
    updateLoadingStep(0, pcts[0]);
    stepDurations.forEach((dur, i) => {
        elapsed += dur;
        setTimeout(() => updateLoadingStep(i + 1 < LOADING_STEPS.length ? i + 1 : i, pcts[Math.min(i + 1, pcts.length - 1)]), elapsed);
    });
}

function hideLoading() {
    if (loadingProgress) loadingProgress.style.width = '100%';
    setTimeout(() => {
        loadingOverlay.classList.remove('show');
        document.body.style.overflow = '';
        if (loadingProgress) loadingProgress.style.width = '0%';
        updateLoadingStep(-1, 0);
    }, 400);
}

/* ── File validation ─────────────────────────────────────────── */
const ALLOWED_TYPES = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
const MAX_SIZE_MB = 10;

function validateFile(file) {
    if (!file) return 'Please select an image file.';
    if (!ALLOWED_TYPES.includes(file.type)) {
        return `Unsupported format: ${file.type || 'unknown'}. Please upload JPG or PNG.`;
    }
    if (file.size > MAX_SIZE_MB * 1024 * 1024) {
        return `File too large (${(file.size / 1048576).toFixed(1)} MB). Maximum is ${MAX_SIZE_MB} MB.`;
    }
    return null;
}

/* ── Image preview ────────────────────────────────────────────── */
function simulateProgress(onComplete) {
    let pct = 0;
    progressWrap.classList.add('show');
    const iv = setInterval(() => {
        pct = Math.min(pct + Math.random() * 18, 95);
        progressFill.style.width = `${pct}%`;
        if (pct >= 95) {
            clearInterval(iv);
            setTimeout(() => {
                progressFill.style.width = '100%';
                setTimeout(() => {
                    progressWrap.classList.remove('show');
                    progressFill.style.width = '0%';
                    onComplete?.();
                }, 300);
            }, 150);
        }
    }, 60);
}

function setPreview(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
        previewImg.alt = `Uploaded image: ${file.name}`;
        uploadPlaceholder.style.display = 'none';
        uploadPreview.classList.add('show');
        simulateProgress();
    };
    reader.readAsDataURL(file);
}

function handleFile(file) {
    const err = validateFile(file);
    if (err) {
        window.showToast(err, 'error');
        return;
    }
    selectedFile = file;
    setPreview(file);
}

/* ── Drag and Drop ─────────────────────────────────────────────── */
uploadZone?.addEventListener('dragenter', (e) => {
    e.preventDefault();
    uploadZone.classList.add('drag-over');
});
uploadZone?.addEventListener('dragleave', (e) => {
    if (!uploadZone.contains(e.relatedTarget)) {
        uploadZone.classList.remove('drag-over');
    }
});
uploadZone?.addEventListener('dragover', (e) => { e.preventDefault(); });
uploadZone?.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
    const file = e.dataTransfer?.files?.[0];
    if (file) handleFile(file);
});

/* ── Click to browse ─────────────────────────────────────────── */
uploadZone?.addEventListener('click', (e) => {
    if (e.target.closest('.upload-preview')) return;
    fileInput?.click();
});

document.getElementById('browse-btn')?.addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput?.click();
});

fileInput?.addEventListener('change', (e) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
    // Reset so same file can be re-selected
    e.target.value = '';
});

/* ── Change image (preview overlay button) ─────────────────────── */
document.getElementById('change-image-btn')?.addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput?.click();
});

/* ── Form validation & submission ─────────────────────────────── */
function validateForm() {
    if (!selectedFile) {
        window.showToast('Please upload a leaf image first.', 'error');
        uploadZone?.scrollIntoView({ behavior: 'smooth', block: 'center' });
        return false;
    }
    const requiredSelects = ['crop-type', 'soil-type', 'climate-zone', 'growth-stage'];
    for (const id of requiredSelects) {
        const el = document.getElementById(id);
        if (!el?.value) {
            window.showToast(`Please select a ${id.replace(/-/g, ' ')}.`, 'error');
            el?.focus();
            return false;
        }
    }
    return true;
}

analyzeForm?.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!validateForm()) return;

    const formData = new FormData();
    formData.append('image', selectedFile);
    formData.append('crop_type', document.getElementById('crop-type').value);
    formData.append('soil_type', document.getElementById('soil-type').value);
    formData.append('climate_zone', document.getElementById('climate-zone').value);
    formData.append('growth_stage', document.getElementById('growth-stage').value);
    formData.append('notes', document.getElementById('notes')?.value || '');

    analyzeBtn.disabled = true;
    showLoading();

    try {
        const res = await fetch('/api/analyze', {
            method: 'POST',
            body: formData,
        });

        hideLoading();

        if (res.status === 413) {
            window.showToast('Image too large. Please upload a smaller image (max 10 MB).', 'error');
            return;
        }
        if (res.status === 415) {
            window.showToast('Unsupported file type. Please upload JPG or PNG.', 'error');
            return;
        }

        let data;
        try {
            data = await res.json();
        } catch {
            window.showToast('Server returned an unexpected response. Please try again.', 'error');
            return;
        }

        if (!res.ok || !data.success) {
            window.showToast(data.error || 'Analysis failed. Please try again.', 'error');
            return;
        }

        // Success — hand off to results.js
        window.renderResults(data, previewImg.src);

    } catch (err) {
        hideLoading();
        if (err.name === 'TypeError') {
            window.showToast('Network error — is the server running at localhost:5000?', 'error');
        } else {
            window.showToast('An unexpected error occurred. Please try again.', 'error');
        }
        console.error('[CropMind] Fetch error:', err);
    } finally {
        analyzeBtn.disabled = false;
    }
});

/* ── Reset / analyze another ──────────────────────────────────── */
window.resetAnalysis = function () {
    selectedFile = null;
    previewImg.src = '';
    uploadPreview.classList.remove('show');
    uploadPlaceholder.style.display = '';
    analyzeForm?.reset();
    document.getElementById('results-panel').innerHTML = getPlaceholderHTML();
    const navH = 68;
    const diagEl = document.getElementById('diagnose');
    if (diagEl) {
        window.scrollTo({ top: diagEl.getBoundingClientRect().top + window.pageYOffset - navH, behavior: 'smooth' });
    }
};

function getPlaceholderHTML() {
    return `
    <div class="results-placeholder">
      <div class="big-icon">🔬</div>
      <h3>Awaiting Analysis</h3>
      <p>Upload a leaf image and fill in the crop context, then click "Analyze Leaf" to get AI-powered nutrient deficiency insights.</p>
    </div>
  `;
}
