'use strict';

const dropZone   = document.getElementById('drop-zone');
const fileInput  = document.getElementById('file-input');
const queryImg   = document.getElementById('query-img');
const queryPreview = document.getElementById('query-preview');
const resultsSection = document.getElementById('results-section');
const resultsGrid = document.getElementById('results-grid');
const voteLabel  = document.getElementById('vote-label');
const kLabel     = document.getElementById('k-label');
const errorMsg   = document.getElementById('error-msg');
const spinner    = document.getElementById('spinner');
const encoderInfo = document.getElementById('encoder-info');
const indexInfo  = document.getElementById('index-info');

// Fetch and display server status
async function fetchStatus() {
  try {
    const res = await fetch('/health');
    const data = await res.json();
    encoderInfo.textContent = `Encoder: ${data.encoder || 'none'} (dim ${data.embed_dim || '?'}) · Device: ${data.device}`;
    indexInfo.textContent = `Index: ${data.index_size.toLocaleString()} patches`;
    kLabel.textContent = data.top_k;
  } catch {
    encoderInfo.textContent = 'Server unreachable';
  }
}

fetchStatus();

// Drag-and-drop
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

function handleFile(file) {
  if (!file.type.startsWith('image/')) {
    showError('Please upload an image file.');
    return;
  }

  // Show query preview
  const reader = new FileReader();
  reader.onload = e => {
    queryImg.src = e.target.result;
    queryPreview.classList.remove('hidden');
  };
  reader.readAsDataURL(file);

  uploadAndSearch(file);
}

async function uploadAndSearch(file) {
  clearError();
  resultsSection.classList.add('hidden');
  spinner.classList.remove('hidden');

  const form = new FormData();
  form.append('file', file);

  try {
    const res = await fetch('/search', { method: 'POST', body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();
    renderResults(data);
  } catch (err) {
    showError(`Search failed: ${err.message}`);
  } finally {
    spinner.classList.add('hidden');
  }
}

function renderResults(data) {
  voteLabel.textContent = data.majority_vote;
  resultsGrid.innerHTML = '';

  for (const r of data.results) {
    const card = document.createElement('div');
    card.className = 'result-card';
    card.innerHTML = `
      <img src="${r.thumb_url}" alt="${r.label}" loading="lazy">
      <div class="card-info">
        <div class="card-rank">#${r.rank}</div>
        <div class="card-label">${r.label}</div>
        <div class="card-score">Score: ${r.score.toFixed(4)}</div>
      </div>`;
    resultsGrid.appendChild(card);
  }

  resultsSection.classList.remove('hidden');
}

function showError(msg) {
  errorMsg.textContent = msg;
  errorMsg.classList.remove('hidden');
}

function clearError() {
  errorMsg.textContent = '';
  errorMsg.classList.add('hidden');
}
