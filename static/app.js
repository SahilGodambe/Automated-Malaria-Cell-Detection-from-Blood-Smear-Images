const form = document.querySelector('#analysis-form');
const input = document.querySelector('#image-input');
const fileName = document.querySelector('#file-name');
const preview = document.querySelector('#preview');
const emptyPreview = document.querySelector('.empty-preview');
const status = document.querySelector('#status');
const resultValue = document.querySelector('#result-value');
const dropzone = document.querySelector('#dropzone');

input.addEventListener('change', () => {
  const file = input.files[0];
  if (!file) return;
  fileName.textContent = file.name;
  preview.src = URL.createObjectURL(file);
  preview.hidden = false;
  emptyPreview.hidden = true;
  status.textContent = 'Ready for analysis';
});

['dragenter', 'dragover'].forEach((eventName) => dropzone.addEventListener(eventName, (event) => {
  event.preventDefault();
  dropzone.classList.add('is-dragging');
}));
['dragleave', 'drop'].forEach((eventName) => dropzone.addEventListener(eventName, (event) => {
  event.preventDefault();
  dropzone.classList.remove('is-dragging');
}));
dropzone.addEventListener('drop', (event) => {
  const [file] = event.dataTransfer.files;
  if (!file || !file.type.startsWith('image/')) return;
  const transfer = new DataTransfer();
  transfer.items.add(file);
  input.files = transfer.files;
  input.dispatchEvent(new Event('change'));
});

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!input.files[0]) return;

  const data = new FormData(form);
  status.textContent = 'Preparing original pipeline...';
  resultValue.textContent = 'Analyzing';

  try {
    const response = await fetch('/api/analyze', { method: 'POST', body: data });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || 'Analysis failed');
    status.textContent = 'Analysis complete';
    resultValue.textContent = payload.summary || 'Result available';
    if (payload.annotated_image) {
      preview.src = payload.annotated_image;
      preview.hidden = false;
      emptyPreview.hidden = true;
    }
  } catch (error) {
    status.textContent = error.message;
    resultValue.textContent = 'Not available';
  }
});
