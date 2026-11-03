(() => {
  const canvas = document.getElementById('drawCanvas');
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  const clearBtn = document.getElementById('clearBtn');
  const predictBtn = document.getElementById('predictBtn');
  const placeholder = document.getElementById('placeholder');
  const statusEl = document.getElementById('modelStatus');
  const message = document.getElementById('message');
  const predictedDigit = document.getElementById('predictedDigit');
  const resultTitle = document.getElementById('resultTitle');
  const resultSubtitle = document.getElementById('resultSubtitle');
  const bars = document.getElementById('bars');
  const confidenceLabel = document.getElementById('confidenceLabel');

  let drawing = false;
  let hasInk = false;
  let session = null;

  function blankCanvas() {
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 22;
  }
  blankCanvas();

  function setStatus(text, ready) {
    statusEl.querySelector('span').textContent = text;
    statusEl.classList.toggle('ready', !!ready);
  }

  async function loadModel() {
    try {
      if (!window.ort) throw new Error('ONNX Runtime library did not load. Check the app files and reload.');
      ort.env.wasm.wasmPaths = './vendor/';
      session = await ort.InferenceSession.create('./MiniAI_1.0.onnx', {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all'
      });
      predictBtn.disabled = false;
      setStatus('AI model ready', true);
      message.textContent = 'Ready. Draw a digit from 0 to 9.';
    } catch (err) {
      setStatus('Model not loaded', false);
      message.textContent = err.message || String(err);
      console.error(err);
    }
  }

  function pointFromEvent(e) {
    const r = canvas.getBoundingClientRect();
    return {
      x: (e.clientX - r.left) * canvas.width / r.width,
      y: (e.clientY - r.top) * canvas.height / r.height
    };
  }

  canvas.addEventListener('pointerdown', e => {
    e.preventDefault();
    drawing = true;
    hasInk = true;
    placeholder.classList.add('hidden');
    canvas.setPointerCapture(e.pointerId);
    const p = pointFromEvent(e);
    ctx.beginPath();
    ctx.moveTo(p.x, p.y);
    ctx.lineTo(p.x + 0.1, p.y + 0.1);
    ctx.stroke();
  });
  canvas.addEventListener('pointermove', e => {
    if (!drawing) return;
    e.preventDefault();
    const p = pointFromEvent(e);
    ctx.lineTo(p.x, p.y);
    ctx.stroke();
  });
  function stopDrawing() { drawing = false; }
  canvas.addEventListener('pointerup', stopDrawing);
  canvas.addEventListener('pointercancel', stopDrawing);
  canvas.addEventListener('lostpointercapture', stopDrawing);

  clearBtn.addEventListener('click', () => {
    blankCanvas();
    hasInk = false;
    placeholder.classList.remove('hidden');
    predictedDigit.textContent = '—';
    resultTitle.textContent = 'Waiting for input';
    resultSubtitle.textContent = 'Draw a digit and tap Recognize.';
    confidenceLabel.textContent = '—';
    bars.innerHTML = '<div class="empty-bars">Your top predictions will appear here.</div>';
    message.textContent = session ? 'Canvas cleared. Draw another digit.' : 'AI model is not ready yet.';
  });

  function preprocess() {
    const image = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = image.data;
    let minX = canvas.width, minY = canvas.height, maxX = -1, maxY = -1;
    for (let y = 0; y < canvas.height; y++) {
      for (let x = 0; x < canvas.width; x++) {
        const idx = (y * canvas.width + x) * 4;
        const v = data[idx];
        if (v > 28) {
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
    }
    if (maxX < minX || maxY < minY) return null;

    const w = maxX - minX + 1;
    const h = maxY - minY + 1;
    const pad = Math.max(12, Math.round(Math.max(w, h) * 0.18));
    const side = Math.max(w, h) + pad * 2;
    const square = document.createElement('canvas');
    square.width = side;
    square.height = side;
    const sctx = square.getContext('2d', { willReadFrequently: true });
    sctx.fillStyle = '#000';
    sctx.fillRect(0, 0, side, side);
    sctx.imageSmoothingEnabled = true;
    sctx.imageSmoothingQuality = 'high';
    sctx.drawImage(canvas, minX, minY, w, h, pad + (Math.max(w,h)-w)/2, pad + (Math.max(w,h)-h)/2, w, h);

    const tiny = document.createElement('canvas');
    tiny.width = 8;
    tiny.height = 8;
    const tctx = tiny.getContext('2d', { willReadFrequently: true });
    tctx.imageSmoothingEnabled = true;
    tctx.imageSmoothingQuality = 'high';
    tctx.drawImage(square, 0, 0, 8, 8);
    const pixels = tctx.getImageData(0, 0, 8, 8).data;
    const values = new Float32Array(64);
    for (let i = 0; i < 64; i++) values[i] = pixels[i * 4] / 255;
    return values;
  }

  predictBtn.addEventListener('click', async () => {
    if (!session) { message.textContent = 'The AI model is not loaded yet.'; return; }
    if (!hasInk) { message.textContent = 'Draw a digit first.'; return; }
    const input = preprocess();
    if (!input) { message.textContent = 'No visible drawing detected. Try again.'; return; }

    predictBtn.disabled = true;
    message.textContent = 'MiniAI is analyzing your drawing…';
    try {
      const tensor = new ort.Tensor('float32', input, [1, 64]);
      const result = await session.run({ input: tensor });
      const logits = Array.from(result.logits.data);
      const max = Math.max(...logits);
      const exps = logits.map(v => Math.exp(v - max));
      const sum = exps.reduce((a,b) => a+b, 0);
      const probs = exps.map(v => v / sum);
      const ranked = probs.map((p, digit) => ({digit, p})).sort((a,b) => b.p - a.p);
      const best = ranked[0];
      predictedDigit.textContent = String(best.digit);
      resultTitle.textContent = `MiniAI thinks this is ${best.digit}`;
      resultSubtitle.textContent = `${(best.p * 100).toFixed(1)}% model confidence`;
      confidenceLabel.textContent = `${(best.p * 100).toFixed(1)}%`;
      bars.innerHTML = ranked.slice(0, 5).map(item =>
        `<div class="bar-row"><b>${item.digit}</b><div class="bar-track"><div class="bar-fill" style="width:${(item.p*100).toFixed(2)}%"></div></div><span class="bar-value">${(item.p*100).toFixed(1)}%</span></div>`
      ).join('');
      message.textContent = 'Prediction complete. Your drawing stayed on this device.';
    } catch (err) {
      message.textContent = `Prediction failed: ${err.message || err}`;
      console.error(err);
    } finally {
      predictBtn.disabled = false;
    }
  });

  if ('serviceWorker' in navigator && location.protocol.startsWith('http')) {
    window.addEventListener('load', () => {
      navigator.serviceWorker.register('./service-worker.js').catch(err => console.warn('Offline worker:', err));
    });
  }
  loadModel();
})();
