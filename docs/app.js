// ============================================================
// app.js — Main entry point, wires everything together
// ============================================================

"use strict";

(async function main() {
  const canvas = document.getElementById('main-canvas');
  const hudText = document.getElementById('hud-text');
  const gpuBadge = document.getElementById('gpu-badge');

  // Sliders
  const sliders = {
    coupling: { el: document.getElementById('sl-coupling'), vEl: document.getElementById('v-coupling') },
    threshold: { el: document.getElementById('sl-threshold'), vEl: document.getElementById('v-threshold') },
    chaos: { el: document.getElementById('sl-chaos'), vEl: document.getElementById('v-chaos') },
    noise: { el: document.getElementById('sl-noise'), vEl: document.getElementById('v-noise') },
    speed: { el: document.getElementById('sl-speed'), vEl: document.getElementById('v-speed') },
  };

  // Wire slider value displays
  for (const [key, s] of Object.entries(sliders)) {
    s.el.addEventListener('input', () => {
      s.vEl.textContent = parseFloat(s.el.value).toFixed(
        key === 'speed' ? 0 : key === 'noise' ? 3 : 2
      );
    });
  }

  // Task button
  const taskBtn = document.getElementById('btn-task');
  taskBtn.addEventListener('click', () => {
    orch.taskActive = !orch.taskActive;
    taskBtn.classList.toggle('active', orch.taskActive);
    taskBtn.textContent = orch.taskActive ? 'BALANCE \u25b6' : 'BALANCE';
  });

  // Feedback slider
  const fbSlider = document.getElementById('sl-feedback');
  const fbVal = document.getElementById('v-feedback');
  fbSlider.addEventListener('input', () => {
    fbVal.textContent = parseFloat(fbSlider.value).toFixed(2);
  });

  // Origin gain slider
  const ogSlider = document.getElementById('sl-origain');
  const ogVal = document.getElementById('v-origain');
  ogSlider.addEventListener('input', () => {
    ogVal.textContent = parseFloat(ogSlider.value).toFixed(1);
  });

  // Build simulation
  const DIM = 16;  // State dimension (lighter for web)
  const constellation = new Constellation(20, 5, DIM, 3);
  const pendulums = [new DoublePendulum(0), new DoublePendulum(1)];
  const orch = new Orchestrator(constellation, pendulums, DIM);

  // WebGPU
  const gpu = new WebGPUAccelerator();
  const gpuOK = await gpu.init();

  if (gpuOK) {
    gpuBadge.className = 'gpu-badge gpu-on';
    gpuBadge.textContent = 'WebGPU';
  } else {
    gpuBadge.className = 'gpu-badge gpu-off';
    gpuBadge.textContent = 'CPU';
  }

  // Renderer
  const renderer = new Renderer(canvas, orch);

  // FPS tracking
  let frameCount = 0, lastFpsTime = performance.now(), fps = 0;
  let stepTime = 0;

  // Animation loop
  function loop() {
    const t0 = performance.now();

    // Read slider values
    orch.coupling = parseFloat(sliders.coupling.el.value);
    orch.threshold = parseFloat(sliders.threshold.el.value);
    orch.chaos = parseFloat(sliders.chaos.el.value);
    orch.noise = parseFloat(sliders.noise.el.value);
    orch.feedbackValue = parseFloat(fbSlider.value);
    orch.originGain = parseFloat(document.getElementById('sl-origain').value);
    const speed = parseInt(sliders.speed.el.value);

    // Simulation steps
    for (let s = 0; s < speed; s++) {
      orch.step();
    }

    stepTime = performance.now() - t0;

    // Render
    renderer.draw();

    // FPS
    frameCount++;
    const now = performance.now();
    if (now - lastFpsTime >= 1000) {
      fps = frameCount;
      frameCount = 0;
      lastFpsTime = now;
    }

    // HUD
    hudText.textContent =
      `tick=${orch.tick} | ${fps}fps | ${stepTime.toFixed(1)}ms/frame | ` +
      `nodes=${constellation.nTotal} | edges=${constellation.edges.length} | ` +
      `coupling=${orch.coupling.toFixed(2)} chaos=${orch.chaos.toFixed(2)}` +
      (orch.taskActive ? ' | TASK: BALANCE' : '') +
      (Math.abs(orch.feedbackValue) > 0.05 ? ` | fb=${orch.feedbackValue.toFixed(2)}` : '');

    requestAnimationFrame(loop);
  }

  hudText.textContent = 'Starting simulation...';
  requestAnimationFrame(loop);

})();
