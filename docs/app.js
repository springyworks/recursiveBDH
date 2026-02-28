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

  // Mass slider
  const massSlider = document.getElementById('sl-mass');
  const massVal = document.getElementById('v-mass');
  massSlider.addEventListener('input', () => {
    massVal.textContent = parseFloat(massSlider.value).toFixed(1);
  });

  // Damping slider
  const dampSlider = document.getElementById('sl-damp');
  const dampVal = document.getElementById('v-damp');
  dampSlider.addEventListener('input', () => {
    dampVal.textContent = parseFloat(dampSlider.value).toFixed(3);
  });

  // Adrenaline slider
  const adrSlider = document.getElementById('sl-adrenaline');
  const adrVal = document.getElementById('v-adrenaline');
  adrSlider.addEventListener('input', () => {
    adrVal.textContent = parseFloat(adrSlider.value).toFixed(2);
  });

  // AUTO button — smart PD controller + slider auto-regulation
  const autoBtn = document.getElementById('btn-auto');
  let autoMode = false;
  let autoRewardEMA = 0;
  let autoTick = 0;
  let autoPrevErr = [0, 0, 0, 0]; // PD errors for θ1,θ2 per pendulum
  autoBtn.addEventListener('click', () => {
    autoMode = !autoMode;
    autoBtn.classList.toggle('active', autoMode);
    autoBtn.textContent = autoMode ? 'AUTO \u25b6' : 'AUTO';
    if (autoMode) {
      autoRewardEMA = 0;
      autoTick = 0;
      autoPrevErr = [0, 0, 0, 0];
      // Force BALANCE on when AUTO is on
      if (!orch.taskActive) {
        orch.taskActive = true;
        taskBtn.classList.add('active');
        taskBtn.textContent = 'BALANCE \u25b6';
      }
    }
  });

  // Build simulation
  const DIM = 16;  // State dimension (lighter for web)
  const constellation = new Constellation(20, 5, DIM, 3);
  const pendulums = [new DoublePendulum(0), new DoublePendulum(1)];
  const orch = new Orchestrator(constellation, pendulums, DIM);

  // Default: BALANCE task active on start
  orch.taskActive = true;
  taskBtn.classList.add('active');
  taskBtn.textContent = 'BALANCE \u25b6';

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
    orch.originGain = parseFloat(ogSlider.value);
    orch.pendMass = parseFloat(massSlider.value);
    orch.pendDamping = parseFloat(dampSlider.value);
    orch.adrenaline = parseFloat(adrSlider.value);
    const speed = parseInt(sliders.speed.el.value);

    // Simulation steps
    for (let s = 0; s < speed; s++) {
      orch.step();
    }

    // AUTO mode: PD balance controller + slider tuning
    if (autoMode) {
      autoTick++;
      const reward = orch._computeTaskReward();
      autoRewardEMA = 0.95 * autoRewardEMA + 0.05 * reward;

      // --- PD balance via cart (origin) control + elbow torque ---
      // Swivel at grip → no tau1, balance by moving the origin (like cart-pole)
      for (let p = 0; p < pendulums.length; p++) {
        const [t1, t2, w1, w2] = pendulums[p].state;
        // sin(θ) → 0 at inverted (π), sign tells tilt direction
        const err1 = Math.sin(t1);
        const err2 = Math.sin(t2);
        const derr1 = err1 - autoPrevErr[p * 2];
        const derr2 = err2 - autoPrevErr[p * 2 + 1];
        autoPrevErr[p * 2] = err1;
        autoPrevErr[p * 2 + 1] = err2;

        // Cart-pole PD: move origin under the centre of mass
        const Kp_cart = 1.2, Kd_cart = 0.4;
        const cartCorr = Kp_cart * err1 + Kd_cart * (derr1 + w1 * 0.15);
        const ox = pendulums[p].originX + cartCorr;
        const oy = pendulums[p].originY + 0.3 * err2;
        pendulums[p].setOrigin(
          clamp(ox, -pendulums[p].originRange, pendulums[p].originRange),
          clamp(oy, -pendulums[p].originRange, pendulums[p].originRange),
          orch.simDt
        );

        // Elbow PD torque only (no tau1 — swivel)
        const Kp_elbow = 6.0, Kd_elbow = 2.0;
        const pdTau2 = -(Kp_elbow * err2 + Kd_elbow * (derr2 + w2 * 0.3));
        const nsub = 2, subDt = orch.simDt / nsub;
        for (let ss = 0; ss < nsub; ss++) {
          pendulums[p].step(subDt, 0, pdTau2 * 0.3);
        }
      }

      // --- Slider auto-tuning every 20 frames ---
      if (autoTick % 20 === 0) {
        const nudge = () => (Math.random() - 0.5) * 0.015;
        const improving = autoRewardEMA > -0.3;

        // Coupling → moderate (0.4-0.9)
        let cp = parseFloat(sliders.coupling.el.value);
        cp = Math.max(0.15, Math.min(1.2, cp + nudge() + (improving ? 0.003 : -0.003)));
        sliders.coupling.el.value = cp;
        sliders.coupling.vEl.textContent = cp.toFixed(2);

        // Threshold → lower when struggling
        let th = parseFloat(sliders.threshold.el.value);
        th = Math.max(0.05, Math.min(0.6, th + nudge() + (improving ? 0.002 : -0.006)));
        sliders.threshold.el.value = th;
        sliders.threshold.vEl.textContent = th.toFixed(2);

        // Chaos → explore when bad, calm down when good
        let ch = parseFloat(sliders.chaos.el.value);
        ch = Math.max(0, Math.min(0.5, ch + nudge() + (autoRewardEMA < -0.5 ? 0.008 : -0.004)));
        sliders.chaos.el.value = ch;
        sliders.chaos.vEl.textContent = ch.toFixed(2);

        // Noise
        let ns = parseFloat(sliders.noise.el.value);
        ns = Math.max(0.005, Math.min(0.15, ns + nudge() * 0.2));
        sliders.noise.el.value = ns;
        sliders.noise.vEl.textContent = ns.toFixed(3);

        // Origin gain → increase when doing well
        let og = parseFloat(ogSlider.value);
        og = Math.max(1.5, Math.min(7, og + nudge() * 3 + (improving ? 0.03 : -0.02)));
        ogSlider.value = og;
        ogVal.textContent = og.toFixed(1);

        // Mass → lighter is easier to balance
        let ms = parseFloat(massSlider.value);
        ms = Math.max(0.3, Math.min(3, ms + nudge() + (improving ? -0.005 : -0.01)));
        massSlider.value = ms;
        massVal.textContent = ms.toFixed(1);

        // Damping → a bit of damping helps stability
        let dp = parseFloat(dampSlider.value);
        dp = Math.max(0.001, Math.min(0.08, dp + nudge() * 0.05 + (improving ? 0.0005 : 0.001)));
        dampSlider.value = dp;
        dampVal.textContent = dp.toFixed(3);

        // Adrenaline → ramp up when learning, calm when stable
        let ad = parseFloat(adrSlider.value);
        ad = Math.max(0.2, Math.min(1.8, ad + nudge() + (improving ? 0.005 : 0.01)));
        adrSlider.value = ad;
        adrVal.textContent = ad.toFixed(2);
      }
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
      (autoMode ? ' | AUTO' : '') +
      (Math.abs(orch.feedbackValue) > 0.05 ? ` | fb=${orch.feedbackValue.toFixed(2)}` : '');

    requestAnimationFrame(loop);
  }

  hudText.textContent = 'Starting simulation...';
  requestAnimationFrame(loop);

})();
