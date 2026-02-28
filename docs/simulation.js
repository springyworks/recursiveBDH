// ============================================================
// simulation.js — Pure JS BDH + DLINOSS + Pendulum simulation
// Runs on CPU; optionally offloads to WebGPU compute shaders
// ============================================================

"use strict";

// ---- Utility ----
function randn() {
  // Box-Muller
  let u1 = Math.random(), u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1 + 1e-12)) * Math.cos(2 * Math.PI * u2);
}
function zeros(n) { return new Float32Array(n); }
function randnArr(n, std = 1) {
  const a = new Float32Array(n);
  for (let i = 0; i < n; i++) a[i] = randn() * std;
  return a;
}
function dot(a, b, n) {
  let s = 0; for (let i = 0; i < n; i++) s += a[i] * b[i]; return s;
}
function norm(a, n) { return Math.sqrt(dot(a, a, n)); }
function addScaled(out, a, b, s, n) {
  for (let i = 0; i < n; i++) out[i] = a[i] + s * b[i];
}
function clamp(x, lo, hi) { return x < lo ? lo : x > hi ? hi : x; }
function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

// ============================================================
// BDH Node — simplified continuous dynamics
// ============================================================
class BDHNode {
  constructor(dim, nodeId) {
    this.dim = dim;
    this.nodeId = nodeId;
    // Encoder/decoder weights (random small)
    this.encoderW = randnArr(dim * dim, 0.02);
    this.decoderW = randnArr(dim * dim, 0.02);
    this.gateW = randnArr(dim * dim, 0.02);
    this.activation = zeros(dim);
    this.bufferLen = 8;
    this.buffer = new Float32Array(this.bufferLen * dim);
    this.bufPtr = 0;
    this.energy = 0;
  }

  // Push to circular buffer
  push(vec) {
    const off = this.bufPtr * this.dim;
    for (let i = 0; i < this.dim; i++) this.buffer[off + i] = vec[i];
    this.bufPtr = (this.bufPtr + 1) % this.bufferLen;
  }

  // Encode: W @ x with ReLU (sparse pulsing)
  encode(x, W) {
    const d = this.dim, out = zeros(d);
    for (let i = 0; i < d; i++) {
      let s = 0;
      for (let j = 0; j < d; j++) s += W[i * d + j] * x[j];
      out[i] = Math.max(0, s); // ReLU sparse
    }
    return out;
  }

  step(input) {
    const d = this.dim;
    // Mix input with activation (layer-norm-like: just normalize)
    const combined = zeros(d);
    let cn = 0;
    for (let i = 0; i < d; i++) {
      combined[i] = input[i] + this.activation[i];
      cn += combined[i] * combined[i];
    }
    cn = Math.sqrt(cn + 1e-8);
    for (let i = 0; i < d; i++) combined[i] /= cn;

    this.push(combined);

    // BDH: encode → gate with buffer mean → decode
    const encoded = this.encode(combined, this.encoderW);

    // Buffer mean
    const bufMean = zeros(d);
    for (let b = 0; b < this.bufferLen; b++) {
      const off = b * d;
      for (let i = 0; i < d; i++) bufMean[i] += this.buffer[off + i];
    }
    for (let i = 0; i < d; i++) bufMean[i] /= this.bufferLen;

    const gated = this.encode(bufMean, this.gateW);

    // Sparse product gating
    const product = zeros(d);
    for (let i = 0; i < d; i++) product[i] = encoded[i] * gated[i];

    // Decode
    const output = zeros(d);
    for (let i = 0; i < d; i++) {
      let s = 0;
      for (let j = 0; j < d; j++) s += this.decoderW[i * d + j] * product[j];
      output[i] = s;
    }

    // Normalize output + residual
    let on = 0;
    for (let i = 0; i < d; i++) {
      output[i] += combined[i];
      on += output[i] * output[i];
    }
    on = Math.sqrt(on + 1e-8);
    for (let i = 0; i < d; i++) {
      output[i] /= on;
      this.activation[i] = output[i];
    }

    this.energy = norm(this.activation, d);
    return output;
  }

  getPulse(threshold) {
    const d = this.dim, pulse = zeros(d);
    for (let i = 0; i < d; i++) {
      if (Math.abs(this.activation[i]) > threshold) {
        pulse[i] = this.activation[i];
      }
    }
    return pulse;
  }
}

// ============================================================
// DLINOSS Node — damped oscillator SSM
// ============================================================
class DLinOSSNode {
  constructor(dim, nodeId) {
    this.dim = dim;
    this.nodeId = nodeId;
    // SSM parameters
    this.A = new Float32Array(dim);
    this.G = new Float32Array(dim);
    this.dt_raw = new Float32Array(dim);
    this.B = randnArr(dim * dim, 1 / Math.sqrt(dim));
    this.C = randnArr(dim * dim, 1 / Math.sqrt(dim));
    this.D_diag = randnArr(dim, 1.0);
    // State
    this.x_state = zeros(dim);
    this.z_state = zeros(dim);
    this.energy = 0;

    // Initialize with oscillatory parameters
    for (let i = 0; i < dim; i++) {
      this.A[i] = 0.5 + Math.random() * 7.5;
      this.G[i] = 0.1 + Math.random() * 1.9;
      this.dt_raw[i] = 0.05;
    }
  }

  reset() {
    for (let i = 0; i < this.dim; i++) {
      this.x_state[i] = randn() * 0.1;
      this.z_state[i] = randn() * 0.1;
    }
  }

  step(u) {
    const d = this.dim;
    // B @ u
    const Bu = zeros(d);
    for (let i = 0; i < d; i++) {
      let s = 0;
      for (let j = 0; j < d; j++) s += this.B[i * d + j] * u[j];
      Bu[i] = s;
    }

    // Damped IMEX1 recurrence
    for (let i = 0; i < d; i++) {
      const dt = sigmoid(this.dt_raw[i]);
      const A = Math.max(0, this.A[i]);
      const G_val = Math.max(0, this.G[i]);
      const S = 1.0 + dt * G_val;
      const z_new = (this.z_state[i] + dt * (-A * this.x_state[i] + Bu[i])) / S;
      const x_new = this.x_state[i] + dt * z_new;
      this.z_state[i] = z_new;
      this.x_state[i] = x_new;
    }

    // C @ x + D * u
    const out = zeros(d);
    for (let i = 0; i < d; i++) {
      let s = 0;
      for (let j = 0; j < d; j++) s += this.C[i * d + j] * this.x_state[j];
      out[i] = s + this.D_diag[i] * u[i];
    }

    this.energy = 0;
    for (let i = 0; i < d; i++) {
      this.energy += 0.5 * this.z_state[i] * this.z_state[i]
                   + 0.5 * Math.max(0, this.A[i]) * this.x_state[i] * this.x_state[i];
    }
    return out;
  }
}

// ============================================================
// Double Pendulum — Lagrangian mechanics
// ============================================================
class DoublePendulum {
  constructor(id, l1 = 1.0, l2 = 0.8, m1 = 1.5, m2 = 1.0) {
    this.id = id;
    this.l1 = l1; this.l2 = l2;
    this.m1 = m1; this.m2 = m2;
    this.g = 9.81; this.damping = 0.02;
    this.state = [Math.PI / 2 + 0.3 * id, Math.PI / 2 - 0.2 * id, 0, 0];

    this.trailX = []; this.trailY = [];
    this.maxTrail = 250;
    this.energy = 0;
  }

  derivatives(s, tau1, tau2) {
    const [t1, t2, w1, w2] = s;
    const { m1, m2, l1, l2, g, damping: b } = this;
    const delta = t2 - t1;
    const cd = Math.cos(delta), sd = Math.sin(delta);
    let den1 = (m1 + m2) * l1 - m2 * l1 * cd * cd;
    let den2 = (l2 / l1) * den1;
    den1 = den1 > 0 ? Math.max(den1, 1e-8) : Math.min(den1, -1e-8);
    den2 = den2 > 0 ? Math.max(den2, 1e-8) : Math.min(den2, -1e-8);

    const dw1 = (m2 * l1 * w1 * w1 * sd * cd + m2 * g * Math.sin(t2) * cd
      + m2 * l2 * w2 * w2 * sd - (m1 + m2) * g * Math.sin(t1) - b * w1 + tau1) / den1;
    const dw2 = (-m2 * l2 * w2 * w2 * sd * cd + (m1 + m2) * g * Math.sin(t1) * cd
      - (m1 + m2) * l1 * w1 * w1 * sd - (m1 + m2) * g * Math.sin(t2) - b * w2 + tau2) / den2;
    return [w1, w2, dw1, dw2];
  }

  step(dt, tau1 = 0, tau2 = 0) {
    tau1 = clamp(tau1, -10, 10);
    tau2 = clamp(tau2, -10, 10);
    const s = this.state;
    const k1 = this.derivatives(s, tau1, tau2);
    const s2 = s.map((v, i) => v + 0.5 * dt * k1[i]);
    const k2 = this.derivatives(s2, tau1, tau2);
    const s3 = s.map((v, i) => v + 0.5 * dt * k2[i]);
    const k3 = this.derivatives(s3, tau1, tau2);
    const s4 = s.map((v, i) => v + dt * k3[i]);
    const k4 = this.derivatives(s4, tau1, tau2);
    for (let i = 0; i < 4; i++) {
      this.state[i] += (dt / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
    }
    // Trail
    const pos = this.getPositions();
    this.trailX.push(pos[2]); this.trailY.push(pos[3]);
    if (this.trailX.length > this.maxTrail) {
      this.trailX.shift(); this.trailY.shift();
    }
    this.computeEnergy();
  }

  getPositions() {
    const [t1, t2] = this.state;
    const x1 = this.l1 * Math.sin(t1);
    const y1 = -this.l1 * Math.cos(t1);
    const x2 = x1 + this.l2 * Math.sin(t2);
    const y2 = y1 - this.l2 * Math.cos(t2);
    return [x1, y1, x2, y2];
  }

  computeEnergy() {
    const [t1, t2, w1, w2] = this.state;
    const { m1, m2, l1, l2, g } = this;
    const T = 0.5 * m1 * (l1 * w1) ** 2
      + 0.5 * m2 * ((l1 * w1) ** 2 + (l2 * w2) ** 2
        + 2 * l1 * l2 * w1 * w2 * Math.cos(t1 - t2));
    const V = -(m1 + m2) * g * l1 * Math.cos(t1) - m2 * g * l2 * Math.cos(t2);
    this.energy = T + V;
  }
}

// ============================================================
// Constellation — scale-free graph of BDH + DLINOSS
// ============================================================
class Constellation {
  constructor(nBDH = 20, nDLinOSS = 5, dim = 16, baM = 3) {
    this.nBDH = nBDH;
    this.nDLinOSS = nDLinOSS;
    this.nTotal = nBDH + nDLinOSS;
    this.dim = dim;

    // Nodes
    this.nodes = [];
    this.nodeTypes = [];
    for (let i = 0; i < nBDH; i++) {
      this.nodes.push(new BDHNode(dim, i));
      this.nodeTypes.push('bdh');
    }
    for (let i = 0; i < nDLinOSS; i++) {
      const d = new DLinOSSNode(dim, nBDH + i);
      d.reset();
      this.nodes.push(d);
      this.nodeTypes.push('dlinoss');
    }

    // Build Barabási-Albert scale-free graph
    this.edges = [];
    this.edgeWeights = [];
    this.adjIn = Array.from({ length: this.nTotal }, () => []);
    this._buildBAGraph(baM);

    // Add cycle back-edges for emergence
    for (let i = 0; i < this.nTotal - 1; i += 5) {
      const j = (i + 7) % this.nTotal;
      this._addEdge(j, i);
    }

    // Sort nodes by degree to identify hubs
    const deg = zeros(this.nTotal);
    for (const [u, v] of this.edges) { deg[u]++; deg[v]++; }
    const ranked = Array.from({ length: this.nTotal }, (_, i) => i)
      .sort((a, b) => deg[b] - deg[a]);
    this.hubIds = ranked.slice(0, 5);

    // State arrays
    this.outputs = Array.from({ length: this.nTotal }, () => zeros(dim));
    this.pulses = Array.from({ length: this.nTotal }, () => zeros(dim));
    this.energies = zeros(this.nTotal);
    this.pulseRates = zeros(this.nTotal);
    this._pulseHistLen = 50;
    this._pulseHist = Array.from({ length: this.nTotal }, () => []);

    // Graph layout (force-directed, precomputed)
    this.positions = this._computeLayout();
  }

  _buildBAGraph(m) {
    // Start with complete graph of m+1 nodes
    for (let i = 0; i <= m && i < this.nTotal; i++) {
      for (let j = i + 1; j <= m && j < this.nTotal; j++) {
        this._addEdge(i, j);
      }
    }
    // Attach remaining nodes with preferential attachment
    const deg = zeros(this.nTotal);
    for (const [u, v] of this.edges) { deg[u]++; deg[v]++; }
    for (let i = m + 1; i < this.nTotal; i++) {
      const targets = new Set();
      let totalDeg = 0;
      for (let j = 0; j < i; j++) totalDeg += deg[j] + 1;
      while (targets.size < m && targets.size < i) {
        let r = Math.random() * totalDeg;
        for (let j = 0; j < i; j++) {
          r -= (deg[j] + 1);
          if (r <= 0) { targets.add(j); break; }
        }
      }
      for (const t of targets) {
        this._addEdge(t, i);
        deg[t]++; deg[i]++;
      }
    }
  }

  _addEdge(u, v) {
    // Avoid duplicates
    for (const [eu, ev] of this.edges) {
      if ((eu === u && ev === v) || (eu === v && ev === u)) return;
    }
    this.edges.push([u, v]);
    this.edgeWeights.push(0.5);
    const idx = this.edges.length - 1;
    this.adjIn[v].push({ from: u, idx });
  }

  _computeLayout() {
    // Simple force-directed layout
    const pos = Array.from({ length: this.nTotal }, () => [
      (Math.random() - 0.5) * 2, (Math.random() - 0.5) * 2
    ]);
    const ITER = 100, K = 0.3, REP = 0.5, GRAV = 0.01;
    for (let iter = 0; iter < ITER; iter++) {
      const dx = zeros(this.nTotal), dy = zeros(this.nTotal);
      // Repulsion
      for (let i = 0; i < this.nTotal; i++) {
        for (let j = i + 1; j < this.nTotal; j++) {
          let ddx = pos[i][0] - pos[j][0], ddy = pos[i][1] - pos[j][1];
          let d2 = ddx * ddx + ddy * ddy + 0.01;
          let f = REP / d2;
          dx[i] += f * ddx; dy[i] += f * ddy;
          dx[j] -= f * ddx; dy[j] -= f * ddy;
        }
      }
      // Attraction along edges
      for (const [u, v] of this.edges) {
        let ddx = pos[v][0] - pos[u][0], ddy = pos[v][1] - pos[u][1];
        let d = Math.sqrt(ddx * ddx + ddy * ddy + 0.01);
        let f = K * (d - 0.4);
        dx[u] += f * ddx / d; dy[u] += f * ddy / d;
        dx[v] -= f * ddx / d; dy[v] -= f * ddy / d;
      }
      // Gravity toward center
      for (let i = 0; i < this.nTotal; i++) {
        dx[i] -= GRAV * pos[i][0];
        dy[i] -= GRAV * pos[i][1];
      }
      const cool = 0.1 * (1 - iter / ITER);
      for (let i = 0; i < this.nTotal; i++) {
        pos[i][0] += cool * dx[i];
        pos[i][1] += cool * dy[i];
      }
    }
    return pos;
  }

  step(coupling, threshold, noiseScale, chaosFactor) {
    const d = this.dim;
    // Gather inputs from predecessors
    const inputs = Array.from({ length: this.nTotal }, () => zeros(d));

    for (let v = 0; v < this.nTotal; v++) {
      for (const { from: u, idx } of this.adjIn[v]) {
        const w = this.edgeWeights[idx] * coupling;
        const pulse = this.pulses[u];
        for (let i = 0; i < d; i++) inputs[v][i] += w * pulse[i];
      }
    }

    // Noise + VCO babbling
    for (let n = 0; n < this.nTotal; n++) {
      for (let i = 0; i < d; i++) {
        inputs[n][i] += randn() * noiseScale;
        inputs[n][i] += chaosFactor * 0.05 * Math.sin(
          (i / d) * 2 * Math.PI * (1 + 0.3 * n)
        );
      }
    }

    // Process nodes
    for (let n = 0; n < this.nTotal; n++) {
      const out = this.nodes[n].step(inputs[n]);
      this.outputs[n] = out;

      // Pulse
      if (this.nodeTypes[n] === 'bdh') {
        this.pulses[n] = this.nodes[n].getPulse(threshold);
      } else {
        const pulse = zeros(d);
        for (let i = 0; i < d; i++) {
          pulse[i] = Math.abs(out[i]) > threshold ? out[i] : 0;
        }
        this.pulses[n] = pulse;
      }

      this.energies[n] = this.nodes[n].energy;

      // Pulse rate tracking
      const fired = norm(this.pulses[n], d) > 0.01;
      this._pulseHist[n].push(fired ? 1 : 0);
      if (this._pulseHist[n].length > this._pulseHistLen)
        this._pulseHist[n].shift();
      let sum = 0;
      for (const v of this._pulseHist[n]) sum += v;
      this.pulseRates[n] = sum / this._pulseHist[n].length;
    }

    // Hebbian update
    this._hebbianUpdate(0.001);
  }

  _hebbianUpdate(lr) {
    const d = this.dim;
    for (let idx = 0; idx < this.edges.length; idx++) {
      const [u, v] = this.edges[idx];
      const pre = norm(this.pulses[u], d);
      const post = norm(this.pulses[v], d);
      const w = this.edgeWeights[idx];
      const delta = lr * (pre * post - w * post * post);
      this.edgeWeights[idx] = clamp(w + delta, 0.01, 2.0);
    }
  }

  inject(nodeId, signal) {
    const d = this.dim;
    for (let i = 0; i < d; i++) {
      this.outputs[nodeId][i] += signal[i];
    }
  }
}

// ============================================================
// Orchestrator — connects constellation to pendulums
// ============================================================
class Orchestrator {
  constructor(constellation, pendulums, dim = 16) {
    this.const = constellation;
    this.pends = pendulums;
    this.dim = dim;
    this.tick = 0;

    // Projections (4 → dim, dim → 2)
    this.pendToNode = randnArr(4 * dim, 0.1);
    this.nodeTorque = randnArr(dim * 2, 0.05);

    // Metrics history
    this.energyHist = [];
    this.pendEnergyHist = [[], []];
    this.activationHist = [];
    this.maxHist = 400;

    // Parameters
    this.coupling = 0.5;
    this.threshold = 0.3;
    this.noise = 0.02;
    this.chaos = 0.1;
    this.simDt = 0.005;
    this.internalSteps = 2;
  }

  step() {
    this.tick++;
    const d = this.dim;
    const hubs = this.const.hubIds;

    // 1. Inject pendulum states into hubs
    for (let p = 0; p < this.pends.length; p++) {
      const ps = this.pends[p].state;
      const signal = zeros(d);
      for (let i = 0; i < d; i++) {
        let s = 0;
        for (let j = 0; j < 4; j++) s += this.pendToNode[j * d + i] * ps[j];
        signal[i] = s * 0.5;
      }
      const ha = hubs[p * 2];
      const hb = hubs[Math.min(p * 2 + 1, hubs.length - 1)];
      this.const.inject(ha, signal);
      this.const.inject(hb, signal);
    }

    // 2. Internal constellation steps
    for (let s = 0; s < this.internalSteps; s++) {
      this.const.step(this.coupling, this.threshold, this.noise, this.chaos);
    }

    // 3. Read hub outputs → torques
    for (let p = 0; p < this.pends.length; p++) {
      const ha = hubs[p * 2];
      const hb = hubs[Math.min(p * 2 + 1, hubs.length - 1)];
      const combined = zeros(d);
      for (let i = 0; i < d; i++) {
        combined[i] = (this.const.outputs[ha][i] + this.const.outputs[hb][i]) * 0.5;
      }
      // Project to 2 torques
      let tau1 = 0, tau2 = 0;
      for (let i = 0; i < d; i++) {
        tau1 += this.nodeTorque[i * 2] * combined[i];
        tau2 += this.nodeTorque[i * 2 + 1] * combined[i];
      }
      tau1 *= 2; tau2 *= 2;

      // Sub-step pendulum
      const nsub = 4, subDt = this.simDt / nsub;
      for (let ss = 0; ss < nsub; ss++) {
        this.pends[p].step(subDt, tau1, tau2);
      }
    }

    // 4. Metrics
    this._recordMetrics();
  }

  _recordMetrics() {
    let totalE = 0;
    const acts = zeros(this.const.nTotal);
    for (let i = 0; i < this.const.nTotal; i++) {
      totalE += this.const.energies[i];
      acts[i] = norm(this.const.outputs[i], this.dim);
    }
    this.energyHist.push(totalE);
    if (this.energyHist.length > this.maxHist) this.energyHist.shift();

    this.activationHist.push(acts);
    if (this.activationHist.length > this.maxHist) this.activationHist.shift();

    for (let p = 0; p < this.pends.length; p++) {
      this.pendEnergyHist[p].push(this.pends[p].energy);
      if (this.pendEnergyHist[p].length > this.maxHist)
        this.pendEnergyHist[p].shift();
    }
  }
}
