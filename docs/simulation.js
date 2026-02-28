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

// Cooley-Tukey radix-2 FFT in-place (re, im are Float32Arrays of length n)
function fft(re, im, n) {
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    while (j & bit) { j ^= bit; bit >>= 1; }
    j ^= bit;
    if (i < j) {
      let t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1;
    const ang = -2 * Math.PI / len;
    const wR = Math.cos(ang), wI = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let curR = 1, curI = 0;
      for (let j = 0; j < half; j++) {
        const uR = re[i+j], uI = im[i+j];
        const vR = re[i+j+half]*curR - im[i+j+half]*curI;
        const vI = re[i+j+half]*curI + im[i+j+half]*curR;
        re[i+j] = uR + vR; im[i+j] = uI + vI;
        re[i+j+half] = uR - vR; im[i+j+half] = uI - vI;
        const tmpR = curR*wR - curI*wI;
        curI = curR*wI + curI*wR;
        curR = tmpR;
      }
    }
  }
}

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
    this.G = new Float32Array(dim);  // learnable damping per dimension
    this.G_grad = new Float32Array(dim); // accumulated gradient for G
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

    // Damped IMEX1 recurrence with learnable G
    for (let i = 0; i < d; i++) {
      const dt = sigmoid(this.dt_raw[i]);
      const A = Math.max(0, this.A[i]);
      const G_val = Math.max(0, this.G[i]);
      const S = 1.0 + dt * G_val;
      const z_old = this.z_state[i];
      const z_new = (z_old + dt * (-A * this.x_state[i] + Bu[i])) / S;
      const x_new = this.x_state[i] + dt * z_new;
      // Gradient of z w.r.t. G: d(z_new)/d(G) = -dt * z_new / S
      this.G_grad[i] = 0.9 * this.G_grad[i] + 0.1 * (-dt * z_new / S) * z_new;
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
    this.g = 9.81; this.damping = 0.005; this.baseDamping = 0.005;
    // Start near inverted (π = balanced on top) with small perturbation
    this.state = [Math.PI + 0.15 * (id - 0.5), Math.PI + 0.1 * (0.5 - id), 0, 0];

    this.trailX = []; this.trailY = [];
    this.maxTrail = 300;
    this.energy = 0;

    // Moving origin = mechanical grab point (pivot location in world space)
    this.originX = 0;
    this.originY = 0;
    this.prevOriginX = 0;
    this.prevOriginY = 0;
    this.originVX = 0;
    this.originVY = 0;
    this.originAX = 0;
    this.originAY = 0;
    this.originRange = 1.5;
  }

  setOrigin(targetX, targetY, dt) {
    // Responsive tracking — the brain can swing hard
    const smooth = 0.25;
    const newX = this.originX + smooth * (targetX - this.originX);
    const newY = this.originY + smooth * (targetY - this.originY);
    const safeDt = Math.max(dt, 0.0005);
    const newVX = (newX - this.prevOriginX) / safeDt;
    const newVY = (newY - this.prevOriginY) / safeDt;
    this.originAX = clamp((newVX - this.originVX) / safeDt, -200, 200);
    this.originAY = clamp((newVY - this.originVY) / safeDt, -200, 200);
    this.prevOriginX = this.originX;
    this.prevOriginY = this.originY;
    this.originVX = newVX;
    this.originVY = newVY;
    this.originX = newX;
    this.originY = newY;
  }

  // Full Lagrangian with non-inertial pivot acceleration
  // θ measured from straight-DOWN: θ=0 hanging, θ=π inverted (balanced)
  // Pivot acceleration (ax,ay) creates pseudo-forces on the pendulum
  derivatives(s, tau1, tau2) {
    const [t1, t2, w1, w2] = s;
    const { m1, m2, l1, l2, g, damping: b, originAX: ax, originAY: ay } = this;
    const delta = t2 - t1;
    const cd = Math.cos(delta), sd = Math.sin(delta);

    // Mass matrix determinant
    const M = (m1 + m2) * l1 - m2 * l1 * cd * cd;
    const safeM = Math.abs(M) < 1e-6 ? (M >= 0 ? 1e-6 : -1e-6) : M;

    // Non-inertial pseudo-forces from pivot grab-point acceleration
    // In y-down convention: (g - ay)*sin(θ) - ax*cos(θ)
    const effG1 = (g - ay) * Math.sin(t1) - ax * Math.cos(t1);
    const effG2 = (g - ay) * Math.sin(t2) - ax * Math.cos(t2);

    const dw1 = (
      m2 * l1 * w1 * w1 * sd * cd
      + m2 * effG2 * cd
      + m2 * l2 * w2 * w2 * sd
      - (m1 + m2) * effG1
      - b * w1 + tau1
    ) / safeM;

    const den2 = (l2 / l1) * safeM;
    const dw2 = (
      -m2 * l2 * w2 * w2 * sd * cd
      + (m1 + m2) * effG1 * cd
      - (m1 + m2) * l1 * w1 * w1 * sd
      - (m1 + m2) * effG2
      - b * w2 + tau2
    ) / den2;

    return [w1, w2, dw1, dw2];
  }

  step(dt, tau1 = 0, tau2 = 0) {
    tau1 = clamp(tau1, -15, 15);
    tau2 = clamp(tau2, -15, 15);
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
    // Wrap angles to [-π, π]
    this.state[0] = ((this.state[0] + 3*Math.PI) % (2*Math.PI)) - Math.PI;
    this.state[1] = ((this.state[1] + 3*Math.PI) % (2*Math.PI)) - Math.PI;
    // Clamp angular velocities for stability
    this.state[2] = clamp(this.state[2], -30, 30);
    this.state[3] = clamp(this.state[3], -30, 30);

    // Trail (tip of second arm in world coords)
    const pos = this.getPositions();
    this.trailX.push(pos[2]); this.trailY.push(pos[3]);
    if (this.trailX.length > this.maxTrail) {
      this.trailX.shift(); this.trailY.shift();
    }
    this.computeEnergy();
  }

  getPositions() {
    const [t1, t2] = this.state;
    const x1 = this.originX + this.l1 * Math.sin(t1);
    const y1 = this.originY + this.l1 * Math.cos(t1);
    const x2 = x1 + this.l2 * Math.sin(t2);
    const y2 = y1 + this.l2 * Math.cos(t2);
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
    // Normalize positions to [-0.85, 0.85] for consistent rendering
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const p of pos) {
      minX = Math.min(minX, p[0]); maxX = Math.max(maxX, p[0]);
      minY = Math.min(minY, p[1]); maxY = Math.max(maxY, p[1]);
    }
    const rx = (maxX - minX) || 1, ry = (maxY - minY) || 1;
    for (const p of pos) {
      p[0] = ((p[0] - minX) / rx - 0.5) * 1.7;
      p[1] = ((p[1] - minY) / ry - 0.5) * 1.7;
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

    // Projections: pendulum state (4) → node input (dim)
    this.pendToNode = randnArr(4 * dim, 0.1);
    // Projections: node output (dim) → origin offsets (4: ox1,oy1,ox2,oy2)
    this.nodeToOrigin = randnArr(dim * 4, 0.05);
    // Small torque output (dim → 2 per pendulum)
    this.nodeTorque = randnArr(dim * 2, 0.02);
    // Task/feedback signal encoders
    this.taskEncoder = randnArr(dim, 0.1);
    this.feedbackEncoder = randnArr(dim, 0.1);

    // Eligibility traces for continuous learning
    this.originTrace = new Float32Array(dim * 4);
    this.torqueTrace = new Float32Array(dim * 2);
    this.traceDecay = 0.92;

    // Control signals
    this.taskActive = false;
    this.feedbackValue = 0; // -1 (bad) to +1 (nice), 0 = indifferent

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
    this.originRange = 1.5;
    this.originGain = 3.0;
    this.adrenaline = 0.5; // energy input model [0..2]
    this.pendMass = 1.5;
    this.pendDamping = 0.005;

    // Spectral analysis: 8 channels × 128 samples
    // Channels: grip_x0, grip_y0, grip_x1, grip_y1, tip_x0, tip_y0, tip_x1, tip_y1
    this.fftLen = 128;
    this.specBufIdx = 0;
    this.specBuffers = Array.from({ length: 8 }, () => new Float32Array(128));
    this.specMagnitudes = Array.from({ length: 8 }, () => new Float32Array(64));
    this.specLabels = ['Grip₀X','Grip₀Y','Grip₁X','Grip₁Y','Tip₀X','Tip₀Y','Tip₁X','Tip₁Y'];

    // Learning metrics
    this.rewardHist = [];
    this.rewardEMA = 0;
    this.rawRewardHist = [];
    this.maxRewardHist = 400;
    this.erraticIndex = 0;
    this._erraticDamp = 1.0;

    // Weight-change magnitude tracker (proxy for "loss")
    this.weightDeltaHist = [];

    // Hub autoscaling: track running magnitude of hub outputs
    this.hubScale = 1.0;
  }

  step() {
    this.tick++;
    const d = this.dim;
    const hubs = this.const.hubIds;

    // 1. Apply mass/damping from sliders to pendulums
    for (const pend of this.pends) {
      pend.m1 = this.pendMass;
      pend.m2 = this.pendMass * 0.667;
      pend.baseDamping = this.pendDamping;
    }

    // 2. Inject pendulum states + task + feedback into hub nodes
    const adrScale = 0.5 + this.adrenaline; // 0.5 .. 2.5
    for (let p = 0; p < this.pends.length; p++) {
      const ps = this.pends[p].state;
      const signal = zeros(d);
      for (let i = 0; i < d; i++) {
        let s = 0;
        for (let j = 0; j < 4; j++) s += this.pendToNode[j * d + i] * ps[j];
        signal[i] = s * 0.5 * adrScale;
        if (this.taskActive) signal[i] += this.taskEncoder[i] * 0.3 * adrScale;
        signal[i] += this.feedbackEncoder[i] * this.feedbackValue * 0.2;
      }
      const ha = hubs[p * 2];
      const hb = hubs[Math.min(p * 2 + 1, hubs.length - 1)];
      this.const.inject(ha, signal);
      this.const.inject(hb, signal);
    }

    // 3. Internal constellation steps
    for (let s = 0; s < this.internalSteps; s++) {
      this.const.step(this.coupling, this.threshold, this.noise * adrScale, this.chaos);
    }

    // 4. Read hub outputs → origin offsets + torques → physics
    //    Autoscaling: normalise combined hub vector so projections stay in useful range
    //    Swivel joint: tau1 = 0 (free pivot at grip); only tau2 (elbow motor)
    for (let p = 0; p < this.pends.length; p++) {
      const ha = hubs[p * 2];
      const hb = hubs[Math.min(p * 2 + 1, hubs.length - 1)];
      const combined = zeros(d);
      for (let i = 0; i < d; i++) {
        combined[i] = (this.const.outputs[ha][i] + this.const.outputs[hb][i]) * 0.5;
      }

      // --- Hub autoscaling: track EMA of output magnitude, normalise ---
      let cNorm = 0;
      for (let i = 0; i < d; i++) cNorm += combined[i] * combined[i];
      cNorm = Math.sqrt(cNorm) + 1e-8;
      this.hubScale = 0.98 * this.hubScale + 0.02 * cNorm;
      const invScale = 1.0 / Math.max(this.hubScale, 0.01);
      for (let i = 0; i < d; i++) combined[i] *= invScale;

      // Project to origin offsets
      let ox = 0, oy = 0;
      for (let i = 0; i < d; i++) {
        ox += this.nodeToOrigin[i * 4 + p * 2] * combined[i];
        oy += this.nodeToOrigin[i * 4 + p * 2 + 1] * combined[i];
      }
      ox = Math.tanh(ox * this.originGain) * this.originRange * this._erraticDamp;
      oy = Math.tanh(oy * this.originGain) * this.originRange * this._erraticDamp;

      // Update eligibility traces (origin)
      for (let i = 0; i < d; i++) {
        this.originTrace[i * 4 + p * 2] =
          this.traceDecay * this.originTrace[i * 4 + p * 2]
          + (1 - this.traceDecay) * combined[i] * ox;
        this.originTrace[i * 4 + p * 2 + 1] =
          this.traceDecay * this.originTrace[i * 4 + p * 2 + 1]
          + (1 - this.traceDecay) * combined[i] * oy;
      }

      // Move the pendulum pivot (physics acceleration effects)
      this.pends[p].setOrigin(ox, oy, this.simDt);

      // Elbow torque only — first joint is a free swivel (no motor)
      let tau2 = 0;
      for (let i = 0; i < d; i++) {
        tau2 += this.nodeTorque[i * 2 + 1] * combined[i];
      }
      tau2 *= 0.5 * adrScale;

      // Eligibility trace for tau2
      for (let i = 0; i < d; i++) {
        this.torqueTrace[i * 2 + 1] =
          this.traceDecay * this.torqueTrace[i * 2 + 1]
          + (1 - this.traceDecay) * combined[i] * tau2;
      }

      // Sub-step pendulum physics (tau1 = 0: free swivel at grip)
      const nsub = 4, subDt = this.simDt / nsub;
      for (let ss = 0; ss < nsub; ss++) {
        this.pends[p].step(subDt, 0, tau2);
      }
    }

    // 5. Spectral analysis & erratic prevention
    this._recordSpecSamples();
    if (this.tick % 16 === 0) this._computeSpectra();
    this._updateErraticDamping();

    // 6. Continuous learning (adrenaline scales LR)
    this._learnFromReward();

    // 7. Learn DLinOSS damping
    this._learnDLinOSSDamping();

    // 8. Metrics
    this._recordMetrics();
  }

  _computeTaskReward() {
    // Balance to top: reward is higher when pendulum tips are above pivot
    // cos(θ) = 1 when hanging down (θ=0), -1 when inverted (θ=π)
    // Reward = -cos(θ) → max when inverted
    let r = 0;
    for (const pend of this.pends) {
      const [t1, t2] = pend.state;
      r += (-Math.cos(t1) - Math.cos(t2));
    }
    return r / (2 * this.pends.length); // normalized to [-1, 1]
  }

  _learnFromReward() {
    let reward = this.feedbackValue;
    if (this.taskActive) {
      reward += this._computeTaskReward() * 0.5;
    }
    if (Math.abs(reward) < 0.01) return;

    const lr = 0.0003 * reward * (0.5 + this.adrenaline);
    const d = this.dim;

    // Update origin projection weights
    for (let i = 0; i < d * 4; i++) {
      this.nodeToOrigin[i] += lr * this.originTrace[i];
      this.nodeToOrigin[i] = clamp(this.nodeToOrigin[i], -2, 2);
    }
    // Update torque projection weights
    for (let i = 0; i < d * 2; i++) {
      this.nodeTorque[i] += lr * 0.5 * this.torqueTrace[i];
      this.nodeTorque[i] = clamp(this.nodeTorque[i], -2, 2);
    }
    // Modulate constellation Hebbian learning
    if (reward > 0) {
      this.const._hebbianUpdate(0.002 * reward);
    }
  }

  _learnDLinOSSDamping() {
    // DLinOSS nodes learn their damping G towards better task reward
    const reward = this.taskActive ? this._computeTaskReward() : this.feedbackValue;
    if (Math.abs(reward) < 0.02) return;
    const lr = 0.0005 * reward * (0.5 + this.adrenaline);
    for (let n = this.const.nBDH; n < this.const.nTotal; n++) {
      const node = this.const.nodes[n];
      for (let i = 0; i < node.dim; i++) {
        node.G[i] += lr * node.G_grad[i];
        node.G[i] = clamp(node.G[i], 0.01, 5.0);
      }
    }
  }

  _recordSpecSamples() {
    const idx = this.specBufIdx;
    for (let p = 0; p < this.pends.length; p++) {
      this.specBuffers[p * 2][idx] = this.pends[p].originX;
      this.specBuffers[p * 2 + 1][idx] = this.pends[p].originY;
      const pos = this.pends[p].getPositions();
      this.specBuffers[4 + p * 2][idx] = pos[2];
      this.specBuffers[4 + p * 2 + 1][idx] = pos[3];
    }
    this.specBufIdx = (idx + 1) % this.fftLen;
  }

  _computeSpectra() {
    const N = this.fftLen;
    const re = new Float32Array(N);
    const im = new Float32Array(N);
    let totalHF = 0, totalEnergy = 0;

    for (let ch = 0; ch < 8; ch++) {
      // Unwrap circular buffer
      for (let i = 0; i < N; i++) {
        re[i] = this.specBuffers[ch][(this.specBufIdx + i) % N];
        im[i] = 0;
      }
      // DC removal (subtract mean)
      let mean = 0;
      for (let i = 0; i < N; i++) mean += re[i];
      mean /= N;
      for (let i = 0; i < N; i++) re[i] -= mean;
      // Hann window
      for (let i = 0; i < N; i++) {
        re[i] *= 0.5 * (1 - Math.cos(2 * Math.PI * i / (N - 1)));
      }
      fft(re, im, N);
      // Magnitude spectrum (N/2 bins)
      for (let i = 0; i < N / 2; i++) {
        this.specMagnitudes[ch][i] = Math.sqrt(re[i] * re[i] + im[i] * im[i]) / N;
      }
      // Erratic metric: ratio of high-freq energy (top 75%) to total
      for (let i = 1; i < N / 2; i++) {
        const m2 = this.specMagnitudes[ch][i] * this.specMagnitudes[ch][i];
        totalEnergy += m2;
        if (i > N / 8) totalHF += m2;
      }
    }
    this.erraticIndex = totalEnergy > 1e-8 ? totalHF / totalEnergy : 0;
  }

  _updateErraticDamping() {
    if (this.erraticIndex > 0.4) {
      const excess = Math.min(1, (this.erraticIndex - 0.4) / 0.4);
      this._erraticDamp = 1 - 0.5 * excess;
      for (const pend of this.pends) {
        pend.damping = pend.baseDamping + 0.08 * excess;
      }
    } else {
      this._erraticDamp += 0.1 * (1 - this._erraticDamp);
      for (const pend of this.pends) {
        pend.damping += 0.2 * (pend.baseDamping - pend.damping);
      }
    }
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

    // Learning metrics
    const reward = this._computeTaskReward();
    this.rewardEMA = 0.98 * this.rewardEMA + 0.02 * reward;
    this.rewardHist.push(this.rewardEMA);
    this.rawRewardHist.push(reward);
    if (this.rewardHist.length > this.maxRewardHist) this.rewardHist.shift();
    if (this.rawRewardHist.length > this.maxRewardHist) this.rawRewardHist.shift();

    // Weight delta magnitude (proxy for gradient/loss)
    let wdelta = 0;
    for (let i = 0; i < this.dim * 4; i++) wdelta += this.originTrace[i] * this.originTrace[i];
    for (let i = 0; i < this.dim * 2; i++) wdelta += this.torqueTrace[i] * this.torqueTrace[i];
    this.weightDeltaHist.push(Math.sqrt(wdelta));
    if (this.weightDeltaHist.length > this.maxRewardHist) this.weightDeltaHist.shift();
  }
}
