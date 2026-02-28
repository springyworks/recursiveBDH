// ============================================================
// renderer.js — Canvas 2D renderer for the constellation
// ============================================================

"use strict";

class Renderer {
  constructor(canvas, orchestrator) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.orch = orchestrator;
    this.resize();
    window.addEventListener('resize', () => this.resize());

    // Layout regions (fractions of canvas)
    // Left half: pendulum (top), energy (bottom)
    // Right half: constellation (top), waterfall + pulse (bottom)
    this.waterfallData = []; // rows of activation arrays
    this.maxWaterfall = 100;
  }

  resize() {
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = window.innerWidth * dpr;
    this.canvas.height = window.innerHeight * dpr;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.W = window.innerWidth;
    this.H = window.innerHeight - 50; // controls bar
  }

  draw() {
    const ctx = this.ctx;
    const W = this.W, H = this.H;
    ctx.clearRect(0, 0, W, H + 50);
    ctx.fillStyle = '#0a0a0f';
    ctx.fillRect(0, 0, W, H + 50);

    const c = this.orch.const;

    // Update waterfall
    const acts = [];
    let maxA = 0.01;
    for (let i = 0; i < c.nTotal; i++) {
      const a = norm(c.outputs[i], c.dim);
      acts.push(a);
      if (a > maxA) maxA = a;
    }
    this.waterfallData.push(acts.map(v => v / maxA));
    if (this.waterfallData.length > this.maxWaterfall)
      this.waterfallData.shift();

    // Panel divisions
    const midX = W * 0.48;
    const midY = H * 0.58;

    // Draw panels
    this._drawPendulums(ctx, 0, 0, midX, midY);
    this._drawConstellation(ctx, midX, 0, W - midX, midY);
    this._drawEnergy(ctx, 0, midY, midX * 0.6, H - midY);
    this._drawPhasePortrait(ctx, midX * 0.6, midY, midX * 0.4, H - midY);
    this._drawWaterfall(ctx, midX, midY, (W - midX) * 0.6, H - midY);
    this._drawPulseRates(ctx, midX + (W - midX) * 0.6, midY, (W - midX) * 0.4, H - midY);

    // Panel borders
    ctx.strokeStyle = '#1a1a2a';
    ctx.lineWidth = 1;
    ctx.strokeRect(0, 0, midX, midY);
    ctx.strokeRect(midX, 0, W - midX, midY);
    ctx.strokeRect(0, midY, midX, H - midY);
    ctx.strokeRect(midX, midY, W - midX, H - midY);
  }

  _drawPendulums(ctx, ox, oy, w, h) {
    ctx.save();
    ctx.beginPath();
    ctx.rect(ox, oy, w, h);
    ctx.clip();

    // Title
    ctx.fillStyle = '#ff6b6b';
    ctx.font = '12px monospace';
    ctx.fillText('Double Pendulum Ballet', ox + 10, oy + 18);

    const cx = ox + w / 2;
    const cy = oy + h * 0.35;
    const scale = Math.min(w, h) * 0.2;
    const offsets = [-w * 0.2, w * 0.2];
    const colors = [['#ff6b6b', '#ff8787', '#ffa8a850'], ['#74c0fc', '#91d5ff', '#b2e0ff50']];

    for (let p = 0; p < this.orch.pends.length; p++) {
      const pend = this.orch.pends[p];
      const pcx = cx + offsets[p];
      const [c1, c2, c3] = colors[p];

      // Draw trail
      if (pend.trailX.length > 2) {
        ctx.beginPath();
        for (let i = 0; i < pend.trailX.length; i++) {
          const tx = pcx + pend.trailX[i] * scale;
          const ty = cy + pend.trailY[i] * scale;
          if (i === 0) ctx.moveTo(tx, ty);
          else ctx.lineTo(tx, ty);
        }
        ctx.strokeStyle = c3;
        ctx.lineWidth = 1;
        ctx.stroke();
      }

      // Draw pendulum
      const [x1, y1, x2, y2] = pend.getPositions();
      const sx1 = pcx + x1 * scale, sy1 = cy + y1 * scale;
      const sx2 = pcx + x2 * scale, sy2 = cy + y2 * scale;

      // Rods
      ctx.beginPath();
      ctx.moveTo(pcx, cy);
      ctx.lineTo(sx1, sy1);
      ctx.strokeStyle = c1;
      ctx.lineWidth = 3;
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(sx1, sy1);
      ctx.lineTo(sx2, sy2);
      ctx.strokeStyle = c2;
      ctx.lineWidth = 2.5;
      ctx.stroke();

      // Joints
      ctx.beginPath();
      ctx.arc(pcx, cy, 4, 0, Math.PI * 2);
      ctx.fillStyle = '#fff';
      ctx.fill();

      ctx.beginPath();
      ctx.arc(sx1, sy1, 7, 0, Math.PI * 2);
      ctx.fillStyle = c1;
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 0.5;
      ctx.stroke();

      ctx.beginPath();
      ctx.arc(sx2, sy2, 5, 0, Math.PI * 2);
      ctx.fillStyle = c2;
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 0.5;
      ctx.stroke();
    }
    ctx.restore();
  }

  _drawConstellation(ctx, ox, oy, w, h) {
    ctx.save();
    ctx.beginPath();
    ctx.rect(ox, oy, w, h);
    ctx.clip();

    ctx.fillStyle = '#51cf66';
    ctx.font = '12px monospace';
    ctx.fillText('BDH+DLINOSS Constellation', ox + 10, oy + 18);

    const c = this.orch.const;
    const cx = ox + w / 2, cy = oy + h / 2 + 10;
    const sc = Math.min(w, h) * 0.38;

    // Find activation range
    let maxAct = 0.01;
    for (let i = 0; i < c.nTotal; i++) {
      const a = norm(c.outputs[i], c.dim);
      if (a > maxAct) maxAct = a;
    }

    // Draw edges
    for (let e = 0; e < c.edges.length; e++) {
      const [u, v] = c.edges[e];
      const w2 = c.edgeWeights[e];
      const alpha = Math.min(w2 / 2, 0.6);
      ctx.beginPath();
      ctx.moveTo(cx + c.positions[u][0] * sc, cy + c.positions[u][1] * sc);
      ctx.lineTo(cx + c.positions[v][0] * sc, cy + c.positions[v][1] * sc);
      ctx.strokeStyle = `rgba(60,60,80,${alpha})`;
      ctx.lineWidth = 0.8;
      ctx.stroke();
    }

    // Draw nodes
    for (let i = 0; i < c.nTotal; i++) {
      const nx = cx + c.positions[i][0] * sc;
      const ny = cy + c.positions[i][1] * sc;
      const a = norm(c.outputs[i], c.dim) / maxAct;
      const isBDH = c.nodeTypes[i] === 'bdh';
      const isHub = c.hubIds.includes(i);

      // Hub glow
      if (isHub) {
        const grad = ctx.createRadialGradient(nx, ny, 2, nx, ny, 18);
        grad.addColorStop(0, isBDH ? 'rgba(255,100,100,0.3)' : 'rgba(100,200,255,0.3)');
        grad.addColorStop(1, 'rgba(0,0,0,0)');
        ctx.fillStyle = grad;
        ctx.fillRect(nx - 18, ny - 18, 36, 36);
      }

      // Node
      const r = isBDH ? 4 + 5 * a : 5 + 4 * a;
      ctx.beginPath();
      if (isBDH) {
        ctx.arc(nx, ny, r, 0, Math.PI * 2);
      } else {
        // Diamond for DLINOSS
        ctx.moveTo(nx, ny - r); ctx.lineTo(nx + r, ny);
        ctx.lineTo(nx, ny + r); ctx.lineTo(nx - r, ny); ctx.closePath();
      }

      // Color by activation (plasma-like for BDH, cool for DLINOSS)
      if (isBDH) {
        const rr = Math.floor(50 + 205 * a);
        const gg = Math.floor(20 + 60 * a);
        const bb = Math.floor(80 + 80 * (1 - a));
        ctx.fillStyle = `rgb(${rr},${gg},${bb})`;
      } else {
        const rr = Math.floor(80 + 100 * (1 - a));
        const gg = Math.floor(150 + 100 * a);
        const bb = Math.floor(200 + 55 * a);
        ctx.fillStyle = `rgb(${rr},${gg},${bb})`;
      }
      ctx.fill();
      ctx.strokeStyle = 'rgba(255,255,255,0.3)';
      ctx.lineWidth = 0.5;
      ctx.stroke();

      // Pulse flash
      if (c.pulseRates[i] > 0.5) {
        ctx.beginPath();
        ctx.arc(nx, ny, r * 1.5, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(255,255,255,0.4)';
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    }
    ctx.restore();
  }

  _drawEnergy(ctx, ox, oy, w, h) {
    ctx.save();
    ctx.beginPath();
    ctx.rect(ox, oy, w, h);
    ctx.clip();

    ctx.fillStyle = '#ff922b';
    ctx.font = '10px monospace';
    ctx.fillText('System Energy', ox + 8, oy + 14);

    const pad = 20;
    const gx = ox + pad, gy = oy + 22;
    const gw = w - pad * 2, gh = h - 30;

    // Constellation energy
    if (this.orch.energyHist.length > 1) {
      this._plotLine(ctx, gx, gy, gw, gh, this.orch.energyHist, '#ff922b');
    }

    // Pendulum energies
    const peColors = ['#ff6b6b', '#74c0fc'];
    for (let p = 0; p < 2; p++) {
      const hist = this.orch.pendEnergyHist[p];
      if (hist.length > 1) {
        this._plotLine(ctx, gx, gy, gw, gh, hist, peColors[p]);
      }
    }
    ctx.restore();
  }

  _drawPhasePortrait(ctx, ox, oy, w, h) {
    ctx.save();
    ctx.beginPath();
    ctx.rect(ox, oy, w, h);
    ctx.clip();

    ctx.fillStyle = '#74c0fc';
    ctx.font = '10px monospace';
    ctx.fillText('Hub Phase Portrait', ox + 8, oy + 14);

    const c = this.orch.const;
    const cx = ox + w / 2, cy = oy + h / 2 + 8;
    const sc = Math.min(w, h) * 0.35;
    const colors = ['#ff6b6b', '#51cf66', '#74c0fc', '#ffd43b'];

    for (let h2 = 0; h2 < Math.min(c.hubIds.length, 4); h2++) {
      const hub = c.hubIds[h2];
      const out = c.outputs[hub];
      // Use first 2 dims as x,y
      const px = cx + out[0] * sc;
      const py = cy + out[1] * sc;
      ctx.beginPath();
      ctx.arc(px, py, 2, 0, Math.PI * 2);
      ctx.fillStyle = colors[h2];
      ctx.fill();
    }
    ctx.restore();
  }

  _drawWaterfall(ctx, ox, oy, w, h) {
    ctx.save();
    ctx.beginPath();
    ctx.rect(ox, oy, w, h);
    ctx.clip();

    ctx.fillStyle = '#ffd43b';
    ctx.font = '10px monospace';
    ctx.fillText('Node Activations', ox + 8, oy + 14);

    const data = this.waterfallData;
    if (data.length < 2) { ctx.restore(); return; }

    const nNodes = data[0].length;
    const pad = 8;
    const gx = ox + pad, gy = oy + 20;
    const gw = w - pad * 2, gh = h - 26;
    const cellW = gw / data.length;
    const cellH = gh / nNodes;

    for (let t = 0; t < data.length; t++) {
      for (let n = 0; n < nNodes; n++) {
        const v = data[t][n];
        // Inferno-like colormap
        const r = Math.floor(Math.min(255, v * 400));
        const g = Math.floor(Math.min(255, v * 150));
        const b = Math.floor(40 + v * 80);
        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.fillRect(gx + t * cellW, gy + n * cellH, cellW + 0.5, cellH + 0.5);
      }
    }
    ctx.restore();
  }

  _drawPulseRates(ctx, ox, oy, w, h) {
    ctx.save();
    ctx.beginPath();
    ctx.rect(ox, oy, w, h);
    ctx.clip();

    ctx.fillStyle = '#20c997';
    ctx.font = '10px monospace';
    ctx.fillText('Pulse Rates', ox + 8, oy + 14);

    const c = this.orch.const;
    const pad = 8;
    const gx = ox + pad, gy = oy + 20;
    const gw = w - pad * 2, gh = h - 26;
    const barW = gw / c.nTotal - 1;

    for (let i = 0; i < c.nTotal; i++) {
      const pr = c.pulseRates[i];
      const bh = pr * gh;
      const color = c.nodeTypes[i] === 'bdh' ? '#20c997' : '#e599f7';
      ctx.fillStyle = color;
      ctx.fillRect(
        gx + i * (barW + 1),
        gy + gh - bh,
        barW, bh
      );
    }
    ctx.restore();
  }

  _plotLine(ctx, x, y, w, h, data, color) {
    if (data.length < 2) return;
    let min = Infinity, max = -Infinity;
    for (const v of data) {
      if (v < min) min = v;
      if (v > max) max = v;
    }
    const range = max - min || 1;

    ctx.beginPath();
    for (let i = 0; i < data.length; i++) {
      const px = x + (i / (data.length - 1)) * w;
      const py = y + h - ((data[i] - min) / range) * h;
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.globalAlpha = 0.7;
    ctx.stroke();
    ctx.globalAlpha = 1;
  }
}
