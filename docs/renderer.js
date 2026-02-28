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

    this.waterfallData = [];
    this.maxWaterfall = 100;
    // Phase portrait trail history
    this.phaseTrails = [];
    this.maxPhaseTrail = 200;

    // Mouse-drag interaction for pendulum pivots
    this._pendPanelGeom = null; // cached every frame
    this._dragPendIdx = -1;     // which pendulum is being dragged (-1 = none)
    this._mouseX = 0;
    this._mouseY = 0;
    this._hoverPendIdx = -1;    // pendulum pivot the cursor is near
    this._setupMouseHandlers();
  }

  _setupMouseHandlers() {
    const c = this.canvas;
    c.addEventListener('mousemove', (e) => this._onMouseMove(e));
    c.addEventListener('mousedown', (e) => this._onMouseDown(e));
    c.addEventListener('mouseup',   (e) => this._onMouseUp(e));
    c.addEventListener('mouseleave',(e) => this._onMouseUp(e));
    // Touch support
    c.addEventListener('touchstart', (e) => {
      e.preventDefault();
      const t = e.touches[0];
      this._onMouseMove(t);
      this._onMouseDown(t);
    }, { passive: false });
    c.addEventListener('touchmove', (e) => {
      e.preventDefault();
      this._onMouseMove(e.touches[0]);
    }, { passive: false });
    c.addEventListener('touchend', (e) => { this._onMouseUp(e); });
  }

  _screenToSim(sx, sy, pendIdx) {
    const g = this._pendPanelGeom;
    if (!g) return [0, 0];
    const pcx = g.cx + g.offsets[pendIdx];
    const pcy = g.cy;
    return [(sx - pcx) / g.scale, (sy - pcy) / g.scale];
  }

  _simToScreen(simX, simY, pendIdx) {
    const g = this._pendPanelGeom;
    if (!g) return [0, 0];
    const pcx = g.cx + g.offsets[pendIdx];
    const pcy = g.cy;
    return [pcx + simX * g.scale, pcy + simY * g.scale];
  }

  _hitTestPivots(sx, sy) {
    // Returns pendulum index if cursor is near its pivot, else -1
    const g = this._pendPanelGeom;
    if (!g) return -1;
    const hitR = 18; // px radius for grab zone
    for (let p = 0; p < this.orch.pends.length; p++) {
      const pend = this.orch.pends[p];
      const [px, py] = this._simToScreen(pend.originX, pend.originY, p);
      const dx = sx - px, dy = sy - py;
      if (dx * dx + dy * dy < hitR * hitR) return p;
    }
    return -1;
  }

  _onMouseMove(e) {
    const rect = this.canvas.getBoundingClientRect();
    this._mouseX = (e.clientX || e.pageX) - rect.left;
    this._mouseY = (e.clientY || e.pageY) - rect.top;
    this._hoverPendIdx = this._hitTestPivots(this._mouseX, this._mouseY);

    if (this._dragPendIdx >= 0) {
      const [sx, sy] = this._screenToSim(this._mouseX, this._mouseY, this._dragPendIdx);
      const pend = this.orch.pends[this._dragPendIdx];
      pend.mouseOriginX = sx;
      pend.mouseOriginY = sy;
    }
    // Cursor style
    this.canvas.style.cursor =
      this._dragPendIdx >= 0 ? 'grabbing' :
      this._hoverPendIdx >= 0 ? 'grab' : 'default';
  }

  _onMouseDown(e) {
    const idx = this._hitTestPivots(this._mouseX, this._mouseY);
    if (idx >= 0) {
      this._dragPendIdx = idx;
      const pend = this.orch.pends[idx];
      pend.mouseDragging = true;
      pend.mouseOriginX = pend.originX;
      pend.mouseOriginY = pend.originY;
      this.canvas.style.cursor = 'grabbing';
    }
  }

  _onMouseUp(e) {
    if (this._dragPendIdx >= 0) {
      this.orch.pends[this._dragPendIdx].mouseDragging = false;
      this._dragPendIdx = -1;
      this.canvas.style.cursor = this._hoverPendIdx >= 0 ? 'grab' : 'default';
    }
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

    // Panel divisions — 5 bottom panels including learning/spectrum
    const midX = W * 0.5;
    const midY = H * 0.50;
    const bh = H - midY;
    const pw0 = W * 0.15, pw1 = W * 0.15, pw2 = W * 0.30, pw3 = W * 0.24, pw4 = W * 0.16;

    // Draw panels
    this._drawPendulums(ctx, 0, 0, midX, midY);
    this._drawConstellation(ctx, midX, 0, W - midX, midY);
    this._drawEnergy(ctx, 0, midY, pw0, bh);
    this._drawPhasePortrait(ctx, pw0, midY, pw1, bh);
    this._drawLearningSpectrum(ctx, pw0 + pw1, midY, pw2, bh);
    this._drawWaterfall(ctx, pw0 + pw1 + pw2, midY, pw3, bh);
    this._drawPulseRates(ctx, pw0 + pw1 + pw2 + pw3, midY, pw4, bh);

    // Panel borders
    ctx.strokeStyle = '#1a1a2a';
    ctx.lineWidth = 1;
    ctx.strokeRect(0, 0, midX, midY);
    ctx.strokeRect(midX, 0, W - midX, midY);
    ctx.strokeRect(0, midY, pw0, bh);
    ctx.strokeRect(pw0, midY, pw1, bh);
    ctx.strokeRect(pw0 + pw1, midY, pw2, bh);
    ctx.strokeRect(pw0 + pw1 + pw2, midY, pw3, bh);
    ctx.strokeRect(pw0 + pw1 + pw2 + pw3, midY, pw4, bh);
  }

  _drawPendulums(ctx, ox, oy, w, h) {
    ctx.save();
    ctx.beginPath();
    ctx.rect(ox, oy, w, h);
    ctx.clip();

    // Title (offset to avoid HUD overlap)
    ctx.fillStyle = '#ff6b6b';
    ctx.font = '12px monospace';
    ctx.fillText('Double Pendulum Ballet', ox + 10, oy + 32);
    ctx.fillStyle = '#555';
    ctx.font = '9px monospace';
    const anyDrag = this.orch.pends.some(p => p.mouseDragging);
    ctx.fillText(
      anyDrag
        ? '\u{1F3AF} Mouse controlling pivot — drag to sling!  Brain resumes on release.'
        : 'Brain moves pivots (cart-pole) — click & drag a pivot to take manual control',
      ox + 10, oy + 44
    );

    const cx = ox + w / 2;
    const cy = oy + h * 0.45;
    const scale = Math.min(w, h) * 0.16;
    const offsets = [-w * 0.22, w * 0.22];
    const colors = [['#ff6b6b', '#ff8787', '#ffa8a850'], ['#74c0fc', '#91d5ff', '#b2e0ff50']];

    // Cache panel geometry for mouse hit-testing
    this._pendPanelGeom = { ox, oy, w, h, cx, cy, scale, offsets };

    for (let p = 0; p < this.orch.pends.length; p++) {
      const pend = this.orch.pends[p];
      const pcx = cx + offsets[p];
      const pcy = cy;
      const [c1, c2, c3] = colors[p];

      // Pivot position (origin controlled by brain)
      const pivotX = pcx + pend.originX * scale;
      const pivotY = pcy + pend.originY * scale;

      // Draw origin range box
      const rangeBox = pend.originRange * scale;
      ctx.strokeStyle = 'rgba(255,255,255,0.08)';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.strokeRect(pcx - rangeBox, pcy - rangeBox, rangeBox * 2, rangeBox * 2);
      ctx.setLineDash([]);

      // Draw trail (tip of second arm)
      if (pend.trailX.length > 2) {
        ctx.beginPath();
        for (let i = 0; i < pend.trailX.length; i++) {
          const tx = pcx + pend.trailX[i] * scale;
          const ty = pcy + pend.trailY[i] * scale;
          if (i === 0) ctx.moveTo(tx, ty);
          else ctx.lineTo(tx, ty);
        }
        ctx.strokeStyle = c3;
        ctx.lineWidth = 1;
        ctx.stroke();
      }

      // Draw pendulum from pivot
      const [x1, y1, x2, y2] = pend.getPositions();
      const sx1 = pcx + x1 * scale, sy1 = pcy + y1 * scale;
      const sx2 = pcx + x2 * scale, sy2 = pcy + y2 * scale;

      // Rods
      ctx.beginPath();
      ctx.moveTo(pivotX, pivotY);
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

      // Pivot crosshair
      ctx.beginPath();
      ctx.moveTo(pivotX - 6, pivotY); ctx.lineTo(pivotX + 6, pivotY);
      ctx.moveTo(pivotX, pivotY - 6); ctx.lineTo(pivotX, pivotY + 6);
      ctx.strokeStyle = 'rgba(255,255,255,0.5)';
      ctx.lineWidth = 1;
      ctx.stroke();
      // Swivel ring — free pivot (no motor at grip joint)
      // Highlight when hovering / dragging
      const isHover = (this._hoverPendIdx === p);
      const isDrag  = (this._dragPendIdx === p);
      if (isDrag) {
        // Pulsing glow while dragging
        ctx.beginPath();
        ctx.arc(pivotX, pivotY, 14, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(255,200,50,0.25)';
        ctx.fill();
        ctx.beginPath();
        ctx.arc(pivotX, pivotY, 8, 0, Math.PI * 2);
        ctx.strokeStyle = '#ffd43b';
        ctx.lineWidth = 2;
        ctx.stroke();
      } else if (isHover) {
        ctx.beginPath();
        ctx.arc(pivotX, pivotY, 12, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(255,200,50,0.5)';
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
      ctx.beginPath();
      ctx.arc(pivotX, pivotY, 5, 0, Math.PI * 2);
      ctx.strokeStyle = isDrag ? '#ffd43b' : 'rgba(255,255,255,0.6)';
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(pivotX, pivotY, 2, 0, Math.PI * 2);
      ctx.fillStyle = isDrag ? '#ffd43b' : '#fff';
      ctx.fill();

      // Joint 1
      ctx.beginPath();
      ctx.arc(sx1, sy1, 7, 0, Math.PI * 2);
      ctx.fillStyle = c1;
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 0.5;
      ctx.stroke();

      // Joint 2 (tip)
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
    ctx.fillStyle = '#555';
    ctx.font = '9px monospace';
    ctx.fillText('Scale-free graph: circles=BDH, diamonds=DLinOSS, glow=hubs', ox + 10, oy + 30);

    const c = this.orch.const;
    const cx = ox + w / 2, cy = oy + h / 2 + 10;
    const sc = Math.min(w, h) * 0.42;

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
    ctx.fillStyle = '#555';
    ctx.font = '8px monospace';
    ctx.fillText('orange=constellation, red/blue=pendulums', ox + 8, oy + 24);

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
    ctx.fillStyle = '#555';
    ctx.font = '8px monospace';
    ctx.fillText('2D projection of hub node activations', ox + 8, oy + 24);

    const c = this.orch.const;
    const cxp = ox + w / 2, cyp = oy + h / 2 + 8;
    const colors = ['#ff6b6b', '#51cf66', '#74c0fc', '#ffd43b'];

    // Record current hub positions
    const frame = [];
    for (let h2 = 0; h2 < Math.min(c.hubIds.length, 4); h2++) {
      const hub = c.hubIds[h2];
      const out = c.outputs[hub];
      frame.push([out[0], out[1]]);
    }
    this.phaseTrails.push(frame);
    if (this.phaseTrails.length > this.maxPhaseTrail)
      this.phaseTrails.shift();

    // --- Data-driven autoscale: find extent of all trail points ---
    let maxExt = 0.001;
    for (let t = 0; t < this.phaseTrails.length; t++) {
      for (let h2 = 0; h2 < this.phaseTrails[t].length; h2++) {
        const pt = this.phaseTrails[t][h2];
        if (!pt) continue;
        const ax = Math.abs(pt[0]), ay = Math.abs(pt[1]);
        if (ax > maxExt) maxExt = ax;
        if (ay > maxExt) maxExt = ay;
      }
    }
    // Scale so data fills ~80% of panel (symmetric)
    const sc = Math.min(w, h) * 0.38 / maxExt;

    // Faint axis cross
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(ox + 8, cyp); ctx.lineTo(ox + w - 8, cyp);
    ctx.moveTo(cxp, oy + 28); ctx.lineTo(cxp, oy + h - 8);
    ctx.stroke();

    // Draw trails
    for (let h2 = 0; h2 < frame.length; h2++) {
      ctx.beginPath();
      for (let t = 0; t < this.phaseTrails.length; t++) {
        const pt = this.phaseTrails[t][h2];
        if (!pt) continue;
        const px = cxp + pt[0] * sc;
        const py = cyp + pt[1] * sc;
        if (t === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.strokeStyle = colors[h2] + '60';
      ctx.lineWidth = 0.8;
      ctx.stroke();

      // Current position (bright dot)
      const cur = frame[h2];
      if (cur) {
        ctx.beginPath();
        ctx.arc(cxp + cur[0] * sc, cyp + cur[1] * sc, 4, 0, Math.PI * 2);
        ctx.fillStyle = colors[h2];
        ctx.fill();
      }
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
    ctx.fillStyle = '#555';
    ctx.font = '8px monospace';
    ctx.fillText('Waterfall: time → × node, brighter=more active', ox + 8, oy + 24);

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
    ctx.fillStyle = '#555';
    ctx.font = '8px monospace';
    ctx.fillText('Firing frequency per node (green=BDH, pink=DLinOSS)', ox + 8, oy + 24);

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

  _drawLearningSpectrum(ctx, ox, oy, w, h) {
    ctx.save();
    ctx.beginPath();
    ctx.rect(ox, oy, w, h);
    ctx.clip();

    ctx.fillStyle = '#e599f7';
    ctx.font = '10px monospace';
    ctx.fillText('Learning & Spectrum', ox + 8, oy + 14);
    ctx.fillStyle = '#555';
    ctx.font = '8px monospace';
    ctx.fillText('reward trend + FFT of grip/tip (DC removed)', ox + 8, oy + 24);

    const orch = this.orch;
    const pad = 8;

    // --- Top 35%: reward curve ---
    const rh = (h - 30) * 0.35;
    const ry = oy + 28;

    if (orch.rewardHist && orch.rewardHist.length > 1) {
      const data = orch.rewardHist;
      let min = -1, max = 1;
      for (const v of data) { if (v < min) min = v; if (v > max) max = v; }
      const range = max - min || 1;

      // Zero line
      const zeroY = ry + rh - ((0 - min) / range) * rh;
      ctx.strokeStyle = 'rgba(255,255,255,0.08)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(ox + pad, zeroY); ctx.lineTo(ox + w - pad, zeroY);
      ctx.stroke();

      // Raw reward (faint dots)
      if (orch.rawRewardHist && orch.rawRewardHist.length > 1) {
        ctx.fillStyle = 'rgba(255,255,255,0.12)';
        const raw = orch.rawRewardHist;
        for (let i = 0; i < raw.length; i++) {
          const px = ox + pad + (i / (raw.length - 1)) * (w - pad * 2);
          const py = ry + rh - ((raw[i] - min) / range) * rh;
          ctx.fillRect(px - 0.5, py - 0.5, 1.5, 1.5);
        }
      }

      // EMA reward line
      ctx.beginPath();
      for (let i = 0; i < data.length; i++) {
        const px = ox + pad + (i / (data.length - 1)) * (w - pad * 2);
        const py = ry + rh - ((data[i] - min) / range) * rh;
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      const lastR = data[data.length - 1];
      ctx.strokeStyle = lastR > 0 ? '#51cf66' : '#ff6b6b';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Weight delta (gradient proxy) — thin yellow line
      if (orch.weightDeltaHist && orch.weightDeltaHist.length > 1) {
        const wd = orch.weightDeltaHist;
        let wdMin = Infinity, wdMax = 0;
        for (const v of wd) { if (v < wdMin) wdMin = v; if (v > wdMax) wdMax = v; }
        const wdR = wdMax - wdMin || 1;
        ctx.beginPath();
        for (let i = 0; i < wd.length; i++) {
          const px = ox + pad + (i / (wd.length - 1)) * (w - pad * 2);
          const py = ry + rh - ((wd[i] - wdMin) / wdR) * rh * 0.8;
          if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        }
        ctx.strokeStyle = '#ffd43b';
        ctx.lineWidth = 0.8;
        ctx.globalAlpha = 0.5;
        ctx.stroke();
        ctx.globalAlpha = 1;
      }

      // Labels
      ctx.fillStyle = lastR > 0 ? '#51cf66' : '#ff6b6b';
      ctx.font = '8px monospace';
      ctx.fillText(`reward: ${lastR.toFixed(3)}`, ox + pad, ry + 10);
      ctx.fillStyle = '#ffd43b';
      ctx.fillText('|\u0394w|', ox + w - 35, ry + 10);
    }

    // --- Erratic meter bar ---
    const meterY = ry + rh + 4;
    const meterH = 6;
    ctx.fillStyle = '#111';
    ctx.fillRect(ox + pad, meterY, w - pad * 2, meterH);
    const ei = orch.erraticIndex || 0;
    const meterW = Math.min(1, ei) * (w - pad * 2);
    ctx.fillStyle = ei > 0.5 ? '#ff6b6b' : ei > 0.3 ? '#ffd43b' : '#51cf66';
    ctx.fillRect(ox + pad, meterY, meterW, meterH);
    ctx.fillStyle = '#888';
    ctx.font = '7px monospace';
    ctx.fillText(`erratic: ${(ei * 100).toFixed(0)}%  damp: ${((orch._erraticDamp || 1) * 100).toFixed(0)}%`, ox + pad, meterY + meterH + 9);

    // --- Bottom 55%: spectrum bars ---
    const sy = meterY + meterH + 14;
    const sh = h - (sy - oy) - 12;

    // Spectrum background
    ctx.fillStyle = '#0d0d18';
    ctx.fillRect(ox + pad - 2, sy - 12, w - pad * 2 + 4, sh + 18);

    if (orch.specMagnitudes && orch.specMagnitudes[0] && orch.specMagnitudes[0].length > 0) {
      const halfW = (w - pad * 3) / 2;

      // Grip spectrum (avg of channels 0-3)
      this._drawSpectrumBars(ctx, ox + pad, sy, halfW, sh,
        orch.specMagnitudes, 0, 4, '#74c0fc', 'Grip FFT');

      // Tip spectrum (avg of channels 4-7)
      this._drawSpectrumBars(ctx, ox + pad * 2 + halfW, sy, halfW, sh,
        orch.specMagnitudes, 4, 8, '#ff922b', 'Tip FFT');
    } else {
      ctx.fillStyle = '#555';
      ctx.font = '9px monospace';
      ctx.fillText('waiting for FFT data...', ox + pad, sy + sh / 2);
    }
    ctx.restore();
  }

  _drawSpectrumBars(ctx, ox, oy, w, h, specs, chStart, chEnd, color, label) {
    const N = specs[0].length; // N/2 bins
    const nBins = Math.min(N - 1, Math.floor(w / 2));
    if (nBins < 1) return;
    const barW = w / nBins;

    // Average magnitudes across channels, skip bin 0 (DC)
    let maxMag = 0.001;
    const avg = new Float32Array(nBins);
    for (let b = 0; b < nBins; b++) {
      for (let ch = chStart; ch < chEnd; ch++) {
        avg[b] += specs[ch][b + 1];
      }
      avg[b] /= (chEnd - chStart);
      if (avg[b] > maxMag) maxMag = avg[b];
    }

    // Baseline
    ctx.strokeStyle = 'rgba(255,255,255,0.1)';
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(ox, oy + h); ctx.lineTo(ox + w, oy + h);
    ctx.stroke();

    // Draw bars
    for (let b = 0; b < nBins; b++) {
      const v = avg[b] / maxMag;
      const bh2 = Math.max(v * h * 0.85, 1.5); // minimum visible height
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.4 + 0.6 * v;
      ctx.fillRect(ox + b * barW, oy + h - bh2, Math.max(barW - 0.5, 1), bh2);
    }
    ctx.globalAlpha = 1;

    // Label
    ctx.fillStyle = color;
    ctx.font = '8px monospace';
    ctx.fillText(label, ox, oy - 2);
    // Frequency axis hint
    ctx.fillStyle = '#444';
    ctx.font = '7px monospace';
    ctx.fillText('lo', ox, oy + h + 8);
    ctx.fillText('hi', ox + w - 10, oy + h + 8);
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
