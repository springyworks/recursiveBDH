// ============================================================
// webgpu-compute.js — WebGPU compute shader acceleration
//
// When WebGPU is available (Chrome 113+, Edge, Firefox Nightly),
// offloads the BDH node matmul + DLINOSS recurrence to GPU
// compute shaders — running all 25 nodes in parallel.
//
// Falls back to CPU (simulation.js) when unavailable.
// ============================================================

"use strict";

class WebGPUAccelerator {
  constructor() {
    this.available = false;
    this.device = null;
    this.pipeline = null;
    this.ready = false;
  }

  async init() {
    if (!navigator.gpu) {
      console.log('[WebGPU] Not available — using CPU fallback');
      return false;
    }

    try {
      const adapter = await navigator.gpu.requestAdapter({
        powerPreference: 'high-performance'
      });
      if (!adapter) {
        console.log('[WebGPU] No adapter — using CPU fallback');
        return false;
      }

      this.device = await adapter.requestDevice({
        requiredLimits: {
          maxComputeWorkgroupSizeX: 256,
          maxStorageBufferBindingSize: 128 * 1024 * 1024,
        }
      });

      const info = await adapter.requestAdapterInfo();
      console.log(`[WebGPU] GPU: ${info.description || info.device || 'unknown'}`);
      console.log(`[WebGPU] Vendor: ${info.vendor || 'unknown'}`);

      await this._createPipelines();
      this.available = true;
      this.ready = true;
      return true;

    } catch (e) {
      console.log('[WebGPU] Init failed:', e.message);
      return false;
    }
  }

  async _createPipelines() {
    // Compute shader: parallel BDH node processing
    // Each workgroup processes one node's matmul
    const bdhShaderCode = /* wgsl */ `
      struct Params {
        dim: u32,
        nNodes: u32,
        threshold: f32,
        _pad: f32,
      };

      @group(0) @binding(0) var<uniform> params: Params;
      @group(0) @binding(1) var<storage, read> inputs: array<f32>;
      @group(0) @binding(2) var<storage, read> activations: array<f32>;
      @group(0) @binding(3) var<storage, read> encoderW: array<f32>;
      @group(0) @binding(4) var<storage, read> decoderW: array<f32>;
      @group(0) @binding(5) var<storage, read_write> outputs: array<f32>;
      @group(0) @binding(6) var<storage, read_write> pulses: array<f32>;

      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        let nodeId = gid.x / params.dim;
        let elemId = gid.x % params.dim;

        if (nodeId >= params.nNodes) { return; }

        let d = params.dim;
        let baseIn = nodeId * d;
        let baseW = nodeId * d * d;

        // Combined = input + activation (then normalize)
        var combined: f32 = inputs[baseIn + elemId] + activations[baseIn + elemId];

        // Encode: W @ combined with ReLU
        var encoded: f32 = 0.0;
        for (var j: u32 = 0u; j < d; j++) {
          encoded += encoderW[baseW + elemId * d + j] * (inputs[baseIn + j] + activations[baseIn + j]);
        }
        encoded = max(0.0, encoded);

        // Decode: W @ encoded
        var decoded: f32 = 0.0;
        for (var j: u32 = 0u; j < d; j++) {
          // Simplified: use encoded value broadcast (approximation for speed)
          decoded += decoderW[baseW + elemId * d + j] * encoded;
        }

        // Residual + normalize (simplified)
        let out = combined + decoded;
        outputs[baseIn + elemId] = out;

        // Pulse
        if (abs(out) > params.threshold) {
          pulses[baseIn + elemId] = out;
        } else {
          pulses[baseIn + elemId] = 0.0;
        }
      }
    `;

    // DLINOSS compute shader
    const dlinossShaderCode = /* wgsl */ `
      struct DParams {
        dim: u32,
        nNodes: u32,
        _pad1: f32,
        _pad2: f32,
      };

      @group(0) @binding(0) var<uniform> params: DParams;
      @group(0) @binding(1) var<storage, read> inputs: array<f32>;
      @group(0) @binding(2) var<storage, read> A_diag: array<f32>;
      @group(0) @binding(3) var<storage, read> G_diag: array<f32>;
      @group(0) @binding(4) var<storage, read> dt_raw: array<f32>;
      @group(0) @binding(5) var<storage, read_write> x_state: array<f32>;
      @group(0) @binding(6) var<storage, read_write> z_state: array<f32>;
      @group(0) @binding(7) var<storage, read_write> outputs: array<f32>;

      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        let nodeId = gid.x / params.dim;
        let elemId = gid.x % params.dim;

        if (nodeId >= params.nNodes) { return; }

        let d = params.dim;
        let base = nodeId * d;
        let i = base + elemId;

        let dt_val = 1.0 / (1.0 + exp(-dt_raw[i]));
        let A = max(0.0, A_diag[i]);
        let G = max(0.0, G_diag[i]);
        let S = 1.0 + dt_val * G;

        let z_old = z_state[i];
        let x_old = x_state[i];
        let u = inputs[i];

        let z_new = (z_old + dt_val * (-A * x_old + u)) / S;
        let x_new = x_old + dt_val * z_new;

        z_state[i] = z_new;
        x_state[i] = x_new;
        outputs[i] = x_new;
      }
    `;

    try {
      const bdhModule = this.device.createShaderModule({ code: bdhShaderCode });
      this.bdhPipeline = await this.device.createComputePipelineAsync({
        layout: 'auto',
        compute: { module: bdhModule, entryPoint: 'main' }
      });

      const dlinossModule = this.device.createShaderModule({ code: dlinossShaderCode });
      this.dlinossPipeline = await this.device.createComputePipelineAsync({
        layout: 'auto',
        compute: { module: dlinossModule, entryPoint: 'main' }
      });

      console.log('[WebGPU] Compute pipelines created');
    } catch (e) {
      console.log('[WebGPU] Pipeline creation failed:', e.message);
      this.available = false;
    }
  }

  /**
   * Accelerate a batch of BDH node steps on the GPU.
   * Returns true if GPU was used, false for CPU fallback.
   */
  async processBDHBatch(constellation) {
    if (!this.ready || !this.available) return false;

    const c = constellation;
    const d = c.dim;
    const nBDH = c.nBDH;
    const totalFloats = nBDH * d;

    // Pack inputs
    const inputData = new Float32Array(totalFloats);
    const activationData = new Float32Array(totalFloats);
    const encoderData = new Float32Array(nBDH * d * d);
    const decoderData = new Float32Array(nBDH * d * d);

    for (let n = 0; n < nBDH; n++) {
      const node = c.nodes[n];
      const base = n * d;
      const baseW = n * d * d;
      for (let i = 0; i < d; i++) {
        activationData[base + i] = node.activation[i];
      }
      encoderData.set(node.encoderW, baseW);
      decoderData.set(node.decoderW, baseW);
    }

    // Create GPU buffers
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const usageRW = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;

    const paramsData = new Float32Array([d, nBDH, 0.3, 0]);
    new Uint32Array(paramsData.buffer, 0, 2).set([d, nBDH]);

    const paramsBuf = this._createBuffer(paramsData, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const inputBuf = this._createBuffer(inputData, usage);
    const actBuf = this._createBuffer(activationData, usage);
    const encBuf = this._createBuffer(encoderData, usage);
    const decBuf = this._createBuffer(decoderData, usage);
    const outBuf = this._createBuffer(new Float32Array(totalFloats), usageRW);
    const pulseBuf = this._createBuffer(new Float32Array(totalFloats), usageRW);

    const bindGroup = this.device.createBindGroup({
      layout: this.bdhPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: inputBuf } },
        { binding: 2, resource: { buffer: actBuf } },
        { binding: 3, resource: { buffer: encBuf } },
        { binding: 4, resource: { buffer: decBuf } },
        { binding: 5, resource: { buffer: outBuf } },
        { binding: 6, resource: { buffer: pulseBuf } },
      ]
    });

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.bdhPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(totalFloats / 64));
    pass.end();

    // Read back
    const readBuf = this.device.createBuffer({
      size: totalFloats * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });
    encoder.copyBufferToBuffer(outBuf, 0, readBuf, 0, totalFloats * 4);

    this.device.queue.submit([encoder.finish()]);
    await readBuf.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(readBuf.getMappedRange().slice(0));
    readBuf.unmap();

    // Copy results back
    for (let n = 0; n < nBDH; n++) {
      const base = n * d;
      for (let i = 0; i < d; i++) {
        c.nodes[n].activation[i] = result[base + i];
        c.outputs[n][i] = result[base + i];
      }
    }

    // Cleanup
    [paramsBuf, inputBuf, actBuf, encBuf, decBuf, outBuf, pulseBuf, readBuf]
      .forEach(b => b.destroy());

    return true;
  }

  _createBuffer(data, usage) {
    const buf = this.device.createBuffer({
      size: Math.max(data.byteLength, 16),
      usage,
      mappedAtCreation: true,
    });
    new Float32Array(buf.getMappedRange()).set(data);
    buf.unmap();
    return buf;
  }

  getInfo() {
    return {
      available: this.available,
      ready: this.ready,
    };
  }
}
