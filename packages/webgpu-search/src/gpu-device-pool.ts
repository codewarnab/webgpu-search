import type { AdapterInfo } from './types';
import { IncompatibleOptionError } from './errors';

export interface AcquiredDeviceContext {
  device: GPUDevice;
  adapterInfo: AdapterInfo | null;
  isShared: boolean;
}

/**
 * Fail-closed power-preference validation.
 * Accepts only 'high-performance' | 'low-power' (or undefined = default).
 * Unknown values throw `IncompatibleOptionError` — even on CPU-only paths
 * where no adapter request is made.
 */
export function assertValidPowerPreference(value: unknown): asserts value is GPUPowerPreference | undefined {
  if (value === undefined) return;
  if (value !== 'high-performance' && value !== 'low-power') {
    throw new IncompatibleOptionError(
      'powerPreference',
      `Unknown powerPreference '${String(value)}'. Expected 'high-performance' or 'low-power'.`
    );
  }
}

export class GpuDevicePool {
  private static sharedDevice: GPUDevice | null = null;
  private static sharedAdapter: GPUAdapter | null = null;
  private static sharedAdapterInfo: AdapterInfo | null = null;
  private static refCount: number = 0;
  private static deviceLostListeners: Set<(reason: string) => void> = new Set();

  /**
   * Check if WebGPU is available in the current environment (DOM or Web Worker).
   */
  static isSupported(): boolean {
    return typeof navigator !== 'undefined' && !!navigator.gpu;
  }

  /**
   * Safely obtain unmasked GPU renderer without throwing ReferenceError in Web Workers or Node.
   */
  static getUnmaskedRenderer(): string {
    if (typeof OffscreenCanvas !== 'undefined') {
      try {
        const canvas = new OffscreenCanvas(1, 1);
        const gl = canvas.getContext('webgl') || (canvas as any).getContext('experimental-webgl');
        if (gl) {
          const ext = (gl as any).getExtension('WEBGL_debug_renderer_info');
          if (ext) {
            return (gl as any).getParameter(ext.UNMASKED_RENDERER_WEBGL) || '';
          }
        }
      } catch {
        // ignore
      }
    } else if (typeof document !== 'undefined') {
      try {
        const canvas = document.createElement('canvas');
        const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
        if (gl) {
          const ext = (gl as any).getExtension('WEBGL_debug_renderer_info');
          if (ext) {
            return (gl as any).getParameter(ext.UNMASKED_RENDERER_WEBGL) || '';
          }
        }
      } catch {
        // ignore
      }
    }
    return '';
  }

  /**
   * Acquire a GPUDevice. Multiplexes a single shared GPUDevice across callers,
   * or wraps a custom user-injected GPUDevice.
   */
  static async acquireDevice(options?: {
    device?: GPUDevice;
    powerPreference?: GPUPowerPreference;
  }): Promise<AcquiredDeviceContext | null> {
    // Fail-closed: unknown preferences throw even when a custom device is
    // injected (no adapter request) or WebGPU is unsupported.
    assertValidPowerPreference(options?.powerPreference);
    // 1. Custom injected device (e.g. testing with vgpu/mock or existing 3D context)
    if (options?.device) {
      const mockInfo: AdapterInfo = {
        vendor: 'Custom / Mock Vendor',
        architecture: 'Custom Architecture',
        device: 'Custom GPUDevice',
        description: 'Injected GPUDevice instance',
        renderer: 'Custom GPUDevice',
        maxBufferSizeMB: 256,
        maxStorageBindingSizeMB: 128,
        maxComputeWorkgroupsPerDimension: 65535,
        maxComputeInvocationsPerWorkgroup: 256,
        hasTimestampQuery: options.device.features ? options.device.features.has('timestamp-query') : false
      };
      return {
        device: options.device,
        adapterInfo: mockInfo,
        isShared: false
      };
    }

    // 2. Reuse existing shared device if healthy
    if (this.sharedDevice) {
      this.refCount++;
      return {
        device: this.sharedDevice,
        adapterInfo: this.sharedAdapterInfo,
        isShared: true
      };
    }

    // 3. Check WebGPU availability
    if (!this.isSupported()) {
      return null;
    }

    try {
      const powerPref = options?.powerPreference || 'high-performance';
      this.sharedAdapter = await navigator.gpu.requestAdapter({ powerPreference: powerPref });
      if (!this.sharedAdapter) {
        this.sharedAdapter = await navigator.gpu.requestAdapter();
      }

      if (!this.sharedAdapter) {
        return null;
      }

      // Inspect adapter info
      let info: any = (this.sharedAdapter as any).info || {};
      if (!info.vendor && !info.device && 'requestAdapterInfo' in this.sharedAdapter) {
        try {
          info = await (this.sharedAdapter as any).requestAdapterInfo();
        } catch {
          info = {};
        }
      }

      const unmaskedRenderer = this.getUnmaskedRenderer();
      let vendor = info.vendor || '';
      let device = info.device || '';
      let architecture = info.architecture || '';

      if (!device && unmaskedRenderer) device = unmaskedRenderer;
      if (!vendor) {
        if (/nvidia/i.test(unmaskedRenderer) || /nvidia/i.test(device)) vendor = 'NVIDIA';
        else if (/intel/i.test(unmaskedRenderer) || /intel/i.test(device)) vendor = 'Intel';
        else if (/amd|radeon/i.test(unmaskedRenderer) || /amd|radeon/i.test(device)) vendor = 'AMD';
        else if (/apple/i.test(unmaskedRenderer) || /apple/i.test(device)) vendor = 'Apple';
        else vendor = 'Unknown GPU Vendor';
      }
      if (!device) device = 'WebGPU Generic Device';
      if (!architecture) architecture = 'Default';

      const limits = this.sharedAdapter.limits;
      const requiredFeatures: GPUFeatureName[] = [];
      const hasTimestamp = this.sharedAdapter.features.has('timestamp-query');
      if (hasTimestamp) {
        requiredFeatures.push('timestamp-query');
      }

      this.sharedAdapterInfo = {
        vendor,
        architecture,
        device,
        description: info.description || unmaskedRenderer || (typeof navigator !== 'undefined' ? navigator.userAgent : 'WebGPU Device'),
        renderer: unmaskedRenderer || device,
        maxBufferSizeMB: Math.round(limits.maxBufferSize / (1024 * 1024)),
        maxStorageBindingSizeMB: Math.round(limits.maxStorageBufferBindingSize / (1024 * 1024)),
        maxComputeWorkgroupsPerDimension: limits.maxComputeWorkgroupsPerDimension,
        maxComputeInvocationsPerWorkgroup: limits.maxComputeInvocationsPerWorkgroup,
        hasTimestampQuery: hasTimestamp
      };

      // Request device with graceful limit fallback
      try {
        this.sharedDevice = await this.sharedAdapter.requestDevice({
          requiredFeatures,
          requiredLimits: {
            maxBufferSize: limits.maxBufferSize,
            maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize,
            maxComputeWorkgroupsPerDimension: limits.maxComputeWorkgroupsPerDimension
          }
        });
      } catch {
        try {
          this.sharedDevice = await this.sharedAdapter.requestDevice({ requiredFeatures });
        } catch {
          this.sharedDevice = await this.sharedAdapter.requestDevice();
        }
      }

      this.refCount = 1;

      // Handle device loss gracefully
      this.sharedDevice.lost.then(lostInfo => {
        const reason = lostInfo.message || 'WebGPU device lost';
        console.warn('WebGPU device lost:', reason);
        this.sharedDevice = null;
        this.sharedAdapter = null;
        this.refCount = 0;
        for (const listener of this.deviceLostListeners) {
          try {
            listener(reason);
          } catch (e) {
            console.error('Error in device lost listener:', e);
          }
        }
      }, () => {});

      return {
        device: this.sharedDevice,
        adapterInfo: this.sharedAdapterInfo,
        isShared: true
      };
    } catch (err) {
      console.warn('Failed to acquire WebGPU device:', err);
      return null;
    }
  }

  /**
   * Release device reference. When reference count drops to 0, shared device is destroyed.
   */
  static releaseDevice(_device: GPUDevice, isShared: boolean = true): void {
    if (!isShared) return;
    this.refCount = Math.max(0, this.refCount - 1);
    if (this.refCount === 0 && this.sharedDevice) {
      try {
        this.sharedDevice.destroy();
      } catch {
        // ignore
      }
      this.sharedDevice = null;
      this.sharedAdapter = null;
    }
  }

  /**
   * Subscribe to device loss events.
   */
  static onDeviceLost(listener: (reason: string) => void): () => void {
    this.deviceLostListeners.add(listener);
    return () => this.deviceLostListeners.delete(listener);
  }

  /**
   * Introspection for reliability proofs (Phase 2): number of live
   * `onDeviceLost` subscriptions. Leak tests assert this returns to
   * baseline after create/destroy cycles (no unguarded DOM — safe in
   * workers/Node/SSR).
   */
  static getListenerCount(): number {
    return this.deviceLostListeners.size;
  }

  /**
   * Introspection for reliability proofs (Phase 2): current shared-device
   * reference count. Injected (non-shared) devices never touch this counter.
   */
  static getRefCount(): number {
    return this.refCount;
  }

  /**
   * True while a shared device is held by the pool.
   */
  static hasSharedDevice(): boolean {
    return this.sharedDevice !== null;
  }

  /**
   * Deterministic device-loss simulation for headless reliability proofs.
   *
   * Clears the shared device/adapter, zeroes the refcount, and notifies
   * every `onDeviceLost` subscriber — the same fan-out the real
   * `device.lost.then` handler performs. Indexes holding injected (mock)
   * devices still observe the transition via their subscription and fall
   * back to CPU with `fallbackReason: 'device-lost'`, so rebuild semantics
   * (`rebuildGpu`) can be proven without executing hardware.
   *
   * Portable: no DOM / `navigator` access — safe in workers/Node/SSR.
   */
  static simulateDeviceLoss(reason: string = 'Simulated device loss'): void {
    this.sharedDevice = null;
    this.sharedAdapter = null;
    this.refCount = 0;
    for (const listener of [...this.deviceLostListeners]) {
      try {
        listener(reason);
      } catch (e) {
        console.error('Error in device lost listener:', e);
      }
    }
  }

  static getAdapterInfo(): AdapterInfo | null {
    return this.sharedAdapterInfo;
  }
}

/** @deprecated Use GpuDevicePool. */
export const WebGPUContextManager = GpuDevicePool;
