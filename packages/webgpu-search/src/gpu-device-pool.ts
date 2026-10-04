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
  private static sharedAdapterInfo: AdapterInfo | null = null;
  private static refCount: number = 0;
  private static inFlight: Promise<GPUDevice | null> | null = null;
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

    // 4. Single in-flight acquisition: concurrent callers share one
    // requestAdapter/requestDevice so the pool never creates (and leaks) a
    // second device or resets the refcount under a live holder. Each caller
    // takes its own reference once the shared acquisition settles.
    if (!this.inFlight) {
      const p = this.createSharedDevice(options?.powerPreference);
      this.inFlight = p;
      const clear = () => {
        if (this.inFlight === p) this.inFlight = null;
      };
      p.then(clear, clear);
    }
    const device = await this.inFlight;
    // Device may have been lost between creation and this continuation.
    if (!device || this.sharedDevice !== device) {
      // Superseded while in flight (e.g. simulated loss): nobody holds a
      // reference to this stale device, so release it instead of leaking.
      if (device && this.sharedDevice !== device) device.destroy();
      return null;
    }
    this.refCount++;
    return {
      device,
      adapterInfo: this.sharedAdapterInfo,
      isShared: true
    };
  }

  /**
   * Request adapter + device and install it as the pooled device with
   * refCount 0 (callers increment). Resolves null on failure; never throws.
   */
  private static async createSharedDevice(
    powerPreference?: GPUPowerPreference
  ): Promise<GPUDevice | null> {
    try {
      const powerPref = powerPreference || 'high-performance';
      let adapter = await navigator.gpu.requestAdapter({ powerPreference: powerPref });
      if (!adapter) {
        adapter = await navigator.gpu.requestAdapter();
      }

      if (!adapter) {
        return null;
      }

      // Inspect adapter info
      let info: any = (adapter as any).info || {};
      if (!info.vendor && !info.device && 'requestAdapterInfo' in adapter) {
        try {
          info = await (adapter as any).requestAdapterInfo();
        } catch {
          info = {};
        }
      }

      const unmaskedRenderer = this.getUnmaskedRenderer();
      let vendor = info.vendor || '';
      let deviceName = info.device || '';
      let architecture = info.architecture || '';
      const rawAdapterType =
        typeof info.type === 'string' && info.type ? String(info.type) : undefined;

      if (!deviceName && unmaskedRenderer) deviceName = unmaskedRenderer;
      if (!vendor) {
        if (/nvidia/i.test(unmaskedRenderer) || /nvidia/i.test(deviceName)) vendor = 'NVIDIA';
        else if (/intel/i.test(unmaskedRenderer) || /intel/i.test(deviceName)) vendor = 'Intel';
        else if (/amd|radeon/i.test(unmaskedRenderer) || /amd|radeon/i.test(deviceName)) vendor = 'AMD';
        else if (/apple/i.test(unmaskedRenderer) || /apple/i.test(deviceName)) vendor = 'Apple';
        else vendor = 'Unknown GPU Vendor';
      }
      if (!deviceName) deviceName = 'WebGPU Generic Device';
      if (!architecture) architecture = 'Default';

      const limits = adapter.limits;
      const requiredFeatures: GPUFeatureName[] = [];
      const hasTimestamp = adapter.features.has('timestamp-query');
      if (hasTimestamp) {
        requiredFeatures.push('timestamp-query');
      }

      const adapterInfo: AdapterInfo = {
        vendor,
        architecture,
        device: deviceName,
        description: info.description || unmaskedRenderer || (typeof navigator !== 'undefined' ? navigator.userAgent : 'WebGPU Device'),
        renderer: unmaskedRenderer || deviceName,
        ...(rawAdapterType ? { adapterType: rawAdapterType } : {}),
        maxBufferSizeMB: Math.round(limits.maxBufferSize / (1024 * 1024)),
        maxStorageBindingSizeMB: Math.round(limits.maxStorageBufferBindingSize / (1024 * 1024)),
        maxComputeWorkgroupsPerDimension: limits.maxComputeWorkgroupsPerDimension,
        maxComputeInvocationsPerWorkgroup: limits.maxComputeInvocationsPerWorkgroup,
        hasTimestampQuery: hasTimestamp
      };

      // Request device with graceful limit fallback
      let device: GPUDevice;
      try {
        device = await adapter.requestDevice({
          requiredFeatures,
          requiredLimits: {
            maxBufferSize: limits.maxBufferSize,
            maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize,
            maxComputeWorkgroupsPerDimension: limits.maxComputeWorkgroupsPerDimension
          }
        });
      } catch {
        try {
          device = await adapter.requestDevice({ requiredFeatures });
        } catch {
          device = await adapter.requestDevice();
        }
      }

      this.sharedAdapterInfo = adapterInfo;
      this.sharedDevice = device;
      this.refCount = 0;

      // Handle device loss gracefully. Identity-checked: a stale handler for
      // a previously pooled device must not tear down the current one, and
      // intentional destroy() (pool release → reason 'destroyed') is not a
      // loss event — listeners would otherwise push healthy indexes to CPU.
      device.lost?.then?.(lostInfo => {
        if (this.sharedDevice !== device) return;
        this.sharedDevice = null;
        this.refCount = 0;
        if ((lostInfo as { reason?: unknown } | undefined)?.reason === 'destroyed') return;
        const reason = lostInfo?.message || 'WebGPU device lost';
        console.warn('WebGPU device lost:', reason);
        for (const listener of [...this.deviceLostListeners]) {
          try {
            listener(reason);
          } catch (e) {
            console.error('Error in device lost listener:', e);
          }
        }
      }, () => {});

      return device;
    } catch (err) {
      console.warn('Failed to acquire WebGPU device:', err);
      return null;
    }
  }

  /**
   * Release a device reference. Only the currently pooled device is
   * refcounted: releasing a stale device (from before a loss/re-acquire) or a
   * non-shared injected device is a no-op, so it can never decrement — and
   * destroy — another holder's device. When the count drops to 0 the shared
   * device is destroyed.
   */
  static releaseDevice(device: GPUDevice, isShared: boolean = true): void {
    if (!isShared) return;
    if (!device || device !== this.sharedDevice) return;
    this.refCount = Math.max(0, this.refCount - 1);
    if (this.refCount === 0) {
      const dev = this.sharedDevice;
      // Null first so the lost handler (reason 'destroyed') sees a stale
      // identity and stays silent.
      this.sharedDevice = null;
      try {
        dev.destroy();
      } catch {
        // ignore
      }
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
