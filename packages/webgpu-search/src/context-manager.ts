import type { AdapterInfo } from './types';

export interface AcquiredDeviceContext {
  device: GPUDevice;
  adapterInfo: AdapterInfo | null;
  isShared: boolean;
}

export class WebGPUContextManager {
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

  static getAdapterInfo(): AdapterInfo | null {
    return this.sharedAdapterInfo;
  }
}
