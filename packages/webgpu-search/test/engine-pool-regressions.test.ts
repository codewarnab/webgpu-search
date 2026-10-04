/**
 * Regressions for GPU device pool / engine lifecycle audit findings:
 *  1. Concurrent GpuDevicePool.acquireDevice shares one in-flight acquisition
 *     (no double requestDevice, refcount counts every holder); releaseDevice
 *     honours device identity; intentional destroy() does not fire
 *     onDeviceLost listeners.
 *  2. Headroom capacity is clamped to min(maxBufferSize,
 *     maxStorageBufferBindingSize); actual data above the limit throws the
 *     budget error (CPU fallback) instead of binding an oversized buffer.
 *  3. Teardown/reallocation during the timestamp readback await surfaces as
 *     AbortError, never a TypeError or a silent empty result.
 *  4. Failed / repeated init() never leaks a pool reference.
 *  5. Native CPU baseline scores in normalized code points (GPU symmetry).
 *
 * Run: bun test packages/webgpu-search/test/engine-pool-regressions.test.ts
 */
import { describe, test, expect, afterEach } from 'bun:test';
import { createMockAdapter } from 'vgpu/mock';
import {
  WebGPUEngine,
  CPUEngine,
  GpuDevicePool,
  computeClampedHeadroomBytes,
  normalizeText,
  scoreSubstringTokens,
} from '../src/index';

const NAV_DESC = Object.getOwnPropertyDescriptor(globalThis, 'navigator');

function restoreNavigator(): void {
  if (NAV_DESC) Object.defineProperty(globalThis, 'navigator', NAV_DESC);
  else delete (globalThis as { navigator?: unknown }).navigator;
}

interface FakeDevice {
  id: number;
  lost: Promise<{ reason: string; message: string }>;
  resolveLost: (info: { reason: string; message: string }) => void;
  destroyed: boolean;
  destroy(): void;
}

/** Install a fake navigator.gpu whose devices resolve `lost` like real WebGPU. */
function installFakeGpu(makeDevice?: () => Promise<any>) {
  const created: FakeDevice[] = [];
  let requestDeviceCalls = 0;
  const limits = {
    maxBufferSize: 1 << 28,
    maxStorageBufferBindingSize: 1 << 27,
    maxComputeWorkgroupsPerDimension: 65535,
    maxComputeInvocationsPerWorkgroup: 256,
  };
  const adapter = {
    features: new Set<string>(),
    limits,
    info: { vendor: 'test', device: 'fake' },
    requestDevice: async () => {
      requestDeviceCalls++;
      if (makeDevice) return makeDevice();
      await new Promise((r) => setTimeout(r, 5));
      let resolveLost!: FakeDevice['resolveLost'];
      const lost = new Promise<{ reason: string; message: string }>((r) => (resolveLost = r));
      const d: FakeDevice = {
        id: created.length + 1,
        lost,
        resolveLost,
        destroyed: false,
        destroy() {
          if (this.destroyed) return;
          this.destroyed = true;
          resolveLost({ reason: 'destroyed', message: 'Device destroyed' });
        },
      };
      created.push(d);
      return d;
    },
  };
  Object.defineProperty(globalThis, 'navigator', {
    value: { gpu: { requestAdapter: async () => adapter } },
    configurable: true,
    writable: true,
  });
  return { created, calls: () => requestDeviceCalls };
}

const tick = () => new Promise((r) => setTimeout(r, 0));

afterEach(() => {
  restoreNavigator();
});

describe('GpuDevicePool concurrency + identity (finding 1)', () => {
  test('concurrent acquires share one device and refcount both holders', async () => {
    const fake = installFakeGpu();
    const [a, b] = await Promise.all([GpuDevicePool.acquireDevice(), GpuDevicePool.acquireDevice()]);
    expect(a).not.toBeNull();
    expect(b).not.toBeNull();
    expect(a!.device).toBe(b!.device);
    expect(fake.calls()).toBe(1);
    expect(GpuDevicePool.getRefCount()).toBe(2);

    GpuDevicePool.releaseDevice(a!.device, true);
    expect((b!.device as unknown as FakeDevice).destroyed).toBe(false);
    expect(GpuDevicePool.getRefCount()).toBe(1);
    expect(GpuDevicePool.hasSharedDevice()).toBe(true);

    GpuDevicePool.releaseDevice(b!.device, true);
    expect((b!.device as unknown as FakeDevice).destroyed).toBe(true);
    expect(GpuDevicePool.getRefCount()).toBe(0);
    expect(GpuDevicePool.hasSharedDevice()).toBe(false);
  });

  test('releasing a stale / foreign device never decrements the pooled device', async () => {
    installFakeGpu();
    const first = await GpuDevicePool.acquireDevice();
    GpuDevicePool.simulateDeviceLoss('stale-test');
    const second = await GpuDevicePool.acquireDevice();
    expect(second!.device).not.toBe(first!.device);
    expect(GpuDevicePool.getRefCount()).toBe(1);

    // Stale holder releases its (lost) device: the live device must survive.
    GpuDevicePool.releaseDevice(first!.device, true);
    GpuDevicePool.releaseDevice({} as GPUDevice, true);
    expect(GpuDevicePool.getRefCount()).toBe(1);
    expect((second!.device as unknown as FakeDevice).destroyed).toBe(false);

    GpuDevicePool.releaseDevice(second!.device, true);
    expect(GpuDevicePool.hasSharedDevice()).toBe(false);
  });

  test('intentional destroy does not fire onDeviceLost; real loss fires once', async () => {
    const fake = installFakeGpu();
    const reasons: string[] = [];
    const unsub = GpuDevicePool.onDeviceLost((r) => reasons.push(r));
    try {
      const a = await GpuDevicePool.acquireDevice();
      GpuDevicePool.releaseDevice(a!.device, true); // refcount 0 -> destroy()
      await tick();
      expect(reasons).toEqual([]);

      const b = await GpuDevicePool.acquireDevice();
      (b!.device as unknown as FakeDevice).resolveLost({ reason: 'unknown', message: 'GPU hung' });
      await tick();
      expect(reasons).toEqual(['GPU hung']);
      expect(GpuDevicePool.hasSharedDevice()).toBe(false);
      expect(GpuDevicePool.getRefCount()).toBe(0);

      // Late loss of an already-replaced device is ignored (identity check).
      const c = await GpuDevicePool.acquireDevice();
      fake.created[0]!.resolveLost({ reason: 'unknown', message: 'stale' });
      await tick();
      expect(reasons).toEqual(['GPU hung']);
      expect(GpuDevicePool.hasSharedDevice()).toBe(true);
      GpuDevicePool.releaseDevice(c!.device, true);
    } finally {
      unsub();
    }
  });
});

async function mockDevice(limits?: Record<string, number>): Promise<GPUDevice> {
  const dev = (await createMockAdapter({ features: ['timestamp-query'] as any }).requestDevice()).gpu as any;
  return (limits ? { ...dev, limits: { ...dev.limits, ...limits } } : dev) as GPUDevice;
}

describe('headroom clamp vs binding limit (finding 2)', () => {
  test('computeClampedHeadroomBytes never exceeds min(maxBufferSize, maxStorageBufferBindingSize)', () => {
    const device = { limits: { maxBufferSize: 256 * 2 ** 20, maxStorageBufferBindingSize: 128 * 2 ** 20 } } as any;
    const bytes = computeClampedHeadroomBytes(160_000_000, { growthFactor: 1.0, device });
    expect(bytes).toBe(128 * 2 ** 20);
    // Unaligned limit floors to a 4-byte multiple.
    const odd = { limits: { maxBufferSize: 1001, maxStorageBufferBindingSize: 1001 } } as any;
    expect(computeClampedHeadroomBytes(5000, { device: odd })).toBe(1000);
  });

  test('oversized tokenCapacity request is clamped at allocation', async () => {
    const limit = 1 << 20;
    const dev = await mockDevice({ maxBufferSize: limit, maxStorageBufferBindingSize: limit });
    const sizes: number[] = [];
    const createBuffer = dev.createBuffer.bind(dev);
    (dev as any).createBuffer = (d: GPUBufferDescriptor) => {
      sizes.push(d.size);
      return createBuffer(d);
    };
    const engine = new WebGPUEngine();
    await engine.init(dev);
    await engine.loadDataset(['alpha', 'beta'], { tokenCapacity: 10_000_000, rowCapacity: 10_000_000 });
    expect(engine.allocatedRecordsBytes).toBeLessThanOrEqual(limit);
    expect(engine.allocatedOffsetsBytes).toBeLessThanOrEqual(limit);
    expect(Math.max(...sizes)).toBeLessThanOrEqual(limit);
    engine.destroy();
  });

  test('actual data above the binding limit throws the budget error', async () => {
    const limit = 70_000; // output buffer (8 + 8192*8) fits; records do not
    const dev = await mockDevice({ maxBufferSize: limit, maxStorageBufferBindingSize: limit });
    const engine = new WebGPUEngine();
    await engine.init(dev);
    const rows = Array.from({ length: 200 }, (_, i) => `row-${i}-`.padEnd(120, 'x'));
    await expect(engine.loadDataset(rows)).rejects.toThrow(/too large/);
    expect(engine.currentSize).toBe(0);
    engine.destroy();
  });
});

describe('teardown during timestamp readback (finding 3)', () => {
  async function readyEngine() {
    const engine = new WebGPUEngine();
    await engine.init(await mockDevice());
    await engine.loadDataset(['hello world', 'help']);
    return engine;
  }

  test('destroy() during timestamp mapAsync rejects with AbortError', async () => {
    const engine = await readyEngine();
    (engine as any).queryStagingBuffer = {
      mapAsync: async () => { engine.destroy(); },
      getMappedRange: () => new ArrayBuffer(16),
      unmap() {},
      destroy() {},
    };
    const err = await engine.search('hel', { mode: 'fuzzy' }).then(() => null, (e) => e);
    expect(err).not.toBeNull();
    expect((err as Error).name).toBe('AbortError');
  });

  test('output reallocation during timestamp mapAsync rejects with AbortError', async () => {
    const engine = await readyEngine();
    (engine as any).queryStagingBuffer = {
      mapAsync: async () => { engine.ensureCandidateCapacity(20_000); },
      getMappedRange: () => new ArrayBuffer(16),
      unmap() {},
      destroy() {},
    };
    const err = await engine.search('hel', { mode: 'fuzzy' }).then(() => null, (e) => e);
    expect((err as Error | null)?.name).toBe('AbortError');
    engine.destroy();
  });
});

describe('init() lifecycle never leaks pool references (finding 4)', () => {
  test('init() twice on the pooled device keeps refcount at 1', async () => {
    const mock = await mockDevice();
    installFakeGpu(async () => mock);
    const engine = new WebGPUEngine();
    expect(await engine.init()).toBe(true);
    expect(GpuDevicePool.getRefCount()).toBe(1);
    expect(await engine.init()).toBe(true);
    expect(GpuDevicePool.getRefCount()).toBe(1);
    engine.destroy();
    expect(GpuDevicePool.getRefCount()).toBe(0);
    expect(GpuDevicePool.hasSharedDevice()).toBe(false);
  });

  test('pipeline creation failure after acquire releases the device', async () => {
    const mock = await mockDevice();
    const broken = {
      ...(mock as any),
      createComputePipelineAsync: async () => { throw new Error('pipeline boom'); },
    };
    installFakeGpu(async () => broken);
    const engine = new WebGPUEngine();
    await expect(engine.init()).rejects.toThrow(/pipeline boom/);
    expect(engine.isReady).toBe(false);
    expect(GpuDevicePool.getRefCount()).toBe(0);
    expect(GpuDevicePool.hasSharedDevice()).toBe(false);
  });
});

describe('native CPU baseline scoring symmetry (finding 5)', () => {
  test('astral prefix scores in code points like the exact/GPU scorer', () => {
    const rec = '\u{1F600}abc';
    const exact = scoreSubstringTokens(normalizeText(rec, true).tokens, normalizeText('abc', true).tokens);
    const native = new CPUEngine().searchNaiveScan([rec], 'abc', 10, false);
    expect(native.totalMatches).toBe(1);
    expect(native.results[0]!.score).toBe(exact.score);
    expect(exact.score).toBe(989);
  });

  test('NFC-equivalent inputs match and score identically', () => {
    const composed = 'café';
    const decomposed = 'café';
    const res = new CPUEngine().searchNaiveScan([decomposed], composed, 10, false);
    expect(res.totalMatches).toBe(1);
    expect(res.results[0]!.score).toBe(1000);
  });
});
