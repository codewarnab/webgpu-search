/**
 * Fail-closed WebGPUEngine.loadDataset without a device.
 *
 * Regression: loadDatasetInternal returned { uploadTimeMs: 0 } when
 * `!device`, silently accepting a dataset with no GPU upload. It must
 * throw WebGPUSearchError instead.
 *
 * Run: bun test packages/webgpu-search/test/webgpu-engine-fail-closed.test.ts
 */
import { describe, test, expect } from 'bun:test';
import { createMockAdapter } from 'vgpu/mock';
import { WebGPUEngine, WebGPUSearchError } from '../src/index';

describe('WebGPUEngine.loadDataset fail-closed without device', () => {
  test('throws WebGPUSearchError instead of returning {uploadTimeMs:0}', async () => {
    const engine = new WebGPUEngine();
    expect(engine.isReady).toBe(false);

    let error: unknown = null;
    try {
      await engine.loadDataset(['alpha', 'beta']);
    } catch (e) {
      error = e;
    }
    expect(error).toBeInstanceOf(WebGPUSearchError);
    expect((error as Error).message).toMatch(/initialized GPU device/);
    // No phantom CPU state left behind.
    expect(engine.currentSize).toBe(0);
    expect(engine.isReady).toBe(false);
    engine.destroy();
  });

  test('searchCold without device throws WebGPUSearchError', async () => {
    const engine = new WebGPUEngine();
    await expect(
      engine.searchCold(['alpha'], 'alpha', { mode: 'substring' })
    ).rejects.toBeInstanceOf(WebGPUSearchError);
    engine.destroy();
  });

  test('loadDataset succeeds after init (no regression)', async () => {
    const adapter = await createMockAdapter({ features: ['timestamp-query'] as never });
    const wrapper = await adapter.requestDevice();
    const mockDevice = ((wrapper as unknown as { gpu: GPUDevice }).gpu ?? (wrapper as unknown as GPUDevice));
    const engine = new WebGPUEngine();
    await engine.init(mockDevice);
    const { uploadTimeMs } = await engine.loadDataset(['alpha', 'beta']);
    expect(typeof uploadTimeMs).toBe('number');
    expect(engine.currentSize).toBe(2);
    engine.destroy();
  });
});
