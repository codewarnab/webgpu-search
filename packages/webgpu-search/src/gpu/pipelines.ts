/**
 * Shader module + compute pipeline setup for the WebGPU engine (verbatim
 * move of the pipeline half of the facade's `setupPipelinesAndBuffers`).
 * Device reads stay live (`state.device`): a device lost mid-setup still
 * throws (fail-closed teardown), never completes init on a stale device.
 */

import SUBSTRING_WGSL from '../shaders/substring.wgsl';
import FUZZY_WGSL from '../shaders/fuzzy.wgsl';
import type { GpuEngineState } from './state';

export class GpuPipelineManager {
  constructor(private readonly state: GpuEngineState) {}

  /**
   * Create substring/fuzzy modules and compute pipelines. Verbatim call
   * order (module → module → pipeline → pipeline): mock/spy compat.
   * `createComputePipelineAsync` when available, sync fallback otherwise.
   */
  async setup(): Promise<void> {
    const substringModule = this.state.device!.createShaderModule({
      label: 'substring',
      code: SUBSTRING_WGSL
    });

    const fuzzyModule = this.state.device!.createShaderModule({
      label: 'fuzzy',
      code: FUZZY_WGSL
    });

    const createPipeline = async (desc: GPUComputePipelineDescriptor) => {
      if (typeof (this.state.device as any).createComputePipelineAsync === 'function') {
        return await this.state.device!.createComputePipelineAsync(desc);
      }
      return (this.state.device as any).createComputePipeline(desc);
    };

    this.state.substringPipeline = await createPipeline({
      label: 'substring',
      layout: 'auto',
      compute: {
        module: substringModule,
        entryPoint: 'main'
      }
    });

    this.state.fuzzyPipeline = await createPipeline({
      label: 'fuzzy',
      layout: 'auto',
      compute: {
        module: fuzzyModule,
        entryPoint: 'main'
      }
    });
  }

  /** Null the pipeline handles (destroy / device loss / re-init). */
  drop(): void {
    this.state.substringPipeline = null;
    this.state.fuzzyPipeline = null;
  }
}
