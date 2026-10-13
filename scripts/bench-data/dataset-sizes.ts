/** Shared interactive benchmark sizes; frozen snapshot fixtures are versioned separately. */
export const DATASET_SIZES = [10_000, 100_000, 200_000, 500_000, 1_000_000, 2_000_000] as const;

export const SITE_DATASET_SIZES = DATASET_SIZES.filter(size => size >= 100_000);
export const BENCHMARK_DATASET_SIZES = DATASET_SIZES.filter(size => size !== 200_000);
