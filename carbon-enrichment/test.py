import pyarrow.dataset as ds
import pyarrow.compute as pc
import numpy as np

path = '.dagster_hf_storage/test_100k'
dataset = ds.dataset(path, format='parquet')

# Streaming statistics
count = 0
sum_x = 0.0
sum_x2 = 0.0
min_x = None
max_x = None

# Keep only a bounded sample for percentile estimation.
# Increase this if you want more accurate percentiles.
SAMPLE_SIZE = 1_000_000
sample = np.empty(SAMPLE_SIZE, dtype=np.float64)
sample_count = 0

scanner = dataset.scanner(
    columns=['sequence_length'],
    batch_size=8192,
    use_threads=True,
)

rng = np.random.default_rng(42)

for batch in scanner.to_batches():
    arr = batch.column(0)

    # Drop nulls
    arr = pc.drop_null(arr)

    if len(arr) == 0:
        continue

    values = arr.to_numpy(zero_copy_only=False).astype(
        np.float64, copy=False
    )

    # Basic statistics
    count += len(values)
    sum_x += np.sum(values)
    sum_x2 += np.sum(values ** 2)

    batch_min = np.min(values)
    batch_max = np.max(values)

    min_x = batch_min if min_x is None else min(min_x, batch_min)
    max_x = batch_max if max_x is None else max(max_x, batch_max)

    # Reservoir sampling for bounded-memory percentiles
    for value in values:
        if sample_count < SAMPLE_SIZE:
            sample[sample_count] = value
        else:
            j = rng.integers(0, count)
            if j < SAMPLE_SIZE:
                sample[j] = value
        sample_count += 1

# Final statistics
mean = sum_x / count
variance = (sum_x2 - (sum_x ** 2) / count) / (count - 1)
stddev = np.sqrt(max(variance, 0.0))

sample = sample[:min(sample_count, SAMPLE_SIZE)]

print('count:  ', count)
print('mean:   ', mean)
print('stddev: ', stddev)
print('min:    ', min_x)
print('max:    ', max_x)

for p in [50, 90, 95, 99]:
    print(f'p{p}:    ', np.percentile(sample, p))