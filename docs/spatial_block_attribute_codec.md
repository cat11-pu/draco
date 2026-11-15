# Spatial Block Attribute Codec

`draco/compression/attributes/spatial_block_attribute_codec.h` provides a small
API for per-block point-cloud attribute prediction. It is independent of the
existing Draco bitstream so callers can use blocked prediction without changing
the on-disk Draco format.

## Interface

```cpp
struct SpatialBlockCodecOptions {
  uint32_t max_points_per_block = 1024;
  double max_reconstruction_error = 0.0;
  std::vector<SpatialBlockPredictor> predictors = {
      SpatialBlockPredictor::kPrevious,
      SpatialBlockPredictor::kLinear};
};

Status SpatialBlockAttributeCodec::Encode(
    const PointCloud& point_cloud, int attribute_id,
    const SpatialBlockCodecOptions& options,
    EncodedSpatialBlockAttribute* encoded);

Status SpatialBlockAttributeCodec::Decode(
    const EncodedSpatialBlockAttribute& encoded,
    std::vector<double>* values);

Status SpatialBlockAttributeCodec::DecodeBlock(
    const EncodedSpatialBlockAttribute& encoded, uint32_t block_index,
    std::vector<double>* values,
    std::vector<uint32_t>* point_indices = nullptr);
```

`Encode()` reads the three-component `POSITION` attribute for spatial ordering
and the selected attribute for prediction. Points are sorted lexicographically
by position, with the original point index as the tie-breaker, and split into
consecutive blocks of at most `max_points_per_block` points.

Each block selects the candidate predictor with the cheapest encoded residual
cost. `kPrevious` uses the immediately preceding reconstructed sample;
`kLinear` extrapolates from the two preceding reconstructed samples and falls
back to `kPrevious` when only one sample is available.

The first sample of every block continues from the previous block. The seed
values are serialized in the block, which has two consequences:

1. Block boundaries never restart prediction at zero.
2. Decoding one block independently and decoding all blocks in one call produce
   bit-identical values, item by item.

`Decode()` returns point-major values in the original point order.
`DecodeBlock()` returns point-major values in spatial block order and can also
return the corresponding original point indices.

## Error contract

For a positive threshold, the quantization step equals
`max_reconstruction_error`. Nearest-step quantization therefore keeps every
component's reconstruction error at or below half the threshold. A threshold of
zero selects exact value coding and produces zero reconstruction error.

The returned `SpatialBlockResidualStats` reports:

- `sum_abs_residual`, `max_abs_residual`, and `rms_residual` over all components
  in the block, measured before quantization against the selected predictor.
- `max_abs_reconstruction_error`, measured against the original values.
- `boundary_abs_residual`, the predictor residual at the first point of the
  block. This is the seam residual and is zero for a perfectly linear signal
  using `kLinear`.
- Block bounding box, point count, first/last point index, and whether previous
  context was available.

The double-valued interface supports integer attributes up to 32 bits and
floating-point attributes. `DT_INT64` and `DT_UINT64` are rejected because they
cannot be represented losslessly by this interface.

## Boundary cases

- Empty point clouds return `INVALID_PARAMETER`.
- A point cloud without a three-component `POSITION` attribute is rejected.
- A single point produces one block with no previous context.
- A point count that is an exact multiple of the block size produces no empty
  trailing block.
- `max_points_per_block == 1` is supported; `kLinear` falls back to `kPrevious`.
- Duplicate positions are ordered deterministically by original point index.
- Zero, negative, NaN, and infinite thresholds follow the validation rules
  above; negative and non-finite values are rejected.
- Non-finite positions or attribute values are rejected.
- Truncated payloads, trailing bytes, duplicate point indices, missing points,
  invalid predictors, and out-of-range block indices return an error.

## Measured error data

The following output was produced by a 37-point float test cloud with two
attribute components, seven points per block, and a threshold of `0.25`.
`Decode()` and per-block `DecodeBlock()` were item-wise identical; the maximum
reconstruction error was `0.125`, exactly half the threshold.

| Block | Predictor | Points | Max residual | Mean abs residual | RMS residual | Max recon error | Boundary residual |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | linear | 7 | 31.125000 | 4.562500 | 10.167075 | 0.125000 | 0.125000 |
| 1 | previous | 7 | 25.625000 | 6.848214 | 9.244146 | 0.125000 | 5.375000 |
| 2 | previous | 7 | 20.375000 | 6.133929 | 7.708341 | 0.125000 | 5.125000 |
| 3 | previous | 7 | 20.375000 | 7.919643 | 9.960357 | 0.125000 | 5.125000 |
| 4 | previous | 7 | 20.375000 | 6.151786 | 7.720494 | 0.125000 | 5.375000 |
| 5 | linear | 2 | 0.750000 | 0.312500 | 0.423896 | 0.125000 | 0.125000 |

For a perfectly linear 16-point test cloud with four points per block and zero
threshold, all four blocks selected `kLinear`, every reconstruction error was
zero, and the seam residual of blocks 1-3 was exactly zero.
