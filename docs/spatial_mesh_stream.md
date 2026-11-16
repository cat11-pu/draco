# Spatial Mesh Streaming

`draco/mesh/spatial_mesh_stream.h` provides a narrow streaming interface for
large triangular meshes:

```cpp
SpatialMeshStreamOptions options;
options.chunk_size = 2.0f;

EncoderBuffer encoded;
SpatialMeshConsistency consistency;
SpatialMeshStreamEncoder::Encode(mesh, options, &encoded, &consistency);

DecoderBuffer input;
input.Init(encoded.data(), encoded.size());
auto stream = SpatialMeshStream::Open(&input).value();

SpatialMeshFrame frame;  // Reuse this object every frame.
stream->DecodeNear(camera_position, view_radius, &frame);
for (const SpatialMeshChunkView& chunk : frame.visible_chunks()) {
  Draw(chunk.positions, chunk.point_ids, chunk.indices, chunk.attributes);
}
```

## Container

The container stores metadata plus one independently decodable Draco mesh
bitstream per spatial chunk. `Open()` reads the container but decompresses
nothing. Chunks are selected by the distance from the camera to their AABB.

Each chunk stream uses sequential mesh encoding and a mesh-wide explicit
quantization origin/range. This gives identical transforms to every chunk and
therefore bit-identical values for each source point shared by chunks. The
container also stores the source point id and source face id for every chunk
element so callers can stitch chunks without spatial hashing.

## Invariants

- A face belongs to exactly one chunk; assignment is by the face centroid.
- A chunk contains exactly the source points referenced by its faces.
- Boundary vertices are keyed by source point id, not by position.
- Every attribute is copied, including its type, components, data type,
  normalization, unique id, and name.
- Isolated source points are not emitted because the stream represents faces.
- `SpatialMeshChunkView` pointers remain valid until the next `DecodeNear()`,
  `Clear()`, `Release()`, or destruction of the frame.

## Buffer Reuse

`SpatialMeshFrame` owns an LRU cache of decoded chunks. A cache hit does not
decompress the chunk or resize its buffers. An evicted slot keeps its vector
capacity and is reused by the next decoded chunk. The hot-path scratch vectors
for selection and visibility are also retained by the frame.

## Verification

`SpatialMeshStreamEncoder::Encode()` decodes every chunk once by default and
compares the raw attribute bytes for all repeated source points. It returns an
error if any boundary value differs. `SpatialMeshStream::VerifyBoundaryVertices()`
exposes the same check at runtime.

The tests cover:

- invalid chunk size, speed, empty mesh, and truncated containers;
- a single-triangle one-chunk mesh;
- negative coordinates and vertices exactly on chunk boundaries;
- AABB touch, epsilon outside, empty camera queries, and cache overflow;
- full decode versus a normal whole-mesh Draco decode, including every position,
  UV, normal, and face-index item;
- 100,000 alternating viewpoints with zero heap growth and zero evictions after
  warm-up.

Run the focused tests with:

```sh
./build/draco_tests --gtest_filter='SpatialMeshStreamTest.*'
```
