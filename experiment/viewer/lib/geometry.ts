import type {
  CoordinateMode,
  PointCloudAsset,
} from "./manifest";

export type RasterScope = "full" | "crop";

export function rasterIndices(
  width: number,
  height: number,
  crop: [number, number, number, number],
  scope: RasterScope,
): Uint32Array {
  if (scope === "full") {
    return Uint32Array.from({ length: width * height }, (_, index) => index);
  }
  const [x0, y0, x1, y1] = crop;
  const indices = new Uint32Array((x1 - x0) * (y1 - y0));
  let offset = 0;
  for (let row = y0; row < y1; row += 1) {
    for (let column = x0; column < x1; column += 1) {
      indices[offset] = row * width + column;
      offset += 1;
    }
  }
  return indices;
}

export function transformPoint(
  x: number,
  y: number,
  z: number,
  asset: PointCloudAsset,
  coordinateMode: CoordinateMode,
): [number, number, number] {
  const scale = coordinateMode === "aligned" ? asset.alignment.scale : 1;
  const zShift =
    coordinateMode === "aligned" ? asset.alignment.zShift : 0;
  return [scale * x, -scale * y, -(scale * z + zShift)];
}

export function pointPositions(
  raw: Float32Array,
  indices: Uint32Array,
  asset: PointCloudAsset,
  coordinateMode: CoordinateMode,
): Float32Array {
  const output = new Float32Array(indices.length * 3);
  indices.forEach((sourceIndex, outputIndex) => {
    const [x, y, z] = transformPoint(
      raw[3 * sourceIndex],
      raw[3 * sourceIndex + 1],
      raw[3 * sourceIndex + 2],
      asset,
      coordinateMode,
    );
    output[3 * outputIndex] = x;
    output[3 * outputIndex + 1] = y;
    output[3 * outputIndex + 2] = z;
  });
  return output;
}

export function finiteRasterIndices(
  raw: Float32Array,
  indices: Uint32Array,
): Uint32Array {
  return indices.filter((index) => {
    const offset = 3 * index;
    return (
      Number.isFinite(raw[offset]) &&
      Number.isFinite(raw[offset + 1]) &&
      Number.isFinite(raw[offset + 2]) &&
      raw[offset + 2] > 0
    );
  });
}

export function pointColors(
  raw: ArrayLike<number>,
  indices: Uint32Array,
  scale = 1,
): Float32Array {
  const output = new Float32Array(indices.length * 3);
  indices.forEach((sourceIndex, outputIndex) => {
    output[3 * outputIndex] = raw[3 * sourceIndex] * scale;
    output[3 * outputIndex + 1] = raw[3 * sourceIndex + 1] * scale;
    output[3 * outputIndex + 2] = raw[3 * sourceIndex + 2] * scale;
  });
  return output;
}

export function finitePositions(values: Float32Array): boolean {
  for (const value of values) {
    if (!Number.isFinite(value)) return false;
  }
  return true;
}
