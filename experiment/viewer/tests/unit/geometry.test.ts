import { describe, expect, it } from "vitest";

import {
  finiteRasterIndices,
  pointPositions,
  rasterIndices,
  transformPoint,
} from "@/lib/geometry";
import type { PointCloudAsset } from "@/lib/manifest";

const asset: PointCloudAsset = {
  url: "/data/exp1/sample.ply",
  pointCount: 4,
  bytes: 10,
  sha256: "a",
  checkpointSha256: "b",
  alignment: { scale: 1, zShift: 0 },
  bounds: { min: [0, 0, 1], max: [1, 1, 2] },
  metrics: {
    full: { pixels: 4, point_rel: 0, depth_rel: 0, "depth_delta_1.01": 1, "depth_delta_1.25": 1, boundary_f1: 1 },
    structure: { pixels: 4, point_rel: 0, depth_rel: 0, "depth_delta_1.01": 1, "depth_delta_1.25": 1, boundary_f1: 1 },
  },
};

describe("InfiniDepth point-cloud geometry", () => {
  it("uses row-major full-image and crop order", () => {
    expect(Array.from(rasterIndices(3, 2, [1, 0, 3, 2], "full"))).toEqual([0, 1, 2, 3, 4, 5]);
    expect(Array.from(rasterIndices(3, 2, [1, 0, 3, 2], "crop"))).toEqual([1, 2, 4, 5]);
  });

  it("uses only the Three.js display-axis conversion", () => {
    expect(transformPoint(1, 2, 3, asset, "raw")).toEqual([1, -2, -3]);
    expect(Array.from(pointPositions(new Float32Array([1, 2, 3]), new Uint32Array([0]), asset, "raw"))).toEqual([1, -2, -3]);
  });

  it("drops non-finite and non-positive-Z points without reordering", () => {
    const raw = new Float32Array([0, 0, 1, Number.NaN, 0, 2, 1, 0, 2, 0, 0, 0]);
    expect(Array.from(finiteRasterIndices(raw, new Uint32Array([0, 1, 2, 3])))).toEqual([0, 2]);
  });
});
