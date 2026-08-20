import { describe, expect, it } from "vitest";

import {
  resolveAsset,
  sampleSplits,
  validateCatalog,
  validateManifest,
  type PointCloudAsset,
  type PointCloudManifest,
  type PointCloudSample,
} from "@/lib/manifest";

const asset: PointCloudAsset = {
  url: "/data/exp1/sample/k0.ply",
  pointCount: 512 * 384,
  validPointCount: 190000,
  bytes: 100,
  sha256: "asset",
  checkpointSha256: "checkpoint",
  alignment: { scale: 1, zShift: 0 },
  bounds: { min: [0, 0, 1], max: [1, 1, 2] },
  metrics: {
    full: { pixels: 190000, point_rel: 0.1, depth_rel: 0.1, "depth_delta_1.01": 0.8, "depth_delta_1.25": 0.95, boundary_f1: 0.7 },
    structure: { pixels: 5000, point_rel: 0.2, depth_rel: 0.2, "depth_delta_1.01": 0.6, "depth_delta_1.25": 0.9, boundary_f1: 0.5 },
  },
};

function sample(index: number): PointCloudSample {
  return {
    id: `sample-${index}`,
    description: "细结构",
    cropXYXY: [0, 0, 192, 192],
    disparityQuantiles: [0.1, 1.0],
    rgbUrl: `/data/exp1/sample-${index}/source_rgb.jpg`,
    groundTruth: { ...asset, url: `/data/exp1/sample-${index}/gt.ply` },
    stages: {
      initial: {
        k0: asset,
        k1: { alias: "initial.k0" },
        k3: { alias: "initial.k0" },
        k5: { alias: "initial.k0" },
      },
      stage1_best: { k0: asset, k1: asset, k3: asset, k5: asset },
      joint_best: { k0: asset, k1: asset, k3: asset, k5: asset },
    },
  };
}

function manifest(): PointCloudManifest {
  return {
    version: 1,
    experiment: "exp1_infinidepth_disparity_ssr_single_image_overfit",
    displayNote: "GT 统计反归一化",
    resolution: { width: 512, height: 384 },
    steps: [0, 1, 3, 5],
    stages: ["initial", "stage1_best", "joint_best"],
    voxelization: {
      depthScale: 200,
      spconvOrder: ["batch", "disparity_bin", "row", "column"],
      depthCoordinate: "round(200 * disparity)",
    },
    samples: [1, 2, 3, 4, 5].map(sample),
  };
}

describe("Exp1 viewer manifest", () => {
  it("validates the five fixed samples", () => {
    expect(() => validateManifest(manifest())).not.toThrow();
  });

  it("resolves zero-initialized iteration aliases to the exact K0 asset", () => {
    const selected = manifest().samples[0];
    expect(resolveAsset(selected, "initial", 5)).toBe(selected.stages.initial.k0);
  });

  it("rejects a wrong fixed-grid point count", () => {
    const value = manifest();
    value.samples[0].groundTruth.pointCount -= 1;
    expect(() => validateManifest(value)).toThrow(/点数/);
  });

  it("rejects assets outside the Exp1 data root", () => {
    const value = manifest();
    value.samples[0].groundTruth.url = "/data/other/gt.ply";
    expect(() => validateManifest(value)).toThrow(/URL/);
  });

  it("accepts ExpN catalog entries without viewer code changes", () => {
    expect(() => validateCatalog({
      version: 1,
      defaultExperiment: "exp12",
      experiments: [{ id: "exp12", label: "Exp12", manifestUrl: "/data/exp12_revision2/manifest.json", summary: "next" }],
    })).not.toThrow();
  });

  it("validates and orders train, val, and test samples", () => {
    const value = manifest();
    value.samples = ["train", "val", "test"].flatMap((split, splitIndex) =>
      [1, 2, 3, 4, 5].map((index) => ({
        ...sample(splitIndex * 5 + index),
        split: split as "train" | "val" | "test",
      })),
    );
    value.samplePolicy = {
      algorithm: "python-random-sample",
      countPerSplit: 5,
      seed: 173,
      splitOrder: ["train", "val", "test"],
    };
    expect(sampleSplits(value)).toEqual(["train", "val", "test"]);
    expect(() => validateManifest(value)).not.toThrow();
  });
});
