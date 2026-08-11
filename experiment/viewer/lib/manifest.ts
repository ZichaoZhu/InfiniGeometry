export const REFINEMENT_STEPS = [0, 1, 3, 5] as const;
export type RefinementStep = (typeof REFINEMENT_STEPS)[number];
export type StageName = string;
export type CoordinateMode = "aligned" | "raw";
export type MetricScope = "full" | "structure";

export type ScopeMetrics = {
  pixels: number;
  point_rel: number;
  depth_rel: number;
  "depth_delta_1.01": number;
  "depth_delta_1.25": number;
  boundary_f1: number;
};

export type PointCloudMetrics = Record<MetricScope, ScopeMetrics>;

export type PointCloudAsset = {
  url: string;
  pointCount: number;
  validPointCount?: number;
  bytes: number;
  sha256: string;
  checkpointSha256: string;
  alignment: { scale: number; zShift: number };
  bounds: { min: [number, number, number]; max: [number, number, number] };
  metrics: PointCloudMetrics;
  pointRelReductionFromK0?: number;
};

export type AssetAlias = { alias: "initial.k0" };
export type StageAssets = Record<`k${RefinementStep}`, PointCloudAsset | AssetAlias>;

export type PointCloudSample = {
  id: string;
  label?: string;
  order?: number;
  description: string;
  cropXYXY: [number, number, number, number];
  disparityQuantiles: [number, number];
  rgbUrl: string;
  websiteEnabled?: boolean;
  groundTruth: PointCloudAsset;
  stages: Record<StageName, StageAssets>;
};

export type PointCloudManifest = {
  version: 1;
  experiment: string;
  displayNote: string;
  resolution: { width: number; height: number };
  steps: RefinementStep[];
  stages: StageName[];
  defaultStages?: { left: StageName; right: StageName };
  websiteSampleOrder?: string[];
  coordinateSpace?: { stored: string; aligned: string; threeDisplay: string };
  voxelization: { depthScale: number; spconvOrder: string[]; depthCoordinate: string };
  samples: PointCloudSample[];
};

export type ExperimentCatalogEntry = {
  id: string;
  label: string;
  shortLabel?: string;
  manifestUrl: string;
  summary: string;
  stageLabels?: Partial<Record<StageName, string>>;
  stageDetails?: Partial<Record<StageName, string>>;
};

export type ExperimentCatalog = {
  version: 1;
  defaultExperiment: string;
  experiments: ExperimentCatalogEntry[];
};

export function isAlias(value: PointCloudAsset | AssetAlias): value is AssetAlias {
  return "alias" in value;
}

export function sampleStages(sample: PointCloudSample): StageName[] {
  return Object.keys(sample.stages);
}

export function resolveAsset(
  sample: PointCloudSample,
  stage: StageName,
  step: RefinementStep,
): PointCloudAsset {
  const selected = sample.stages[stage]?.[`k${step}`];
  if (!selected) throw new Error(`${sample.id} 缺少 ${stage} K=${step} 点云`);
  if (!isAlias(selected)) return selected;
  const initial = sample.stages.initial?.k0;
  if (!initial || isAlias(initial)) throw new Error("initial K=0 不能是资产别名");
  return initial;
}

export function stageUsesAlias(
  sample: PointCloudSample,
  stage: StageName,
  step: RefinementStep,
): boolean {
  const asset = sample.stages[stage]?.[`k${step}`];
  return asset ? isAlias(asset) : false;
}

export function defaultStage(
  manifest: PointCloudManifest,
  sample: PointCloudSample,
  pane: "left" | "right",
): StageName {
  const stages = sampleStages(sample);
  const requested = manifest.defaultStages?.[pane];
  if (requested && stages.includes(requested)) return requested;
  const preferred = pane === "left" ? "initial" : stages.at(-1);
  if (preferred && stages.includes(preferred)) return preferred;
  if (!stages.length) throw new Error(`${sample.id} 没有可显示阶段`);
  return stages[0];
}

function validateMetrics(metrics: PointCloudMetrics, sampleId: string): void {
  for (const scope of ["full", "structure"] as const) {
    const value = metrics[scope];
    if (!value || !Number.isFinite(value.point_rel) || !Number.isFinite(value.depth_rel) || !Number.isFinite(value["depth_delta_1.01"]) || !Number.isFinite(value.boundary_f1)) {
      throw new Error(`${sampleId} 缺少 MoGe3 点云指标`);
    }
  }
}

export function validateManifest(manifest: PointCloudManifest): void {
  if (manifest.version !== 1 || !manifest.samples.length || !manifest.stages.length || !manifest.steps.length) {
    throw new Error("查看器 manifest 缺少样本、阶段或 K 登记");
  }
  if (manifest.resolution.width <= 0 || manifest.resolution.height <= 0) {
    throw new Error("查看器分辨率无效");
  }
  const expectedPoints = manifest.resolution.width * manifest.resolution.height;
  const ids = new Set<string>();
  for (const sample of manifest.samples) {
    if (ids.has(sample.id)) throw new Error(`样本重复: ${sample.id}`);
    ids.add(sample.id);
    const assets = [sample.groundTruth];
    for (const stage of sampleStages(sample)) {
      for (const step of manifest.steps) assets.push(resolveAsset(sample, stage, step));
    }
    for (const asset of assets) {
      if (asset.pointCount !== expectedPoints) throw new Error(`${sample.id} 点数与固定网格不一致`);
      if (!asset.url.match(/^\/data\/exp[0-9]+\//)) throw new Error(`${sample.id} 资产 URL 越界`);
      validateMetrics(asset.metrics, sample.id);
    }
  }
}

export function validateCatalog(catalog: ExperimentCatalog): void {
  if (catalog.version !== 1 || !catalog.experiments.length) throw new Error("实验目录为空或版本无效");
  const ids = new Set(catalog.experiments.map((entry) => entry.id));
  if (ids.size !== catalog.experiments.length || !ids.has(catalog.defaultExperiment)) {
    throw new Error("实验目录存在重复项或默认实验无效");
  }
  for (const entry of catalog.experiments) {
    if (!entry.manifestUrl.match(/^\/data\/exp[0-9]+\/manifest\.json$/)) {
      throw new Error(`实验 manifest URL 无效: ${entry.manifestUrl}`);
    }
  }
}
