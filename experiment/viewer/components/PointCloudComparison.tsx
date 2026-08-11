"use client";

import { useEffect, useMemo, useState, type ReactNode } from "react";

import { CloudScene, type CameraSnapshot, type InteractionMode } from "@/components/CloudScene";
import type { RasterScope } from "@/lib/geometry";
import { dataUrl } from "@/lib/dataUrl";
import {
  defaultStage,
  REFINEMENT_STEPS,
  resolveAsset,
  sampleStages,
  stageUsesAlias,
  validateCatalog,
  validateManifest,
  type ExperimentCatalog,
  type ExperimentCatalogEntry,
  type MetricScope,
  type PointCloudAsset,
  type PointCloudManifest,
  type PointCloudSample,
  type RefinementStep,
  type StageName,
} from "@/lib/manifest";

type PaneState = {
  stage: StageName;
  step: RefinementStep;
  fitNonce: number;
  interaction: InteractionMode;
};

const FALLBACK_STAGE_LABELS: Record<string, string> = {
  initial: "官方初始",
  stage1_best: "Detach 最佳",
  joint_best: "联合最佳",
};

function percent(value: number, digits = 2): string {
  return `${(100 * value).toFixed(digits)}%`;
}

function stageLabel(experiment: ExperimentCatalogEntry, stage: StageName): string {
  return experiment.stageLabels?.[stage] ?? FALLBACK_STAGE_LABELS[stage] ?? stage;
}

function defaultStep(manifest: PointCloudManifest, preferred: RefinementStep): RefinementStep {
  return manifest.steps.includes(preferred) ? preferred : manifest.steps[0];
}

function Segment<T extends string | number>({
  label,
  value,
  values,
  onChange,
  format = String,
  testId,
}: {
  label: string;
  value: T;
  values: readonly T[];
  onChange: (value: T) => void;
  format?: (value: T) => string;
  testId: string;
}) {
  return (
    <div className="segment-wrap" data-testid={testId}>
      <span className="segment-label">{label}</span>
      <div className="segment">
        {values.map((item) => (
          <button type="button" key={item} className={item === value ? "active" : ""} aria-pressed={item === value} onClick={() => onChange(item)}>
            {format(item)}
          </button>
        ))}
      </div>
    </div>
  );
}

function AssetMetrics({ asset, scope }: { asset: PointCloudAsset; scope: MetricScope }) {
  const metrics = asset.metrics[scope];
  return (
    <dl className="metrics-grid">
      <div><dt>点图 Rel</dt><dd>{percent(metrics.point_rel, 3)}</dd></div>
      <div><dt>深度 Rel</dt><dd>{percent(metrics.depth_rel, 3)}</dd></div>
      <div><dt>δ1.01</dt><dd>{percent(metrics["depth_delta_1.01"])}</dd></div>
      <div><dt>边界 F1</dt><dd>{percent(metrics.boundary_f1)}</dd></div>
    </dl>
  );
}

function ScenePanel({
  id,
  kicker,
  title,
  detail,
  asset,
  sample,
  manifest,
  fitNonce,
  onFit,
  interaction,
  syncEnabled,
  cameraSnapshot,
  onCameraChange,
  scope,
  controls,
  children,
}: {
  id: string;
  kicker: string;
  title: string;
  detail: string;
  asset: PointCloudAsset;
  sample: PointCloudSample;
  manifest: PointCloudManifest;
  fitNonce: number;
  onFit: () => void;
  interaction: InteractionMode;
  syncEnabled: boolean;
  cameraSnapshot: CameraSnapshot | null;
  onCameraChange: (snapshot: CameraSnapshot) => void;
  scope: RasterScope;
  controls?: ReactNode;
  children?: ReactNode;
}) {
  const metricScope: MetricScope = scope === "full" ? "full" : "structure";
  return (
    <section className={`viewer-pane ${id === "ground-truth" ? "ground-truth-pane" : ""}`} data-testid={`viewer-${id}`}>
      <header className="pane-header">
        <div>
          <span className="pane-kicker">{kicker}</span>
          <h2>{title}</h2>
          <small className="stage-detail">{detail}</small>
        </div>
        <button className="fit-button" type="button" onClick={onFit}>适配视野</button>
      </header>
      {controls && <div className="pane-controls">{controls}</div>}
      {children}
      <div className="canvas-shell">
        <CloudScene
          panelId={id}
          asset={asset}
          manifest={manifest}
          sample={sample}
          coordinateMode="raw"
          rasterScope={scope}
          syncEnabled={syncEnabled}
          cameraSnapshot={cameraSnapshot}
          onCameraChange={onCameraChange}
          fitNonce={fitNonce}
          interactionMode={interaction}
        />
        <div className="canvas-hint">左键{interaction === "rotate" ? "旋转" : "平移"} · 中键拖动/滚轮缩放</div>
      </div>
      <AssetMetrics asset={asset} scope={metricScope} />
      <footer className="asset-meta">
        <span>{(asset.validPointCount ?? asset.pointCount).toLocaleString("zh-CN")} 有效点</span>
        <span>真实尺度 · 米</span>
        <span title={asset.checkpointSha256}>{id === "ground-truth" ? "GT" : "ckpt"} {asset.checkpointSha256.slice(0, 8)}</span>
      </footer>
    </section>
  );
}

function PredictionPanel({
  id,
  pane,
  setPane,
  sample,
  manifest,
  experiment,
  syncEnabled,
  cameraSnapshot,
  onCameraChange,
  scope,
}: {
  id: "left" | "right";
  pane: PaneState;
  setPane: (value: PaneState) => void;
  sample: PointCloudSample;
  manifest: PointCloudManifest;
  experiment: ExperimentCatalogEntry;
  syncEnabled: boolean;
  cameraSnapshot: CameraSnapshot | null;
  onCameraChange: (snapshot: CameraSnapshot) => void;
  scope: RasterScope;
}) {
  const asset = resolveAsset(sample, pane.stage, pane.step);
  const stages = sampleStages(sample);
  const label = stageLabel(experiment, pane.stage);
  return (
    <ScenePanel
      id={id}
      kicker={`窗口 ${id === "left" ? "B" : "C"} · 预测点云`}
      title={`${label} · K=${pane.step}`}
      detail={experiment.stageDetails?.[pane.stage] ?? "固定查询网格上的 disparity 稀疏三维 Refiner 输出"}
      asset={asset}
      sample={sample}
      manifest={manifest}
      fitNonce={pane.fitNonce}
      onFit={() => setPane({ ...pane, fitNonce: pane.fitNonce + 1 })}
      interaction={pane.interaction}
      syncEnabled={syncEnabled}
      cameraSnapshot={cameraSnapshot}
      onCameraChange={onCameraChange}
      scope={scope}
      controls={<>
        <Segment label="阶段" value={pane.stage} values={stages} format={(stage) => stageLabel(experiment, stage)} onChange={(stage) => setPane({ ...pane, stage })} testId={`${id}-stage`} />
        <Segment label="精修" value={pane.step} values={manifest.steps} format={(step) => `K=${step}`} onChange={(step) => setPane({ ...pane, step })} testId={`${id}-k`} />
        <Segment label="鼠标左键" value={pane.interaction} values={["rotate", "pan"] as const} format={(value) => value === "rotate" ? "旋转" : "平移"} onChange={(interaction) => setPane({ ...pane, interaction })} testId={`${id}-interaction`} />
      </>}
    >
      {stageUsesAlias(sample, pane.stage, pane.step) && <div className="identity-note">资源别名：{label} K={pane.step} 与 K=0 完全一致。</div>}
      {asset.pointRelReductionFromK0 !== undefined && <div className={asset.pointRelReductionFromK0 >= 0 ? "gain-note improved" : "gain-note degraded"}>较 K=0 {asset.pointRelReductionFromK0 >= 0 ? "改善" : "退化"} {percent(Math.abs(asset.pointRelReductionFromK0))}</div>}
    </ScenePanel>
  );
}

function orderedSamples(manifest: PointCloudManifest): PointCloudSample[] {
  const samples = manifest.samples.filter((sample) => sample.websiteEnabled !== false);
  const lookup = new Map(samples.map((sample) => [sample.id, sample]));
  return (manifest.websiteSampleOrder ?? samples.map((sample) => sample.id)).map((id) => lookup.get(id)).filter((sample): sample is PointCloudSample => Boolean(sample));
}

function LoadingPage({ error }: { error?: string }) {
  return <main className={`boot-screen ${error ? "error-screen" : ""}`} role={error ? "alert" : undefined}><div className="boot-mark">ID</div><p>{error ?? "正在读取点云清单…"}</p></main>;
}

export function PointCloudComparison() {
  const [catalog, setCatalog] = useState<ExperimentCatalog | null>(null);
  const [experimentId, setExperimentId] = useState("");
  const [manifest, setManifest] = useState<PointCloudManifest | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sampleId, setSampleId] = useState("");
  const [scope, setScope] = useState<RasterScope>("crop");
  const [syncEnabled, setSyncEnabled] = useState(true);
  const [cameraSnapshot, setCameraSnapshot] = useState<CameraSnapshot | null>(null);
  const [groundFit, setGroundFit] = useState(0);
  const [groundInteraction, setGroundInteraction] = useState<InteractionMode>("rotate");
  const [left, setLeft] = useState<PaneState>({ stage: "initial", step: 0, fitNonce: 0, interaction: "rotate" });
  const [right, setRight] = useState<PaneState>({ stage: "joint_best", step: 3, fitNonce: 0, interaction: "rotate" });

  useEffect(() => {
    const controller = new AbortController();
    fetch(dataUrl("/data/experiments.json"), { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error(`实验目录请求失败：HTTP ${response.status}`);
        const value = (await response.json()) as ExperimentCatalog;
        validateCatalog(value);
        setCatalog(value);
        setExperimentId(value.defaultExperiment);
      })
      .catch((reason: unknown) => {
        if (!(reason instanceof DOMException && reason.name === "AbortError")) setError(reason instanceof Error ? reason.message : String(reason));
      });
    return () => controller.abort();
  }, []);

  const experiment = catalog?.experiments.find((entry) => entry.id === experimentId);
  useEffect(() => {
    if (!experiment) return;
    const controller = new AbortController();
    setManifest(null);
    setError(null);
    fetch(dataUrl(experiment.manifestUrl), { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error(`${experiment.label} 清单请求失败：HTTP ${response.status}`);
        const loaded = (await response.json()) as PointCloudManifest;
        if (loaded.experiment !== experiment.id) throw new Error(`实验清单不匹配：${loaded.experiment}`);
        validateManifest(loaded);
        const selected = orderedSamples(loaded)[0];
        if (!selected) throw new Error(`${experiment.label} 没有可显示样本`);
        setManifest(loaded);
        setSampleId(selected.id);
        setCameraSnapshot(null);
        setGroundFit((value) => value + 1);
        setLeft((pane) => ({ ...pane, stage: defaultStage(loaded, selected, "left"), step: defaultStep(loaded, 0), fitNonce: pane.fitNonce + 1 }));
        setRight((pane) => ({ ...pane, stage: defaultStage(loaded, selected, "right"), step: defaultStep(loaded, 3), fitNonce: pane.fitNonce + 1 }));
      })
      .catch((reason: unknown) => {
        if (!(reason instanceof DOMException && reason.name === "AbortError")) setError(reason instanceof Error ? reason.message : String(reason));
      });
    return () => controller.abort();
  }, [experiment]);

  const samples = useMemo(() => manifest ? orderedSamples(manifest) : [], [manifest]);
  const sample = samples.find((item) => item.id === sampleId) ?? samples[0];
  if (error) return <LoadingPage error={error} />;
  if (!catalog || !manifest || !experiment || !sample) return <LoadingPage />;

  const resetView = (nextSample: PointCloudSample) => {
    setSampleId(nextSample.id);
    setCameraSnapshot(null);
    setGroundFit((value) => value + 1);
    setLeft((value) => ({ ...value, stage: defaultStage(manifest, nextSample, "left"), fitNonce: value.fitNonce + 1 }));
    setRight((value) => ({ ...value, stage: defaultStage(manifest, nextSample, "right"), fitNonce: value.fitNonce + 1 }));
  };

  return (
    <main className="app-shell">
      <header className="hero">
        <div><span className="eyebrow">INFINIDEPTH / INTERACTIVE GEOMETRY LAB</span><h1>Disparity Refiner<br />多实验点云对比器</h1><p>在统一相机与渲染设置下，对照 GT、训练阶段和 K 次精修的稠密点云。全部指标遵循 MoGe3 3D 点云评测口径。</p></div>
        <div className="hero-badge"><span>{experiment.shortLabel ?? experiment.label} · 当前样本</span><strong>{sample.label ?? `样本 ${(samples.indexOf(sample) + 1).toString().padStart(2, "0")}`}</strong><small>{sample.description}</small></div>
      </header>

      <nav className="experiment-switcher" aria-label="实验切换"><div className="switcher-heading"><span>实验入口</span><strong>{experiment.label}</strong></div>{catalog.experiments.map((entry) => <button type="button" key={entry.id} className={entry.id === experiment.id ? "active" : ""} aria-pressed={entry.id === experiment.id} data-testid={`experiment-${entry.id}`} onClick={() => { if (entry.id !== experiment.id) setExperimentId(entry.id); }}><strong>{entry.shortLabel ?? entry.label}</strong><small>{entry.summary}</small></button>)}</nav>
      <div className="experiment-summary"><strong>{experiment.label}</strong><span>{manifest.displayNote}</span></div>

      <nav className="sample-switcher" aria-label="图片切换">{samples.map((item, index) => <button type="button" key={item.id} className={item.id === sample.id ? "active" : ""} aria-pressed={item.id === sample.id} data-testid={`sample-${item.order ?? index + 1}`} onClick={() => resetView(item)}><span className="sample-thumbnail"><img src={dataUrl(item.rgbUrl)} alt="" /><span className="sample-hover-preview" aria-hidden="true"><img src={dataUrl(item.rgbUrl)} alt="" /><span>{item.description}</span></span></span><span><strong>{item.label ?? `样本 ${(index + 1).toString().padStart(2, "0")}`}</strong><small className="sample-id">{item.id}</small><small>{item.description}</small></span></button>)}</nav>

      <section className="global-toolbar"><Segment label="范围" value={scope} values={["crop", "full"] as const} format={(value) => value === "crop" ? "细结构裁剪" : "完整场景"} onChange={(value) => { setScope(value); setCameraSnapshot(null); }} testId="raster-scope" /><label className={`sync-toggle ${syncEnabled ? "active" : ""}`}><input type="checkbox" checked={syncEnabled} onChange={(event) => setSyncEnabled(event.target.checked)} /><span className="toggle-track" />相机同步</label></section>

      <div className="comparison-grid">
        <ScenePanel id="ground-truth" kicker="窗口 A · 真实点云" title="Hypersim Ground Truth" detail="由真实深度与相机内参反投影，不经过 Base 或 Refiner" asset={sample.groundTruth} sample={sample} manifest={manifest} fitNonce={groundFit} onFit={() => setGroundFit((value) => value + 1)} interaction={groundInteraction} syncEnabled={syncEnabled} cameraSnapshot={cameraSnapshot} onCameraChange={setCameraSnapshot} scope={scope} controls={<Segment label="鼠标左键" value={groundInteraction} values={["rotate", "pan"] as const} format={(value) => value === "rotate" ? "旋转" : "平移"} onChange={setGroundInteraction} testId="ground-truth-interaction" />} />
        <PredictionPanel id="left" pane={left} setPane={setLeft} sample={sample} manifest={manifest} experiment={experiment} syncEnabled={syncEnabled} cameraSnapshot={cameraSnapshot} onCameraChange={setCameraSnapshot} scope={scope} />
        <PredictionPanel id="right" pane={right} setPane={setRight} sample={sample} manifest={manifest} experiment={experiment} syncEnabled={syncEnabled} cameraSnapshot={cameraSnapshot} onCameraChange={setCameraSnapshot} scope={scope} />
      </div>
      <footer className="page-footer"><div><strong>指标口径</strong><span>Point Rel、Depth Rel、δ1.01 与 depth-boundary F1 均按 MoGe3 的点云评测定义导出。</span></div><div><strong>归档范围</strong><span>{experiment.label} 已开放 {samples.length} 张样本；后续 ExpN 只需加入 `experiments.json` 即可切换。</span></div></footer>
    </main>
  );
}
