import { expect, test } from "@playwright/test";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";

const expRoot = path.resolve(process.cwd(), "../../../MoGe-v3-reproduction/experiment/exp6_5_moge2_ssr_vs_official_moge3");

async function colors(page: import("@playwright/test").Page, panel: string) {
  return page.getByTestId(`viewer-${panel}`).locator("canvas").evaluate((canvas) => {
    const gl = (canvas as HTMLCanvasElement).getContext("webgl2");
    if (!gl) return 0;
    const pixels = new Uint8Array(gl.drawingBufferWidth * gl.drawingBufferHeight * 4);
    gl.readPixels(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    const values = new Set<string>();
    for (let i = 0; i < pixels.length; i += 400) values.add(`${pixels[i]},${pixels[i + 1]},${pixels[i + 2]}`);
    return values.size;
  });
}

test("Exp6-5 four panes, fixed methods, best/final and K controls", async ({ page }) => {
  test.setTimeout(150_000);
  const entries = JSON.parse(await readFile(path.join(expRoot, "viewer_entries.json"), "utf8"));
  // Before CUDA export, exercise UI with explicitly mocked, real historical PLYs.
  // EXP65_REAL_ASSETS=1 instead validates the actual published manifests and clouds.
  if (process.env.EXP65_REAL_ASSETS !== "1") {
    const catalog = JSON.parse(await readFile("public/data/experiments.json", "utf8"));
    catalog.experiments = catalog.experiments.filter((e: { id: string }) => !e.id.startsWith("exp6_5_"));
    catalog.experiments.push(...entries);
    await page.route("**/data/experiments.json", (route) => route.fulfill({ json: catalog }));
    const source = JSON.parse(await readFile("public/data/exp6_4_official_ssr_comparison/manifest.json", "utf8"));
    for (const entry of entries) {
      const fixture = structuredClone(source);
      fixture.experiment = entry.id;
      fixture.stages = ["initial", "self_best", "self_final", "official_best", "official_final"];
      fixture.defaultStages = { left: "initial", right: "self_best" };
      fixture.metricsNote = "仅可视化 · UI 测试使用历史资产 fixture";
      fixture.samples = fixture.samples.map((sample: { stages: Record<string, Record<string, object>> }) => {
        const originals = sample.stages;
        const template = Object.values(originals).find((s) => !("alias" in s.k3))!;
        sample.stages = Object.fromEntries(fixture.stages.map((stage: string) => [stage,
          Object.fromEntries(Object.entries(template).map(([k, asset]) => {
            const clean = { ...asset } as Record<string, unknown>;
            delete clean.metrics;
            delete clean.displayMetrics;
            delete clean.pointRelReductionFromK0;
            return [k, clean];
          })),
        ]));
        return sample;
      });
      await page.route(`**${entry.manifestUrl}`, (route) => route.fulfill({ json: fixture }));
    }
  }
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  await page.getByTestId("experiment-exp6_5_hypersim").click();
  await expect(page.locator("canvas")).toHaveCount(4);
  await expect(page.getByTestId("viewer-left")).toContainText("MoGe2 原始基座 · K=0");
  await expect(page.getByTestId("left-k").getByRole("button")).toHaveCount(1);
  await expect(page.getByTestId("viewer-right")).toContainText("自实现 SSR · 最佳 7,500 · K=3");
  await expect(page.getByTestId("viewer-reference")).toContainText("官方 SSR · 最佳 25,000 · K=3");
  await expect(page.getByTestId("right-stage").getByRole("button")).toHaveCount(2);
  await expect(page.getByTestId("reference-stage").getByRole("button")).toHaveCount(2);
  for (const panel of ["ground-truth", "left", "right", "reference"]) {
    await page.getByTestId(`viewer-${panel}`).scrollIntoViewIfNeeded();
    await expect.poll(() => colors(page, panel), { timeout: 30_000 }).toBeGreaterThan(2);
  }
  await page.getByTestId("right-stage").getByRole("button", { name: "自实现 SSR · 最终 30,000" }).click();
  await page.getByTestId("right-k").getByRole("button", { name: "K=0" }).click();
  await page.getByTestId("reference-k").getByRole("button", { name: "K=5" }).click();
  await page.getByTestId("sample-split").getByRole("button", { name: "VAL" }).click();
  await expect(page.getByTestId("viewer-right")).toContainText("自实现 SSR · 最终 30,000 · K=0");
  await expect(page.getByTestId("viewer-reference")).toContainText("官方 SSR · 最佳 25,000 · K=5");
  await expect(page.locator(".sample-switcher button")).toHaveCount(5);
  await page.getByTestId("sample-split").getByRole("button", { name: "TEST" }).click();
  await expect(page.locator(".sample-switcher button")).toHaveCount(5);
  await page.getByTestId("experiment-exp6_5_ood").click();
  await expect(page.getByTestId("viewer-right")).toContainText("自实现 SSR · 最佳 7,500 · K=3");
  await expect(page.getByTestId("viewer-reference")).toContainText("官方 SSR · 最佳 25,000 · K=3");
  if (process.env.EXP65_REAL_ASSETS === "1") {
    await expect(page.locator(".sample-switcher button")).toHaveCount(20);
    await expect(page.getByTestId("sample-split")).toHaveCount(0);
    await page.getByTestId("sample-20").click();
    await expect(page.getByTestId("sample-20")).toContainText("Sintel/market_6/frame_0035");
  }
  await page.locator(".sample-switcher button").first().hover();
  await expect(page.locator(".sample-hover-preview").first()).toBeVisible();
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.locator("canvas")).toHaveCount(4);
  await expect(page.locator(".canvas-error")).toHaveCount(0);
  expect(errors).toEqual([]);
});

test("Exp6-5 all 35 real samples render without accumulating old clouds", async ({ page }) => {
  test.skip(process.env.EXP65_REAL_ASSETS !== "1", "Requires the actual completed CUDA export");
  test.setTimeout(360_000);
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const directory = test.info().outputPath("acceptance");
  await mkdir(directory, { recursive: true });
  const prefix = "acceptance";
  await page.goto("/");
  await page.setViewportSize({ width: 1440, height: 1100 });
  const session = await page.context().newCDPSession(page);
  const memory: { sample: number; usedBytes: number }[] = [];
  let seen = 0;
  for (const [cohort, count] of [["hypersim", 15], ["ood", 20]] as const) {
    await page.getByTestId(`experiment-exp6_5_${cohort}`).click();
    await expect(page.getByTestId("viewer-right")).toContainText("自实现 SSR · 最佳 7,500 · K=3");
    for (let order = 1; order <= count; order++) {
      if (cohort === "hypersim" && [6, 11].includes(order)) {
        await page.getByTestId("sample-split").getByRole("button", { name: order === 6 ? "VAL" : "TEST" }).click();
      }
      await page.getByTestId(`sample-${order}`).click();
      for (const [panel, filename] of [["ground-truth", "gt"], ["left", "initial_k0"], ["right", "self_best_k3"], ["reference", "official_best_k3"]]) {
        await page.getByTestId(`viewer-${panel}`).scrollIntoViewIfNeeded();
        await expect(page.getByTestId(`viewer-${panel}`).locator("canvas")).toHaveAttribute("data-scene-source",
          new RegExp(`exp6_5_${cohort}_20260921_r1/${String(order).padStart(2, "0")}/${filename}\\.ply`));
        await expect.poll(() => colors(page, panel), { timeout: 30_000 }).toBeGreaterThan(2);
      }
      seen++;
      await expect(page.locator(".canvas-error")).toHaveCount(0);
      await expect(page.locator(".metrics-grid")).toHaveCount(0);
      if ([1, 15, 35].includes(seen)) {
        await session.send("HeapProfiler.collectGarbage");
        const heap = await session.send("Runtime.getHeapUsage");
        memory.push({ sample: seen, usedBytes: heap.usedSize + (heap.backingStorageSize ?? 0) });
      }
      if (order === 1) {
        await page.mouse.move(0, 0);
        await page.locator(".comparison-grid").screenshot({ path: path.join(directory, `${prefix}-${cohort}-clouds.png`) });
      }
    }
  }
  expect(seen).toBe(35);
  expect(errors).toEqual([]);
  expect(memory.at(-1)!.usedBytes - memory[0].usedBytes).toBeLessThan(64 * 1024 * 1024);
  await writeFile(path.join(directory, `${prefix}-memory.json`), JSON.stringify(memory, null, 2));
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.locator(".comparison-grid")).toHaveCSS("grid-template-columns", /^\S+$/);
  await page.locator(".comparison-grid").screenshot({ path: path.join(directory, `${prefix}-mobile-clouds.png`) });
});
