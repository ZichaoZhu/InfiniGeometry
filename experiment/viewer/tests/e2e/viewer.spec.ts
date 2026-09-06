import { expect, test } from "@playwright/test";
import { mkdir } from "node:fs/promises";
import path from "node:path";

async function canvasColorCount(page: import("@playwright/test").Page, testId: string) {
  return page.getByTestId(testId).locator("canvas").evaluate((canvas) => {
    const element = canvas as HTMLCanvasElement;
    const context = element.getContext("webgl2") ?? element.getContext("webgl");
    if (!context) return 0;
    const pixels = new Uint8Array(element.width * element.height * 4);
    context.readPixels(0, 0, element.width, element.height, context.RGBA, context.UNSIGNED_BYTE, pixels);
    const colors = new Set<string>();
    const stride = Math.max(4, Math.floor(pixels.length / 2000 / 4) * 4);
    for (let index = 0; index < pixels.length; index += stride) {
      colors.add(`${pixels[index]},${pixels[index + 1]},${pixels[index + 2]}`);
    }
    return colors.size;
  });
}

test("switches experiments and renders the three point-cloud windows", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: /Disparity Refiner/ })).toBeVisible();
  await expect(page.locator("canvas")).toHaveCount(3);
  const exp2 = page.getByTestId("experiment-exp2_infinidepth_disparity_ssr_hypersim100_overfit");
  await expect(exp2).toBeVisible();
  await exp2.click();
  await expect(exp2).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByTestId("viewer-left")).toContainText("官方初始 · K=0");
  await expect(page.getByTestId("viewer-right")).toContainText("联合最佳 · K=3");
  await expect(page.getByText(/GT 的 2%\/98% disparity 统计反归一化/)).toBeVisible();
  await expect(page.getByText("点图 Rel").first()).toBeVisible();
  await page.getByTestId("sample-1").hover();
  await expect(page.getByTestId("sample-1").locator(".sample-hover-preview")).toBeVisible();

  for (const panel of ["viewer-ground-truth", "viewer-left", "viewer-right"]) {
    await page.getByTestId(panel).scrollIntoViewIfNeeded();
    await expect.poll(
      () => canvasColorCount(page, panel),
      { timeout: 30_000 },
    ).toBeGreaterThan(2);
  }

  const leftCanvas = page.getByTestId("viewer-left").locator("canvas");
  const initialSource = await leftCanvas.getAttribute("data-scene-source");
  await page.getByTestId("left-stage").getByRole("button", { name: "Detach 最佳" }).click();
  await page.getByTestId("left-k").getByRole("button", { name: "K=5" }).click();
  await expect(page.getByTestId("viewer-left")).toContainText("Detach 最佳 · K=5");
  await expect(leftCanvas).not.toHaveAttribute("data-scene-source", initialSource ?? "");

  await page.getByTestId("left-interaction").getByRole("button", { name: "平移" }).click();
  await expect(page.getByTestId("left-interaction").getByRole("button", { name: "平移" })).toHaveAttribute("aria-pressed", "true");
  await page.getByTestId("sample-5").click();
  await expect(page.getByTestId("sample-5")).toContainText("玻璃楼梯细拉杆");
  await expect(page.locator(".canvas-error")).toHaveCount(0);

  const screenshotDirectory = path.join(process.cwd(), "test-results", "viewer-acceptance");
  await mkdir(screenshotDirectory, { recursive: true });
  await page.screenshot({ path: path.join(screenshotDirectory, "desktop.png"), fullPage: true });
});

test("shows five configured samples for each Exp3 dataset split", async ({ page }) => {
  await page.goto("/");
  const exp3 = page.getByTestId("experiment-exp3_infinidepth_disparity_ssr_hypersim_full");
  await exp3.click();
  await expect(exp3).toHaveAttribute("aria-pressed", "true");
  const split = page.getByTestId("sample-split");
  await expect(split.getByRole("button")).toHaveCount(3);
  await expect(split.getByRole("button", { name: "TRAIN" })).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator(".sample-switcher button")).toHaveCount(5);
  await expect(page.getByTestId("sample-1")).toContainText("ai_019_004_cam_00_frame.0000");
  await expect(page.getByTestId("sample-1")).toContainText("楼梯扶手与平行栏杆");
  await expect(page.getByTestId("raster-scope")).toContainText("细结构裁剪");
  for (const panel of ["viewer-ground-truth", "viewer-left", "viewer-right"]) {
    await page.getByTestId(panel).scrollIntoViewIfNeeded();
    await expect.poll(
      () => canvasColorCount(page, panel),
      { timeout: 30_000 },
    ).toBeGreaterThan(2);
  }
  const screenshotDirectory = path.join(process.cwd(), "test-results", "viewer-acceptance");
  await mkdir(screenshotDirectory, { recursive: true });
  await page.screenshot({ path: path.join(screenshotDirectory, "exp3-train-desktop.png"), fullPage: true });

  await split.getByRole("button", { name: "VAL" }).click();
  await expect(page.getByTestId("sample-6")).toContainText("Val 01");
  await expect(page.locator(".sample-switcher button")).toHaveCount(5);
  await split.getByRole("button", { name: "TEST" }).click();
  await expect(page.getByTestId("sample-11")).toContainText("Test 01");
  await expect(page.locator(".sample-switcher button")).toHaveCount(5);

  for (const panel of ["viewer-ground-truth", "viewer-left", "viewer-right"]) {
    await expect.poll(
      () => canvasColorCount(page, panel),
      { timeout: 30_000 },
    ).toBeGreaterThan(2);
  }
  await expect(page.locator(".canvas-error")).toHaveCount(0);
  await page.screenshot({ path: path.join(screenshotDirectory, "exp3-desktop.png"), fullPage: true });
});

test("shows the Exp4 LiDAR best checkpoint across all three dataset splits", async ({ page }) => {
  await page.goto("/");
  const exp4 = page.getByTestId("experiment-exp4_infinidepth_lidar_refiner_hypersim_full");
  await exp4.click();
  await expect(exp4).toHaveAttribute("aria-pressed", "true");
  const split = page.getByTestId("sample-split");
  await expect(split.getByRole("button")).toHaveCount(3);
  await expect(page.getByTestId("sample-1")).toContainText("ai_019_004_cam_00_frame.0000");
  await expect(page.getByTestId("viewer-left")).toContainText("LiDAR 初始 · K=0");
  await expect(page.getByTestId("viewer-right")).toContainText("SSR 最佳 22.5k · K=3");
  await expect(page.getByTestId("viewer-reference")).toContainText("RGB-only · Detach 最佳 · K=3");

  for (const panel of ["viewer-ground-truth", "viewer-left", "viewer-right", "viewer-reference"]) {
    await page.getByTestId(panel).scrollIntoViewIfNeeded();
    await expect.poll(() => canvasColorCount(page, panel), { timeout: 30_000 }).toBeGreaterThan(2);
  }
  const screenshotDirectory = path.join(process.cwd(), "test-results", "viewer-acceptance");
  await mkdir(screenshotDirectory, { recursive: true });
  await page.screenshot({ path: path.join(screenshotDirectory, "exp4-rgb-lidar-desktop.png"), fullPage: true });
  await page.getByTestId("reference-version").getByRole("button", { name: "LiDAR" }).click();
  await expect(page.getByTestId("viewer-reference")).toContainText("LiDAR · SSR 最佳 22.5k · K=3");
  await page.getByTestId("viewer-right").locator("canvas").screenshot({ path: path.join(screenshotDirectory, "exp4-k3-canvas.png") });
  await split.getByRole("button", { name: "VAL" }).click();
  await expect(page.getByTestId("sample-6")).toContainText("Val 01");
  await split.getByRole("button", { name: "TEST" }).click();
  await expect(page.getByTestId("sample-11")).toContainText("Test 01");
  await page.getByTestId("right-k").getByRole("button", { name: "K=5" }).click();
  await expect(page.getByTestId("viewer-right")).toContainText("SSR 最佳 22.5k · K=5");
  for (const panel of ["viewer-ground-truth", "viewer-left", "viewer-right", "viewer-reference"]) {
    await page.getByTestId(panel).scrollIntoViewIfNeeded();
    await expect.poll(() => canvasColorCount(page, panel), { timeout: 30_000 }).toBeGreaterThan(2);
  }
  await expect(page.locator(".canvas-error")).toHaveCount(0);
});

test("shows ten Waymo FRONT and SIDE visualization samples", async ({ page }) => {
  await page.goto("/");
  const exp5 = page.getByTestId("experiment-exp5_waymo");
  await exp5.click();
  await expect(exp5).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator(".sample-switcher button")).toHaveCount(10, { timeout: 30_000 });
  await expect(page.getByTestId("sample-1")).toContainText("Waymo FRONT 01");
  await expect(page.getByTestId("viewer-ground-truth")).toContainText("Waymo held-out TOP LiDAR");
  await expect(page.getByText("展示 FRONT 五张样例的 held-out TOP LiDAR 指标；SIDE 五张仅有定性可视化，暂无对应正式逐图报告。指标不用于 checkpoint 选择。")).toHaveCount(1);
  await expect(page.getByText("Inverse-depth MAE").first()).toBeVisible();
  await page.getByTestId("right-k").getByRole("button", { name: "K=5" }).click();
  await expect(page.getByTestId("viewer-right")).toContainText("Waymo Zero-shot · K=5");
  for (const panel of ["viewer-ground-truth", "viewer-left", "viewer-right"]) {
    await page.getByTestId(panel).scrollIntoViewIfNeeded();
    await expect(page.getByTestId(panel).locator("canvas")).toHaveAttribute(
      "data-scene-source",
      /exp5_waymo_val202_seed173/,
      { timeout: 30_000 },
    );
  }
  await expect(page.locator(".canvas-error")).toHaveCount(0);
  const screenshotDirectory = path.join(process.cwd(), "test-results", "viewer-acceptance");
  await mkdir(screenshotDirectory, { recursive: true });
  await page.screenshot({ path: path.join(screenshotDirectory, "exp5-waymo-desktop.png"), fullPage: true });
  await page.getByTestId("sample-6").click();
  await expect(page.getByTestId("sample-6")).toContainText("Waymo SIDE 01");
  await expect(page.getByTestId("sample-6")).toContainText("SIDE_RIGHT");
  await page.getByTestId("right-k").getByRole("button", { name: "K=3" }).click();
  await expect(page.getByTestId("viewer-right")).toContainText("Waymo Zero-shot · K=3");
  for (const panel of ["viewer-ground-truth", "viewer-left", "viewer-right"]) {
    await page.getByTestId(panel).scrollIntoViewIfNeeded();
    await expect(page.getByTestId(panel).locator("canvas")).toHaveAttribute(
      "data-scene-source",
      /exp5_waymo_side_selected_20260903/,
      { timeout: 30_000 },
    );
    await expect.poll(() => canvasColorCount(page, panel), { timeout: 30_000 }).toBeGreaterThan(2);
  }
  await expect(page.locator(".canvas-error")).toHaveCount(0);
  await page.screenshot({ path: path.join(screenshotDirectory, "exp5-waymo-side-desktop.png"), fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.locator(".sample-switcher button")).toHaveCount(10);
  for (const panel of ["viewer-ground-truth", "viewer-left", "viewer-right"]) {
    await page.getByTestId(panel).scrollIntoViewIfNeeded();
    await expect.poll(() => canvasColorCount(page, panel), { timeout: 30_000 }).toBeGreaterThan(2);
  }
  await page.screenshot({ path: path.join(screenshotDirectory, "exp5-waymo-side-mobile.png"), fullPage: true });
});

test("shows ten selected ETH3D point-cloud samples", async ({ page }) => {
  await page.goto("/");
  const exp5 = page.getByTestId("experiment-exp5_eth3d");
  await exp5.click();
  await expect(exp5).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator(".sample-switcher button")).toHaveCount(10, { timeout: 30_000 });
  await expect(page.getByTestId("sample-1")).toContainText("ETH3D 01");
  await expect(page.getByTestId("viewer-ground-truth")).toContainText("ETH3D Ground Truth");
  await expect(page.getByText("Local Point Rel").first()).toBeVisible();
  await page.getByTestId("right-k").getByRole("button", { name: "K=3" }).click();
  await expect(page.getByTestId("viewer-right")).toContainText("ETH3D Zero-shot · K=3");
  for (const panel of ["viewer-ground-truth", "viewer-left", "viewer-right"]) {
    await page.getByTestId(panel).scrollIntoViewIfNeeded();
    await expect(page.getByTestId(panel).locator("canvas")).toHaveAttribute(
      "data-scene-source",
      /exp5_eth3d_selected_20260903_r3/,
      { timeout: 30_000 },
    );
    await expect.poll(() => canvasColorCount(page, panel), { timeout: 30_000 }).toBeGreaterThan(2);
  }
  await expect(page.locator(".canvas-error")).toHaveCount(0);
});

test("shows ten fixed Exp6-1 official MoGe-3 samples", async ({ page }) => {
  await page.goto("/");
  const exp6 = page.getByTestId("experiment-exp6_1_moge3_official_reproduction");
  await exp6.click();
  await expect(exp6).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByRole("heading", { name: /MoGe-3/ })).toBeVisible();
  await expect(page.locator(".sample-switcher button")).toHaveCount(10, { timeout: 30_000 });
  await expect(page.getByTestId("sample-1")).toContainText("NYUv2");
  await expect(page.getByTestId("sample-3")).toContainText("DSC_6487");
  await expect(page.getByTestId("sample-4")).toContainText("office_02");
  await expect(page.getByTestId("sample-6")).toContainText("00022_00193_outdoor_320_020");
  await expect(page.getByTestId("sample-8")).toContainText("000290");
  await expect(page.getByTestId("sample-10")).toContainText("DDAD");
  await expect(page.getByTestId("sample-10")).toContainText("CAMERA_06");
  await expect(page.getByTestId("viewer-left")).toContainText("官方 ViT-L · K=0");
  await expect(page.getByTestId("viewer-right")).toContainText("官方 ViT-L · K=3");
  await expect(page.getByText("Affine Depth Rel").first()).toBeVisible();
  await page.getByTestId("right-k").getByRole("button", { name: "K=5" }).click();
  await expect(page.getByTestId("viewer-right")).toContainText("官方 ViT-L · K=5");
  for (const sample of [3, 4, 6, 8, 10]) {
    await page.getByTestId(`sample-${sample}`).click();
    for (const panel of ["viewer-ground-truth", "viewer-left", "viewer-right"]) {
      await page.getByTestId(panel).scrollIntoViewIfNeeded();
      await expect(page.getByTestId(panel).locator("canvas")).toHaveAttribute(
        "data-scene-source",
        /exp6_1_moge3_official_reproduction_r3/,
        { timeout: 30_000 },
      );
      await expect.poll(() => canvasColorCount(page, panel), { timeout: 30_000 }).toBeGreaterThan(2);
    }
  }
  await expect(page.locator(".canvas-error")).toHaveCount(0);
  await page.evaluate(() => (document.activeElement as HTMLElement | null)?.blur());
  await page.mouse.move(1, 1);
  await expect(page.locator(".sample-hover-preview").last()).toBeHidden();
  const screenshotDirectory = path.join(process.cwd(), "test-results", "viewer-acceptance");
  await mkdir(screenshotDirectory, { recursive: true });
  await page.screenshot({ path: path.join(screenshotDirectory, "exp6-1-desktop.png"), fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({ path: path.join(screenshotDirectory, "exp6-1-mobile.png"), fullPage: true });
});

test("lists and selects the Waymo side-camera previews", async ({ page }) => {
  await page.goto("/data/waymo_gallery_exp5_val202_side_20260903_r2/index.html");
  await expect(page.getByRole("heading", { name: "Waymo Val202 · SIDE 选图" })).toBeVisible();
  await expect(page.locator(".card")).toHaveCount(404, { timeout: 30_000 });
  await page.locator(".card").first().locator("img").click();
  await expect(page.locator(".summary")).toContainText("已选择 1 张");
  await page.getByPlaceholder("搜索编号、地点、时段、文件名").fill("Night");
  await expect(page.locator(".card:not(.hidden)")).toHaveCount(38);
  await page.getByRole("combobox").nth(0).selectOption("SIDE_RIGHT");
  await expect(page.locator(".card:not(.hidden)")).toHaveCount(19);
  const screenshotDirectory = path.join(process.cwd(), "test-results", "viewer-acceptance");
  await mkdir(screenshotDirectory, { recursive: true });
  await page.screenshot({ path: path.join(screenshotDirectory, "waymo-gallery-night.png"), fullPage: true });
});

test("lists and limits ETH3D selection to ten completed evaluation inputs", async ({ page }) => {
  await page.goto("/data/eth3d_gallery_exp5_highres_train_20260903_r1/index.html");
  await expect(page.getByRole("heading", { name: "ETH3D Exp5 · 选图" })).toBeVisible();
  await expect(page.locator(".card")).toHaveCount(454, { timeout: 30_000 });
  await page.getByRole("combobox").selectOption("courtyard");
  await expect(page.locator(".card:not(.hidden)")).toHaveCount(38);
  await page.getByPlaceholder("搜索序号、场景或样本 ID").fill("DSC_0286");
  await expect(page.locator(".card:not(.hidden)")).toHaveCount(1);
  await page.getByPlaceholder("搜索序号、场景或样本 ID").fill("");
  for (let index = 0; index < 11; index += 1) await page.locator(".card:not(.hidden) .thumb").nth(index).click();
  await expect(page.locator("#summary")).toContainText("已选择 10/10 张");
  await expect(page.locator("#warning")).toContainText("最多选择 10 张");
});

test("keeps controls and canvases separated on a mobile viewport", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await page.getByTestId("experiment-exp3_infinidepth_disparity_ssr_hypersim_full").click();
  await expect(page.getByTestId("sample-split")).toBeVisible();
  await expect(page.locator("canvas")).toHaveCount(3);
  for (const panel of ["viewer-ground-truth", "viewer-left", "viewer-right"]) {
    await page.getByTestId(panel).scrollIntoViewIfNeeded();
    await expect.poll(() => canvasColorCount(page, panel), { timeout: 30_000 }).toBeGreaterThan(2);
  }
  const panels = page.locator(".viewer-pane");
  await expect(panels).toHaveCount(3);
  const first = await panels.nth(0).boundingBox();
  const second = await panels.nth(1).boundingBox();
  expect(first).not.toBeNull();
  expect(second).not.toBeNull();
  expect(second!.y).toBeGreaterThan(first!.y + first!.height - 1);
  await expect(page.locator(".canvas-error")).toHaveCount(0);
  const screenshotDirectory = path.join(process.cwd(), "test-results", "viewer-acceptance");
  await mkdir(screenshotDirectory, { recursive: true });
  await page.screenshot({ path: path.join(screenshotDirectory, "mobile.png"), fullPage: true });
});
