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
    await expect.poll(() => canvasColorCount(page, panel)).toBeGreaterThan(2);
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

test("keeps controls and canvases separated on a mobile viewport", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await expect(page.locator("canvas")).toHaveCount(3);
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
