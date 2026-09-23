import { expect, test } from "@playwright/test";
import { existsSync, mkdirSync, readFileSync } from "node:fs";
import path from "node:path";

const galleryUrl = "/data/exp6_1_rgb_gallery_20260915_r1/index.html";
const commands = path.resolve("../../../MoGe-v3-reproduction/experiment/exp6_1_moge3_official_reproduction/04_checkpoint_visualization/commands");

test("Exp6-1 RGB gallery fixture filters, paginates, and retains cross-dataset selection", async ({ page }) => {
  // UI-only records; never exported or published as real dataset candidates.
  const source = JSON.parse(readFileSync("public/data/exp6_1_exp3_rgb_20260915_r2/manifest.json", "utf8"));
  const samples = Array.from({ length: 101 }, (_, index) => ({
    order: index + 1, dataset: index < 100 ? "NYUv2" : "KITTI", scene: "test scene",
    id: `${index < 100 ? "NYUv2" : "KITTI"}/fixture-${index + 1}`,
    previewUrl: source.samples[index < 100 ? 0 : 1].rgbUrl,
  }));
  await page.route(`**${galleryUrl}`, route => route.fulfill({ contentType: "text/html", body: readFileSync(path.join(commands, "image_gallery.html"), "utf8") }));
  await page.route("**/exp6_1_rgb_gallery_20260915_r1/gallery.json", route => route.fulfill({ json: { samples } }));
  await page.goto(galleryUrl);
  await expect(page.locator(".card")).toHaveCount(96);
  await page.locator(".card button").first().click();
  await page.getByRole("button", { name: "下一页" }).click();
  await expect(page.locator(".card")).toHaveCount(5);
  await page.getByRole("combobox", { name: "数据集", exact: true }).selectOption("KITTI");
  await expect(page.locator(".card")).toHaveCount(1);
  await page.locator(".card button").click();
  await page.getByRole("button", { name: "复制已选样本 ID" }).click();
  await expect(page.getByRole("textbox", { name: "已选样本 ID" })).toHaveValue("NYUv2/fixture-1\nKITTI/fixture-101");
  await page.reload();
  await expect(page.locator("#summary")).toContainText("已选择 2 张");
  await page.getByLabel("仅看已选").check();
  await expect(page.locator(".card")).toHaveCount(2);
  await page.getByRole("searchbox").fill("KITTI/fixture-101");
  await expect(page.locator(".card")).toHaveCount(1);
  await expect(page.locator(".thumb")).toHaveJSProperty("complete", true);
  await expect(page.locator(".thumb")).not.toHaveJSProperty("naturalWidth", 0);
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.locator(".grid")).toHaveCSS("grid-template-columns", /^\S+$/);
  await page.getByRole("button", { name: "清空选择" }).click();
  await expect(page.locator("#summary")).toContainText("已选择 0 张");
});

test("Exp6-1 RGB gallery real candidates load from all five existing datasets", async ({ page }) => {
  test.skip(!process.env.PLAYWRIGHT_BASE_URL && !existsSync(`public${galleryUrl}`), "Real server previews have not been synchronized yet");
  await page.goto(galleryUrl);
  await expect(page.locator("#summary")).toContainText("2924/2924", { timeout: 30_000 });
  for (const [dataset, count] of Object.entries({ NYUv2: 654, KITTI: 652, ETH3D: 454, "iBims-1": 100, Sintel: 1064 })) {
    await page.getByRole("combobox", { name: "数据集", exact: true }).selectOption(dataset);
    await expect(page.locator("#summary")).toContainText(`匹配 ${count}/2924`);
    await expect.poll(() => page.locator(".thumb").first().evaluate(image => (image as HTMLImageElement).naturalWidth)).toBeGreaterThan(0);
  }
  await page.getByRole("combobox", { name: "数据集", exact: true }).selectOption("iBims-1");
  await page.getByRole("searchbox").fill("office_02");
  await expect(page.locator(".card")).toHaveCount(1);
  await page.locator(".card button").click();
  await page.getByRole("button", { name: "复制已选样本 ID" }).click();
  await expect(page.getByRole("textbox", { name: "已选样本 ID" })).toHaveValue("iBims-1/office_02");
  await page.getByRole("searchbox").fill("");
  const screenshots = test.info().outputPath("acceptance");
  mkdirSync(screenshots, { recursive: true });
  await page.screenshot({ path: path.join(screenshots, "exp6-1-gallery-desktop.png") });
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.locator(".grid")).toHaveCSS("grid-template-columns", /^\S+$/);
  await page.screenshot({ path: path.join(screenshots, "exp6-1-gallery-mobile.png") });
});
