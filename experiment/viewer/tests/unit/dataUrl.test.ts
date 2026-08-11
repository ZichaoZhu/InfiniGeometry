import { afterEach, describe, expect, it, vi } from "vitest";

afterEach(() => {
  vi.unstubAllEnvs();
  vi.resetModules();
});

describe("point-cloud data URL", () => {
  it("keeps relative data paths when no asset origin is configured", async () => {
    vi.stubEnv("NEXT_PUBLIC_POINT_CLOUD_ASSET_ORIGIN", "");
    const { dataUrl } = await import("@/lib/dataUrl");
    expect(dataUrl("/data/exp1/manifest.json")).toBe(
      "/data/exp1/manifest.json",
    );
  });

  it("prefixes relative paths for a separate static asset origin", async () => {
    vi.stubEnv(
      "NEXT_PUBLIC_POINT_CLOUD_ASSET_ORIGIN",
      "https://assets.example.test/",
    );
    const { dataUrl } = await import("@/lib/dataUrl");
    expect(dataUrl("/data/exp1/manifest.json")).toBe(
      "https://assets.example.test/data/exp1/manifest.json",
    );
    expect(dataUrl("https://example.com/cloud.ply")).toBe(
      "https://example.com/cloud.ply",
    );
  });
});
