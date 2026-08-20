import { defineConfig, devices } from "@playwright/test";

const remoteBaseUrl = process.env.PLAYWRIGHT_BASE_URL;

export default defineConfig({
  testDir: "./tests/e2e",
  timeout: 45_000,
  expect: { timeout: 10_000 },
  use: {
    baseURL: remoteBaseUrl ?? "http://127.0.0.1:3219",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  webServer: remoteBaseUrl ? undefined : {
    command: "npm run dev -- --hostname 127.0.0.1 --port 3219",
    url: "http://127.0.0.1:3219",
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
});
