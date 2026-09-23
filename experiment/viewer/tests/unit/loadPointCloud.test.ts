import { afterEach, expect, it, vi } from "vitest";
import { loadPointCloud } from "@/lib/loadPointCloud";

afterEach(() => vi.unstubAllGlobals());

const ply = "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n0 0 1\n";

it("loads a PLY without a global cache and disposes it on cleanup", async () => {
  const fetcher = vi.fn<typeof fetch>(async () => new Response(ply));
  vi.stubGlobal("fetch", fetcher);
  const onLoad = vi.fn();
  const onError = vi.fn();
  const release = loadPointCloud("/one.ply", onLoad, onError);
  await vi.waitFor(() => expect(onLoad).toHaveBeenCalledOnce());
  const geometry = onLoad.mock.calls[0][0];
  expect(geometry.getAttribute("position").count).toBe(1);
  const dispose = vi.spyOn(geometry, "dispose");
  release();
  expect(dispose).toHaveBeenCalledOnce();
  expect(fetcher.mock.calls[0][1]?.signal?.aborted).toBe(true);
  expect(onError).not.toHaveBeenCalled();
});

it("does not publish a stale response after switching images", async () => {
  let resolve!: (response: Response) => void;
  vi.stubGlobal("fetch", vi.fn(() => new Promise<Response>((done) => { resolve = done; })));
  const onLoad = vi.fn();
  const onError = vi.fn();
  const release = loadPointCloud("/stale.ply", onLoad, onError);
  release();
  resolve(new Response(ply));
  await new Promise((done) => setTimeout(done, 0));
  expect(onLoad).not.toHaveBeenCalled();
  expect(onError).not.toHaveBeenCalled();
});

it("reports HTTP errors instead of parsing an error page", async () => {
  vi.stubGlobal("fetch", vi.fn(async () => new Response("missing", { status: 404 })));
  const onError = vi.fn();
  const release = loadPointCloud("/missing.ply", vi.fn(), onError);
  await vi.waitFor(() => expect(onError).toHaveBeenCalledWith(expect.objectContaining({ message: expect.stringContaining("404") })));
  release();
});
