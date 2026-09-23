import { BufferGeometry } from "three";
import { PLYLoader } from "three/addons/loaders/PLYLoader.js";

// ponytail: no decoded-PLY cache; the browser may still cache HTTP responses.
export function loadPointCloud(
  url: string,
  onLoad: (geometry: BufferGeometry) => void,
  onError: (error: Error) => void,
): () => void {
  const controller = new AbortController();
  let geometry: BufferGeometry | undefined;
  void fetch(url, { signal: controller.signal })
    .then(async (response) => {
      if (!response.ok) throw new Error(`点云请求失败：HTTP ${response.status}`);
      const buffer = await response.arrayBuffer();
      if (controller.signal.aborted) return;
      geometry = new PLYLoader().parse(buffer);
      onLoad(geometry);
    })
    .catch((error: unknown) => {
      if (!controller.signal.aborted) {
        onError(error instanceof Error ? error : new Error(String(error)));
      }
    });
  return () => {
    controller.abort();
    geometry?.dispose();
  };
}
