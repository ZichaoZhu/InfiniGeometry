"use client";

import dynamic from "next/dynamic";

const PointCloudComparison = dynamic(
  () =>
    import("@/components/PointCloudComparison").then(
      (module) => module.PointCloudComparison,
    ),
  {
    ssr: false,
    loading: () => (
      <main className="boot-screen">
        <div className="boot-mark">ID</div>
        <p>正在载入 InfiniDepth 实验查看器…</p>
      </main>
    ),
  },
);

export default function Home() {
  return <PointCloudComparison />;
}
