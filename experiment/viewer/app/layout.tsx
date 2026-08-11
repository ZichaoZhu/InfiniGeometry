import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "InfiniDepth Disparity Refiner 实验查看器",
  description: "比较 Exp1 的 GT、官方初始、Detach 最佳和联合最佳点云",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
