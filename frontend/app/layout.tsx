import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "隔日沖預測",
  description: "輸入台股代號，由模型預測該股明日漲跌傾向"
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="zh-Hant">
      <body>{children}</body>
    </html>
  );
}
