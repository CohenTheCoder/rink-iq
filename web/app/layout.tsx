import type { Metadata } from "next";
import { Barlow, Barlow_Condensed, Share_Tech_Mono } from "next/font/google";
import "./globals.css";

const body = Barlow({ subsets: ["latin"], weight: ["400", "500", "600"], variable: "--f-body" });
const display = Barlow_Condensed({ subsets: ["latin"], weight: ["600", "800"], variable: "--f-display" });
const mono = Share_Tech_Mono({ subsets: ["latin"], weight: "400", variable: "--f-mono" });

export const metadata: Metadata = {
  title: "RinkIQ · hockey film in, stats out",
  description: "Upload hockey game film. RinkIQ finds the rink, tracks every skater, reads jersey numbers and grades the players.",
  robots: { index: false, follow: false },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${body.variable} ${display.variable} ${mono.variable}`}>
      <body>
        <div className="rink-bg" aria-hidden />
        <main className="wrap">{children}</main>
        <footer className="foot">
          RinkIQ · computer vision on game film · <a href="https://github.com/CohenTheCoder/rink-iq">source</a>
        </footer>
      </body>
    </html>
  );
}
