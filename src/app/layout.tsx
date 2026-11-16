import type { Metadata } from "next";
import { Chakra_Petch, Exo_2, JetBrains_Mono, Michroma } from "next/font/google";
import { PrefsSync } from "@/features/shell/display-menu";
import "./globals.css";

// Body: Exo 2 (geometric, technical). Display: Chakra Petch (angular HUD lettering).
// Brand: Michroma (wide, instrument-panel wordmark). Mono: JetBrains Mono (code, numbers).
const body = Exo_2({ variable: "--font-body", subsets: ["latin"] });
const display = Chakra_Petch({
  variable: "--font-display-face",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
});
const brand = Michroma({ variable: "--font-brand-face", subsets: ["latin"], weight: "400" });
const mono = JetBrains_Mono({ variable: "--font-mono-face", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "NeuroScope",
  description:
    "Drop a Jupyter notebook and watch its model architecture run as a live 3D simulation.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${body.variable} ${display.variable} ${brand.variable} ${mono.variable} h-full antialiased`}
    >
      <body className="flex min-h-full flex-col">
        <PrefsSync />
        {children}
      </body>
    </html>
  );
}
