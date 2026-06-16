import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "NeuralRetail Intelligence Platform",
  description: "Retail AI platform — demand forecasting, churn prediction, inventory intelligence · AMX-DS-2026-04",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
