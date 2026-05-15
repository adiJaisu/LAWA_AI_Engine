import 'primereact/resources/themes/lara-light-indigo/theme.css';
import 'primereact/resources/primereact.min.css';
import 'primeicons/primeicons.css';

import React, { useState, useEffect } from "react";
import ReactDOM from "react-dom/client";
import { AuthProvider } from "./context/AuthContext";
import AppRoutes from "./routes/routes";
import { BrowserRouter } from "react-router-dom";
import { ToastProvider } from './providers/toastProvider';

const FontLoader = ({ children }: { children: React.ReactNode }) => {
  const [fontsReady, setFontsReady] = useState(false);

  useEffect(() => {
    Promise.all([
      document.fonts.load('1em "Material Icons"'),
      document.fonts.load('1em "Material Symbols Outlined"'),
    ])
      .then(() => setFontsReady(true))
      .catch(() => {
        setFontsReady(true);
      });
  }, []);

  if (!fontsReady) {
    return (
      <div style={{
        display: "flex", flexDirection: "column", justifyContent: "center", alignItems: "center",
        height: "100vh", width: "100vw", backgroundColor: "#f7fafc", gap: "16px"
      }}>
        <div style={{
          width: "52px", height: "52px", border: "5px solid #e2e8f0", borderTop: "5px solid #4299e1",
          borderRadius: "50%", animation: "spin 0.8s linear infinite"
        }} />
        <p style={{ color: "#718096", fontSize: "14px", fontFamily: "sans-serif" }}>Loading...</p>
        <style>{`@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }`}</style>
      </div>
    );
  }

  return <>{children}</>;
};

const rootElement = document.getElementById("root");
if (!rootElement) throw new Error("Root element not found");

ReactDOM.createRoot(rootElement).render(
  <React.StrictMode>
    <FontLoader>
      <AuthProvider>
        <ToastProvider>
          <BrowserRouter>
            <AppRoutes />
          </BrowserRouter>
        </ToastProvider>
      </AuthProvider>
    </FontLoader>
  </React.StrictMode>
);