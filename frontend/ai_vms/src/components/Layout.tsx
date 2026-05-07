import type { ReactNode } from "react";
import Header from "./Header";
import Sidebar from "./Sidebar";
import Footer from "./Footer";

const Layout = ({ children }: { children: ReactNode }) => {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100vh",
        overflow: "hidden"
      }}
    >
      <Header />

      <div style={{ display: "flex", flex: 1, overflow: "hidden" }}>
        <Sidebar />

        <main
          style={{
            flex: 1,
            overflow: "hidden",
            background: "var(--bg-main)",
            display: "flex",
            flexDirection: "column"
          }}
        >
          {children}
        </main>
      </div>

      <Footer />
    </div>
  );
};

export default Layout;
