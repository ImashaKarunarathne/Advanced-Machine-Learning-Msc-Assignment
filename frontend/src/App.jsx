import { useState } from "react";
import OverviewPanel from "./components/OverviewPanel.jsx";
import ClassificationPanel from "./components/ClassificationPanel.jsx";
import TimeSeriesPanel from "./components/TimeSeriesPanel.jsx";
import RegressionPanel from "./components/RegressionPanel.jsx";

const TABS = [
  { id: "overview", label: "Overview" },
  { id: "classification", label: "RF vs XGBoost" },
  { id: "timeseries", label: "RF + ARIMA" },
  { id: "regression", label: "RF + ANN" },
];

export default function App() {
  const [tab, setTab] = useState("overview");

  return (
    <div className="app">
      <header className="topbar">
        <div>
          <h1>Random Forest Dashboard</h1>
          <p className="subtitle">Applied ML hands-on project · local datasets</p>
        </div>
      </header>

      <nav className="tabs" role="tablist">
        {TABS.map((t) => (
          <button
            key={t.id}
            role="tab"
            aria-selected={tab === t.id}
            className={tab === t.id ? "tab active" : "tab"}
            onClick={() => setTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </nav>

      <main>
        {tab === "overview" && <OverviewPanel />}
        {tab === "classification" && <ClassificationPanel />}
        {tab === "timeseries" && <TimeSeriesPanel />}
        {tab === "regression" && <RegressionPanel />}
      </main>
    </div>
  );
}
