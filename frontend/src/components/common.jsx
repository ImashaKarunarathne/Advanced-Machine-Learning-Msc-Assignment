export const COLORS = {
  rf: "#2f6fdb",
  xgb: "#e0823d",
  arima: "#7a8699",
  hybrid: "#14906b",
  ann: "#9b59b6",
  rfAnn: "#14906b",
  actual: "#1f2937",
};

export function pct(value) {
  return `${(value * 100).toFixed(1)}%`;
}

export function Kpi({ label, value, hint }) {
  return (
    <div className="kpi">
      <span className="kpi-label">{label}</span>
      <span className="kpi-value">{value}</span>
      {hint && <span className="kpi-hint">{hint}</span>}
    </div>
  );
}

export function Card({ title, subtitle, children, wide = false }) {
  return (
    <section className={wide ? "card wide" : "card"}>
      <header className="card-header">
        <h2>{title}</h2>
        {subtitle && <p>{subtitle}</p>}
      </header>
      {children}
    </section>
  );
}

export function StatusMessage({ loading, error }) {
  if (error) return <p className="status error">Could not load data: {error}</p>;
  if (loading) return <p className="status">Training models and computing results…</p>;
  return null;
}
