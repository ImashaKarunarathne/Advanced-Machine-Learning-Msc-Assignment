import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../api.js";
import { useFetch } from "../hooks/useFetch.js";
import { Card, COLORS, Kpi, StatusMessage } from "./common.jsx";

const MODELS = [
  { key: "random_forest", label: "Random Forest", color: COLORS.rf, sampleKey: "rf" },
  { key: "ann", label: "ANN (MLP)", color: COLORS.ann, sampleKey: "ann" },
  { key: "rf_ann", label: "RF + ANN", color: COLORS.rfAnn, sampleKey: "rf_ann" },
];

export default function RegressionPanel() {
  const { data, error, loading } = useFetch(() => api.regression(), []);

  if (error || loading || !data) return <StatusMessage loading={loading} error={error} />;

  const best = MODELS.reduce((a, b) => (data.models[a.key].rmse <= data.models[b.key].rmse ? a : b));
  const barData = MODELS.map((m) => ({
    model: m.label,
    MAE: data.models[m.key].mae,
    RMSE: data.models[m.key].rmse,
  }));

  const max = Math.max(...data.samples.map((s) => Math.max(s.actual, s.rf, s.ann, s.rf_ann))) * 1.05;

  return (
    <>
      <div className="kpi-row">
        <Kpi label="Best model (RMSE)" value={best.label} />
        <Kpi label="RF + ANN R²" value={data.models.rf_ann.r2.toFixed(3)} />
        <Kpi label="Random Forest R²" value={data.models.random_forest.r2.toFixed(3)} />
        <Kpi label="Test set" value={data.test_rows.toLocaleString()} hint={`target: ${data.target}`} />
      </div>

      <div className="grid">
        <Card wide title="Predicted vs actual length of stay" subtitle="Points on the dashed line are perfect predictions">
          <div className="chart">
            <ResponsiveContainer width="100%" height={340}>
              <ScatterChart margin={{ top: 10, right: 20, bottom: 10, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis type="number" dataKey="actual" name="Actual" domain={[0, max]} label={{ value: "Actual (days)", position: "insideBottom", offset: -4 }} />
                <YAxis type="number" domain={[0, max]} label={{ value: "Predicted (days)", angle: -90, position: "insideLeft" }} />
                <Tooltip cursor={{ strokeDasharray: "3 3" }} />
                <Legend />
                <ReferenceLine segment={[{ x: 0, y: 0 }, { x: max, y: max }]} stroke="#9ca3af" strokeDasharray="6 4" />
                {MODELS.map((m) => (
                  <Scatter key={m.key} name={m.label} data={data.samples} dataKey={m.sampleKey} fill={m.color} fillOpacity={0.55} />
                ))}
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card wide title="Error comparison" subtitle="Lower is better. Units: days.">
          <div className="chart">
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={barData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="model" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="MAE" fill="#94a3b8" />
                <Bar dataKey="RMSE" fill="#334155" />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <table className="metrics-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>MAE</th>
                <th>RMSE</th>
                <th>R²</th>
              </tr>
            </thead>
            <tbody>
              {MODELS.map((m) => (
                <tr key={m.key} className={m.key === best.key ? "best-row" : undefined}>
                  <td>{m.label}</td>
                  <td>{data.models[m.key].mae.toFixed(3)}</td>
                  <td>{data.models[m.key].rmse.toFixed(3)}</td>
                  <td>{data.models[m.key].r2.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="note">
            The gaps between models are small on this data. These numbers come from a single train/test split, so
            they are not a significance test.
          </p>
        </Card>
      </div>
    </>
  );
}
