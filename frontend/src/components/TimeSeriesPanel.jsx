import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../api.js";
import { useFetch } from "../hooks/useFetch.js";
import { Card, COLORS, Kpi, StatusMessage } from "./common.jsx";

// `key` matches the metrics block; `series` matches the field in each forecast row.
const MODELS = [
  { key: "arima", series: "arima", label: "ARIMA", color: COLORS.arima, dash: "6 4" },
  { key: "random_forest", series: "rf", label: "Random Forest", color: COLORS.rf, dash: "2 3" },
  { key: "rf_arima", series: "rf_arima", label: "RF + ARIMA", color: COLORS.hybrid, dash: null },
];

export default function TimeSeriesPanel() {
  const { data, error, loading } = useFetch(() => api.timeseries(), []);

  if (error || loading || !data) return <StatusMessage loading={loading} error={error} />;

  const best = MODELS.reduce((a, b) => (data.models[a.key].mae <= data.models[b.key].mae ? a : b));
  const improvement = 1 - data.models.rf_arima.mae / data.models.arima.mae;

  return (
    <>
      <div className="kpi-row">
        <Kpi label="Best model (MAE)" value={best.label} />
        <Kpi label="RF+ARIMA MAE" value={data.models.rf_arima.mae.toFixed(1)} hint="thousand subscribers" />
        <Kpi label="Improvement over ARIMA" value={`${(improvement * 100).toFixed(1)}%`} hint="lower MAE" />
        <Kpi label="ARIMA order" value={`(${data.arima_order.join(", ")})`} hint={`AIC ${data.arima_aic}`} />
      </div>

      <div className="grid">
        <Card
          wide
          title="Test-period forecasts"
          subtitle={`One-step-ahead forecasts for the last ${data.test_months} months (trained on ${data.train_months})`}
        >
          <div className="chart">
            <ResponsiveContainer width="100%" height={340}>
              <LineChart data={data.series}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="month" tick={{ fontSize: 12 }} />
                <YAxis domain={["auto", "auto"]} />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="actual" name="Actual" stroke={COLORS.actual} strokeWidth={2.5} dot={false} />
                {MODELS.map((m) => (
                  <Line
                    key={m.key}
                    type="monotone"
                    dataKey={m.series}
                    name={m.label}
                    stroke={m.color}
                    strokeDasharray={m.dash ?? undefined}
                    strokeWidth={2}
                    dot={false}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card wide title="Error metrics" subtitle="Lower is better. Errors in thousand subscribers.">
          <table className="metrics-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>MAE</th>
                <th>RMSE</th>
                <th>MAPE (%)</th>
              </tr>
            </thead>
            <tbody>
              {MODELS.map((m) => (
                <tr key={m.key} className={m.key === best.key ? "best-row" : undefined}>
                  <td>{m.label}</td>
                  <td>{data.models[m.key].mae.toFixed(2)}</td>
                  <td>{data.models[m.key].rmse.toFixed(2)}</td>
                  <td>{data.models[m.key].mape.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="note">
            The random forest alone cannot extrapolate the upward trend, which is why it lags. Adding it on top of
            ARIMA lets it correct the nonlinear shocks that ARIMA misses.
          </p>
        </Card>
      </div>
    </>
  );
}
