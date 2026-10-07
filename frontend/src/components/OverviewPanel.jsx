import { api } from "../api.js";
import { useFetch } from "../hooks/useFetch.js";
import { Card, Kpi, pct, StatusMessage } from "./common.jsx";

export default function OverviewPanel() {
  const { data, error, loading } = useFetch(() => api.datasets(), []);

  if (error || loading || !data) return <StatusMessage loading={loading} error={error} />;

  const totalRows = data.reduce((sum, d) => sum + d.rows, 0);

  return (
    <>
      <div className="kpi-row">
        <Kpi label="Datasets" value={data.length} hint="local CSV files" />
        <Kpi label="Total rows" value={totalRows.toLocaleString()} />
        <Kpi label="Missing values" value={data.reduce((s, d) => s + d.missing_values, 0)} hint="across all datasets" />
      </div>

      <div className="grid">
        {data.map((d) => (
          <Card key={d.key} title={d.title} subtitle={d.description}>
            <dl className="facts">
              <dt>Task</dt>
              <dd className="capitalize">{d.task}</dd>
              <dt>Target</dt>
              <dd><code>{d.target}</code></dd>
              <dt>Rows</dt>
              <dd>{d.rows.toLocaleString()}</dd>
              <dt>Columns</dt>
              <dd>{d.columns.length}</dd>
              {d.positive_rate !== undefined && (
                <>
                  <dt>Positive rate</dt>
                  <dd>{pct(d.positive_rate)}</dd>
                </>
              )}
              {d.target_mean !== undefined && (
                <>
                  <dt>Target mean</dt>
                  <dd>{d.target_mean}</dd>
                </>
              )}
            </dl>
          </Card>
        ))}
      </div>
    </>
  );
}
