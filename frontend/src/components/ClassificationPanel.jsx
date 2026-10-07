import { useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../api.js";
import { useFetch } from "../hooks/useFetch.js";
import { Card, COLORS, Kpi, pct, StatusMessage } from "./common.jsx";

const DATASETS = [
  { key: "telecom_churn", label: "Telecom churn" },
  { key: "patient_readmission", label: "Patient readmission" },
];

const METRICS = [
  { key: "accuracy", label: "Accuracy" },
  { key: "precision", label: "Precision" },
  { key: "recall", label: "Recall" },
  { key: "f1", label: "F1" },
  { key: "roc_auc", label: "ROC-AUC" },
];

const MODEL_LABEL = { random_forest: "Random Forest", xgboost: "XGBoost" };

export default function ClassificationPanel() {
  const [dataset, setDataset] = useState(DATASETS[0].key);
  const { data, error, loading } = useFetch(() => api.classification(dataset), [dataset]);

  const picker = (
    <div className="toolbar">
      <label htmlFor="dataset">Dataset</label>
      <select id="dataset" value={dataset} onChange={(e) => setDataset(e.target.value)}>
        {DATASETS.map((d) => (
          <option key={d.key} value={d.key}>{d.label}</option>
        ))}
      </select>
    </div>
  );

  if (error || loading || !data) {
    return (
      <>
        {picker}
        <StatusMessage loading={loading} error={error} />
      </>
    );
  }

  const { random_forest: rf, xgboost: xgb } = data.models;
  const barData = METRICS.map((m) => ({
    metric: m.label,
    "Random Forest": rf.metrics[m.key],
    XGBoost: xgb.metrics[m.key],
  }));
  const winnerLabel = MODEL_LABEL[data.best_by_f1];

  return (
    <>
      {picker}

      <div className="kpi-row">
        <Kpi label="Best model (F1)" value={winnerLabel} />
        <Kpi label="RF ROC-AUC" value={rf.metrics.roc_auc.toFixed(3)} />
        <Kpi label="XGBoost ROC-AUC" value={xgb.metrics.roc_auc.toFixed(3)} />
        <Kpi label="Test set" value={data.test_rows.toLocaleString()} hint={`positive rate ${pct(data.test_positive_rate)}`} />
      </div>

      <div className="grid">
        <Card wide title="Metric comparison" subtitle="Held-out test set, threshold 0.5">
          <div className="chart">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={barData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="metric" />
                <YAxis domain={[0, 1]} />
                <Tooltip />
                <Legend />
                <Bar dataKey="Random Forest" fill={COLORS.rf} />
                <Bar dataKey="XGBoost" fill={COLORS.xgb} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <table className="metrics-table">
            <thead>
              <tr>
                <th>Model</th>
                {METRICS.map((m) => <th key={m.key}>{m.label}</th>)}
                <th>Train time (s)</th>
              </tr>
            </thead>
            <tbody>
              {[["random_forest", rf], ["xgboost", xgb]].map(([key, model]) => (
                <tr key={key} className={key === data.best_by_f1 ? "best-row" : undefined}>
                  <td>{MODEL_LABEL[key]}</td>
                  {METRICS.map((m) => <td key={m.key}>{model.metrics[m.key].toFixed(3)}</td>)}
                  <td>{model.metrics.train_seconds.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>

        {[["Random Forest", rf], ["XGBoost", xgb]].map(([label, model]) => (
          <Card key={label} title={`${label}: confusion matrix`} subtitle="Rows = actual, columns = predicted">
            <ConfusionMatrix cm={model.confusion_matrix} />
          </Card>
        ))}

        {[["Random Forest", rf, COLORS.rf], ["XGBoost", xgb, COLORS.xgb]].map(([label, model, color]) => (
          <Card key={label} title={`${label}: top features`} subtitle="Feature importance (top 10)">
            <div className="chart">
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={model.top_features} layout="vertical" margin={{ left: 40 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis type="number" />
                  <YAxis type="category" dataKey="feature" width={150} tick={{ fontSize: 12 }} />
                  <Tooltip />
                  <Bar dataKey="importance" fill={color} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Card>
        ))}
      </div>
    </>
  );
}

function ConfusionMatrix({ cm }) {
  return (
    <div className="confusion">
      <div />
      <div className="cm-head">Pred 0</div>
      <div className="cm-head">Pred 1</div>
      <div className="cm-head">Actual 0</div>
      <div className="cm-cell good">{cm.tn.toLocaleString()}<span>TN</span></div>
      <div className="cm-cell bad">{cm.fp.toLocaleString()}<span>FP</span></div>
      <div className="cm-head">Actual 1</div>
      <div className="cm-cell bad">{cm.fn.toLocaleString()}<span>FN</span></div>
      <div className="cm-cell good">{cm.tp.toLocaleString()}<span>TP</span></div>
    </div>
  );
}
