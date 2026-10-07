// Thin wrapper around the FastAPI backend. Paths are relative so the Vite proxy handles them.

async function getJson(path) {
  const response = await fetch(path);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      detail = (await response.json()).detail ?? detail;
    } catch {
      // Non-JSON error body; keep the status text.
    }
    throw new Error(`${response.status}: ${detail}`);
  }
  return response.json();
}

export const api = {
  datasets: () => getJson("/api/datasets"),
  classification: (dataset) => getJson(`/api/classification/${dataset}`),
  timeseries: () => getJson("/api/hybrid/timeseries"),
  regression: () => getJson("/api/hybrid/regression"),
};
