export type Metric = { mean: number; std: number };
export type Run = {
  dataset: string;
  task: string;
  mechanism: string;
  param_name: string | null;
  param_value: number;
  n_seeds: number;
  metrics: Record<string, Metric>;
};

export type Dataset = {
  id: string;
  name: string;
  n_users: number | null;
  n_pois: number | null;
  n_checkins: number | null;
};

export type Mechanism = {
  id: string;
  name: string;
  acts_at: string;
  param_name: string | null;
};

async function get<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${path} ${res.status}`);
  return res.json();
}

export const api = {
  health: () => get<{ status: string }>("/health"),
  datasets: () => get<Dataset[]>("/datasets"),
  mechanisms: () => get<Mechanism[]>("/mechanisms"),
  runs: (q: { dataset?: string; task?: string; mechanism?: string } = {}) => {
    const p = new URLSearchParams();
    Object.entries(q).forEach(([k, v]) => v && p.set(k, v));
    return get<Run[]>(`/runs?${p.toString()}`);
  },
  pareto: (dataset: string, task: string, utility: string, risk: string) =>
    get<{ points: Record<string, number | string>[]; front: Record<string, number | string>[] }>(
      `/pareto?dataset=${dataset}&task=${task}&utility=${utility}&risk=${risk}`,
    ),
  compliance: (dataset: string) =>
    get<{
      items: { id: string; obligation: string; act_reference: string; control: string; test: string }[];
      status: { item_id: string; mechanism_id: string; status: string; evidence: string; obligation: string }[];
    }>(`/compliance?dataset=${dataset}`),
  perturb: (body: { mechanism: string; param: number; n_points: number }) =>
    fetch("/perturb", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then((r) => r.json()),
  recommend: (body: { task: string; max_risk: number; min_utility: number }) =>
    fetch("/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then((r) => r.json()),
};
