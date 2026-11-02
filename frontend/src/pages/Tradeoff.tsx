import { useMemo, useState, type ReactNode } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import { api, type Run } from "../api";

const MECH_COLOR: Record<string, string> = {
  M0: "#1d3557",
  M1: "#457b9d",
  M2: "#2a9d8f",
  M3: "#c9a227",
  M4: "#c45c26",
  M5: "#6d597a",
  M6: "#b56576",
  M7: "#4a4e69",
};

function metric(run: Run, name: string): number | null {
  return run.metrics[name]?.mean ?? null;
}

export default function Tradeoff() {
  const [dataset, setDataset] = useState("synthetic");
  const [task, setTask] = useState<"next_poi" | "hotspot">("next_poi");
  const runsQ = useQuery({ queryKey: ["runs", dataset, task], queryFn: () => api.runs({ dataset, task }) });
  const utilKey = task === "next_poi" ? "acc10" : "hotspot_f1";
  const riskKey = task === "next_poi" ? "mia_auc" : "reid_top1_p4";
  const paretoQ = useQuery({
    queryKey: ["pareto", dataset, task, utilKey, riskKey],
    queryFn: () => api.pareto(dataset, task, utilKey, riskKey),
  });

  const byMech = useMemo(() => {
    const map = new Map<string, { param_value: number; util: number; risk: number | null }[]>();
    for (const r of runsQ.data ?? []) {
      const util = metric(r, utilKey);
      if (util === null) continue;
      const arr = map.get(r.mechanism) ?? [];
      arr.push({ param_value: r.param_value, util, risk: metric(r, riskKey) });
      map.set(r.mechanism, arr);
    }
    for (const arr of map.values()) arr.sort((a, b) => a.param_value - b.param_value);
    return map;
  }, [runsQ.data, utilKey, riskKey]);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end gap-4">
        <label className="text-sm">
          Dataset
          <select
            className="ml-2 rounded border border-ink/20 bg-white px-2 py-1"
            value={dataset}
            onChange={(e) => setDataset(e.target.value)}
          >
            <option value="synthetic">synthetic</option>
            <option value="foursquare_nyc">foursquare_nyc</option>
            <option value="gowalla">gowalla</option>
          </select>
        </label>
        <label className="text-sm">
          Task
          <select
            className="ml-2 rounded border border-ink/20 bg-white px-2 py-1"
            value={task}
            onChange={(e) => setTask(e.target.value as "next_poi" | "hotspot")}
          >
            <option value="next_poi">next-POI</option>
            <option value="hotspot">hotspot</option>
          </select>
        </label>
      </div>
      {!runsQ.data?.length && (
        <p className="rounded-lg border border-accent/30 bg-accent/10 px-4 py-3 text-sm">
          No runs yet. From the repo root: <code>python -m lpdp demo</code>
        </p>
      )}
      <div className="grid gap-6 lg:grid-cols-2">
        <ChartCard title={`Fig. 5 — ${task === "next_poi" ? "Acc@10" : "Hotspot F1"} vs parameter`}>
          <ResponsiveContainer width="100%" height={280}>
            <LineChart>
              <CartesianGrid strokeDasharray="3 3" stroke="#ddd" />
              <XAxis dataKey="param_value" type="number" allowDuplicatedCategory={false} />
              <YAxis />
              <Tooltip />
              <Legend />
              {[...byMech.entries()].map(([mech, rows]) => (
                <Line
                  key={mech}
                  data={rows}
                  dataKey="util"
                  name={mech}
                  stroke={MECH_COLOR[mech] ?? "#333"}
                  dot
                  type="monotone"
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>
        <ChartCard title={`Fig. 6 — ${riskKey} vs parameter`}>
          <ResponsiveContainer width="100%" height={280}>
            <LineChart>
              <CartesianGrid strokeDasharray="3 3" stroke="#ddd" />
              <XAxis dataKey="param_value" type="number" allowDuplicatedCategory={false} />
              <YAxis />
              <Tooltip />
              <Legend />
              {[...byMech.entries()].map(([mech, rows]) => (
                <Line
                  key={mech}
                  data={rows.filter((r) => r.risk !== null)}
                  dataKey="risk"
                  name={mech}
                  stroke={MECH_COLOR[mech] ?? "#333"}
                  dot
                  type="monotone"
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>
      <ChartCard title="Fig. 7 — Pareto front (utility vs empirical risk)">
        <ResponsiveContainer width="100%" height={320}>
          <ScatterChart>
            <CartesianGrid strokeDasharray="3 3" stroke="#ddd" />
            <XAxis dataKey="risk" name={riskKey} type="number" />
            <YAxis dataKey="utility" name={utilKey} type="number" />
            <ZAxis range={[60, 60]} />
            <Tooltip />
            <Legend />
            <Scatter
              name="all"
              data={(paretoQ.data?.points ?? []).map((p) => ({
                ...p,
                utility: p[utilKey],
                risk: p[riskKey],
              }))}
              fill="#457b9d"
            />
            <Scatter
              name="Pareto"
              data={(paretoQ.data?.front ?? []).map((p) => ({
                ...p,
                utility: p[utilKey],
                risk: p[riskKey],
              }))}
              fill="#c45c26"
            />
          </ScatterChart>
        </ResponsiveContainer>
      </ChartCard>
    </div>
  );
}

function ChartCard({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="rounded-xl border border-ink/10 bg-white p-4">
      <h3 className="mb-3 font-serif text-lg">{title}</h3>
      {children}
    </section>
  );
}
