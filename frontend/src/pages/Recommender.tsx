import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { api } from "../api";

type Rec = {
  dataset: string;
  mechanism: string;
  param_name: string;
  param_value: number;
  utility: number;
  risk: number;
};

export default function Recommender() {
  const [task, setTask] = useState("next_poi");
  const [maxRisk, setMaxRisk] = useState(0.55);
  const [minUtil, setMinUtil] = useState(0.1);
  const mut = useMutation({
    mutationFn: () => api.recommend({ task, max_risk: maxRisk, min_utility: minUtil }),
  });
  const rows: Rec[] = mut.data ?? [];

  return (
    <div className="space-y-5">
      <p className="max-w-2xl text-ink/75">
        Pick a use case. Mechanisms that meet both the utility floor and the risk ceiling are ranked
        by utility. Thresholds are project policy, not legal standards.
      </p>
      <form
        className="flex flex-wrap items-end gap-3 rounded-xl border border-ink/10 bg-white p-4"
        onSubmit={(e) => {
          e.preventDefault();
          mut.mutate();
        }}
      >
        <label className="text-sm">
          Task
          <select
            className="ml-2 rounded border border-ink/20 px-2 py-1"
            value={task}
            onChange={(e) => setTask(e.target.value)}
          >
            <option value="next_poi">next-POI</option>
            <option value="hotspot">hotspot</option>
          </select>
        </label>
        <label className="text-sm">
          Min utility
          <input
            type="number"
            step="0.01"
            className="ml-2 w-24 rounded border border-ink/20 px-2 py-1"
            value={minUtil}
            onChange={(e) => setMinUtil(Number(e.target.value))}
          />
        </label>
        <label className="text-sm">
          Max risk
          <input
            type="number"
            step="0.01"
            className="ml-2 w-24 rounded border border-ink/20 px-2 py-1"
            value={maxRisk}
            onChange={(e) => setMaxRisk(Number(e.target.value))}
          />
        </label>
        <button className="rounded-full bg-ink px-4 py-1.5 text-sm text-paper" type="submit">
          Recommend
        </button>
      </form>
      {mut.isSuccess && rows.length === 0 && (
        <p className="text-sm text-ink/60">No mechanism met both constraints on the logged runs.</p>
      )}
      {rows.length > 0 && (
        <table className="w-full overflow-hidden rounded-xl border border-ink/10 bg-white text-left text-sm">
          <thead className="bg-ink/5">
            <tr>
              <th className="px-3 py-2">#</th>
              <th className="px-3 py-2">Mechanism</th>
              <th className="px-3 py-2">Parameter</th>
              <th className="px-3 py-2">Utility</th>
              <th className="px-3 py-2">Risk</th>
              <th className="px-3 py-2">Dataset</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={`${r.mechanism}-${r.param_value}-${i}`} className="border-t border-ink/5">
                <td className="px-3 py-2">{i + 1}</td>
                <td className="px-3 py-2 font-medium">{r.mechanism}</td>
                <td className="px-3 py-2">
                  {r.param_name} = {r.param_value}
                </td>
                <td className="px-3 py-2">{r.utility.toFixed(3)}</td>
                <td className="px-3 py-2">{r.risk.toFixed(3)}</td>
                <td className="px-3 py-2">{r.dataset}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
