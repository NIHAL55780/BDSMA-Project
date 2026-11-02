import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "../api";

const MECHS = ["M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7"];
const TONE: Record<string, string> = {
  pass: "bg-teal/15 text-teal",
  partial: "bg-amber-100 text-amber-800",
  fail: "bg-accent/15 text-accent",
  na: "bg-ink/5 text-ink/50",
};

export default function Checklist() {
  const [dataset, setDataset] = useState("synthetic");
  const q = useQuery({ queryKey: ["compliance", dataset], queryFn: () => api.compliance(dataset) });
  const items = q.data?.items ?? [];
  const lookup = useMemo(() => {
    const m = new Map<string, { status: string; evidence: string }>();
    for (const row of q.data?.status ?? []) {
      m.set(`${row.item_id}|${row.mechanism_id}`, { status: row.status, evidence: row.evidence });
    }
    return m;
  }, [q.data]);

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <p className="max-w-2xl text-ink/75">
          Fig. 9 — obligation × mechanism. Cells are computed from checklist rules and attack results,
          not filled in by hand. This is an academic mapping, not a legal opinion.
        </p>
        <select
          className="rounded border border-ink/20 bg-white px-2 py-1 text-sm"
          value={dataset}
          onChange={(e) => setDataset(e.target.value)}
        >
          <option value="synthetic">synthetic</option>
          <option value="foursquare_nyc">foursquare_nyc</option>
          <option value="gowalla">gowalla</option>
        </select>
      </div>
      <div className="overflow-x-auto rounded-xl border border-ink/10 bg-white">
        <table className="w-full min-w-[720px] text-left text-sm">
          <thead className="bg-ink/5">
            <tr>
              <th className="px-3 py-2">Obligation</th>
              {MECHS.map((m) => (
                <th key={m} className="px-2 py-2 text-center">
                  {m}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {items.map((it) => (
              <tr key={it.id} className="border-t border-ink/5">
                <td className="px-3 py-2">
                  <div className="font-medium">{it.obligation}</div>
                  <div className="text-xs text-ink/50">{it.act_reference}</div>
                </td>
                {MECHS.map((m) => {
                  const cell = lookup.get(`${it.id}|${m}`);
                  const st = cell?.status ?? "—";
                  return (
                    <td key={m} className="px-1 py-2 text-center" title={cell?.evidence}>
                      <span className={`inline-block rounded-full px-2 py-0.5 text-xs ${TONE[st] ?? "bg-ink/5"}`}>
                        {st}
                      </span>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-xs text-ink/50">Hover a cell for evidence. DPDP Rules 2025 / fiduciary dates are documented in the paper outline.</p>
    </div>
  );
}
