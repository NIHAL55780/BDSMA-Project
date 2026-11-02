import { useQuery } from "@tanstack/react-query";
import { api } from "../api";

const stages = [
  { title: "Data", body: "Foursquare NYC, Gowalla (city subset), or a synthetic demo. User split for MIA; time split for utility." },
  { title: "Mechanisms", body: "M0 raw · M1 grid cloaking · M2 k-anonymity · M3 planar Laplace · M4 DP-SGD · M5 FedAvg · M6 DP-FedAvg · M7 DP histogram." },
  { title: "Tasks", body: "T1 next-POI (GRU Acc@k, MRR) and T2 hotspot detection (F1 vs the raw hotspot set)." },
  { title: "Attacks", body: "A1 spatial-temporal re-identification on released points; A2 loss-threshold membership inference on trained models." },
  { title: "DPDP", body: "Obligation → control → test. Security-safeguards pass/fail uses attack thresholds in policy.yaml." },
];

export default function Overview() {
  const ds = useQuery({ queryKey: ["datasets"], queryFn: api.datasets });
  return (
    <div className="space-y-8">
      <section className="max-w-3xl">
        <h2 className="font-serif text-3xl">A shared yardstick for location privacy</h2>
        <p className="mt-3 text-lg leading-relaxed text-ink/80">
          Location check-ins are useful for mobility and urban analytics, and easy to re-identify.
          This project runs six privacy mechanisms on the same datasets and the same two tasks,
          attacks every output, and maps the measured risk onto obligations in India’s DPDP Act, 2023.
        </p>
        <p className="mt-3 text-ink/70">
          Epsilon is not comparable across mechanisms, so Pareto plots use empirical attack success as
          the common privacy axis.
        </p>
      </section>
      <section className="grid gap-3 md:grid-cols-5">
        {stages.map((s, i) => (
          <article key={s.title} className="rounded-xl border border-ink/10 bg-white p-4">
            <p className="text-xs uppercase tracking-widest text-teal">0{i + 1}</p>
            <h3 className="mt-1 font-serif text-xl">{s.title}</h3>
            <p className="mt-2 text-sm leading-relaxed text-ink/70">{s.body}</p>
          </article>
        ))}
      </section>
      <section>
        <h3 className="font-serif text-2xl">Datasets</h3>
        <div className="mt-3 overflow-hidden rounded-xl border border-ink/10 bg-white">
          <table className="w-full text-left text-sm">
            <thead className="bg-ink/5 text-ink/70">
              <tr>
                <th className="px-4 py-2">Id</th>
                <th className="px-4 py-2">Users</th>
                <th className="px-4 py-2">POIs</th>
                <th className="px-4 py-2">Check-ins</th>
              </tr>
            </thead>
            <tbody>
              {(ds.data ?? []).map((d) => (
                <tr key={d.id} className="border-t border-ink/5">
                  <td className="px-4 py-2 font-medium">{d.name}</td>
                  <td className="px-4 py-2">{d.n_users ?? "—"}</td>
                  <td className="px-4 py-2">{d.n_pois ?? "—"}</td>
                  <td className="px-4 py-2">{d.n_checkins ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-2 text-sm text-ink/60">
          Run <code className="rounded bg-ink/5 px-1">python -m lpdp demo</code> to fill the synthetic
          row, or preprocess Foursquare NYC / Gowalla for the paper tables.
        </p>
      </section>
    </div>
  );
}
