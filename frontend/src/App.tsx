import { NavLink, Route, Routes } from "react-router-dom";
import Overview from "./pages/Overview";
import Tradeoff from "./pages/Tradeoff";
import MapDemo from "./pages/MapDemo";
import Checklist from "./pages/Checklist";
import Recommender from "./pages/Recommender";

const links = [
  { to: "/", label: "Overview" },
  { to: "/tradeoff", label: "Trade-off" },
  { to: "/map", label: "Map demo" },
  { to: "/checklist", label: "DPDP checklist" },
  { to: "/recommend", label: "Recommender" },
];

export default function App() {
  return (
    <div className="min-h-screen">
      <header className="border-b border-ink/10 bg-ink text-paper">
        <div className="mx-auto flex max-w-6xl flex-wrap items-end justify-between gap-4 px-6 py-5">
          <div>
            <p className="text-xs uppercase tracking-[0.2em] text-paper/60">BDSMA · Paper 6</p>
            <h1 className="font-serif text-2xl">Location privacy under the DPDP Act</h1>
          </div>
          <nav className="flex flex-wrap gap-1">
            {links.map((l) => (
              <NavLink
                key={l.to}
                to={l.to}
                end={l.to === "/"}
                className={({ isActive }) =>
                  `rounded-full px-3 py-1.5 text-sm ${isActive ? "bg-paper text-ink" : "text-paper/80 hover:bg-paper/10"}`
                }
              >
                {l.label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>
      <main className="mx-auto max-w-6xl px-6 py-8">
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/tradeoff" element={<Tradeoff />} />
          <Route path="/map" element={<MapDemo />} />
          <Route path="/checklist" element={<Checklist />} />
          <Route path="/recommend" element={<Recommender />} />
        </Routes>
      </main>
      <footer className="mx-auto max-w-6xl px-6 pb-10 text-sm text-ink/60">
        Indicative academic mapping, not legal advice. Numbers come from{" "}
        <code className="rounded bg-ink/5 px-1">results/runs.csv</code> — never invented.
      </footer>
    </div>
  );
}
