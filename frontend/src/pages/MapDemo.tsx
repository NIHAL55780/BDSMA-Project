import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { CircleMarker, MapContainer, TileLayer, Tooltip } from "react-leaflet";
import { api } from "../api";

type Pt = { name: string; lat: number; lon: number };

export default function MapDemo() {
  const [mechanism, setMechanism] = useState("M3");
  const [param, setParam] = useState(2);
  const mut = useMutation({
    mutationFn: () => api.perturb({ mechanism, param, n_points: 24 }),
  });
  const raw: Pt[] = mut.data?.raw ?? [];
  const pert: Pt[] = mut.data?.perturbed ?? [];

  return (
    <div className="space-y-4">
      <p className="max-w-2xl text-ink/75">
        Public NYC landmarks (not user check-ins) run through a release mechanism. M1 snaps to a grid
        cell; M3 adds planar Laplace noise then snaps to the nearest landmark.
      </p>
      <div className="flex flex-wrap items-end gap-3">
        <label className="text-sm">
          Mechanism
          <select
            className="ml-2 rounded border border-ink/20 bg-white px-2 py-1"
            value={mechanism}
            onChange={(e) => setMechanism(e.target.value)}
          >
            <option value="M0">M0 raw</option>
            <option value="M1">M1 cloaking (metres)</option>
            <option value="M3">M3 geo-ind (ε per km)</option>
          </select>
        </label>
        <label className="text-sm">
          Parameter
          <input
            type="number"
            className="ml-2 w-24 rounded border border-ink/20 bg-white px-2 py-1"
            value={param}
            onChange={(e) => setParam(Number(e.target.value))}
          />
        </label>
        <button
          className="rounded-full bg-ink px-4 py-1.5 text-sm text-paper"
          onClick={() => mut.mutate()}
        >
          Perturb
        </button>
      </div>
      <div className="h-[480px] overflow-hidden rounded-xl border border-ink/10">
        <MapContainer center={[40.75, -73.99]} zoom={12} scrollWheelZoom>
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a>'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          {raw.map((p) => (
            <CircleMarker key={`r-${p.name}`} center={[p.lat, p.lon]} radius={7} pathOptions={{ color: "#12263a" }}>
              <Tooltip>{p.name} (raw)</Tooltip>
            </CircleMarker>
          ))}
          {pert.map((p, i) => (
            <CircleMarker key={`p-${p.name}-${i}`} center={[p.lat, p.lon]} radius={7} pathOptions={{ color: "#c45c26" }}>
              <Tooltip>{p.name} (perturbed)</Tooltip>
            </CircleMarker>
          ))}
        </MapContainer>
      </div>
      <p className="text-sm text-ink/60">Navy = raw landmark. Amber = released location.</p>
    </div>
  );
}
