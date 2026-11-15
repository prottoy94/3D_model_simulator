/** Times a full simulation of the three gallery models: pnpm tsx scripts/bench-sim.mts */
import { buildRuntimeIR } from "../src/core/ir/build";
import { alexnetIR, vgg16IR, miniVitIR } from "../src/core/ir/examples";
import { simulate } from "../src/core/sim/run";
import { summarize } from "../src/core/sim/view";
for (const [name, make, id] of [["vit", miniVitIR, "m_vit"], ["alexnet", alexnetIR, "m_alexnet"], ["vgg16", vgg16IR, "m_vgg16"]] as const) {
  const built = buildRuntimeIR(make());
  if (!built.ok) throw new Error("bad");
  const t = performance.now();
  let slow = "";
  let worst = 0;
  let viewMs = 0;
  for (const r of simulate(built.ir, id, { kind: "shapes" })) {
    if (r.ms > worst) { worst = r.ms; slow = r.nodeId; }
    const v = performance.now();
    summarize(r, built.ir.nodes[r.nodeId]!.op === "input");
    viewMs += performance.now() - v;
  }
  console.log(name, "total", ((performance.now() - t) / 1000).toFixed(1), "s; slowest", slow, (worst / 1000).toFixed(1), "s; views", (viewMs / 1000).toFixed(2), "s");
}
