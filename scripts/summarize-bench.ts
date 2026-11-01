// Prints the results of a fetched dedibox run as Markdown tables: one table per benchmark group,
// one column per configuration (build target and threads), and the median time of every
// benchmark with its speedup over binom_split (or over thread_rng for the generators).
//
//   bun run scripts/summarize-bench.ts bench-results/dedibox/<run>

import { Glob } from "bun";
import { dirname, basename, join } from "node:path";

const run = process.argv[2];
if (!run) {
  console.error("usage: bun run scripts/summarize-bench.ts bench-results/dedibox/<run>");
  process.exit(2);
}

type BenchmarkJson = { group_id: string; function_id: string | null; value_str: string | null };

// group -> row -> configuration -> median in ns
const medians = new Map<string, Map<string, Map<string, number>>>();
const configurations = new Set<string>();

for await (const path of new Glob("results/criterion-*/**/benchmark.json").scan(run)) {
  const dir = join(run, dirname(path));
  const configuration = basename(dir);
  // Criterion keeps the latest run in new/ and the previous one in base/, next to the baselines.
  if (configuration === "new" || configuration === "base" || configuration === "change") continue;
  const info: BenchmarkJson = await Bun.file(join(dir, "benchmark.json")).json();
  const estimates = await Bun.file(join(dir, "estimates.json")).json();
  const row = [info.function_id, info.value_str].filter((s) => s).join(" @ ");
  configurations.add(configuration);
  const group = medians.get(info.group_id) ?? new Map();
  medians.set(info.group_id, group);
  const cells = group.get(row) ?? new Map();
  group.set(row, cells);
  cells.set(configuration, estimates.median.point_estimate);
}

const columns = [...configurations].sort();

function format(ns: number): string {
  if (ns >= 1e6) return `${(ns / 1e6).toFixed(1)} ms`;
  if (ns >= 1e3) return `${(ns / 1e3).toFixed(1)} µs`;
  return `${ns.toFixed(0)} ns`;
}

for (const [group, rows] of [...medians].sort(([a], [b]) => a.localeCompare(b))) {
  console.log(`\n### ${group}\n`);
  console.log(`| benchmark | ${columns.join(" | ")} |`);
  console.log(`|:--|${columns.map(() => "--:").join("|")}|`);
  const base = group.startsWith("rng_fill") ? "thread_rng" : group.startsWith("pois") ? "pois_gen" : "binom_split";
  for (const [row, cells] of [...rows].sort(([a], [b]) => a.localeCompare(b, undefined, { numeric: true }))) {
    const [, value] = row.split(" @ ");
    const baseRow = rows.get(value ? `${base} @ ${value}` : base);
    const text = columns.map((c) => {
      const ns = cells.get(c);
      if (ns === undefined) return "";
      const ref = baseRow?.get(c);
      return ref ? `${format(ns)} (${(ref / ns).toFixed(1)}×)` : format(ns);
    });
    console.log(`| ${row} | ${text.join(" | ")} |`);
  }
}
