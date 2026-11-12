/** Cola de jobs: uno a la vez, estado en data/jobs.json.
 *
 * De 1 en 1 a proposito, no por simplicidad: hay una GPU con 8 GB y dos
 * separaciones simultaneas la revientan. El cuello de botella es fisico, asi
 * que no hay nada que escalar y un array con un while basta. Nada de Redis.
 */
import { randomUUID } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import * as worker from "./worker.ts";
import type { Analysis, Job, Preset, Stem } from "../types.ts";

export const DATA = join(import.meta.dirname, "..", "data");
const DB = join(DATA, "jobs.json");
const LIVE: Job["state"][] = ["preparing", "analyzing", "separating", "refining"];

const jobs = new Map<string, Job>();
const listeners = new Set<(j: Job) => void>();
let running = false;

export const dirOf = (id: string) => join(DATA, id);

/** El archivo subido, cuya extension no conocemos de antemano. */
function inputOf(id: string): string | null {
  const dir = dirOf(id);
  if (!existsSync(dir)) return null;
  const f = readdirSync(dir).find((f) => f.startsWith("input."));
  return f ? join(dir, f) : null;
}

mkdirSync(DATA, { recursive: true });
load();

function load() {
  if (!existsSync(DB)) return;
  try {
    for (const j of JSON.parse(readFileSync(DB, "utf8")) as Job[]) {
      // Un job a medio procesar no se puede reanudar: su estado se fue con el
      // proceso Python. Los que seguian en cola si sobreviven, porque el
      // archivo de entrada esta en disco.
      if (LIVE.includes(j.state)) {
        j.state = "error";
        j.error = "interrumpido al reiniciar el servidor";
      }
      jobs.set(j.id, j);
    }
  } catch (e) {
    console.error("jobs.json ilegible, se empieza de cero:", e);
  }
}

const save = () => writeFileSync(DB, JSON.stringify([...jobs.values()], null, 2));

function update(j: Job, patch: Partial<Job>) {
  Object.assign(j, patch);
  // El progreso tickea muchas veces por segundo: no se persiste, solo se emite.
  if (!Object.keys(patch).every((k) => k === "progress" || k === "stage")) save();
  for (const l of listeners) l(j);
}

export const list = () => [...jobs.values()].sort((a, b) => b.createdAt - a.createdAt);
export const get = (id: string) => jobs.get(id);

export function onUpdate(fn: (j: Job) => void) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

export function create(name: string, preset: Preset): Job {
  const job: Job = {
    id: randomUUID().slice(0, 8),
    name,
    preset,
    state: "queued",
    progress: 0,
    createdAt: Date.now(),
  };
  jobs.set(job.id, job);
  mkdirSync(dirOf(job.id), { recursive: true });
  save();
  for (const l of listeners) l(job);
  return job;
}

export function remove(id: string) {
  const j = jobs.get(id);
  if (!j) return false;
  if (LIVE.includes(j.state)) return false; // no se borra lo que la GPU esta usando
  jobs.delete(id);
  rmSync(dirOf(id), { recursive: true, force: true });
  save();
  return true;
}

export function enqueue(id: string) {
  const j = jobs.get(id);
  if (!j) return;
  update(j, { state: "queued", progress: 0, error: undefined });
  void pump();
}

async function runJob(j: Job, input: string) {
  const dir = dirOf(j.id);
  const mix = join(dir, "mix.flac");
  try {
    // 1. decodificar UNA vez a 44.1k lossless. Todo lo demas parte de aqui.
    update(j, { state: "preparing", progress: 0, stage: "decodificando a 44.1 kHz" });
    const prep = await worker.call<{ duration: number; sr_in: number }>({
      cmd: "prepare", src: input, out: mix,
    });

    // 2. BPM y key del mix: lo que el usuario ve en los primeros segundos.
    update(j, { state: "analyzing", progress: 0, stage: "BPM y key" });
    const analysis = await worker.call<Analysis>({ cmd: "analyze", path: mix });
    update(j, { analysis: { ...analysis, duration: prep.duration } });

    // 3. separacion (la parte larga).
    update(j, { state: "separating", progress: 0 });
    const sep = await worker.call<{ stems: Record<Stem, string>; residual_db: number }>(
      { cmd: "separate", path: mix, out: dir, preset: j.preset },
      (progress, stage) => update(j, { progress, stage }),
    );

    // 4. refinado gratis: BPM sobre drums, key sobre bass+other.
    update(j, {
      state: "refining",
      progress: 0,
      stage: "recalculando sobre los stems",
      stems: Object.keys(sep.stems) as Stem[],
      residualDb: sep.residual_db,
    });
    const refined = await worker.call<Analysis>({ cmd: "analyze_stems", stems: sep.stems });

    update(j, { state: "done", progress: 1, stage: undefined, refined });
  } catch (e) {
    update(j, { state: "error", stage: undefined, error: (e as Error).message });
  }
}

async function pump() {
  if (running) return;
  running = true;
  try {
    for (;;) {
      const j = list().reverse().find((x) => x.state === "queued");
      if (!j) break;
      const input = inputOf(j.id);
      if (!input) {
        update(j, { state: "error", error: "falta el archivo de entrada en disco" });
        continue;
      }
      await runJob(j, input);
    }
  } finally {
    running = false;
  }
}

// Reanudar lo que quedo en cola de una ejecucion anterior.
if (list().some((j) => j.state === "queued")) void pump();
