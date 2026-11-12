/** Puente con el proceso Python.
 *
 * El proceso NO muere entre canciones: cargar un checkpoint de RoFormer cuesta
 * 5-20s y con una cola eso se pagaria en cada archivo. Arranca una vez y se
 * queda leyendo lineas JSON de stdin.
 *
 * Protocolo (una linea JSON por mensaje):
 *   ->  {"cmd":"separate","path":"...","out":"...","preset":"max"}
 *   <-  {"progress":0.42,"stage":"roformer"}      (0..n veces)
 *   <-  {"result":{...}}  |  {"error":"..."}      (exactamente 1, cierra)
 */
import { spawn, type ChildProcess } from "node:child_process";
import { createInterface } from "node:readline";
import { homedir } from "node:os";
import { join } from "node:path";

const WIN = process.platform === "win32";
const PYTHON =
  process.env.STEMTOOL_PYTHON ??
  join(homedir(), ".venvs", "stemtool", WIN ? "Scripts" : "bin",
       WIN ? "python.exe" : "python");
const SCRIPT = join(import.meta.dirname, "..", "py", "worker.py");

export type OnProgress = (progress: number, stage?: string) => void;

let proc: ChildProcess | null = null;
let pending:
  | { resolve: (v: any) => void; reject: (e: Error) => void; onProgress?: OnProgress }
  | null = null;

function settle(fn: "resolve" | "reject", value: any) {
  const p = pending;
  pending = null;
  p?.[fn](value);
}

function start() {
  proc = spawn(PYTHON, ["-u", SCRIPT], { stdio: ["pipe", "pipe", "pipe"] });

  createInterface({ input: proc.stdout! }).on("line", (line) => {
    if (!line.trim()) return;
    let msg: any;
    try {
      msg = JSON.parse(line);
    } catch {
      // stdout contaminado (un print suelto, un warning de torch): se registra
      // y se ignora, no se mata el job por eso.
      console.error(`[py] ${line}`);
      return;
    }
    if (msg.progress !== undefined) pending?.onProgress?.(msg.progress, msg.stage);
    else if (msg.error !== undefined) settle("reject", new Error(msg.error));
    else if (msg.result !== undefined) settle("resolve", msg.result);
    else if (msg.log !== undefined) console.error(`[py] ${msg.log}`);
  });

  proc.stderr!.on("data", (d) => process.stderr.write(`[py] ${d}`));

  proc.on("exit", (code, signal) => {
    proc = null;
    settle("reject", new Error(`el worker Python murio (code=${code} signal=${signal})`));
  });
  proc.on("error", (e) => {
    proc = null;
    settle("reject", new Error(`no se pudo arrancar Python en ${PYTHON}: ${e.message}`));
  });
}

/** Una sola peticion en vuelo: la cola secuencial ya lo garantiza. */
export function call<T = any>(
  req: Record<string, unknown> & { cmd: string },
  onProgress?: OnProgress,
): Promise<T> {
  if (!proc) start();
  if (pending) return Promise.reject(new Error("worker ocupado"));
  return new Promise<T>((resolve, reject) => {
    pending = { resolve, reject, onProgress };
    proc!.stdin!.write(JSON.stringify(req) + "\n");
  });
}

export function stop() {
  proc?.kill();
  proc = null;
}

/** Proceso aparte y efimero para tareas baratas (zip, export).
 *
 * No van por el worker persistente a proposito: ese puede estar 5 minutos
 * ocupado con la GPU, y comprimir cuatro FLAC no tiene por que esperar.
 */
export function oneShot(args: string[]): Promise<void> {
  return new Promise((resolve, reject) => {
    const p = spawn(PYTHON, [SCRIPT, ...args], { stdio: ["ignore", "ignore", "pipe"] });
    let err = "";
    p.stderr!.on("data", (d) => (err += d));
    p.on("error", reject);
    p.on("exit", (code) =>
      code === 0 ? resolve() : reject(new Error(err.trim() || `salio con code ${code}`)),
    );
  });
}

/** Arranca Python ya, para que el primer job no pague los ~10s de import. */
export function warmUp() {
  if (!proc) start();
}
