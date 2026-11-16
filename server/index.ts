/** Servidor local. node:http a pelo, cero dependencias de runtime.
 *
 * La subida es PUT con el archivo como cuerpo crudo, no multipart: el navegador
 * manda un File como body de un fetch sin mas, y asi no hay que parsear
 * boundaries (~80 lineas delicadas) para un solo archivo y dos parametros.
 */
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { createReadStream, createWriteStream, existsSync, statSync } from "node:fs";
import { pipeline } from "node:stream/promises";
import { basename, extname, join, resolve, sep } from "node:path";
import * as q from "./queue.ts";
import * as worker from "./worker.ts";
import { STEMS, bestAnalysis, type Job, type Preset, type Stem } from "../types.ts";

const PORT = Number(process.env.PORT ?? 4747);
const WEB = resolve(import.meta.dirname, "..", "web");
const MAX_BYTES = 600 * 1024 * 1024;

// Frontera de confianza: lo que llega del navegador se valida contra listas
// cerradas, nunca se concatena a una ruta tal cual.
const AUDIO_EXT = new Set([".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg",
                           ".opus", ".aif", ".aiff", ".wma"]);
const MIME: Record<string, string> = {
  ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8", ".json": "application/json",
  ".flac": "audio/flac", ".wav": "audio/wav", ".mp3": "audio/mpeg",
  ".zip": "application/zip", ".svg": "image/svg+xml",
  ".ttf": "font/ttf", ".woff2": "font/woff2",
};

const json = (res: ServerResponse, code: number, body: unknown) => {
  res.writeHead(code, { "content-type": "application/json; charset=utf-8" });
  res.end(JSON.stringify(body));
};

/** Resuelve rel dentro de root, o null si se escapa (path traversal). */
function safeJoin(root: string, rel: string): string | null {
  const p = resolve(root, "." + (rel.startsWith("/") ? rel : "/" + rel));
  return p === root || p.startsWith(root + sep) ? p : null;
}

function sendFile(res: ServerResponse, path: string, downloadAs?: string) {
  if (!existsSync(path) || !statSync(path).isFile()) return json(res, 404, { error: "no existe" });
  const headers: Record<string, string> = {
    "content-type": MIME[extname(path).toLowerCase()] ?? "application/octet-stream",
    "content-length": String(statSync(path).size),
  };
  if (downloadAs) {
    headers["content-disposition"] =
      `attachment; filename*=UTF-8''${encodeURIComponent(downloadAs)}`;
  }
  res.writeHead(200, headers);
  createReadStream(path).pipe(res);
}

const stemOf = (s: string | null): Stem | null =>
  (STEMS as readonly string[]).includes(s ?? "") ? (s as Stem) : null;

/** "cancion [120 BPM, A minor]": el analisis viaja en el nombre del archivo.
 *
 * Se usa el refinado cuando existe (bestAnalysis), que es el que mide el BPM
 * sobre la bateria sola. Si el analisis fallo, se queda el nombre pelado en
 * vez de meter un "undefined BPM" en el titulo.
 */
function titleOf(job: Job) {
  const base = job.name.replace(/\.[^.]+$/, "");
  const a = bestAnalysis(job);
  const tags: string[] = [];
  // Un decimal, el mismo que muestra la tarjeta: redondear a entero dejaria
  // "125 BPM" en el nombre de una cancion que la UI llama de 124.5.
  if (a.bpm) tags.push(`${a.bpm.toFixed(1)} BPM`);
  if (a.key) tags.push(a.key);
  return tags.length ? `${base} [${tags.join(", ")}]` : base;
}

async function upload(req: IncomingMessage, res: ServerResponse, url: URL) {
  const name = basename(url.searchParams.get("name") ?? "audio");
  const preset: Preset = url.searchParams.get("preset") === "max" ? "max" : "fast";
  const ext = extname(name).toLowerCase();
  if (!AUDIO_EXT.has(ext)) {
    return json(res, 415, { error: `extension no soportada: ${ext || "(ninguna)"}` });
  }
  if (Number(req.headers["content-length"] ?? 0) > MAX_BYTES) {
    return json(res, 413, { error: "archivo demasiado grande" });
  }

  const job = q.create(name, preset);
  let seen = 0;
  req.on("data", (c: Buffer) => {
    seen += c.length;
    if (seen > MAX_BYTES) req.destroy(new Error("archivo demasiado grande"));
  });
  try {
    await pipeline(req, createWriteStream(join(q.dirOf(job.id), "input" + ext)));
  } catch (e) {
    q.remove(job.id);
    return json(res, 413, { error: (e as Error).message });
  }
  q.enqueue(job.id);
  json(res, 202, { id: job.id });
}

function sse(res: ServerResponse) {
  res.writeHead(200, {
    "content-type": "text/event-stream; charset=utf-8",
    "cache-control": "no-cache",
    connection: "keep-alive",
  });
  const send = (j: unknown) => res.write(`data: ${JSON.stringify(j)}\n\n`);
  // Del mas viejo al mas nuevo: el cliente hace prepend de cada tarjeta, asi
  // que mandarlos en orden descendente dejaria la lista justo al reves.
  for (const j of q.list().reverse()) send(j);
  const off = q.onUpdate(send);
  // Un comentario cada 25s para que ningun proxy ni el SO corten el socket.
  const ping = setInterval(() => res.write(": ping\n\n"), 25_000);
  res.on("close", () => {
    off();
    clearInterval(ping);
  });
}

const FORMATS = { flac: ".flac", wav24: ".wav", mp3: ".mp3" } as const;
type Format = keyof typeof FORMATS;

const formatOf = (s: string | null): Format =>
  s === "mp3" || s === "wav24" ? s : "flac";

/** Un stem convertido al formato pedido, cacheado en disco. */
async function materialize(dir: string, base: string, format: Format) {
  const src = join(dir, `${base}.flac`);
  if (!existsSync(src)) return null;
  if (format === "flac") return src;
  const out = join(dir, `${base}${FORMATS[format]}`);
  if (!existsSync(out)) await worker.oneShot(["export", src, out, format]);
  return out;
}

/** Metronomo al BPM detectado, generado la primera vez que se pide.
 *
 * Se genera aqui y no al separar porque depende del BPM refinado (el que se
 * mide sobre la bateria sola) y porque cuesta medio segundo: no hay nada que
 * adelantar.
 */
async function clickTrack(dir: string, job: Job, format: Format) {
  const a = bestAnalysis(job);
  if (!a.bpm || !a.duration) return null;
  // El BPM va en el nombre del cacheado: si el analisis cambia, el archivo es
  // otro. Con un "click.flac" fijo, un reanalisis deja en disco un metronomo
  // al tempo viejo y se sirve tal cual -- justo el error que este archivo
  // existe para delatar.
  const base = `click-${a.bpm}`;
  const src = join(dir, `${base}.flac`);
  if (!existsSync(src)) {
    await worker.oneShot(["click", src, String(a.bpm),
                          String(a.beat_offset ?? 0), String(a.duration)]);
  }
  return materialize(dir, base, format);
}

/** Descarga de los stems marcados: uno va directo, varios en un zip.
 *
 * El formato se aplica aqui y no en el cliente porque el navegador bloquea
 * varias descargas programaticas seguidas: mejor un zip que cinco clics.
 */
async function download(res: ServerResponse, id: string, url: URL) {
  const job = q.get(id);
  if (!job) return json(res, 404, { error: "job no existe" });
  const format = formatOf(url.searchParams.get("format"));
  const stems = (url.searchParams.get("pick") ?? "")
    .split(",").map(stemOf).filter((s): s is Stem => s !== null);

  const dir = q.dirOf(id);
  const title = titleOf(job);
  // Se emparejan stem y ruta ANTES de filtrar: con un indice sobre el array ya
  // filtrado, si falta un stem los nombres se desplazan y salen cruzados.
  const files = (await Promise.all(stems.map(
    async (s): Promise<[string, string] | null> => {
      const f = await materialize(dir, s, format);
      return f ? [`${title} - ${s}${FORMATS[format]}`, f] : null;
    }))).filter((p): p is [string, string] => p !== null);

  if (url.searchParams.get("click") === "1") {
    const click = await clickTrack(dir, job, format);
    if (click) files.push([`${title} - metronomo${FORMATS[format]}`, click]);
  }
  if (!files.length) return json(res, 404, { error: "no hay nada que descargar" });
  if (files.length === 1) return sendFile(res, files[0][1], files[0][0]);

  const zip = join(dir, `stems-${format}.zip`);
  // Pares nombre/ruta: los nombres de dentro del zip tambien llevan BPM y key,
  // que es lo que queda al descomprimir en una carpeta con otras canciones.
  await worker.oneShot(["zip", zip, ...files.flat()]);
  sendFile(res, zip, `${title} - stems.zip`);
}

const ROUTE = /^\/api\/jobs\/([0-9a-f]{4,12})(?:\/(stem|mix|download))?$/;

createServer(async (req, res) => {
  const url = new URL(req.url ?? "/", `http://${req.headers.host ?? "localhost"}`);
  const path = url.pathname;
  try {
    if (path === "/api/jobs") {
      if (req.method === "PUT") return await upload(req, res, url);
      if (req.method === "GET") return json(res, 200, q.list());
    }
    if (path === "/api/events" && req.method === "GET") return sse(res);

    const m = ROUTE.exec(path);
    if (m) {
      const [, id, action] = m;
      if (!action && req.method === "DELETE") {
        return q.remove(id)
          ? json(res, 200, { ok: true })
          : json(res, 409, { error: "no se puede borrar un job en proceso" });
      }
      if (req.method === "GET") {
        if (action === "stem") {
          const stem = stemOf(url.searchParams.get("name"));
          if (!stem) return json(res, 400, { error: "stem invalido" });
          return sendFile(res, join(q.dirOf(id), `${stem}.flac`));
        }
        if (action === "mix") return sendFile(res, join(q.dirOf(id), "mix.flac"));
        if (action === "download") return await download(res, id, url);
      }
    }

    // estaticos
    if (req.method === "GET") {
      const file = safeJoin(WEB, path === "/" ? "/index.html" : path);
      if (file) return sendFile(res, file);
    }
    json(res, 404, { error: "ruta no encontrada" });
  } catch (e) {
    console.error(e);
    if (!res.headersSent) json(res, 500, { error: (e as Error).message });
  }
}).listen(PORT, "127.0.0.1", () => {
  console.log(`stemtool -> http://localhost:${PORT}`);
  worker.warmUp(); // los ~10s de importar torch se pagan ahora, no en el primer job
});
