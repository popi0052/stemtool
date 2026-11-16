import type { Analysis, Job, Stem } from "../types.ts";

// La lista vive aqui y no importada de types.ts porque la UI manda en el orden
// y el color de presentacion; y asi el import de tipos se borra al compilar.
const STEMS: Stem[] = ["vocals", "drums", "bass", "other"];
const COLOR: Record<Stem, string> = {
  vocals: "var(--vocals)", drums: "var(--drums)",
  bass: "var(--bass)", other: "var(--other)",
};
// En minusculas y con tildes: las versalitas las pone el CSS, que si sabe
// pasar "duración" a mayusculas sin romper el acento.
const STATE: Record<Job["state"], string> = {
  queued: "en cola", preparing: "decodificando", analyzing: "BPM y key",
  separating: "separando", refining: "refinando",
  done: "listo", error: "error",
};

const $ = <T extends Element>(s: string) => document.querySelector(s) as T;
const listEl = $<HTMLDivElement>("#list");
const emptyEl = $<HTMLParagraphElement>("#empty");
const dropEl = $<HTMLDivElement>("#drop");
const fileEl = $<HTMLInputElement>("#file");
const browseEl = $<HTMLButtonElement>("#browse");

/** Hyperscript minimo. Devuelve nodos reales, no HTML en texto: asi repintar
 *  una tarjeta no destruye el reproductor ni las casillas marcadas. */
function h<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  props: Record<string, unknown> = {},
  ...kids: (Node | string)[]
): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (v === undefined || v === null) continue;
    if (k === "class") el.className = String(v);
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v as EventListener);
    else el.setAttribute(k, String(v));
  }
  el.append(...kids);
  return el;
}

const mmss = (t: number) =>
  `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;

// --- subida ------------------------------------------------------------------

async function upload(file: File) {
  const preset = $<HTMLInputElement>("input[name=preset]:checked").value;
  const q = `name=${encodeURIComponent(file.name)}&preset=${preset}`;
  const r = await fetch(`/api/jobs?${q}`, { method: "PUT", body: file });
  if (!r.ok) {
    const { error } = await r.json().catch(() => ({ error: r.statusText }));
    alert(`${file.name}: ${error}`);
  }
}

const pickFiles = () => fileEl.click();
dropEl.addEventListener("click", pickFiles);
dropEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    pickFiles();
  }
});
// El boton y el input viven dentro del dropzone: sin esto su click vuelve a
// burbujear al contenedor y se abre el dialogo dos veces.
browseEl.addEventListener("click", (e) => {
  e.stopPropagation();
  pickFiles();
});
fileEl.addEventListener("click", (e) => e.stopPropagation());
fileEl.addEventListener("change", () => {
  for (const f of fileEl.files ?? []) void upload(f);
  fileEl.value = "";
});

// Contador en vez de un booleano: al pasar por encima de un hijo salta un
// dragleave del padre, y con una bandera simple el marco parpadea.
let dragDepth = 0;
const paintDrag = () => dropEl.classList.toggle("over", dragDepth > 0);
dropEl.addEventListener("dragenter", (e) => {
  e.preventDefault();
  dragDepth++;
  paintDrag();
});
dropEl.addEventListener("dragover", (e) => e.preventDefault());
dropEl.addEventListener("dragleave", () => {
  dragDepth = Math.max(0, dragDepth - 1);
  paintDrag();
});
dropEl.addEventListener("drop", (e) => {
  e.preventDefault();
  dragDepth = 0;
  paintDrag();
  for (const f of (e as DragEvent).dataTransfer?.files ?? []) void upload(f);
});

// --- reproductor -------------------------------------------------------------

class Player {
  private ctx?: AudioContext;
  private buffers = new Map<Stem, AudioBuffer>();
  private sources = new Map<Stem, AudioBufferSourceNode>();
  private gains = new Map<Stem, GainNode>();
  private meters = new Map<Stem, AnalyserNode>();
  private scratch = new Uint8Array(256);
  private startedAt = 0;
  private offset = 0;
  playing = false;
  duration = 0;
  muted = new Set<Stem>();
  solo: Stem | null = null;
  onTick?: () => void;

  constructor(private jobId: string, private stems: Stem[]) {}

  get position() {
    if (!this.playing || !this.ctx) return this.offset;
    const t = this.offset + (this.ctx.currentTime - this.startedAt);
    return Math.min(this.duration, Math.max(this.offset, t));
  }

  private async load() {
    this.ctx ??= new AudioContext();
    if (this.buffers.size) return;
    await Promise.all(this.stems.map(async (s) => {
      const r = await fetch(`/api/jobs/${this.jobId}/stem?name=${s}`);
      if (!r.ok) return;
      const buf = await this.ctx!.decodeAudioData(await r.arrayBuffer());
      this.buffers.set(s, buf);
      this.duration = Math.max(this.duration, buf.duration);
    }));
  }

  private applyGains() {
    if (!this.ctx) return;
    for (const [s, g] of this.gains) {
      const on = this.solo ? this.solo === s : !this.muted.has(s);
      // Rampa de 10ms: cambiar la ganancia de golpe produce un click audible.
      g.gain.setTargetAtTime(on ? 1 : 0, this.ctx.currentTime, 0.01);
    }
  }

  async play(from = this.offset) {
    await this.load();
    await this.ctx!.resume();
    this.halt();
    this.offset = from >= this.duration ? 0 : from;

    // Los 4 arrancan en el MISMO instante del reloj del contexto, asi van
    // sincronizados a nivel de muestra. Cuatro elementos <audio> derivan.
    const at = this.ctx!.currentTime + 0.06;
    for (const s of this.stems) {
      const buf = this.buffers.get(s);
      if (!buf) continue;
      const src = this.ctx!.createBufferSource();
      src.buffer = buf;
      const g = this.gains.get(s) ?? this.ctx!.createGain();
      // El medidor va ANTES de la ganancia: un stem silenciado sigue
      // enseniando su nivel, que es como se detecta una fuga entre pistas.
      let m = this.meters.get(s);
      if (!m) {
        m = this.ctx!.createAnalyser();
        m.fftSize = 256;
        this.meters.set(s, m);
      }
      g.connect(this.ctx!.destination);
      m.connect(g);
      src.connect(m);
      this.gains.set(s, g);
      this.sources.set(s, src);
      src.start(at, this.offset);
    }
    this.startedAt = at;
    this.playing = true;
    this.applyGains();

    const first = this.sources.values().next().value;
    if (first) {
      first.onended = () => {
        this.halt();
        this.offset = 0;
        this.onTick?.();
      };
    }
    this.tick();
  }

  /** Nivel RMS 0..1 de un stem, para el medidor. */
  level(s: Stem) {
    const m = this.meters.get(s);
    if (!m || !this.playing) return 0;
    m.getByteTimeDomainData(this.scratch);
    let sum = 0;
    for (const v of this.scratch) {
      const x = (v - 128) / 128;
      sum += x * x;
    }
    return Math.sqrt(sum / this.scratch.length);
  }

  /** Para las fuentes sin tocar la posicion ni notificar. */
  private halt() {
    for (const src of this.sources.values()) {
      src.onended = null;
      try { src.stop(); } catch { /* ya paradas */ }
    }
    this.sources.clear();
    this.playing = false;
  }

  pause() {
    this.offset = this.position;
    this.halt();
    this.onTick?.();
  }

  async seek(t: number) {
    const was = this.playing;
    this.halt();
    this.offset = t;
    if (was) await this.play(t);
    else this.onTick?.();
  }

  toggleMute(s: Stem) {
    this.muted.has(s) ? this.muted.delete(s) : this.muted.add(s);
    this.applyGains();
  }

  toggleSolo(s: Stem) {
    this.solo = this.solo === s ? null : s;
    this.applyGains();
  }

  private tick = () => {
    if (!this.playing) return;
    this.onTick?.();
    requestAnimationFrame(this.tick);
  };
}

// --- tarjetas ----------------------------------------------------------------

class Card {
  el: HTMLDivElement;
  private nameEl = h("span", { class: "name" });
  private tagEl = h("span", { class: "tag" });
  private factsEl = h("div", { class: "facts" });
  private fillEl = h("i");
  private barEl: HTMLDivElement;
  private statusEl = h("div", { class: "status" });
  private stemsEl = h("div", { class: "stems" });
  private actionsEl = h("div", { class: "actions" });
  private picks = new Map<Stem, HTMLInputElement>();
  private rows = new Map<Stem, HTMLDivElement>();
  private levelEls = new Map<Stem, HTMLElement>();
  player?: Player;
  private seekEl?: HTMLInputElement;
  private timeEl?: HTMLSpanElement;
  private playEl?: HTMLButtonElement;
  private dlEl?: HTMLButtonElement;
  private clickEl?: HTMLInputElement;

  constructor(private job: Job) {
    this.barEl = h("div", { class: "bar" }, this.fillEl);
    this.el = h("div", { class: "card" },
      h("div", { class: "top" }, this.nameEl, this.tagEl,
        h("button", { class: "x", title: "borrar", "aria-label": "borrar",
                      onclick: () => this.destroy() }, "✕")),
      this.factsEl, this.barEl, this.statusEl, this.stemsEl, this.actionsEl);
    this.update(job);
  }

  private async destroy() {
    this.player?.pause();
    const r = await fetch(`/api/jobs/${this.job.id}`, { method: "DELETE" });
    if (r.ok) {
      cards.delete(this.job.id);
      this.el.remove();
      emptyEl.hidden = cards.size > 0;
    } else {
      alert((await r.json().catch(() => ({}))).error ?? "no se pudo borrar");
    }
  }

  update(job: Job) {
    this.job = job;
    this.nameEl.textContent = job.name;
    this.tagEl.textContent = job.preset === "max" ? "máxima" : "rápido";

    this.factsEl.replaceChildren(...this.facts(job));

    const done = job.state === "done";
    const bad = job.state === "error";
    // La barra roja de la izquierda: lo que esta vivo se ve de un vistazo.
    this.el.classList.toggle("live", !done && !bad);
    this.el.classList.toggle("bad", bad);
    this.barEl.hidden = done || bad;
    this.fillEl.style.width = `${Math.round((job.progress || 0) * 100)}%`;
    this.statusEl.className = `status${bad ? " err" : ""}`;
    this.statusEl.textContent = bad
      ? `error: ${job.error ?? "desconocido"}`
      : done ? "" : [STATE[job.state], job.stage].filter(Boolean).join(" · ");

    if (done && job.stems?.length && !this.player) this.buildStems(job.stems);
  }

  /** Un dato con su etiqueta debajo: numero grande, nombre pequenio. */
  private fact(k: string, v: Node | string, extra?: Node) {
    const val = h("span", { class: "v" }, v);
    if (extra) val.append(extra);
    return h("div", {}, h("span", { class: "k" }, k), val);
  }

  /** duracion, BPM y key. El margen manda: si es estrecho, se dice. */
  private facts(job: Job): Node[] {
    const a: Analysis = { ...job.analysis, ...job.refined };
    if (!a.bpm && !a.key && a.duration === undefined) return [];
    const out: Node[] = [];
    if (a.duration !== undefined) out.push(this.fact("duración", mmss(a.duration)));
    if (a.bpm) {
      // Si el BPM medido sobre la bateria se aparta del del mix, se avisa:
      // uno de los dos esta leyendo mal el compas.
      const drift = job.refined?.bpm && job.analysis?.bpm
        && Math.abs(job.refined.bpm - job.analysis.bpm) > 1
        ? h("span", { class: "warn" }, `mix ${job.analysis.bpm}`)
        : undefined;
      out.push(this.fact("BPM", String(a.bpm), drift));
    }
    if (a.key) {
      const unsure = (a.key_margin ?? 1) < 0.05;
      const alt = a.key_alternatives?.[0];
      out.push(this.fact("key",
        unsure && alt ? `${a.key} / ${alt}` : a.key,
        unsure ? h("span", { class: "warn" }, "poco claro") : undefined));
    }
    if (job.residualDb !== undefined) {
      const d = h("div", { title: "suma de stems menos mix: más negativo es mejor" },
        h("span", { class: "k" }, "null test"),
        h("span", { class: "v soft" }, `${job.residualDb} dB`));
      out.push(d);
    }
    return out;
  }

  private buildStems(available: Stem[]) {
    const stems = STEMS.filter((s) => available.includes(s));
    this.player = new Player(this.job.id, stems);

    for (const s of stems) {
      const pick = h("input", {
        type: "checkbox", "aria-label": `incluir ${s} en la descarga`,
      }) as HTMLInputElement;
      pick.checked = true;
      pick.addEventListener("change", () => this.syncDownload());
      this.picks.set(s, pick);

      const level = h("i");
      this.levelEls.set(s, level);
      const mute = h("button", { title: "silenciar (M)", "aria-label": `silenciar ${s}` }, "M");
      const solo = h("button", { title: "solo (S)", "aria-label": `solo ${s}` }, "S");
      const row = h("div", {
        class: "stem", tabindex: "0", role: "group", "aria-label": s,
      }, pick,
        h("span", { class: "chip", style: `background:${COLOR[s]}` }),
        h("span", { class: "label" }, s),
        h("span", { class: "meter", style: `color:${COLOR[s]}` }, level),
        mute, solo);
      mute.addEventListener("click", () => {
        this.player!.toggleMute(s);
        this.paintStems();
      });
      solo.addEventListener("click", () => {
        this.player!.toggleSolo(s);
        this.paintStems();
      });
      // M y S sobre la fila enfocada: el atajo que usa cualquier DAW.
      row.addEventListener("keydown", (e) => {
        const k = (e as KeyboardEvent).key.toLowerCase();
        if (k !== "m" && k !== "s") return;
        e.preventDefault();
        k === "m" ? this.player!.toggleMute(s) : this.player!.toggleSolo(s);
        this.paintStems();
      });
      this.rows.set(s, row);
      this.stemsEl.append(row);
    }
    this.stemsEl.classList.add("on");

    this.playEl = h("button", { class: "play", "aria-label": "reproducir" }, "▶");
    this.playEl.addEventListener("click", () => this.togglePlay());

    this.seekEl = h("input", {
      type: "range", class: "seek", min: "0", max: "1000", value: "0",
      "aria-label": "posición",
    }) as HTMLInputElement;
    this.seekEl.addEventListener("input", () => {
      const p = this.player!;
      if (p.duration) void p.seek((+this.seekEl!.value / 1000) * p.duration);
    });
    this.timeEl = h("span", { class: "time" }, "0:00");

    const fmtSel = h("select", { "aria-label": "formato" },
      h("option", { value: "flac" }, "FLAC"),
      h("option", { value: "wav24" }, "WAV 24"),
      h("option", { value: "mp3" }, "MP3 320")) as HTMLSelectElement;
    // Marcado por defecto: el BPM es un dato que conviene comprobar, no
    // creerse. Si el clic se despega de la musica, el numero esta mal.
    const click = h("input", { type: "checkbox" }) as HTMLInputElement;
    click.checked = true;
    click.addEventListener("change", () => this.syncDownload());
    this.clickEl = click;
    const clickLabel = h("label", {
      class: "click",
      title: "pista de clic a tempo fijo con el BPM detectado, para comprobar que lo sigue",
    }, click, "metrónomo");

    const dl = h("button", { class: "dl" }, "Descargar") as HTMLButtonElement;
    dl.addEventListener("click", () => {
      const pick = [...this.picks].filter(([, c]) => c.checked).map(([s]) => s);
      if (!pick.length && !click.checked) return;
      // Navegacion normal: el navegador bloquea varias descargas seguidas, por
      // eso el servidor zipea cuando hay mas de uno.
      location.href = `/api/jobs/${this.job.id}/download?pick=${pick.join(",")}`
        + `&format=${fmtSel.value}${click.checked ? "&click=1" : ""}`;
    });
    this.dlEl = dl;

    this.actionsEl.append(this.playEl, this.seekEl, this.timeEl,
      h("div", { class: "row2" }, clickLabel, fmtSel, dl));
    this.player.onTick = () => this.paintTransport();
    this.paintStems();
    this.paintTransport();
  }

  togglePlay() {
    const p = this.player;
    if (!p) return;
    p.playing ? p.pause() : void p.play();
    this.paintTransport();
  }

  private syncDownload() {
    if (!this.dlEl) return;
    const n = [...this.picks.values()].filter((c) => c.checked).length
      + (this.clickEl?.checked ? 1 : 0);
    this.dlEl.disabled = n === 0;
    this.dlEl.textContent = n > 1 ? `Descargar ${n} (zip)` : "Descargar";
  }

  private paintStems() {
    const p = this.player!;
    for (const [s, row] of this.rows) {
      const on = p.solo ? p.solo === s : !p.muted.has(s);
      row.classList.toggle("off", !on);
      const [mute, solo] = [...row.querySelectorAll("button")];
      mute.classList.toggle("on", p.muted.has(s));
      mute.setAttribute("aria-pressed", String(p.muted.has(s)));
      solo.classList.toggle("on", p.solo === s);
      solo.setAttribute("aria-pressed", String(p.solo === s));
    }
    this.syncDownload();
  }

  private paintTransport() {
    const p = this.player!;
    this.playEl!.textContent = p.playing ? "⏸" : "▶";
    this.playEl!.setAttribute("aria-label", p.playing ? "pausa" : "reproducir");
    // La duracion del analisis sirve de respaldo: el total se puede enseniar
    // antes de descargar los stems, que es cuando el reproductor la sabe.
    const total = p.duration || this.job.refined?.duration
      || this.job.analysis?.duration || 0;
    this.timeEl!.textContent = total
      ? `${mmss(p.position)} / ${mmss(total)}`
      : "0:00";
    if (p.duration) {
      const pct = (p.position / p.duration) * 100;
      this.seekEl!.style.setProperty("--p", `${pct}%`);
      if (document.activeElement !== this.seekEl) {
        this.seekEl!.value = String(Math.round(pct * 10));
      }
    }
    // Raiz cuadrada: el RMS crudo deja el medidor pegado abajo casi siempre.
    for (const [s, el] of this.levelEls) {
      el.style.transform = `scaleX(${Math.min(1, Math.sqrt(p.level(s)) * 1.6)})`;
    }
  }
}

// --- estado en vivo ----------------------------------------------------------

const cards = new Map<string, Card>();

function render(job: Job) {
  const existing = cards.get(job.id);
  if (existing) return existing.update(job);
  const card = new Card(job);
  cards.set(job.id, card);
  listEl.prepend(card.el);
  emptyEl.hidden = true;
}

// Espacio = reproducir la tarjeta enfocada, o la primera que tenga audio.
document.addEventListener("keydown", (e) => {
  if (e.key !== " " || e.ctrlKey || e.metaKey || e.altKey) return;
  const t = e.target as HTMLElement | null;
  if (t && (t.closest("button, input, select, textarea, [contenteditable]"))) return;
  const el = t?.closest(".card");
  const card = el
    ? [...cards.values()].find((c) => c.el === el)
    : [...cards.values()].find((c) => c.player);
  if (!card?.player) return;
  e.preventDefault();
  card.togglePlay();
});

const es = new EventSource("/api/events");
es.addEventListener("message", (e) => render(JSON.parse(e.data) as Job));
es.addEventListener("error", () => {
  // EventSource reconecta solo; solo se avisa si el servidor se fue del todo.
  if (es.readyState === EventSource.CLOSED) {
    emptyEl.hidden = false;
    emptyEl.className = "empty lost";
    emptyEl.textContent = "Sin conexión con el servidor";
  }
});
