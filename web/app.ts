import type { Analysis, Job, Stem } from "../types.ts";

// La lista vive aqui y no importada de types.ts porque la UI manda en el orden
// y el color de presentacion; y asi el import de tipos se borra al compilar.
const STEMS: Stem[] = ["vocals", "drums", "bass", "other"];
const COLOR: Record<Stem, string> = {
  vocals: "var(--vocals)", drums: "var(--drums)",
  bass: "var(--bass)", other: "var(--other)",
};
const STATE: Record<Job["state"], string> = {
  queued: "en cola", preparing: "decodificando", analyzing: "analizando BPM y key",
  separating: "separando", refining: "refinando sobre los stems",
  done: "listo", error: "error",
};

const $ = <T extends Element>(s: string) => document.querySelector(s) as T;
const listEl = $<HTMLDivElement>("#list");
const emptyEl = $<HTMLParagraphElement>("#empty");
const dropEl = $<HTMLDivElement>("#drop");
const fileEl = $<HTMLInputElement>("#file");

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

dropEl.addEventListener("click", () => fileEl.click());
// El input vive dentro del dropzone: sin esto su click sintetico vuelve a
// burbujear al contenedor y se abre el dialogo dos veces.
fileEl.addEventListener("click", (e) => e.stopPropagation());
fileEl.addEventListener("change", () => {
  for (const f of fileEl.files ?? []) void upload(f);
  fileEl.value = "";
});
for (const ev of ["dragenter", "dragover"]) {
  dropEl.addEventListener(ev, (e) => {
    e.preventDefault();
    dropEl.classList.add("over");
  });
}
for (const ev of ["dragleave", "drop"]) {
  dropEl.addEventListener(ev, (e) => {
    e.preventDefault();
    dropEl.classList.remove("over");
  });
}
dropEl.addEventListener("drop", (e) => {
  for (const f of (e as DragEvent).dataTransfer?.files ?? []) void upload(f);
});

// --- reproductor -------------------------------------------------------------

class Player {
  private ctx?: AudioContext;
  private buffers = new Map<Stem, AudioBuffer>();
  private sources = new Map<Stem, AudioBufferSourceNode>();
  private gains = new Map<Stem, GainNode>();
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
      g.connect(this.ctx!.destination);
      src.connect(g);
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
  private player?: Player;
  private seekEl?: HTMLInputElement;
  private timeEl?: HTMLSpanElement;
  private playEl?: HTMLButtonElement;
  private dlEl?: HTMLButtonElement;

  constructor(private job: Job) {
    this.barEl = h("div", { class: "bar" }, this.fillEl);
    this.el = h("div", { class: "card" },
      h("div", { class: "top" }, this.nameEl, this.tagEl,
        h("button", { class: "x", title: "borrar", onclick: () => this.destroy() }, "✕")),
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
    this.tagEl.textContent = job.preset === "max" ? "maxima calidad" : "rapido";

    this.factsEl.replaceChildren(...this.facts(job));

    const done = job.state === "done";
    this.barEl.hidden = done || job.state === "error";
    this.fillEl.style.width = `${Math.round((job.progress || 0) * 100)}%`;
    this.statusEl.className = `status${job.state === "error" ? " err" : ""}`;
    this.statusEl.textContent = job.state === "error"
      ? `error: ${job.error ?? "desconocido"}`
      : done ? "" : [STATE[job.state], job.stage].filter(Boolean).join(" · ");

    if (done && job.stems?.length && !this.player) this.buildStems(job.stems);
  }

  /** duracion, BPM y key. El margen manda: si es estrecho, se dice. */
  private facts(job: Job): Node[] {
    const a: Analysis = { ...job.analysis, ...job.refined };
    if (!a.bpm && !a.key && a.duration === undefined) return [];
    const out: Node[] = [];
    if (a.duration !== undefined) {
      out.push(h("div", {}, "dur ", h("b", {}, mmss(a.duration))));
    }
    if (a.bpm) {
      out.push(h("div", {}, h("b", {}, String(a.bpm)), " BPM"));
    }
    if (a.key) {
      const unsure = (a.key_margin ?? 1) < 0.05;
      const alt = a.key_alternatives?.[0];
      out.push(h("div", {},
        h("b", {}, unsure && alt ? `${a.key} / ${alt}` : a.key),
        unsure ? h("span", { class: "warn" }, " poco claro") : ""));
    }
    if (job.refined?.bpm && job.analysis?.bpm
        && Math.abs(job.refined.bpm - job.analysis.bpm) > 1) {
      out.push(h("div", { class: "warn" }, `(del mix: ${job.analysis.bpm})`));
    }
    if (job.residualDb !== undefined) {
      out.push(h("div", { title: "suma de stems menos mix: mas negativo es mejor" },
        "null ", h("b", {}, `${job.residualDb} dB`)));
    }
    return out;
  }

  private buildStems(available: Stem[]) {
    const stems = STEMS.filter((s) => available.includes(s));
    this.player = new Player(this.job.id, stems);

    for (const s of stems) {
      const pick = h("input", { type: "checkbox", checked: "" }) as HTMLInputElement;
      pick.checked = true;
      pick.addEventListener("change", () => this.syncDownload());
      this.picks.set(s, pick);

      const mute = h("button", { title: "silenciar" }, "M");
      const solo = h("button", { title: "solo" }, "S");
      const row = h("div", { class: "stem" },
        pick,
        h("span", { class: "dot", style: `background:${COLOR[s]}` }),
        h("span", { class: "label" }, s),
        mute, solo);
      mute.addEventListener("click", () => {
        this.player!.toggleMute(s);
        this.paintStems();
      });
      solo.addEventListener("click", () => {
        this.player!.toggleSolo(s);
        this.paintStems();
      });
      this.rows.set(s, row);
      this.stemsEl.append(row);
    }
    this.stemsEl.classList.add("on");

    this.playEl = h("button", { class: "ghost" }, "▶ Reproducir");
    this.playEl.addEventListener("click", () => {
      const p = this.player!;
      p.playing ? p.pause() : void p.play();
      this.paintTransport();
    });

    this.seekEl = h("input", {
      type: "range", class: "seek", min: "0", max: "1000", value: "0",
    }) as HTMLInputElement;
    this.seekEl.addEventListener("input", () => {
      const p = this.player!;
      if (p.duration) void p.seek((+this.seekEl!.value / 1000) * p.duration);
    });
    this.timeEl = h("span", { class: "time" }, "0:00");

    const fmtSel = h("select", {},
      h("option", { value: "flac" }, "FLAC"),
      h("option", { value: "wav24" }, "WAV 24"),
      h("option", { value: "mp3" }, "MP3 320")) as HTMLSelectElement;
    const dl = h("button", { class: "primary" }, "Descargar") as HTMLButtonElement;
    dl.addEventListener("click", () => {
      const pick = [...this.picks].filter(([, c]) => c.checked).map(([s]) => s);
      if (!pick.length) return;
      // Navegacion normal: el navegador bloquea varias descargas seguidas, por
      // eso el servidor zipea cuando hay mas de uno.
      location.href =
        `/api/jobs/${this.job.id}/download?pick=${pick.join(",")}&format=${fmtSel.value}`;
    });
    this.dlEl = dl;

    this.actionsEl.append(this.playEl, this.seekEl, this.timeEl, fmtSel, dl);
    this.player.onTick = () => this.paintTransport();
    this.paintStems();
    this.paintTransport();
  }

  private syncDownload() {
    if (!this.dlEl) return;
    const n = [...this.picks.values()].filter((c) => c.checked).length;
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
      solo.classList.toggle("on", p.solo === s);
    }
    this.syncDownload();
  }

  private paintTransport() {
    const p = this.player!;
    this.playEl!.textContent = p.playing ? "⏸ Pausa" : "▶ Reproducir";
    this.timeEl!.textContent = p.duration
      ? `${mmss(p.position)} / ${mmss(p.duration)}`
      : "0:00";
    if (p.duration && document.activeElement !== this.seekEl) {
      this.seekEl!.value = String(Math.round((p.position / p.duration) * 1000));
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

const es = new EventSource("/api/events");
es.addEventListener("message", (e) => render(JSON.parse(e.data) as Job));
es.addEventListener("error", () => {
  // EventSource reconecta solo; solo se avisa si el servidor se fue del todo.
  if (es.readyState === EventSource.CLOSED) {
    emptyEl.hidden = false;
    emptyEl.textContent = "se perdio la conexion con el servidor";
  }
});
