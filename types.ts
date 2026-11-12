/** Tipos compartidos entre servidor y navegador. */

export const STEMS = ["vocals", "drums", "bass", "other"] as const;
export type Stem = (typeof STEMS)[number];

export type Preset = "fast" | "max";
export type JobState =
  | "queued"
  | "preparing"
  | "analyzing"
  | "separating"
  | "refining"
  | "done"
  | "error";

export interface Analysis {
  duration?: number;
  bpm?: number;
  /** Lo que dijo librosa antes de la guarda de octava. Util para depurar. */
  bpm_raw?: number;
  bpm_confidence?: number;
  bpm_alternatives?: number[];
  key?: string | null;
  key_confidence?: number;
  /** Distancia a la segunda candidata: margen pequeno = no te lo creas. */
  key_margin?: number;
  key_alternatives?: string[];
}

export interface Job {
  id: string;
  name: string;
  preset: Preset;
  state: JobState;
  /** 0..1 dentro de la etapa actual. */
  progress: number;
  stage?: string;
  /** Analisis del mix, disponible a los pocos segundos de subir. */
  analysis?: Analysis;
  /** Recalculado sobre los stems: drums manda en BPM, bass+other en key. */
  refined?: Analysis;
  stems?: Stem[];
  /** Null test: suma de stems menos mix, en dB. Mas negativo = mejor. */
  residualDb?: number;
  error?: string;
  createdAt: number;
}

/** El analisis que la UI debe mostrar: el refinado si existe. */
export function bestAnalysis(j: Job): Analysis {
  return { ...j.analysis, ...j.refined };
}
