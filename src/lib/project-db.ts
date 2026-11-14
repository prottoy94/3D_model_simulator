import Dexie, { type EntityTable } from "dexie";
import type { Notebook } from "@/core/notebook";

/**
 * A notebook the user opened, stored locally in IndexedDB. Only the stripped,
 * text-only notebook is kept; the analysis is recomputed on load (it takes
 * milliseconds). Nothing here ever leaves the browser.
 */
export type StoredProject = {
  /** sha256 of the stripped notebook content. Re-opening the same file reuses the entry. */
  hash: string;
  name: string;
  notebook: Notebook;
  originalBytes: number;
  openedAt: number;
};

let database: (Dexie & { projects: EntityTable<StoredProject, "hash"> }) | null = null;

function db() {
  if (!database) {
    const instance = new Dexie("neuroscope") as Dexie & {
      projects: EntityTable<StoredProject, "hash">;
    };
    instance.version(1).stores({ projects: "hash, openedAt" });
    database = instance;
  }
  return database;
}

export async function saveProject(project: StoredProject): Promise<void> {
  await db().projects.put(project);
}

export async function loadLatestProject(): Promise<StoredProject | undefined> {
  return db().projects.orderBy("openedAt").last();
}

export async function sha256Hex(text: string): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}
