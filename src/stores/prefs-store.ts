import { create } from "zustand";

export type Theme = "dark" | "light";
export type Palette = "default" | "safe";

type Saved = { theme: Theme; palette: Palette; tourSeen: boolean };

type PrefsState = Saved & {
  /** False until the saved choices have been read, so nothing flashes on first paint. */
  loaded: boolean;
  /** Whether the guided tour is on screen, and which step. */
  tourStep: number | null;
  load: () => void;
  setTheme: (theme: Theme) => void;
  setPalette: (palette: Palette) => void;
  startTour: () => void;
  setTourStep: (step: number) => void;
  endTour: () => void;
};

const STORAGE_KEY = "neuroscope.prefs";

function read(): Partial<Saved> {
  try {
    return JSON.parse(window.localStorage.getItem(STORAGE_KEY) ?? "{}") as Partial<Saved>;
  } catch {
    return {};
  }
}

function write({ theme, palette, tourSeen }: Saved): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify({ theme, palette, tourSeen }));
  } catch {
    // Private mode: the choice just does not survive a reload.
  }
}

/** The tokens in tokens.css switch on these two attributes. */
function apply({ theme, palette }: Pick<Saved, "theme" | "palette">): void {
  const root = document.documentElement;
  if (theme === "light") root.dataset.theme = "light";
  else delete root.dataset.theme;
  if (palette === "safe") root.dataset.palette = "safe";
  else delete root.dataset.palette;
  root.style.colorScheme = theme;
}

export const usePrefsStore = create<PrefsState>((set, get) => ({
  theme: "dark",
  palette: "default",
  tourSeen: false,
  loaded: false,
  tourStep: null,

  load: () => {
    if (get().loaded) return;
    const saved = read();
    const next: Saved = {
      theme: saved.theme === "light" ? "light" : "dark",
      palette: saved.palette === "safe" ? "safe" : "default",
      tourSeen: saved.tourSeen === true,
    };
    apply(next);
    set({ ...next, loaded: true });
  },

  setTheme: (theme) => {
    set({ theme });
    apply(get());
    write(get());
  },
  setPalette: (palette) => {
    set({ palette });
    apply(get());
    write(get());
  },

  startTour: () => set({ tourStep: 0 }),
  setTourStep: (tourStep) => set({ tourStep }),
  endTour: () => {
    set({ tourStep: null, tourSeen: true });
    write(get());
  },
}));

/**
 * Changes whenever the colors on screen change. WebGL materials, Monaco and 2D
 * canvases read tokens once, so they take this as a `key` (or effect dependency)
 * to read them again.
 */
export const useSkin = () => usePrefsStore((state) => `${state.theme}-${state.palette}`);
