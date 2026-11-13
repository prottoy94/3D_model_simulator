import { create } from "zustand";

export type ViewMode = "3d" | "2d";

type UiState = {
  viewMode: ViewMode;
  setViewMode: (mode: ViewMode) => void;
};

export const useUiStore = create<UiState>((set) => ({
  viewMode: "3d",
  setViewMode: (viewMode) => set({ viewMode }),
}));
