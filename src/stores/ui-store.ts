import { create } from "zustand";

export type ViewMode = "3d" | "2d";
export type InspectorTab = "layer" | "tensor" | "notebook";

type UiState = {
  viewMode: ViewMode;
  inspectorTab: InspectorTab;
  setViewMode: (mode: ViewMode) => void;
  setInspectorTab: (tab: InspectorTab) => void;
};

export const useUiStore = create<UiState>((set) => ({
  viewMode: "3d",
  inspectorTab: "layer",
  setViewMode: (viewMode) => set({ viewMode }),
  setInspectorTab: (inspectorTab) => set({ inspectorTab }),
}));
