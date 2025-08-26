import { create } from 'zustand';
import { immer } from 'zustand/middleware/immer';
import type { 
  AppStore, 
  DetectionTask, 
  MapViewState, 
  AnalysisConfig, 
  SystemStatus,
  DEFAULT_MAP_CONFIG 
} from '@types/index';

export const useAppStore = create<AppStore>()(
  immer((set) => ({
    // UI State
    sidebarCollapsed: false,
    setSidebarCollapsed: (collapsed: boolean) =>
      set((state) => {
        state.sidebarCollapsed = collapsed;
      }),

    // Current Task
    currentTask: null,
    setCurrentTask: (task: DetectionTask | null) =>
      set((state) => {
        state.currentTask = task;
      }),

    // Map State
    mapViewState: {
      center: [39.9042, 116.4074], // 默认北京
      zoom: 10,
    },
    setMapViewState: (viewState: Partial<MapViewState>) =>
      set((state) => {
        Object.assign(state.mapViewState, viewState);
      }),

    // Analysis Settings
    analysisConfig: {
      minChangeArea: 100,
      confidenceThreshold: 0.5,
      registrationMethod: 'auto',
      detectionSensitivity: 'medium',
      enableAdvancedFiltering: true,
    },
    setAnalysisConfig: (config: Partial<AnalysisConfig>) =>
      set((state) => {
        Object.assign(state.analysisConfig, config);
      }),

    // System Status
    systemStatus: null,
    setSystemStatus: (status: SystemStatus) =>
      set((state) => {
        state.systemStatus = status;
      }),
  }))
);