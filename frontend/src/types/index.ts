// 地理测绘变化检测系统 - 类型定义

export interface GPSCoordinate {
  latitude: number;
  longitude: number;
  altitude?: number;
}

export interface ImageInfo {
  id: string;
  filename: string;
  size: number;
  format: string;
  dimensions: {
    width: number;
    height: number;
  };
  gps?: GPSCoordinate;
  timestamp: string;
  url?: string;
}

export interface ChangeRegion {
  id: string;
  area: number;
  centroid: [number, number]; // [lat, lng]
  boundingBox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  confidence: number;
  changeType: 'building' | 'vegetation' | 'road' | 'water' | 'unknown' | 'large_area' | 'small_object' | 'linear_structure';
  geoCoordinates?: {
    bounds: [[number, number], [number, number]]; // [[south, west], [north, east]]
    polygon?: Array<[number, number]>; // 多边形顶点坐标
  };
}

export interface RegistrationResult {
  confidenceScore: number;
  featureMatches: number;
  inlierRatio: number;
  registrationError: number;
  transformApplied: boolean;
}

export interface DetectionResult {
  taskId: string;
  timestamp: string;
  description?: string;
  algorithmType: 'simple' | 'advanced';
  
  // GPS信息
  gpsInfo: {
    image1?: GPSCoordinate;
    image2?: GPSCoordinate;
  };
  
  // 配准结果
  registrationResults: RegistrationResult;
  
  // 检测结果
  detectionResults: {
    changeRegionsCount: number;
    totalChangeAreaPixels: number;
    changePercentage: number;
    processingTime?: number;
  };
  
  // 变化区域详情
  changeRegions: ChangeRegion[];
  
  // 结果图像
  resultImageUrl: string;
  
  status: 'pending' | 'processing' | 'completed' | 'error';
}

export interface DetectionTask {
  id: string;
  image1: ImageInfo;
  image2: ImageInfo;
  description?: string;
  algorithmType: 'simple' | 'advanced';
  status: 'pending' | 'processing' | 'completed' | 'error';
  result?: DetectionResult;
  createdAt: string;
  completedAt?: string;
  error?: string;
}

export interface MapViewState {
  center: [number, number]; // [lat, lng]
  zoom: number;
  bounds?: [[number, number], [number, number]]; // [[south, west], [north, east]]
}

export interface LayerConfig {
  id: string;
  name: string;
  type: 'tile' | 'marker' | 'polygon' | 'heatmap';
  visible: boolean;
  opacity: number;
  zIndex: number;
  url?: string; // for tile layers
  data?: any; // for data layers
}

export interface MapConfig {
  defaultCenter: [number, number];
  defaultZoom: number;
  minZoom: number;
  maxZoom: number;
  baseLayers: LayerConfig[];
  overlayLayers: LayerConfig[];
}

export interface AnalysisConfig {
  minChangeArea: number;
  confidenceThreshold: number;
  registrationMethod: 'sift' | 'surf' | 'orb' | 'auto';
  detectionSensitivity: 'low' | 'medium' | 'high';
  enableAdvancedFiltering: boolean;
}

export interface SystemStatus {
  status: 'healthy' | 'warning' | 'error';
  timestamp: string;
  services: {
    changeDetection: 'running' | 'stopped' | 'error';
    fileStorage: 'available' | 'unavailable';
    database?: 'connected' | 'disconnected';
  };
  statistics?: {
    totalTasks: number;
    completedTasks: number;
    errorTasks: number;
    averageProcessingTime: number;
  };
}

export interface DroneInfo {
  id: string;
  name: string;
  status: 'online' | 'offline' | 'flying' | 'error';
  lastSeen: string;
  location?: GPSCoordinate;
  battery?: number;
  altitude?: number;
  speed?: number;
}

export interface DroneDataStream {
  droneId: string;
  timestamp: string;
  image: string; // base64 or URL
  gps: GPSCoordinate;
  metadata: {
    altitude: number;
    heading: number;
    speed: number;
    battery: number;
  };
}

// API Response Types
export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: {
    code: string;
    message: string;
    details?: any;
  };
  timestamp: string;
}

export interface UploadResponse extends ApiResponse<DetectionResult> {}

export interface HealthCheckResponse extends ApiResponse<SystemStatus> {}

export interface TaskListResponse extends ApiResponse<DetectionTask[]> {}

// Hook Types
export interface UseDetectionTasksReturn {
  tasks: DetectionTask[];
  loading: boolean;
  error: string | null;
  createTask: (image1: File, image2: File, options?: Partial<DetectionTask>) => Promise<DetectionTask>;
  getTask: (taskId: string) => DetectionTask | undefined;
  refreshTasks: () => Promise<void>;
}

export interface UseMapStateReturn {
  viewState: MapViewState;
  setViewState: (viewState: Partial<MapViewState>) => void;
  layers: LayerConfig[];
  toggleLayer: (layerId: string) => void;
  setLayerOpacity: (layerId: string, opacity: number) => void;
  addLayer: (layer: LayerConfig) => void;
  removeLayer: (layerId: string) => void;
}

// Component Props Types
export interface MapComponentProps {
  height?: string | number;
  width?: string | number;
  center?: [number, number];
  zoom?: number;
  detectionResults?: DetectionResult[];
  onRegionSelect?: (region: ChangeRegion) => void;
  onMapClick?: (coordinates: [number, number]) => void;
  showControls?: boolean;
  showLegend?: boolean;
}

export interface ImageUploadProps {
  onImagesSelected: (image1: File, image2: File) => void;
  loading?: boolean;
  disabled?: boolean;
  maxFileSize?: number; // in bytes
  supportedFormats?: string[];
}

export interface ResultDisplayProps {
  result: DetectionResult;
  showDetails?: boolean;
  onRegionClick?: (region: ChangeRegion) => void;
  onExport?: (format: 'json' | 'csv' | 'pdf') => void;
}

// Store Types (for Zustand)
export interface AppStore {
  // UI State
  sidebarCollapsed: boolean;
  setSidebarCollapsed: (collapsed: boolean) => void;
  
  // Current Task
  currentTask: DetectionTask | null;
  setCurrentTask: (task: DetectionTask | null) => void;
  
  // Map State
  mapViewState: MapViewState;
  setMapViewState: (viewState: Partial<MapViewState>) => void;
  
  // Settings
  analysisConfig: AnalysisConfig;
  setAnalysisConfig: (config: Partial<AnalysisConfig>) => void;
  
  // System
  systemStatus: SystemStatus | null;
  setSystemStatus: (status: SystemStatus) => void;
}

// Utility Types
export type Coordinates = [number, number]; // [lat, lng]
export type BoundingBox = [Coordinates, Coordinates]; // [[south, west], [north, east]]
export type ChangeTypeColor = Record<ChangeRegion['changeType'], string>;

// Constants
export const CHANGE_TYPE_COLORS: ChangeTypeColor = {
  building: '#f44336',
  vegetation: '#4caf50',
  road: '#9e9e9e',
  water: '#2196f3',
  unknown: '#ff9800',
  large_area: '#e91e63',
  small_object: '#9c27b0',
  linear_structure: '#607d8b'
};

export const CONFIDENCE_LEVELS = {
  HIGH: 0.8,
  MEDIUM: 0.5,
  LOW: 0.3
} as const;

export const DEFAULT_MAP_CONFIG: MapConfig = {
  defaultCenter: [39.9042, 116.4074], // Beijing
  defaultZoom: 10,
  minZoom: 1,
  maxZoom: 20,
  baseLayers: [
    {
      id: 'osm',
      name: 'OpenStreetMap',
      type: 'tile',
      visible: true,
      opacity: 1,
      zIndex: 0,
      url: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png'
    }
  ],
  overlayLayers: []
};