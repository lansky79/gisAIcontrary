import React, { useEffect, useRef, useMemo } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Circle, useMap } from 'react-leaflet';
import { Card, Tag, Space, Tooltip, Button } from 'antd';
import { ZoomInOutlined, FullscreenOutlined } from '@ant-design/icons';
import L, { LatLngBounds } from 'leaflet';
import 'leaflet/dist/leaflet.css';
import type { 
  MapComponentProps, 
  DetectionResult, 
  ChangeRegion,
  CHANGE_TYPE_COLORS,
  CONFIDENCE_LEVELS 
} from '@types/index';

// 修复 Leaflet 默认图标问题
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
});

// 变化类型颜色映射
const CHANGE_TYPE_COLORS = {
  building: '#f44336',
  vegetation: '#4caf50',
  road: '#9e9e9e',
  water: '#2196f3',
  unknown: '#ff9800',
  large_area: '#e91e63',
  small_object: '#9c27b0',
  linear_structure: '#607d8b'
};

// 置信度级别
const CONFIDENCE_LEVELS = {
  HIGH: 0.8,
  MEDIUM: 0.5,
  LOW: 0.3
} as const;

// 地图控制组件
const MapControls: React.FC<{
  onZoomToFit?: () => void;
  onFullscreen?: () => void;
}> = ({ onZoomToFit, onFullscreen }) => {
  return (
    <div 
      style={{
        position: 'absolute',
        top: 10,
        right: 10,
        zIndex: 1000,
        display: 'flex',
        flexDirection: 'column',
        gap: 8,
      }}
    >
      {onZoomToFit && (
        <Button
          size="small"
          icon={<ZoomInOutlined />}
          onClick={onZoomToFit}
          style={{ backgroundColor: 'white' }}
        >
          适应视图
        </Button>
      )}
      {onFullscreen && (
        <Button
          size="small"
          icon={<FullscreenOutlined />}
          onClick={onFullscreen}
          style={{ backgroundColor: 'white' }}
        >
          全屏
        </Button>
      )}
    </div>
  );
};

// 变化区域图例组件
const ChangeLegend: React.FC<{ visible?: boolean }> = ({ visible = true }) => {
  if (!visible) return null;

  return (
    <div 
      style={{
        position: 'absolute',
        bottom: 20,
        left: 20,
        zIndex: 1000,
        background: 'white',
        padding: 12,
        borderRadius: 6,
        boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
        fontSize: 12,
        minWidth: 150,
      }}
    >
      <div style={{ fontWeight: 'bold', marginBottom: 8 }}>变化类型</div>
      {Object.entries(CHANGE_TYPE_COLORS).map(([type, color]) => (
        <div key={type} style={{ display: 'flex', alignItems: 'center', margin: '4px 0' }}>
          <div 
            style={{
              width: 12,
              height: 12,
              backgroundColor: color,
              borderRadius: '50%',
              marginRight: 8,
              border: '1px solid rgba(0,0,0,0.2)',
            }}
          />
          <span>{getChangeTypeLabel(type as keyof typeof CHANGE_TYPE_COLORS)}</span>
        </div>
      ))}
    </div>
  );
};

// 获取变化类型中文标签
const getChangeTypeLabel = (type: keyof typeof CHANGE_TYPE_COLORS): string => {
  const labels = {
    building: '建筑物',
    vegetation: '植被',
    road: '道路',
    water: '水体',
    unknown: '未知',
    large_area: '大面积',
    small_object: '小目标',
    linear_structure: '线性结构'
  };
  return labels[type] || type;
};

// 获取置信度级别
const getConfidenceLevel = (confidence: number): 'high' | 'medium' | 'low' => {
  if (confidence >= CONFIDENCE_LEVELS.HIGH) return 'high';
  if (confidence >= CONFIDENCE_LEVELS.MEDIUM) return 'medium';
  return 'low';
};

// 获取置信度颜色
const getConfidenceColor = (confidence: number): string => {
  const level = getConfidenceLevel(confidence);
  switch (level) {
    case 'high': return '#52c41a';
    case 'medium': return '#faad14';
    case 'low': return '#ff4d4f';
    default: return '#d9d9d9';
  }
};

// 变化区域标记组件
const ChangeRegionMarker: React.FC<{
  region: ChangeRegion;
  onRegionClick?: (region: ChangeRegion) => void;
}> = ({ region, onRegionClick }) => {
  // 如果没有地理坐标，不显示标记
  if (!region.geoCoordinates?.bounds) {
    return null;
  }

  const [center] = region.geoCoordinates.bounds;
  const position: [number, number] = [center[0], center[1]];
  
  // 根据面积计算圆圈半径
  const radius = Math.max(10, Math.min(100, Math.sqrt(region.area) / 10));
  
  const color = CHANGE_TYPE_COLORS[region.changeType] || '#ff9800';
  const confidenceLevel = getConfidenceLevel(region.confidence);

  return (
    <>
      {/* 变化区域圆圈 */}
      <Circle
        center={position}
        radius={radius}
        pathOptions={{
          color: color,
          fillColor: color,
          fillOpacity: 0.3,
          weight: 2,
        }}
        eventHandlers={{
          click: () => onRegionClick?.(region),
        }}
      />
      
      {/* 中心标记点 */}
      <Marker 
        position={position}
        eventHandlers={{
          click: () => onRegionClick?.(region),
        }}
        icon={L.divIcon({
          className: 'change-marker',
          html: `<div style="
            background: ${color};
            width: 16px;
            height: 16px;
            border-radius: 50%;
            border: 2px solid white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
          "></div>`,
          iconSize: [16, 16],
          iconAnchor: [8, 8],
        })}
      >
        <Popup>
          <Card 
            size="small" 
            title={`变化区域 #${region.id}`}
            style={{ minWidth: 250 }}
          >
            <Space direction="vertical" size="small" style={{ width: '100%' }}>
              <div>
                <strong>变化类型:</strong>{' '}
                <Tag color={color}>
                  {getChangeTypeLabel(region.changeType)}
                </Tag>
              </div>
              
              <div>
                <strong>面积:</strong> {region.area.toFixed(0)} 像素
              </div>
              
              <div>
                <strong>置信度:</strong>{' '}
                <Tag color={getConfidenceColor(region.confidence)}>
                  {(region.confidence * 100).toFixed(1)}%
                </Tag>
              </div>
              
              <div>
                <strong>中心坐标:</strong><br />
                纬度: {region.centroid[0].toFixed(6)}<br />
                经度: {region.centroid[1].toFixed(6)}
              </div>
              
              {region.geoCoordinates?.bounds && (
                <div>
                  <strong>边界范围:</strong><br />
                  {region.geoCoordinates.bounds.map((coord, idx) => (
                    <div key={idx} style={{ fontSize: 11, color: '#666' }}>
                      {coord[0].toFixed(4)}, {coord[1].toFixed(4)}
                    </div>
                  ))}
                </div>
              )}
            </Space>
          </Card>
        </Popup>
      </Marker>
    </>
  );
};

// 地图视图适配组件
const MapViewAdapter: React.FC<{
  detectionResults?: DetectionResult[];
  center?: [number, number];
  zoom?: number;
}> = ({ detectionResults, center, zoom }) => {
  const map = useMap();

  useEffect(() => {
    if (center && zoom) {
      map.setView(center, zoom);
    } else if (detectionResults && detectionResults.length > 0) {
      // 自动适配到包含所有变化区域的视图
      const allBounds: [number, number][] = [];
      
      detectionResults.forEach(result => {
        result.changeRegions.forEach(region => {
          if (region.geoCoordinates?.bounds) {
            allBounds.push(...region.geoCoordinates.bounds);
          }
        });
      });
      
      if (allBounds.length > 0) {
        const latLngBounds = new LatLngBounds(allBounds);
        map.fitBounds(latLngBounds, { padding: [20, 20] });
      }
    }
  }, [map, detectionResults, center, zoom]);

  return null;
};

// 主地图组件
const MapComponent: React.FC<MapComponentProps> = ({
  height = '100%',
  width = '100%',
  center = [39.9042, 116.4074], // 默认北京
  zoom = 10,
  detectionResults = [],
  onRegionSelect,
  onMapClick,
  showControls = true,
  showLegend = true,
}) => {
  const mapRef = useRef<L.Map>(null);

  // 处理地图点击事件
  const handleMapClick = (e: L.LeafletMouseEvent) => {
    const { lat, lng } = e.latlng;
    onMapClick?.([lat, lng]);
  };

  // 缩放到适应所有变化区域
  const handleZoomToFit = () => {
    if (!mapRef.current || !detectionResults.length) return;

    const allBounds: [number, number][] = [];
    detectionResults.forEach(result => {
      result.changeRegions.forEach(region => {
        if (region.geoCoordinates?.bounds) {
          allBounds.push(...region.geoCoordinates.bounds);
        }
      });
    });

    if (allBounds.length > 0) {
      const latLngBounds = new LatLngBounds(allBounds);
      mapRef.current.fitBounds(latLngBounds, { padding: [20, 20] });
    }
  };

  // 全屏功能
  const handleFullscreen = () => {
    if (mapRef.current) {
      const mapContainer = mapRef.current.getContainer().parentElement;
      if (mapContainer) {
        if (document.fullscreenElement) {
          document.exitFullscreen();
        } else {
          mapContainer.requestFullscreen();
        }
      }
    }
  };

  // 收集所有变化区域用于渲染
  const allChangeRegions = useMemo(() => {
    const regions: ChangeRegion[] = [];
    detectionResults.forEach(result => {
      regions.push(...result.changeRegions);
    });
    return regions;
  }, [detectionResults]);

  return (
    <div style={{ position: 'relative', height, width }}>
      <MapContainer
        ref={mapRef}
        center={center}
        zoom={zoom}
        style={{ height: '100%', width: '100%' }}
        zoomControl={true}
        scrollWheelZoom={true}
        doubleClickZoom={true}
        boxZoom={true}
        eventHandlers={{
          click: handleMapClick,
        }}
      >
        {/* 底图图层 */}
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        {/* 视图适配器 */}
        <MapViewAdapter 
          detectionResults={detectionResults}
          center={center}
          zoom={zoom}
        />

        {/* 渲染所有变化区域 */}
        {allChangeRegions.map((region, index) => (
          <ChangeRegionMarker
            key={`${region.id}-${index}`}
            region={region}
            onRegionClick={onRegionSelect}
          />
        ))}
      </MapContainer>

      {/* 地图控件 */}
      {showControls && (
        <MapControls
          onZoomToFit={allChangeRegions.length > 0 ? handleZoomToFit : undefined}
          onFullscreen={handleFullscreen}
        />
      )}

      {/* 图例 */}
      {showLegend && allChangeRegions.length > 0 && (
        <ChangeLegend visible={true} />
      )}

      {/* 无数据提示 */}
      {detectionResults.length === 0 && (
        <div 
          style={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            zIndex: 1000,
            background: 'rgba(255, 255, 255, 0.9)',
            padding: 20,
            borderRadius: 8,
            textAlign: 'center',
            color: '#666',
          }}
        >
          <div style={{ fontSize: 16, marginBottom: 8 }}>暂无检测结果</div>
          <div style={{ fontSize: 12 }}>上传图像并执行变化检测后，结果将在此显示</div>
        </div>
      )}
    </div>
  );
};

export default MapComponent;