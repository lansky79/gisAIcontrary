#!/usr/bin/env python3
"""
变化区域标注和报告生成模块 - 第一部分
包含核心数据结构和图像标注功能
"""

import os
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)

@dataclass
class AnnotationInfo:
    """标注信息"""
    id: str
    region_id: str
    annotation_type: str  # 'manual', 'auto', 'corrected'
    label: str
    confidence: float
    user_id: Optional[str] = None
    timestamp: datetime = None
    notes: str = ""

@dataclass
class ChangeRegionDetailed:
    """详细的变化区域信息"""
    id: str
    area_pixels: float
    area_real: float  # 实际面积（平方米）
    centroid: Tuple[float, float]  # GPS坐标
    bounding_box: Dict[str, int]
    polygon_coords: List[Tuple[float, float]]  # GPS多边形坐标
    change_type: str
    confidence: float
    annotations: List[AnnotationInfo]
    detected_at: datetime
    verified: bool = False
    verification_user: Optional[str] = None
    verification_time: Optional[datetime] = None

class ImageAnnotator:
    """图像标注器"""
    
    def __init__(self):
        self.colors = {
            'building': (0, 0, 255),      # 红色
            'vegetation': (0, 255, 0),    # 绿色
            'road': (128, 128, 128),      # 灰色
            'water': (255, 0, 0),         # 蓝色
            'unknown': (0, 255, 255),     # 黄色
            'large_area': (255, 0, 255),  # 品红
            'small_object': (128, 0, 128), # 紫色
            'linear_structure': (0, 128, 128) # 青色
        }
    
    def annotate_change_regions(self, 
                               base_image: np.ndarray,
                               regions: List[Dict[str, Any]],
                               show_confidence: bool = True,
                               show_labels: bool = True,
                               show_area: bool = True) -> np.ndarray:
        """在图像上标注变化区域"""
        
        annotated_image = base_image.copy()
        
        for region in regions:
            # 获取区域信息
            change_type = region.get('change_type', 'unknown')
            confidence = region.get('confidence', 0.0)
            area = region.get('area', 0)
            bbox = region.get('bounding_box', {})
            
            # 获取颜色
            color = self.colors.get(change_type, (0, 255, 255))
            
            # 绘制边界框
            if bbox and all(key in bbox for key in ['x', 'y', 'width', 'height']):
                x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
                
                # 根据置信度调整线条粗细
                thickness = max(1, int(confidence * 5))
                cv2.rectangle(annotated_image, (x, y), (x + w, y + h), color, thickness)
                
                # 添加半透明填充
                overlay = annotated_image.copy()
                cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
                alpha = 0.2 * confidence
                annotated_image = cv2.addWeighted(annotated_image, 1 - alpha, overlay, alpha, 0)
                
                # 添加文本标注
                label_parts = []
                if show_labels:
                    label_parts.append(self._get_change_type_label(change_type))
                if show_confidence:
                    label_parts.append(f"{confidence*100:.1f}%")
                if show_area:
                    label_parts.append(f"{area:.0f}px")
                
                if label_parts:
                    label = " | ".join(label_parts)
                    
                    # 计算文本大小
                    font_scale = 0.6
                    font_thickness = 1
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
                    )
                    
                    # 绘制文本背景
                    text_x, text_y = x, y - 10 if y > 30 else y + h + 20
                    cv2.rectangle(annotated_image, 
                                (text_x - 2, text_y - text_height - 2),
                                (text_x + text_width + 2, text_y + baseline + 2),
                                (0, 0, 0), -1)
                    
                    # 绘制文本
                    cv2.putText(annotated_image, label, (text_x, text_y),
                              cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
        
        return annotated_image
    
    def create_legend(self, 
                     image_width: int, 
                     change_types: List[str],
                     title: str = "变化类型图例") -> np.ndarray:
        """创建图例"""
        
        legend_height = len(change_types) * 30 + 60
        legend_width = 200
        
        legend = np.ones((legend_height, legend_width, 3), dtype=np.uint8) * 255
        
        # 标题
        cv2.putText(legend, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # 图例项
        y_start = 50
        for i, change_type in enumerate(change_types):
            y = y_start + i * 30
            color = self.colors.get(change_type, (0, 255, 255))
            
            # 绘制颜色块
            cv2.rectangle(legend, (10, y - 8), (30, y + 8), color, -1)
            cv2.rectangle(legend, (10, y - 8), (30, y + 8), (0, 0, 0), 1)
            
            # 绘制标签
            label = self._get_change_type_label(change_type)
            cv2.putText(legend, label, (35, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return legend
    
    def _get_change_type_label(self, change_type: str) -> str:
        """获取变化类型的中文标签"""
        labels = {
            'building': '建筑物',
            'vegetation': '植被',
            'road': '道路',
            'water': '水体',
            'unknown': '未知',
            'large_area': '大面积',
            'small_object': '小目标',
            'linear_structure': '线性结构'
        }
        return labels.get(change_type, change_type)

class StatisticsCalculator:
    """统计计算器"""
    
    @staticmethod
    def calculate_detection_statistics(regions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算检测统计信息"""
        
        if not regions:
            return {
                'total_regions': 0,
                'total_area': 0,
                'average_area': 0,
                'change_type_distribution': {},
                'confidence_distribution': {
                    'high': 0, 'medium': 0, 'low': 0
                },
                'size_distribution': {
                    'large': 0, 'medium': 0, 'small': 0
                }
            }
        
        total_area = sum(region.get('area', 0) for region in regions)
        average_area = total_area / len(regions)
        
        # 变化类型分布
        change_type_count = {}
        for region in regions:
            change_type = region.get('change_type', 'unknown')
            change_type_count[change_type] = change_type_count.get(change_type, 0) + 1
        
        # 置信度分布
        confidence_dist = {'high': 0, 'medium': 0, 'low': 0}
        for region in regions:
            confidence = region.get('confidence', 0)
            if confidence >= 0.8:
                confidence_dist['high'] += 1
            elif confidence >= 0.5:
                confidence_dist['medium'] += 1
            else:
                confidence_dist['low'] += 1
        
        # 尺寸分布
        areas = [region.get('area', 0) for region in regions]
        area_75 = np.percentile(areas, 75) if areas else 0
        area_25 = np.percentile(areas, 25) if areas else 0
        
        size_dist = {'large': 0, 'medium': 0, 'small': 0}
        for area in areas:
            if area >= area_75:
                size_dist['large'] += 1
            elif area >= area_25:
                size_dist['medium'] += 1
            else:
                size_dist['small'] += 1
        
        return {
            'total_regions': len(regions),
            'total_area': total_area,
            'average_area': average_area,
            'median_area': np.median(areas) if areas else 0,
            'max_area': max(areas) if areas else 0,
            'min_area': min(areas) if areas else 0,
            'change_type_distribution': change_type_count,
            'confidence_distribution': confidence_dist,
            'size_distribution': size_dist
        }

class ReportGenerator:
    """报告生成器"""
    
    def __init__(self):
        self.annotator = ImageAnnotator()
        self.stats_calculator = StatisticsCalculator()
    
    def generate_annotated_image(self, 
                               result_image_path: str,
                               regions: List[Dict[str, Any]],
                               output_path: Optional[str] = None) -> str:
        """生成标注图像"""
        
        # 读取结果图像
        result_image = cv2.imread(result_image_path)
        if result_image is None:
            raise ValueError(f"Cannot read image: {result_image_path}")
        
        # 标注变化区域
        annotated_image = self.annotator.annotate_change_regions(
            result_image, regions, 
            show_confidence=True, 
            show_labels=True, 
            show_area=True
        )
        
        # 添加图例
        if regions:
            change_types = list(set(region.get('change_type', 'unknown') for region in regions))
            legend = self.annotator.create_legend(result_image.shape[1], change_types)
            
            # 将图例添加到图像右侧
            annotated_image = np.hstack([annotated_image, legend])
        
        # 保存标注图像
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"results/annotated/annotated_{timestamp}.jpg"
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, annotated_image)
        
        return output_path
    
    def generate_json_report(self, 
                           detection_result: Dict[str, Any],
                           output_path: Optional[str] = None) -> str:
        """生成JSON格式报告"""
        
        # 计算详细统计信息
        regions = detection_result.get('change_regions', [])
        statistics = self.stats_calculator.calculate_detection_statistics(regions)
        
        # 构建报告数据
        report_data = {
            'report_id': str(uuid.uuid4()),
            'generated_at': datetime.now().isoformat(),
            'task_id': detection_result.get('task_id', ''),
            'algorithm_type': detection_result.get('algorithmType', 'unknown'),
            'detection_summary': {
                'total_regions': len(regions),
                'change_percentage': detection_result.get('detectionResults', {}).get('changePercentage', 0),
                'total_change_area': detection_result.get('detectionResults', {}).get('totalChangeAreaPixels', 0),
                'processing_time': detection_result.get('detectionResults', {}).get('processingTime', 0)
            },
            'registration_info': detection_result.get('registrationResults', {}),
            'change_regions': regions,
            'statistics': statistics,
            'metadata': {
                'gps_info': detection_result.get('gpsInfo', {}),
                'timestamp': detection_result.get('timestamp', ''),
                'description': detection_result.get('description', '')
            }
        }
        
        # 保存JSON报告
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"results/reports/report_{timestamp}.json"
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        return output_path