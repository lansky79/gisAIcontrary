#!/usr/bin/env python3
"""
地理测绘变化检测系统 - 核心算法模块
包含高精度图像配准和智能变化检测算法
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from scipy import ndimage
from skimage import filters, morphology, measure, segmentation
from skimage.feature import peak_local_maxima
import math

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RegistrationResult:
    """图像配准结果数据类"""
    transform_matrix: np.ndarray
    aligned_image: np.ndarray
    confidence_score: float
    feature_matches: int
    inlier_ratio: float
    registration_error: float

@dataclass
class ChangeRegion:
    """变化区域数据类"""
    contour: np.ndarray
    area: float
    centroid: Tuple[float, float]
    bounding_box: Tuple[int, int, int, int]  # x, y, w, h
    confidence: float
    change_type: str

@dataclass
class DetectionResult:
    """变化检测结果数据类"""
    change_mask: np.ndarray
    change_regions: List[ChangeRegion]
    total_change_area: float
    change_percentage: float
    confidence_map: np.ndarray
    processing_time: float

class AdvancedImageRegistration:
    """高级图像配准算法"""
    
    def __init__(self):
        self.sift = cv2.SIFT_create(nfeatures=2000)
        self.surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400)
        self.orb = cv2.ORB_create(nfeatures=3000)
        
        # FLANN匹配器参数
        FLANN_INDEX_KDTREE = 1
        self.flann_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        self.search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(self.flann_params, self.search_params)
        
    def extract_features_multi_method(self, image: np.ndarray) -> Dict[str, Tuple]:
        """使用多种方法提取特征点"""
        
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        features = {}
        
        try:
            # SIFT特征 - 尺度不变
            kp_sift, des_sift = self.sift.detectAndCompute(gray, None)
            features['sift'] = (kp_sift, des_sift)
            logger.info(f"SIFT检测到 {len(kp_sift)} 个特征点")
        except Exception as e:
            logger.warning(f"SIFT特征提取失败: {e}")
            features['sift'] = ([], None)
            
        try:
            # SURF特征 - 快速稳健
            kp_surf, des_surf = self.surf.detectAndCompute(gray, None)
            features['surf'] = (kp_surf, des_surf)
            logger.info(f"SURF检测到 {len(kp_surf)} 个特征点")
        except Exception as e:
            logger.warning(f"SURF特征提取失败: {e}")
            features['surf'] = ([], None)
            
        try:
            # ORB特征 - 实时性能
            kp_orb, des_orb = self.orb.detectAndCompute(gray, None)
            features['orb'] = (kp_orb, des_orb)
            logger.info(f"ORB检测到 {len(kp_orb)} 个特征点")
        except Exception as e:
            logger.warning(f"ORB特征提取失败: {e}")
            features['orb'] = ([], None)
            
        return features
    
    def match_features_robust(self, features1: Dict, features2: Dict) -> List[cv2.DMatch]:
        """稳健的特征匹配"""
        
        all_matches = []
        
        # 尝试SIFT匹配
        if (features1['sift'][1] is not None and features2['sift'][1] is not None and 
            len(features1['sift'][1]) > 10 and len(features2['sift'][1]) > 10):
            
            try:
                matches_sift = self.matcher.knnMatch(
                    features1['sift'][1], features2['sift'][1], k=2
                )
                
                # Lowe's比率测试
                good_matches = []
                for match_pair in matches_sift:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.7 * n.distance:
                            good_matches.append(m)
                
                # 添加特征点信息
                for match in good_matches:
                    match.queryIdx_method = 'sift'
                    match.trainIdx_method = 'sift'
                    match.kp1 = features1['sift'][0][match.queryIdx]
                    match.kp2 = features2['sift'][0][match.trainIdx]
                    
                all_matches.extend(good_matches)
                logger.info(f"SIFT匹配到 {len(good_matches)} 个点对")
                
            except Exception as e:
                logger.warning(f"SIFT匹配失败: {e}")
        
        # 尝试SURF匹配（如果SIFT匹配不足）
        if (len(all_matches) < 20 and 
            features1['surf'][1] is not None and features2['surf'][1] is not None):
            
            try:
                matches_surf = self.matcher.knnMatch(
                    features1['surf'][1], features2['surf'][1], k=2
                )
                
                good_matches = []
                for match_pair in matches_surf:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.7 * n.distance:
                            good_matches.append(m)
                
                for match in good_matches:
                    match.queryIdx_method = 'surf'
                    match.trainIdx_method = 'surf'
                    match.kp1 = features1['surf'][0][match.queryIdx]
                    match.kp2 = features2['surf'][0][match.trainIdx]
                    
                all_matches.extend(good_matches)
                logger.info(f"SURF匹配到 {len(good_matches)} 个点对")
                
            except Exception as e:
                logger.warning(f"SURF匹配失败: {e}")
        
        # ORB作为备选方案
        if (len(all_matches) < 10 and 
            features1['orb'][1] is not None and features2['orb'][1] is not None):
            
            try:
                bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches_orb = bf_matcher.match(features1['orb'][1], features2['orb'][1])
                matches_orb = sorted(matches_orb, key=lambda x: x.distance)
                
                # 取前50个最佳匹配
                good_matches = matches_orb[:50]
                
                for match in good_matches:
                    match.queryIdx_method = 'orb'
                    match.trainIdx_method = 'orb'
                    match.kp1 = features1['orb'][0][match.queryIdx]
                    match.kp2 = features2['orb'][0][match.trainIdx]
                    
                all_matches.extend(good_matches)
                logger.info(f"ORB匹配到 {len(good_matches)} 个点对")
                
            except Exception as e:
                logger.warning(f"ORB匹配失败: {e}")
        
        return all_matches
    
    def estimate_transform_robust(self, matches: List[cv2.DMatch], 
                                 image_shape: Tuple[int, int]) -> RegistrationResult:
        """稳健的变换矩阵估计"""
        
        if len(matches) < 4:
            logger.warning("匹配点不足，无法进行配准")
            return RegistrationResult(
                transform_matrix=np.eye(3),
                aligned_image=np.zeros(image_shape),
                confidence_score=0.0,
                feature_matches=len(matches),
                inlier_ratio=0.0,
                registration_error=float('inf')
            )
        
        # 提取匹配点坐标
        src_pts = np.float32([match.kp2.pt for match in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([match.kp1.pt for match in matches]).reshape(-1, 1, 2)
        
        best_transform = None
        best_inliers = 0
        best_error = float('inf')
        
        # 尝试不同的变换模型
        transform_methods = [
            ('homography', cv2.findHomography),
            ('affine', cv2.estimateAffinePartial2D),
        ]
        
        for method_name, method_func in transform_methods:
            try:
                if method_name == 'homography':
                    M, mask = method_func(
                        src_pts, dst_pts, 
                        cv2.RANSAC, 
                        ransacReprojThreshold=3.0,
                        maxIters=2000,
                        confidence=0.995
                    )
                    if M is not None:
                        # 转换为3x3矩阵
                        if M.shape[0] == 2:
                            transform_matrix = np.vstack([M, [0, 0, 1]])
                        else:
                            transform_matrix = M
                else:  # affine
                    result = method_func(
                        src_pts, dst_pts,
                        cv2.RANSAC,
                        ransacReprojThreshold=3.0,
                        maxIters=2000,
                        confidence=0.995
                    )
                    if result[0] is not None:
                        M, mask = result[0], result[1]
                        # 将仿射变换转换为3x3齐次矩阵
                        transform_matrix = np.vstack([M, [0, 0, 1]])
                    else:
                        continue
                
                if mask is not None:
                    inliers = np.sum(mask)
                    inlier_ratio = inliers / len(matches)
                    
                    # 计算重投影误差
                    if transform_matrix is not None:
                        projected_pts = cv2.perspectiveTransform(src_pts, transform_matrix)
                        errors = np.sqrt(np.sum((projected_pts - dst_pts) ** 2, axis=2))
                        mean_error = np.mean(errors[mask.flatten() == 1])
                        
                        # 选择最佳变换
                        if inliers > best_inliers or (inliers == best_inliers and mean_error < best_error):
                            best_transform = transform_matrix
                            best_inliers = inliers
                            best_error = mean_error
                            
                        logger.info(f"{method_name}变换: 内点数={inliers}, 比例={inlier_ratio:.3f}, 误差={mean_error:.2f}")
                        
            except Exception as e:
                logger.warning(f"{method_name}变换估计失败: {e}")
                continue
        
        # 计算配准置信度
        if best_transform is not None and best_inliers > 0:
            inlier_ratio = best_inliers / len(matches)
            confidence = min(1.0, inlier_ratio * math.exp(-best_error / 5.0))
        else:
            confidence = 0.0
            best_transform = np.eye(3)
            
        return RegistrationResult(
            transform_matrix=best_transform,
            aligned_image=np.zeros(image_shape),  # 稍后填充
            confidence_score=confidence,
            feature_matches=len(matches),
            inlier_ratio=best_inliers / len(matches) if len(matches) > 0 else 0.0,
            registration_error=best_error
        )
    
    def register_images_advanced(self, img1: np.ndarray, img2: np.ndarray) -> RegistrationResult:
        """高级图像配准主方法"""
        
        logger.info("开始高级图像配准...")
        
        # 1. 多方法特征提取
        features1 = self.extract_features_multi_method(img1)
        features2 = self.extract_features_multi_method(img2)
        
        # 2. 稳健特征匹配
        matches = self.match_features_robust(features1, features2)
        
        if len(matches) < 4:
            logger.warning("特征匹配失败，返回单位变换")
            return RegistrationResult(
                transform_matrix=np.eye(3),
                aligned_image=img2.copy(),
                confidence_score=0.0,
                feature_matches=0,
                inlier_ratio=0.0,
                registration_error=float('inf')
            )
        
        # 3. 变换矩阵估计
        result = self.estimate_transform_robust(matches, img1.shape[:2])
        
        # 4. 应用变换
        if result.confidence_score > 0.1:
            h, w = img1.shape[:2]
            aligned_img = cv2.warpPerspective(img2, result.transform_matrix, (w, h))
            result.aligned_image = aligned_img
            logger.info(f"配准完成，置信度: {result.confidence_score:.3f}")
        else:
            result.aligned_image = img2.copy()
            logger.warning("配准置信度过低，使用原图")
        
        return result

class AdvancedChangeDetection:
    """高级变化检测算法"""
    
    def __init__(self):
        self.min_change_area = 100  # 最小变化区域面积
        self.noise_filter_size = 5  # 噪声过滤核大小
        
    def preprocess_for_detection(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """变化检测预处理"""
        
        # 确保图像尺寸一致
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        if (h1, w1) != (h2, w2):
            # 调整到最小公共尺寸
            min_h, min_w = min(h1, h2), min(w1, w2)
            img1 = cv2.resize(img1, (min_w, min_h))
            img2 = cv2.resize(img2, (min_w, min_h))
        
        # 转换为灰度图
        if len(img1.shape) == 3:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = img1.copy()
            
        if len(img2.shape) == 3:
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            gray2 = img2.copy()
        
        # 直方图均衡化（减少光照影响）
        gray1 = cv2.equalizeHist(gray1)
        gray2 = cv2.equalizeHist(gray2)
        
        # 高斯滤波降噪
        gray1 = cv2.GaussianBlur(gray1, (5, 5), 1.5)
        gray2 = cv2.GaussianBlur(gray2, (5, 5), 1.5)
        
        return gray1, gray2
    
    def multi_scale_detection(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """多尺度变化检测"""
        
        scales = [1.0, 0.5, 0.25]  # 不同尺度
        change_maps = []
        
        for scale in scales:
            if scale != 1.0:
                # 缩放图像
                h, w = img1.shape[:2]
                new_h, new_w = int(h * scale), int(w * scale)
                scaled_img1 = cv2.resize(img1, (new_w, new_h))
                scaled_img2 = cv2.resize(img2, (new_w, new_h))
            else:
                scaled_img1, scaled_img2 = img1, img2
            
            # 计算差分
            diff = cv2.absdiff(scaled_img1, scaled_img2)
            
            # 自适应阈值
            threshold = np.percentile(diff, 95)  # 95%分位数作为阈值
            _, binary = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
            
            # 缩放回原尺寸
            if scale != 1.0:
                binary = cv2.resize(binary, (w, h))
            
            change_maps.append(binary)
        
        # 融合多尺度结果
        combined = np.zeros_like(change_maps[0])
        for change_map in change_maps:
            combined = cv2.bitwise_or(combined, change_map)
        
        return combined
    
    def morphological_refinement(self, binary_mask: np.ndarray) -> np.ndarray:
        """形态学精细化处理"""
        
        # 去除小噪声
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_small)
        
        # 填充小洞
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        filled = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_medium)
        
        # 连接相近区域
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        connected = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel_large)
        
        return connected
    
    def extract_change_regions(self, change_mask: np.ndarray, 
                              min_area: int = None) -> List[ChangeRegion]:
        """提取变化区域"""
        
        if min_area is None:
            min_area = self.min_change_area
        
        # 查找轮廓
        contours, _ = cv2.findContours(
            change_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        change_regions = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area >= min_area:
                # 计算质心
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                else:
                    cx, cy = 0, 0
                
                # 边界框
                x, y, w, h = cv2.boundingRect(contour)
                
                # 计算置信度（基于区域规整度）
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    confidence = min(1.0, circularity + 0.2)  # 基础置信度
                else:
                    confidence = 0.5
                
                # 简单的变化类型分类
                aspect_ratio = w / h if h > 0 else 1
                if area > 10000:  # 大区域
                    if aspect_ratio > 3 or aspect_ratio < 0.33:
                        change_type = "linear_structure"  # 可能是道路、河流等
                    else:
                        change_type = "large_area"  # 可能是建筑群、农田等
                elif area > 1000:
                    change_type = "building"  # 可能是建筑物
                else:
                    change_type = "small_object"  # 小目标
                
                region = ChangeRegion(
                    contour=contour,
                    area=area,
                    centroid=(cx, cy),
                    bounding_box=(x, y, w, h),
                    confidence=confidence,
                    change_type=change_type
                )
                
                change_regions.append(region)
        
        # 按面积排序
        change_regions.sort(key=lambda r: r.area, reverse=True)
        
        return change_regions
    
    def create_confidence_map(self, img1: np.ndarray, img2: np.ndarray, 
                             change_mask: np.ndarray) -> np.ndarray:
        """创建置信度图"""
        
        # 计算梯度强度（边缘信息）
        grad1 = cv2.Sobel(img1, cv2.CV_64F, 1, 1, ksize=3)
        grad2 = cv2.Sobel(img2, cv2.CV_64F, 1, 1, ksize=3)
        grad_strength = np.abs(grad1) + np.abs(grad2)
        grad_strength = cv2.GaussianBlur(grad_strength, (5, 5), 0)
        
        # 归一化到0-255
        grad_strength = cv2.normalize(grad_strength, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # 结合变化掩码和梯度信息
        confidence_map = np.zeros_like(change_mask, dtype=np.float32)
        confidence_map[change_mask > 0] = grad_strength[change_mask > 0] / 255.0
        
        # 高斯平滑
        confidence_map = cv2.GaussianBlur(confidence_map, (7, 7), 2.0)
        
        return (confidence_map * 255).astype(np.uint8)
    
    def detect_changes_advanced(self, img1: np.ndarray, img2: np.ndarray) -> DetectionResult:
        """高级变化检测主方法"""
        
        import time
        start_time = time.time()
        
        logger.info("开始高级变化检测...")
        
        # 1. 图像预处理
        processed_img1, processed_img2 = self.preprocess_for_detection(img1, img2)
        
        # 2. 多尺度变化检测
        change_mask = self.multi_scale_detection(processed_img1, processed_img2)
        
        # 3. 形态学精细化
        refined_mask = self.morphological_refinement(change_mask)
        
        # 4. 提取变化区域
        change_regions = self.extract_change_regions(refined_mask)
        
        # 5. 创建置信度图
        confidence_map = self.create_confidence_map(processed_img1, processed_img2, refined_mask)
        
        # 6. 计算统计信息
        total_pixels = refined_mask.shape[0] * refined_mask.shape[1]
        total_change_area = np.sum(refined_mask > 0)
        change_percentage = (total_change_area / total_pixels) * 100
        
        processing_time = time.time() - start_time
        
        logger.info(f"变化检测完成，耗时: {processing_time:.2f}秒")
        logger.info(f"检测到 {len(change_regions)} 个变化区域，总变化比例: {change_percentage:.2f}%")
        
        return DetectionResult(
            change_mask=refined_mask,
            change_regions=change_regions,
            total_change_area=float(total_change_area),
            change_percentage=change_percentage,
            confidence_map=confidence_map,
            processing_time=processing_time
        )

# 算法集成类
class GeoChangeDetectionEngine:
    """地理变化检测引擎 - 集成配准和检测算法"""
    
    def __init__(self):
        self.registration = AdvancedImageRegistration()
        self.detection = AdvancedChangeDetection()
        
    def process_image_pair(self, img1: np.ndarray, img2: np.ndarray, 
                          gps_hints: Optional[Dict] = None) -> Dict[str, Any]:
        """处理图像对，返回完整的分析结果"""
        
        logger.info("开始处理图像对...")
        
        # 1. 图像配准
        registration_result = self.registration.register_images_advanced(img1, img2)
        
        # 2. 变化检测
        if registration_result.confidence_score > 0.3:
            # 使用配准后的图像
            detection_result = self.detection.detect_changes_advanced(
                img1, registration_result.aligned_image
            )
        else:
            # 配准失败，使用原图像（可能效果较差）
            logger.warning("配准置信度较低，使用原图像进行检测")
            detection_result = self.detection.detect_changes_advanced(img1, img2)
        
        # 3. 整合结果
        result = {
            'registration': {
                'confidence_score': registration_result.confidence_score,
                'feature_matches': registration_result.feature_matches,
                'inlier_ratio': registration_result.inlier_ratio,
                'registration_error': registration_result.registration_error,
                'transform_applied': registration_result.confidence_score > 0.3
            },
            'detection': {
                'change_regions_count': len(detection_result.change_regions),
                'total_change_area_pixels': detection_result.total_change_area,
                'change_percentage': detection_result.change_percentage,
                'processing_time': detection_result.processing_time
            },
            'change_regions': [
                {
                    'area': region.area,
                    'centroid': region.centroid,
                    'bounding_box': region.bounding_box,
                    'confidence': region.confidence,
                    'change_type': region.change_type
                }
                for region in detection_result.change_regions
            ],
            'visualization_data': {
                'change_mask': detection_result.change_mask,
                'confidence_map': detection_result.confidence_map,
                'aligned_image': registration_result.aligned_image
            }
        }
        
        logger.info("图像对处理完成")
        return result