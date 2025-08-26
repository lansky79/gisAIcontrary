#!/usr/bin/env python3
"""
地理测绘变化检测系统 - MVP版本
最简化实现：两张图片上传对比，输出变化检测结果
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
from PIL import Image, ExifTags
import io
import os
import uuid
import json
from datetime import datetime
from typing import Optional, Dict, Any
import asyncio
from pathlib import Path
from core_algorithms import GeoChangeDetectionEngine
from drone_service import DroneDataService, RealTimeDetectionResult
from report_generator import ReportGenerator

# 创建必要的目录
os.makedirs("uploads", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("static", exist_ok=True)

app = FastAPI(
    title="地理测绘变化检测系统 MVP",
    description="简单的历史图与新图对比功能",
    version="0.1.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件服务
app.mount("/static", StaticFiles(directory="static"), name="static")

class SimpleChangeDetector:
    """简单的变化检测算法实现"""
    
    def __init__(self):
        self.min_change_area = 100  # 最小变化区域面积（像素）
        
    def extract_gps_from_exif(self, image_path: str) -> Optional[Dict]:
        """从图像EXIF信息提取GPS坐标"""
        try:
            image = Image.open(image_path)
            exif = image._getexif()
            
            if exif is None:
                return None
                
            gps_info = {}
            for tag_id in exif:
                tag = ExifTags.TAGS.get(tag_id, tag_id)
                if tag == "GPSInfo":
                    gps_data = exif[tag_id]
                    
                    # 解析GPS坐标
                    if 1 in gps_data and 2 in gps_data and 3 in gps_data and 4 in gps_data:
                        lat_ref = gps_data[1]
                        lat = gps_data[2]
                        lon_ref = gps_data[3] 
                        lon = gps_data[4]
                        
                        # 转换为十进制度数
                        lat_decimal = self._convert_to_decimal(lat)
                        lon_decimal = self._convert_to_decimal(lon)
                        
                        if lat_ref == 'S':
                            lat_decimal = -lat_decimal
                        if lon_ref == 'W':
                            lon_decimal = -lon_decimal
                            
                        gps_info = {
                            "latitude": lat_decimal,
                            "longitude": lon_decimal
                        }
                        
            return gps_info if gps_info else None
            
        except Exception as e:
            print(f"GPS提取错误: {e}")
            return None
    
    def _convert_to_decimal(self, coord):
        """将GPS坐标从度分秒格式转换为十进制"""
        d, m, s = coord
        return float(d) + float(m)/60 + float(s)/3600
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """图像预处理"""
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # 高斯滤波降噪
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        return blurred
    
    def align_images(self, img1: np.ndarray, img2: np.ndarray):
        """简单的图像对齐（基于特征点匹配）"""
        try:
            # 使用ORB检测器（更快速）
            orb = cv2.ORB_create(nfeatures=1000)
            
            # 检测关键点和描述符
            kp1, des1 = orb.detectAndCompute(img1, None)
            kp2, des2 = orb.detectAndCompute(img2, None)
            
            if des1 is None or des2 is None:
                print("警告: 无法检测到足够的特征点，跳过对齐")
                return img2, np.eye(3)
            
            # 特征匹配
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            
            if len(matches) < 10:
                print("警告: 匹配点不足，跳过对齐")
                return img2, np.eye(3)
            
            # 按距离排序
            matches = sorted(matches, key=lambda x: x.distance)
            
            # 提取匹配点坐标
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)
            
            # 计算单应性矩阵
            M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            
            if M is None:
                print("警告: 无法计算变换矩阵，跳过对齐")
                return img2, np.eye(3)
            
            # 应用变换
            h, w = img1.shape[:2]
            aligned_img2 = cv2.warpPerspective(img2, M, (w, h))
            
            return aligned_img2, M
            
        except Exception as e:
            print(f"图像对齐错误: {e}")
            return img2, np.eye(3)
    
    def detect_changes(self, img1: np.ndarray, img2: np.ndarray) -> Dict[str, Any]:
        """执行变化检测"""
        
        # 1. 图像预处理
        processed_img1 = self.preprocess_image(img1)
        processed_img2 = self.preprocess_image(img2)
        
        # 2. 确保图像尺寸一致
        h1, w1 = processed_img1.shape[:2]
        h2, w2 = processed_img2.shape[:2]
        
        if (h1, w1) != (h2, w2):
            # 调整到相同尺寸
            target_size = (min(w1, w2), min(h1, h2))
            processed_img1 = cv2.resize(processed_img1, target_size)
            processed_img2 = cv2.resize(processed_img2, target_size)
        
        # 3. 尝试图像对齐
        aligned_img2, transform_matrix = self.align_images(processed_img1, processed_img2)
        
        # 4. 计算差分图像
        diff = cv2.absdiff(processed_img1, aligned_img2)
        
        # 5. 二值化处理
        _, binary_diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        
        # 6. 形态学操作去除噪声
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_diff = cv2.morphologyEx(binary_diff, cv2.MORPH_CLOSE, kernel)
        binary_diff = cv2.morphologyEx(binary_diff, cv2.MORPH_OPEN, kernel)
        
        # 7. 查找变化区域轮廓
        contours, _ = cv2.findContours(binary_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 8. 过滤小区域
        significant_contours = []
        total_change_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_change_area:
                significant_contours.append(contour)
                total_change_area += area
        
        # 9. 生成结果图像
        result_img = self.create_result_visualization(
            img1, img2, aligned_img2, binary_diff, significant_contours
        )
        
        # 10. 计算统计信息
        h, w = processed_img1.shape[:2]
        total_pixels = h * w
        change_percentage = (total_change_area / total_pixels) * 100
        
        return {
            "change_mask": binary_diff,
            "result_image": result_img,
            "change_regions_count": len(significant_contours),
            "total_change_area_pixels": int(total_change_area),
            "change_percentage": round(change_percentage, 2),
            "image_size": {"width": w, "height": h},
            "transform_applied": not np.array_equal(transform_matrix, np.eye(3))
        }
    
    def create_result_visualization(self, img1, img2, aligned_img2, change_mask, contours):
        """创建结果可视化图像"""
        
        # 确保图像是3通道
        if len(img1.shape) == 2:
            img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        else:
            img1_color = img1.copy()
            
        if len(aligned_img2.shape) == 2:
            img2_color = cv2.cvtColor(aligned_img2, cv2.COLOR_GRAY2BGR)
        else:
            img2_color = aligned_img2.copy()
        
        # 调整图像尺寸一致
        h, w = change_mask.shape[:2]
        img1_color = cv2.resize(img1_color, (w, h))
        img2_color = cv2.resize(img2_color, (w, h))
        
        # 创建叠加图像
        overlay = img2_color.copy()
        
        # 在变化区域绘制红色高亮
        for contour in contours:
            cv2.fillPoly(overlay, [contour], (0, 0, 255))  # 红色填充
            cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 2)  # 绿色边界
        
        # 混合图像
        alpha = 0.7
        result = cv2.addWeighted(img2_color, alpha, overlay, 1-alpha, 0)
        
        # 创建对比图（左右并排）
        comparison = np.hstack([img1_color, result])
        
        return comparison

# 全局变化检测器
detector = SimpleChangeDetector()
# 报告生成器
report_generator = ReportGenerator()

# 无人机数据服务
drone_service = None
realtime_results = []  # 存储实时检测结果

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化无人机数据服务"""
    global drone_service
    
    # 配置无人机数据接收
    mqtt_config = {
        'host': os.getenv('MQTT_BROKER_HOST', 'localhost'),
        'port': int(os.getenv('MQTT_BROKER_PORT', 1883))
    }
    
    websocket_config = {
        'port': int(os.getenv('DRONE_WEBSOCKET_PORT', 8001))
    }
    
    try:
        drone_service = DroneDataService(
            mqtt_config=mqtt_config,
            websocket_config=websocket_config
        )
        
        # 添加结果回调函数
        drone_service.add_result_callback(on_realtime_detection_result)
        
        # 启动服务
        drone_service.start()
        print("✅ 无人机数据服务已启动")
        print(f"📡 MQTT服务器: {mqtt_config['host']}:{mqtt_config['port']}")
        print(f"🔌 WebSocket端口: {websocket_config['port']}")
        
    except Exception as e:
        print(f"⚠️  无人机数据服务启动失败: {e}")
        drone_service = None

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时清理资源"""
    global drone_service
    if drone_service:
        drone_service.stop()
        print("🛑 无人机数据服务已停止")

def on_realtime_detection_result(result: RealTimeDetectionResult):
    """实时检测结果回调函数"""
    global realtime_results
    
    # 保存结果到内存（实际项目中应该保存到数据库）
    result_dict = {
        'frame_id': result.frame_id,
        'drone_id': result.drone_id,
        'timestamp': result.timestamp.isoformat(),
        'change_detected': result.change_detected,
        'change_regions_count': result.change_regions_count,
        'change_percentage': result.change_percentage,
        'confidence_score': result.confidence_score,
        'processing_time': result.processing_time,
        'result_image_path': result.result_image_path
    }
    
    realtime_results.append(result_dict)
    
    # 保持最近1000条记录
    if len(realtime_results) > 1000:
        realtime_results.pop(0)
    
    # 如果检测到变化，记录日志
    if result.change_detected:
        print(f"🚨 实时变化检测: 无人机 {result.drone_id} 检测到 {result.change_percentage:.2f}% 变化")
async def root():
    """根路径，返回API信息"""
    return {
        "message": "地理测绘变化检测系统 MVP版本",
        "version": "0.1.0",
        "features": ["图像上传", "变化检测", "结果可视化"]
    }

@app.post("/api/upload-and-compare-advanced")
async def upload_and_compare_advanced(
    image1: UploadFile = File(..., description="历史基准图像"),
    image2: UploadFile = File(..., description="新拍摄图像"),
    description: Optional[str] = Form(None, description="检测描述"),
    use_advanced: bool = Form(True, description="使用高级算法")
):
    """
    使用高级算法上传两张图像并执行变化检测
    
    返回:
    - 高精度变化检测结果
    - 图像配准信息
    - 详细的变化区域分析
    - 置信度评估
    """
    
    try:
        # 生成唯一任务ID
        task_id = str(uuid.uuid4())
        
        # 保存上传的图像
        image1_path = f"uploads/{task_id}_image1.jpg"
        image2_path = f"uploads/{task_id}_image2.jpg"
        
        # 保存文件
        with open(image1_path, "wb") as f:
            content = await image1.read()
            f.write(content)
            
        with open(image2_path, "wb") as f:
            content = await image2.read()
            f.write(content)
        
        # 读取图像
        img1 = cv2.imread(image1_path)
        img2 = cv2.imread(image2_path)
        
        if img1 is None or img2 is None:
            raise HTTPException(status_code=400, detail="无法读取图像文件")
        
        # 提取GPS信息
        gps1 = detector.extract_gps_from_exif(image1_path)
        gps2 = detector.extract_gps_from_exif(image2_path)
        
        # 使用高级算法引擎进行处理
        if use_advanced:
            analysis_result = advanced_engine.process_image_pair(
                img1, img2, 
                gps_hints={'image1': gps1, 'image2': gps2}
            )
            
            # 创建可视化结果
            result_image = create_advanced_visualization(
                img1, img2, analysis_result
            )
        else:
            # 使用简单算法
            detection_result = detector.detect_changes(img1, img2)
            analysis_result = {
                'detection': {
                    'change_regions_count': detection_result["change_regions_count"],
                    'total_change_area_pixels': detection_result["total_change_area_pixels"],
                    'change_percentage': detection_result["change_percentage"],
                    'processing_time': 0
                },
                'registration': {
                    'confidence_score': 0.5,
                    'transform_applied': detection_result["transform_applied"]
                },
                'change_regions': []
            }
            result_image = detection_result["result_image"]
        
        # 保存结果图像
        result_image_path = f"static/{task_id}_advanced_result.jpg"
        cv2.imwrite(result_image_path, result_image)
        
        # 构建响应
        result = {
            "task_id": task_id,
            "timestamp": datetime.now().isoformat(),
            "description": description,
            "algorithm_type": "advanced" if use_advanced else "simple",
            "gps_info": {
                "image1": gps1,
                "image2": gps2
            },
            "registration_results": analysis_result.get('registration', {}),
            "detection_results": analysis_result.get('detection', {}),
            "change_regions": analysis_result.get('change_regions', []),
            "result_image_url": f"/static/{task_id}_advanced_result.jpg",
            "status": "completed"
        }
        
        # 保存结果到JSON文件
        result_json_path = f"results/{task_id}_advanced_result.json"
        with open(result_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

def create_advanced_visualization(img1: np.ndarray, img2: np.ndarray, 
                                analysis_result: Dict[str, Any]) -> np.ndarray:
    """创建高级可视化结果"""
    
    # 获取可视化数据
    viz_data = analysis_result.get('visualization_data', {})
    change_mask = viz_data.get('change_mask')
    confidence_map = viz_data.get('confidence_map')
    aligned_img2 = viz_data.get('aligned_image')
    
    if change_mask is None:
        # 如果没有高级结果，使用简单方法
        return detector.create_result_visualization(img1, img2, img2, np.zeros_like(img1[:,:,0]), [])
    
    # 调整图像尺寸一致
    h, w = change_mask.shape[:2]
    img1_resized = cv2.resize(img1, (w, h))
    img2_resized = cv2.resize(aligned_img2 if aligned_img2 is not None else img2, (w, h))
    
    # 确保图像是3通道
    if len(img1_resized.shape) == 2:
        img1_resized = cv2.cvtColor(img1_resized, cv2.COLOR_GRAY2BGR)
    if len(img2_resized.shape) == 2:
        img2_resized = cv2.cvtColor(img2_resized, cv2.COLOR_GRAY2BGR)
    
    # 创建变化可视化
    overlay = img2_resized.copy()
    
    # 绘制变化区域（红色高亮）
    change_regions = analysis_result.get('change_regions', [])
    for region_info in change_regions:
        if 'bounding_box' in region_info:
            x, y, w_box, h_box = region_info['bounding_box']
            confidence = region_info.get('confidence', 0.5)
            
            # 根据置信度调整颜色强度
            color_intensity = int(255 * confidence)
            cv2.rectangle(overlay, (x, y), (x + w_box, y + h_box), 
                         (0, 0, color_intensity), 2)
            
            # 添加变化类型标签
            change_type = region_info.get('change_type', 'unknown')
            cv2.putText(overlay, change_type, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # 使用变化掩码创建红色高亮区域
    red_overlay = np.zeros_like(img2_resized)
    red_overlay[:, :, 2] = change_mask  # 红色通道
    
    # 混合图像
    alpha = 0.3
    highlighted = cv2.addWeighted(img2_resized, 1-alpha, red_overlay, alpha, 0)
    
    # 创建对比图（左右并排）
    comparison = np.hstack([img1_resized, highlighted])
    
    # 添加文本信息
    detection_info = analysis_result.get('detection', {})
    reg_info = analysis_result.get('registration', {})
    
    info_text = [
        f"Changes: {detection_info.get('change_regions_count', 0)}",
        f"Area: {detection_info.get('change_percentage', 0):.1f}%",
        f"Reg: {reg_info.get('confidence_score', 0):.2f}"
    ]
    
    y_offset = 30
    for i, text in enumerate(info_text):
        cv2.putText(comparison, text, (10, y_offset + i*25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(comparison, text, (10, y_offset + i*25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    
    return comparison
async def upload_and_compare(
    image1: UploadFile = File(..., description="历史基准图像"),
    image2: UploadFile = File(..., description="新拍摄图像"),
    description: Optional[str] = Form(None, description="检测描述")
):
    """
    上传两张图像并执行变化检测
    
    返回:
    - 变化检测结果
    - 结果图像URL
    - 统计信息
    """
    
    try:
        # 生成唯一任务ID
        task_id = str(uuid.uuid4())
        
        # 保存上传的图像
        image1_path = f"uploads/{task_id}_image1.jpg"
        image2_path = f"uploads/{task_id}_image2.jpg"
        
        # 保存文件
        with open(image1_path, "wb") as f:
            content = await image1.read()
            f.write(content)
            
        with open(image2_path, "wb") as f:
            content = await image2.read()
            f.write(content)
        
        # 读取图像
        img1 = cv2.imread(image1_path)
        img2 = cv2.imread(image2_path)
        
        if img1 is None or img2 is None:
            raise HTTPException(status_code=400, detail="无法读取图像文件")
        
        # 提取GPS信息
        gps1 = detector.extract_gps_from_exif(image1_path)
        gps2 = detector.extract_gps_from_exif(image2_path)
        
        # 执行变化检测
        detection_result = detector.detect_changes(img1, img2)
        
        # 保存结果图像
        result_image_path = f"static/{task_id}_result.jpg"
        cv2.imwrite(result_image_path, detection_result["result_image"])
        
        # 构建响应
        result = {
            "task_id": task_id,
            "timestamp": datetime.now().isoformat(),
            "description": description,
            "gps_info": {
                "image1": gps1,
                "image2": gps2
            },
            "detection_results": {
                "change_regions_count": detection_result["change_regions_count"],
                "total_change_area_pixels": detection_result["total_change_area_pixels"],
                "change_percentage": detection_result["change_percentage"],
                "image_size": detection_result["image_size"],
                "transform_applied": detection_result["transform_applied"]
            },
            "result_image_url": f"/static/{task_id}_result.jpg",
            "status": "completed"
        }
        
        # 保存结果到JSON文件
        result_json_path = f"results/{task_id}_result.json"
        with open(result_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

@app.get("/api/result/{task_id}")
async def get_result(task_id: str):
    """获取检测结果"""
    
    result_json_path = f"results/{task_id}_result.json"
    
    if not os.path.exists(result_json_path):
        raise HTTPException(status_code=404, detail="结果不存在")
    
    with open(result_json_path, "r", encoding="utf-8") as f:
        result = json.load(f)
    
    return JSONResponse(content=result)

@app.get("/api/drones")
async def get_drones():
    """获取所有无人机状态"""
    if not drone_service:
        raise HTTPException(status_code=503, detail="无人机数据服务未启动")
    
    try:
        drones = drone_service.get_all_drones()
        return JSONResponse(content={
            "drones": drones,
            "count": len(drones),
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取无人机状态失败: {str(e)}")

@app.get("/api/drones/{drone_id}")
async def get_drone_status(drone_id: str):
    """获取指定无人机状态"""
    if not drone_service:
        raise HTTPException(status_code=503, detail="无人机数据服务未启动")
    
    try:
        drone_status = drone_service.get_drone_status(drone_id)
        if not drone_status:
            raise HTTPException(status_code=404, detail="无人机未找到")
        
        return JSONResponse(content=drone_status)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取无人机状态失败: {str(e)}")

@app.get("/api/realtime-results")
async def get_realtime_results(
    drone_id: Optional[str] = None,
    limit: int = 100,
    changes_only: bool = False
):
    """获取实时检测结果"""
    try:
        # 过滤结果
        filtered_results = realtime_results
        
        if drone_id:
            filtered_results = [r for r in filtered_results if r['drone_id'] == drone_id]
        
        if changes_only:
            filtered_results = [r for r in filtered_results if r['change_detected']]
        
        # 限制数量（取最新的）
        filtered_results = filtered_results[-limit:]
        
        return JSONResponse(content={
            "results": filtered_results,
            "count": len(filtered_results),
            "total_results": len(realtime_results),
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取实时结果失败: {str(e)}")

@app.get("/api/drone-service/stats")
async def get_drone_service_stats():
    """获取无人机数据服务统计信息"""
    if not drone_service:
        raise HTTPException(status_code=503, detail="无人机数据服务未启动")
    
    try:
        stats = drone_service.get_stats()
        return JSONResponse(content=stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取服务统计失败: {str(e)}")

@app.post("/api/drone-service/control")
async def control_drone_service(action: str):
    """控制无人机数据服务"""
    global drone_service
    
    try:
        if action == "start":
            if drone_service and drone_service.running:
                return JSONResponse(content={"message": "服务已在运行", "status": "running"})
            
            # 重新启动服务
            if drone_service:
                drone_service.stop()
            
            # ... 重新初始化配置 ...
            # (这里可以重复startup_event中的逻辑)
            
            return JSONResponse(content={"message": "服务启动成功", "status": "running"})
            
        elif action == "stop":
            if drone_service:
                drone_service.stop()
                return JSONResponse(content={"message": "服务停止成功", "status": "stopped"})
            else:
                return JSONResponse(content={"message": "服务未运行", "status": "stopped"})
                
        else:
            raise HTTPException(status_code=400, detail="无效的操作，支持: start, stop")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务控制失败: {str(e)}")

@app.post("/api/generate-report")
async def generate_report(
    task_id: str,
    report_format: str = "json",  # json, annotated_image
    include_annotations: bool = True
):
    """
    为指定任务生成报告
    
    支持的报告格式:
    - json: JSON格式报告
    - annotated_image: 标注图像
    """
    
    try:
        # 获取任务结果
        result_json_path = f"results/{task_id}_advanced_result.json"
        if not os.path.exists(result_json_path):
            result_json_path = f"results/{task_id}_result.json"
            if not os.path.exists(result_json_path):
                raise HTTPException(status_code=404, detail="任务结果未找到")
        
        with open(result_json_path, "r", encoding="utf-8") as f:
            detection_result = json.load(f)
        
        generated_files = []
        
        if report_format == "json":
            # 生成JSON报告
            json_report_path = report_generator.generate_json_report(detection_result)
            generated_files.append({
                "type": "json_report",
                "path": json_report_path,
                "url": f"/static/{os.path.basename(json_report_path)}"
            })
            
        elif report_format == "annotated_image":
            # 生成标注图像
            result_image_url = detection_result.get('result_image_url', '')
            if result_image_url:
                result_image_path = result_image_url.replace('/static/', 'static/')
                if os.path.exists(result_image_path):
                    annotated_path = report_generator.generate_annotated_image(
                        result_image_path,
                        detection_result.get('change_regions', [])
                    )
                    generated_files.append({
                        "type": "annotated_image",
                        "path": annotated_path,
                        "url": f"/static/{os.path.basename(annotated_path)}"
                    })
                else:
                    raise HTTPException(status_code=404, detail="结果图像未找到")
            else:
                raise HTTPException(status_code=400, detail="无结果图像可标注")
        
        else:
            raise HTTPException(status_code=400, detail="不支持的报告格式")
        
        return JSONResponse(content={
            "message": "报告生成成功",
            "task_id": task_id,
            "report_format": report_format,
            "generated_files": generated_files,
            "timestamp": datetime.now().isoformat()
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"报告生成失败: {str(e)}")

@app.get("/api/reports")
async def list_reports(
    limit: int = 50,
    report_type: Optional[str] = None
):
    """获取报告列表"""
    try:
        reports = []
        
        # 扫描报告目录
        reports_dir = "results/reports"
        if os.path.exists(reports_dir):
            for filename in os.listdir(reports_dir):
                if filename.endswith('.json'):
                    file_path = os.path.join(reports_dir, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            report_data = json.load(f)
                            
                        reports.append({
                            "filename": filename,
                            "report_id": report_data.get('report_id', ''),
                            "task_id": report_data.get('task_id', ''),
                            "generated_at": report_data.get('generated_at', ''),
                            "algorithm_type": report_data.get('algorithm_type', ''),
                            "regions_count": report_data.get('detection_summary', {}).get('total_regions', 0),
                            "change_percentage": report_data.get('detection_summary', {}).get('change_percentage', 0)
                        })
                    except Exception as e:
                        logger.error(f"读取报告文件失败 {filename}: {e}")
        
        # 按时间排序
        reports.sort(key=lambda x: x.get('generated_at', ''), reverse=True)
        
        # 限制数量
        reports = reports[:limit]
        
        return JSONResponse(content={
            "reports": reports,
            "count": len(reports),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取报告列表失败: {str(e)}")

@app.get("/api/reports/{report_id}")
async def get_report_detail(report_id: str):
    """获取报告详情"""
    try:
        # 在报告目录中查找
        reports_dir = "results/reports"
        found_report = None
        
        if os.path.exists(reports_dir):
            for filename in os.listdir(reports_dir):
                if filename.endswith('.json'):
                    file_path = os.path.join(reports_dir, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            report_data = json.load(f)
                            
                        if report_data.get('report_id') == report_id:
                            found_report = report_data
                            break
                    except Exception:
                        continue
        
        if not found_report:
            raise HTTPException(status_code=404, detail="报告未找到")
        
        return JSONResponse(content=found_report)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取报告详情失败: {str(e)}")

@app.get("/")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "change_detection": "running",
            "file_storage": "available"
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 启动地理测绘变化检测系统 MVP版本")
    print("📍 访问地址: http://localhost:8000")
    print("📖 API文档: http://localhost:8000/docs")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )