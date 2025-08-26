#!/usr/bin/env python3
"""
无人机数据接收和实时处理服务
支持MQTT协议接收无人机数据流，实时进行变化检测
"""

import asyncio
import json
import base64
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
import numpy as np
import cv2
from io import BytesIO
import threading
import queue
import time

# MQTT客户端（如果需要）
try:
    import paho.mqtt.client as mqtt
except ImportError:
    mqtt = None
    print("Warning: paho-mqtt not installed. MQTT functionality disabled.")

# WebSocket支持
try:
    import websockets
    import websocket
except ImportError:
    websockets = None
    print("Warning: websockets not installed. WebSocket functionality disabled.")

from core_algorithms import GeoChangeDetectionEngine

logger = logging.getLogger(__name__)

@dataclass
class DroneInfo:
    """无人机基本信息"""
    id: str
    name: str
    status: str  # 'online', 'offline', 'flying', 'error'
    last_seen: datetime
    location: Optional[Dict[str, float]] = None  # {'latitude': xx, 'longitude': xx}
    battery: Optional[int] = None
    altitude: Optional[float] = None
    speed: Optional[float] = None

@dataclass
class DroneDataFrame:
    """无人机数据帧"""
    drone_id: str
    timestamp: datetime
    image_data: bytes  # 原始图像数据
    gps_coordinates: Dict[str, float]  # GPS坐标
    metadata: Dict[str, Any]  # 其他元数据

@dataclass
class RealTimeDetectionResult:
    """实时检测结果"""
    frame_id: str
    drone_id: str
    timestamp: datetime
    change_detected: bool
    change_regions_count: int
    change_percentage: float
    confidence_score: float
    processing_time: float
    result_image_path: Optional[str] = None

class CircularBuffer:
    """循环缓冲区，用于存储历史帧"""
    
    def __init__(self, maxsize: int = 100):
        self.maxsize = maxsize
        self.buffer: Dict[str, List[DroneDataFrame]] = {}
        self.locks: Dict[str, threading.Lock] = {}
    
    def add_frame(self, drone_id: str, frame: DroneDataFrame):
        """添加新帧到指定无人机的缓冲区"""
        if drone_id not in self.buffer:
            self.buffer[drone_id] = []
            self.locks[drone_id] = threading.Lock()
        
        with self.locks[drone_id]:
            self.buffer[drone_id].append(frame)
            if len(self.buffer[drone_id]) > self.maxsize:
                self.buffer[drone_id].pop(0)
    
    def get_latest_frame(self, drone_id: str) -> Optional[DroneDataFrame]:
        """获取指定无人机的最新帧"""
        if drone_id not in self.buffer or not self.buffer[drone_id]:
            return None
        
        with self.locks[drone_id]:
            return self.buffer[drone_id][-1] if self.buffer[drone_id] else None
    
    def get_reference_frame(self, drone_id: str, max_age_seconds: int = 300) -> Optional[DroneDataFrame]:
        """获取参考帧（用于变化检测）"""
        if drone_id not in self.buffer:
            return None
        
        current_time = datetime.now()
        with self.locks[drone_id]:
            # 寻找合适的参考帧（不太旧，且距离当前帧有一定间隔）
            for frame in reversed(self.buffer[drone_id][:-1]):  # 排除最新帧
                age = (current_time - frame.timestamp).total_seconds()
                if 30 <= age <= max_age_seconds:  # 30秒到5分钟之间
                    return frame
            
            # 如果没有找到合适的，返回最旧的帧
            return self.buffer[drone_id][0] if self.buffer[drone_id] else None

class MQTTDroneReceiver:
    """MQTT协议无人机数据接收器"""
    
    def __init__(self, broker_host: str = "localhost", broker_port: int = 1883):
        if mqtt is None:
            raise ImportError("paho-mqtt is required for MQTT functionality")
        
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.client = mqtt.Client()
        self.data_queue = queue.Queue()
        self.running = False
        
        # 设置回调函数
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect
    
    def _on_connect(self, client, userdata, flags, rc):
        """MQTT连接回调"""
        if rc == 0:
            logger.info(f"Connected to MQTT broker at {self.broker_host}:{self.broker_port}")
            # 订阅无人机数据主题
            client.subscribe("drone/+/data")  # 订阅所有无人机的数据
            client.subscribe("drone/+/status")  # 订阅所有无人机的状态
        else:
            logger.error(f"Failed to connect to MQTT broker, return code {rc}")
    
    def _on_message(self, client, userdata, msg):
        """MQTT消息回调"""
        try:
            topic_parts = msg.topic.split('/')
            if len(topic_parts) >= 3:
                drone_id = topic_parts[1]
                message_type = topic_parts[2]
                
                payload = json.loads(msg.payload.decode())
                
                if message_type == "data":
                    # 处理无人机数据
                    frame = self._parse_drone_data(drone_id, payload)
                    if frame:
                        self.data_queue.put(frame)
                elif message_type == "status":
                    # 处理无人机状态
                    logger.info(f"Drone {drone_id} status: {payload}")
                    
        except Exception as e:
            logger.error(f"Error processing MQTT message: {e}")
    
    def _on_disconnect(self, client, userdata, rc):
        """MQTT断开连接回调"""
        logger.warning(f"Disconnected from MQTT broker, return code {rc}")
    
    def _parse_drone_data(self, drone_id: str, payload: Dict) -> Optional[DroneDataFrame]:
        """解析无人机数据"""
        try:
            # 解码图像数据
            image_b64 = payload.get('image_data', '')
            if not image_b64:
                return None
            
            image_data = base64.b64decode(image_b64)
            
            # 解析GPS坐标
            gps_data = payload.get('gps', {})
            if not gps_data.get('latitude') or not gps_data.get('longitude'):
                logger.warning(f"Missing GPS data for drone {drone_id}")
                return None
            
            # 创建数据帧
            frame = DroneDataFrame(
                drone_id=drone_id,
                timestamp=datetime.now(),
                image_data=image_data,
                gps_coordinates=gps_data,
                metadata=payload.get('metadata', {})
            )
            
            return frame
            
        except Exception as e:
            logger.error(f"Error parsing drone data: {e}")
            return None
    
    def start(self):
        """启动MQTT接收器"""
        self.running = True
        try:
            self.client.connect(self.broker_host, self.broker_port, 60)
            self.client.loop_start()
            logger.info("MQTT drone receiver started")
        except Exception as e:
            logger.error(f"Failed to start MQTT receiver: {e}")
            self.running = False
    
    def stop(self):
        """停止MQTT接收器"""
        self.running = False
        self.client.loop_stop()
        self.client.disconnect()
        logger.info("MQTT drone receiver stopped")
    
    def get_data(self, timeout: float = 1.0) -> Optional[DroneDataFrame]:
        """获取接收到的数据"""
        try:
            return self.data_queue.get(timeout=timeout)
        except queue.Empty:
            return None

class WebSocketDroneReceiver:
    """WebSocket协议无人机数据接收器"""
    
    def __init__(self, port: int = 8001):
        if websockets is None:
            raise ImportError("websockets is required for WebSocket functionality")
        
        self.port = port
        self.data_queue = queue.Queue()
        self.running = False
        self.server = None
        self.clients = set()
    
    async def _handle_client(self, websocket, path):
        """处理WebSocket客户端连接"""
        self.clients.add(websocket)
        logger.info(f"New drone client connected: {websocket.remote_address}")
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    frame = self._parse_websocket_data(data)
                    if frame:
                        self.data_queue.put(frame)
                except Exception as e:
                    logger.error(f"Error processing WebSocket message: {e}")
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.remove(websocket)
            logger.info(f"Drone client disconnected: {websocket.remote_address}")
    
    def _parse_websocket_data(self, data: Dict) -> Optional[DroneDataFrame]:
        """解析WebSocket数据"""
        try:
            drone_id = data.get('drone_id')
            if not drone_id:
                return None
            
            # 解码图像数据
            image_b64 = data.get('image_data', '')
            if not image_b64:
                return None
            
            image_data = base64.b64decode(image_b64)
            
            # 解析GPS坐标
            gps_data = data.get('gps', {})
            if not gps_data.get('latitude') or not gps_data.get('longitude'):
                return None
            
            frame = DroneDataFrame(
                drone_id=drone_id,
                timestamp=datetime.now(),
                image_data=image_data,
                gps_coordinates=gps_data,
                metadata=data.get('metadata', {})
            )
            
            return frame
            
        except Exception as e:
            logger.error(f"Error parsing WebSocket data: {e}")
            return None
    
    async def start_async(self):
        """异步启动WebSocket服务器"""
        self.running = True
        self.server = await websockets.serve(
            self._handle_client, 
            "0.0.0.0", 
            self.port
        )
        logger.info(f"WebSocket drone receiver started on port {self.port}")
    
    def start(self):
        """启动WebSocket接收器"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.start_async())
        loop.run_forever()
    
    def stop(self):
        """停止WebSocket接收器"""
        self.running = False
        if self.server:
            self.server.close()
    
    def get_data(self, timeout: float = 1.0) -> Optional[DroneDataFrame]:
        """获取接收到的数据"""
        try:
            return self.data_queue.get(timeout=timeout)
        except queue.Empty:
            return None

class RealTimeDroneProcessor:
    """实时无人机数据处理器"""
    
    def __init__(self, 
                 detection_engine: Optional[GeoChangeDetectionEngine] = None,
                 buffer_size: int = 100):
        self.detection_engine = detection_engine or GeoChangeDetectionEngine()
        self.frame_buffer = CircularBuffer(buffer_size)
        self.drone_registry: Dict[str, DroneInfo] = {}
        self.processing_stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'detected_changes': 0,
            'average_processing_time': 0.0
        }
        
        # 结果回调函数
        self.result_callbacks: List[Callable[[RealTimeDetectionResult], None]] = []
        
        # 处理参数
        self.min_frame_interval = 5.0  # 最小帧间隔（秒）
        self.change_threshold = 0.1  # 变化阈值
        
    def add_result_callback(self, callback: Callable[[RealTimeDetectionResult], None]):
        """添加结果回调函数"""
        self.result_callbacks.append(callback)
    
    def update_drone_status(self, drone_id: str, status_data: Dict[str, Any]):
        """更新无人机状态"""
        if drone_id not in self.drone_registry:
            self.drone_registry[drone_id] = DroneInfo(
                id=drone_id,
                name=status_data.get('name', f'Drone {drone_id}'),
                status='offline',
                last_seen=datetime.now()
            )
        
        drone = self.drone_registry[drone_id]
        drone.status = status_data.get('status', 'unknown')
        drone.last_seen = datetime.now()
        drone.battery = status_data.get('battery')
        drone.altitude = status_data.get('altitude')
        drone.speed = status_data.get('speed')
        
        if 'location' in status_data:
            drone.location = status_data['location']
    
    def process_frame(self, frame: DroneDataFrame) -> Optional[RealTimeDetectionResult]:
        """处理单帧数据"""
        start_time = time.time()
        
        try:
            # 更新统计
            self.processing_stats['total_frames'] += 1
            
            # 添加到缓冲区
            self.frame_buffer.add_frame(frame.drone_id, frame)
            
            # 更新无人机状态
            self.update_drone_status(frame.drone_id, {
                'status': 'flying',
                'location': frame.gps_coordinates
            })
            
            # 获取参考帧进行变化检测
            reference_frame = self.frame_buffer.get_reference_frame(frame.drone_id)
            if not reference_frame:
                logger.info(f"No reference frame available for drone {frame.drone_id}")
                return None
            
            # 检查时间间隔
            time_diff = (frame.timestamp - reference_frame.timestamp).total_seconds()
            if time_diff < self.min_frame_interval:
                return None
            
            # 解码图像
            current_image = self._decode_image(frame.image_data)
            reference_image = self._decode_image(reference_frame.image_data)
            
            if current_image is None or reference_image is None:
                logger.error("Failed to decode images")
                return None
            
            # 执行变化检测
            detection_result = self.detection_engine.process_image_pair(
                reference_image, 
                current_image,
                gps_hints={
                    'image1': reference_frame.gps_coordinates,
                    'image2': frame.gps_coordinates
                }
            )
            
            # 分析结果
            change_detected = detection_result['detection']['change_percentage'] > self.change_threshold
            
            # 创建结果对象
            result = RealTimeDetectionResult(
                frame_id=str(uuid.uuid4()),
                drone_id=frame.drone_id,
                timestamp=frame.timestamp,
                change_detected=change_detected,
                change_regions_count=detection_result['detection']['change_regions_count'],
                change_percentage=detection_result['detection']['change_percentage'],
                confidence_score=detection_result['registration']['confidence_score'],
                processing_time=time.time() - start_time
            )
            
            # 保存结果图像（如果检测到变化）
            if change_detected:
                result.result_image_path = self._save_result_image(
                    detection_result, frame.drone_id, result.frame_id
                )
                self.processing_stats['detected_changes'] += 1
            
            # 更新统计
            self.processing_stats['processed_frames'] += 1
            self._update_processing_stats(result.processing_time)
            
            # 调用回调函数
            for callback in self.result_callbacks:
                try:
                    callback(result)
                except Exception as e:
                    logger.error(f"Error in result callback: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return None
    
    def _decode_image(self, image_data: bytes) -> Optional[np.ndarray]:
        """解码图像数据"""
        try:
            # 从字节数据创建图像
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return image
        except Exception as e:
            logger.error(f"Error decoding image: {e}")
            return None
    
    def _save_result_image(self, detection_result: Dict, drone_id: str, frame_id: str) -> str:
        """保存检测结果图像"""
        try:
            import os
            
            # 创建结果目录
            result_dir = f"results/realtime/{drone_id}"
            os.makedirs(result_dir, exist_ok=True)
            
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{frame_id}.jpg"
            filepath = os.path.join(result_dir, filename)
            
            # 获取可视化数据并保存
            viz_data = detection_result.get('visualization_data', {})
            if 'result_image' in viz_data:
                cv2.imwrite(filepath, viz_data['result_image'])
            
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving result image: {e}")
            return ""
    
    def _update_processing_stats(self, processing_time: float):
        """更新处理统计信息"""
        current_avg = self.processing_stats['average_processing_time']
        processed_count = self.processing_stats['processed_frames']
        
        # 计算移动平均
        new_avg = (current_avg * (processed_count - 1) + processing_time) / processed_count
        self.processing_stats['average_processing_time'] = new_avg
    
    def get_drone_status(self, drone_id: str) -> Optional[DroneInfo]:
        """获取无人机状态"""
        return self.drone_registry.get(drone_id)
    
    def get_all_drones(self) -> List[DroneInfo]:
        """获取所有无人机状态"""
        return list(self.drone_registry.values())
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        return self.processing_stats.copy()

class DroneDataService:
    """无人机数据服务主类"""
    
    def __init__(self, 
                 mqtt_config: Optional[Dict] = None,
                 websocket_config: Optional[Dict] = None):
        
        # 初始化接收器
        self.receivers = []
        
        if mqtt_config:
            try:
                mqtt_receiver = MQTTDroneReceiver(
                    broker_host=mqtt_config.get('host', 'localhost'),
                    broker_port=mqtt_config.get('port', 1883)
                )
                self.receivers.append(mqtt_receiver)
            except ImportError:
                logger.warning("MQTT receiver not available")
        
        if websocket_config:
            try:
                ws_receiver = WebSocketDroneReceiver(
                    port=websocket_config.get('port', 8001)
                )
                self.receivers.append(ws_receiver)
            except ImportError:
                logger.warning("WebSocket receiver not available")
        
        # 初始化处理器
        self.processor = RealTimeDroneProcessor()
        
        # 运行状态
        self.running = False
        self.worker_threads = []
    
    def add_result_callback(self, callback: Callable[[RealTimeDetectionResult], None]):
        """添加结果回调函数"""
        self.processor.add_result_callback(callback)
    
    def start(self):
        """启动无人机数据服务"""
        if self.running:
            return
        
        self.running = True
        logger.info("Starting drone data service...")
        
        # 启动接收器
        for receiver in self.receivers:
            if hasattr(receiver, 'start'):
                if isinstance(receiver, WebSocketDroneReceiver):
                    # WebSocket需要在新线程中启动
                    thread = threading.Thread(target=receiver.start, daemon=True)
                    thread.start()
                    self.worker_threads.append(thread)
                else:
                    receiver.start()
        
        # 启动数据处理循环
        process_thread = threading.Thread(target=self._processing_loop, daemon=True)
        process_thread.start()
        self.worker_threads.append(process_thread)
        
        logger.info("Drone data service started")
    
    def stop(self):
        """停止无人机数据服务"""
        if not self.running:
            return
        
        logger.info("Stopping drone data service...")
        self.running = False
        
        # 停止接收器
        for receiver in self.receivers:
            if hasattr(receiver, 'stop'):
                receiver.stop()
        
        # 等待工作线程结束
        for thread in self.worker_threads:
            thread.join(timeout=5.0)
        
        logger.info("Drone data service stopped")
    
    def _processing_loop(self):
        """数据处理主循环"""
        logger.info("Data processing loop started")
        
        while self.running:
            try:
                # 从所有接收器获取数据
                for receiver in self.receivers:
                    frame = receiver.get_data(timeout=0.1)
                    if frame:
                        result = self.processor.process_frame(frame)
                        if result and result.change_detected:
                            logger.info(
                                f"Change detected by drone {result.drone_id}: "
                                f"{result.change_percentage:.2f}% change, "
                                f"{result.change_regions_count} regions"
                            )
                
                # 短暂休眠避免CPU过度使用
                time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                time.sleep(1.0)
        
        logger.info("Data processing loop stopped")
    
    def get_drone_status(self, drone_id: str) -> Optional[Dict]:
        """获取无人机状态"""
        drone_info = self.processor.get_drone_status(drone_id)
        return asdict(drone_info) if drone_info else None
    
    def get_all_drones(self) -> List[Dict]:
        """获取所有无人机状态"""
        drones = self.processor.get_all_drones()
        return [asdict(drone) for drone in drones]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取服务统计信息"""
        return {
            'running': self.running,
            'active_receivers': len([r for r in self.receivers if hasattr(r, 'running') and r.running]),
            'processing_stats': self.processor.get_processing_stats()
        }