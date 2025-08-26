#!/usr/bin/env python3
"""
无人机数据模拟器
用于测试实时无人机数据接收和处理功能
"""

import asyncio
import json
import base64
import time
import random
import cv2
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
import threading
import argparse

# 根据可用的库选择通信方式
try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    print("Warning: paho-mqtt not available, MQTT simulation disabled")

try:
    import websocket
    import websockets
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    print("Warning: websockets not available, WebSocket simulation disabled")

class DroneSimulator:
    """无人机数据模拟器"""
    
    def __init__(self, drone_id: str, initial_position: Dict[str, float]):
        self.drone_id = drone_id
        self.position = initial_position.copy()
        self.altitude = 100.0
        self.speed = 15.0  # m/s
        self.battery = 100
        self.status = "flying"
        
        # 运动参数
        self.direction = random.uniform(0, 360)  # 度
        self.direction_change_interval = 30  # 秒
        self.last_direction_change = time.time()
        
        # 图像生成参数
        self.image_size = (640, 480)
        self.base_scene = self.generate_base_scene()
        
    def generate_base_scene(self) -> np.ndarray:
        """生成基础场景图像"""
        h, w = self.image_size
        
        # 创建基础背景（绿色代表植被）
        scene = np.ones((h, w, 3), dtype=np.uint8) * 50
        scene[:, :, 1] = 100  # 绿色通道
        
        # 添加一些固定建筑物
        buildings = [
            (100, 100, 80, 60),   # x, y, w, h
            (300, 200, 100, 80),
            (500, 150, 60, 40),
            (200, 350, 120, 90),
        ]
        
        for x, y, w, h in buildings:
            if x + w < scene.shape[1] and y + h < scene.shape[0]:
                cv2.rectangle(scene, (x, y), (x + w, y + h), (200, 200, 200), -1)
        
        # 添加道路
        cv2.rectangle(scene, (0, h//2 - 15), (w, h//2 + 15), (128, 128, 128), -1)
        cv2.rectangle(scene, (w//2 - 15, 0), (w//2 + 15, h), (128, 128, 128), -1)
        
        return scene
    
    def update_position(self, delta_time: float):
        """更新无人机位置"""
        
        # 定期改变方向
        current_time = time.time()
        if current_time - self.last_direction_change > self.direction_change_interval:
            self.direction += random.uniform(-45, 45)
            self.direction = self.direction % 360
            self.last_direction_change = current_time
        
        # 计算位移
        distance = self.speed * delta_time  # 米
        
        # 转换为经纬度变化（粗略估算）
        lat_change = (distance * np.cos(np.radians(self.direction))) / 111320  # 1度纬度约111320米
        lng_change = (distance * np.sin(np.radians(self.direction))) / (111320 * np.cos(np.radians(self.position['latitude'])))
        
        self.position['latitude'] += lat_change
        self.position['longitude'] += lng_change
        
        # 更新高度（小幅度变化）
        self.altitude += random.uniform(-2, 2)
        self.altitude = max(50, min(200, self.altitude))
        
        # 电池消耗
        self.battery -= delta_time / 60  # 每分钟消耗1%
        self.battery = max(0, self.battery)
        
        if self.battery < 20:
            self.status = "low_battery"
        elif self.battery == 0:
            self.status = "offline"
    
    def generate_current_image(self) -> np.ndarray:
        """生成当前场景图像"""
        scene = self.base_scene.copy()
        
        # 添加一些随机变化（模拟真实场景的变化）
        # 1. 随机添加小车辆
        if random.random() < 0.3:  # 30%概率
            x = random.randint(0, scene.shape[1] - 20)
            y = random.randint(scene.shape[0]//2 - 10, scene.shape[0]//2 + 10)
            cv2.rectangle(scene, (x, y), (x + 15, y + 8), (0, 0, 255), -1)
        
        # 2. 随机改变植被颜色（季节变化）
        vegetation_factor = 0.8 + 0.4 * random.random()
        mask = (scene[:, :, 1] > scene[:, :, 0]) & (scene[:, :, 1] > scene[:, :, 2])
        scene[mask, 1] = np.clip(scene[mask, 1] * vegetation_factor, 0, 255)
        
        # 3. 添加光照变化
        brightness = 0.8 + 0.4 * random.random()
        scene = np.clip(scene * brightness, 0, 255).astype(np.uint8)
        
        # 4. 模拟无人机拍摄的变化（不同角度、高度）
        # 轻微的几何变换
        angle = random.uniform(-2, 2)
        scale = random.uniform(0.98, 1.02)
        
        center = (scene.shape[1]//2, scene.shape[0]//2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        scene = cv2.warpAffine(scene, M, (scene.shape[1], scene.shape[0]))
        
        # 5. 模拟一些明显的变化（新建筑物）
        if random.random() < 0.1:  # 10%概率出现新建筑
            x = random.randint(50, scene.shape[1] - 100)
            y = random.randint(50, scene.shape[0] - 100)
            w = random.randint(30, 80)
            h = random.randint(30, 80)
            cv2.rectangle(scene, (x, y), (x + w, y + h), (255, 255, 255), -1)
        
        return scene
    
    def get_current_data(self) -> Dict[str, Any]:
        """获取当前无人机数据"""
        # 生成图像
        image = self.generate_current_image()
        
        # 编码图像为JPEG
        _, encoded_image = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        image_b64 = base64.b64encode(encoded_image.tobytes()).decode('utf-8')
        
        return {
            'drone_id': self.drone_id,
            'timestamp': datetime.now().isoformat(),
            'image_data': image_b64,
            'gps': {
                'latitude': self.position['latitude'],
                'longitude': self.position['longitude'],
                'altitude': self.altitude
            },
            'metadata': {
                'battery': int(self.battery),
                'speed': self.speed,
                'direction': self.direction,
                'status': self.status
            }
        }

class MQTTSimulator:
    """MQTT协议模拟器"""
    
    def __init__(self, broker_host: str = "localhost", broker_port: int = 1883):
        if not MQTT_AVAILABLE:
            raise ImportError("paho-mqtt is required for MQTT simulation")
        
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.client = mqtt.Client()
        self.connected = False
        
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
    
    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.connected = True
            print(f"✅ Connected to MQTT broker at {self.broker_host}:{self.broker_port}")
        else:
            print(f"❌ Failed to connect to MQTT broker, return code {rc}")
    
    def _on_disconnect(self, client, userdata, rc):
        self.connected = False
        print(f"📡 Disconnected from MQTT broker")
    
    def connect(self):
        try:
            self.client.connect(self.broker_host, self.broker_port, 60)
            self.client.loop_start()
            
            # 等待连接
            timeout = 5
            while not self.connected and timeout > 0:
                time.sleep(0.1)
                timeout -= 0.1
            
            return self.connected
        except Exception as e:
            print(f"❌ MQTT connection error: {e}")
            return False
    
    def disconnect(self):
        self.client.loop_stop()
        self.client.disconnect()
    
    def send_drone_data(self, drone_id: str, data: Dict[str, Any]):
        if not self.connected:
            return False
        
        try:
            topic = f"drone/{drone_id}/data"
            payload = json.dumps(data)
            
            result = self.client.publish(topic, payload)
            return result.rc == mqtt.MQTT_ERR_SUCCESS
        except Exception as e:
            print(f"❌ Error sending MQTT data: {e}")
            return False
    
    def send_drone_status(self, drone_id: str, status: Dict[str, Any]):
        if not self.connected:
            return False
        
        try:
            topic = f"drone/{drone_id}/status"
            payload = json.dumps(status)
            
            result = self.client.publish(topic, payload)
            return result.rc == mqtt.MQTT_ERR_SUCCESS
        except Exception as e:
            print(f"❌ Error sending MQTT status: {e}")
            return False

class WebSocketSimulator:
    """WebSocket协议模拟器"""
    
    def __init__(self, server_url: str = "ws://localhost:8001"):
        if not WEBSOCKET_AVAILABLE:
            raise ImportError("websocket-client is required for WebSocket simulation")
        
        self.server_url = server_url
        self.ws = None
        self.connected = False
    
    def connect(self):
        try:
            self.ws = websocket.WebSocket()
            self.ws.connect(self.server_url)
            self.connected = True
            print(f"✅ Connected to WebSocket server at {self.server_url}")
            return True
        except Exception as e:
            print(f"❌ WebSocket connection error: {e}")
            return False
    
    def disconnect(self):
        if self.ws:
            self.ws.close()
            self.connected = False
    
    def send_drone_data(self, data: Dict[str, Any]):
        if not self.connected or not self.ws:
            return False
        
        try:
            payload = json.dumps(data)
            self.ws.send(payload)
            return True
        except Exception as e:
            print(f"❌ Error sending WebSocket data: {e}")
            return False

class DroneFleetSimulator:
    """无人机集群模拟器"""
    
    def __init__(self, 
                 num_drones: int = 3,
                 communication_type: str = "mqtt",
                 mqtt_config: Optional[Dict] = None,
                 websocket_config: Optional[Dict] = None):
        
        self.num_drones = num_drones
        self.communication_type = communication_type
        self.running = False
        
        # 创建无人机模拟器
        self.drones = []
        base_position = {"latitude": 39.9042, "longitude": 116.4074}
        
        for i in range(num_drones):
            # 在基准位置周围随机分布
            position = {
                "latitude": base_position["latitude"] + random.uniform(-0.01, 0.01),
                "longitude": base_position["longitude"] + random.uniform(-0.01, 0.01)
            }
            drone = DroneSimulator(f"drone_{i+1:03d}", position)
            self.drones.append(drone)
        
        # 创建通信客户端
        self.communication_client = None
        
        if communication_type == "mqtt" and MQTT_AVAILABLE:
            config = mqtt_config or {}
            self.communication_client = MQTTSimulator(
                broker_host=config.get('host', 'localhost'),
                broker_port=config.get('port', 1883)
            )
        elif communication_type == "websocket" and WEBSOCKET_AVAILABLE:
            config = websocket_config or {}
            self.communication_client = WebSocketSimulator(
                server_url=config.get('url', 'ws://localhost:8001')
            )
        
        if not self.communication_client:
            raise ValueError(f"Communication type '{communication_type}' not available or not supported")
    
    def start_simulation(self, 
                        data_interval: float = 5.0,
                        status_interval: float = 30.0,
                        duration: Optional[float] = None):
        """启动模拟"""
        
        print(f"🚁 启动 {self.num_drones} 架无人机模拟器")
        print(f"📡 通信方式: {self.communication_type}")
        print(f"⏰ 数据发送间隔: {data_interval}秒")
        
        # 连接通信客户端
        if not self.communication_client.connect():
            print("❌ 通信连接失败，退出模拟")
            return
        
        self.running = True
        start_time = time.time()
        last_data_time = 0
        last_status_time = 0
        
        try:
            while self.running:
                current_time = time.time()
                delta_time = current_time - start_time if start_time else 0
                
                # 更新所有无人机位置
                for drone in self.drones:
                    drone.update_position(data_interval)
                
                # 发送数据
                if current_time - last_data_time >= data_interval:
                    self._send_drone_data()
                    last_data_time = current_time
                
                # 发送状态
                if current_time - last_status_time >= status_interval:
                    self._send_drone_status()
                    last_status_time = current_time
                
                # 检查持续时间
                if duration and (current_time - start_time) >= duration:
                    print(f"⏱️  模拟时间结束 ({duration}秒)")
                    break
                
                time.sleep(1.0)  # 1秒循环
                
        except KeyboardInterrupt:
            print("\\n🛑 用户中断模拟")
        finally:
            self.stop_simulation()
    
    def stop_simulation(self):
        """停止模拟"""
        self.running = False
        if self.communication_client:
            self.communication_client.disconnect()
        print("📡 无人机模拟器已停止")
    
    def _send_drone_data(self):
        """发送无人机数据"""
        for drone in self.drones:
            if drone.status == "offline":
                continue
            
            data = drone.get_current_data()
            
            if self.communication_type == "mqtt":
                success = self.communication_client.send_drone_data(drone.drone_id, data)
            else:  # websocket
                success = self.communication_client.send_drone_data(data)
            
            if success:
                print(f"📤 {drone.drone_id}: 数据已发送 (位置: {drone.position['latitude']:.6f}, {drone.position['longitude']:.6f})")
            else:
                print(f"❌ {drone.drone_id}: 数据发送失败")
    
    def _send_drone_status(self):
        """发送无人机状态"""
        for drone in self.drones:
            status = {
                'drone_id': drone.drone_id,
                'status': drone.status,
                'battery': int(drone.battery),
                'altitude': drone.altitude,
                'speed': drone.speed,
                'location': drone.position
            }
            
            if self.communication_type == "mqtt":
                success = self.communication_client.send_drone_status(drone.drone_id, status)
                if success:
                    print(f"📊 {drone.drone_id}: 状态已发送 (电池: {int(drone.battery)}%)")

def main():
    parser = argparse.ArgumentParser(description="无人机数据模拟器")
    parser.add_argument("--drones", type=int, default=3, help="无人机数量")
    parser.add_argument("--protocol", choices=["mqtt", "websocket"], default="mqtt", help="通信协议")
    parser.add_argument("--interval", type=float, default=5.0, help="数据发送间隔（秒）")
    parser.add_argument("--duration", type=float, help="模拟持续时间（秒）")
    parser.add_argument("--mqtt-host", default="localhost", help="MQTT服务器地址")
    parser.add_argument("--mqtt-port", type=int, default=1883, help="MQTT服务器端口")
    parser.add_argument("--ws-url", default="ws://localhost:8001", help="WebSocket服务器URL")
    
    args = parser.parse_args()
    
    print("🚁 无人机数据模拟器")
    print("=" * 50)
    
    # 配置参数
    mqtt_config = {
        'host': args.mqtt_host,
        'port': args.mqtt_port
    }
    
    websocket_config = {
        'url': args.ws_url
    }
    
    try:
        simulator = DroneFleetSimulator(
            num_drones=args.drones,
            communication_type=args.protocol,
            mqtt_config=mqtt_config,
            websocket_config=websocket_config
        )
        
        simulator.start_simulation(
            data_interval=args.interval,
            duration=args.duration
        )
        
    except Exception as e:
        print(f"❌ 模拟器启动失败: {e}")

if __name__ == "__main__":
    main()