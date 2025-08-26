#!/usr/bin/env python3
"""
地理测绘变化检测系统 MVP 测试脚本
用于验证核心功能是否正常工作
"""

import requests
import os
import json
import time
from PIL import Image
import numpy as np
import cv2

# 测试配置
API_BASE = "http://localhost:8000"
TEST_DATA_DIR = "test_data"

def create_test_images():
    """创建测试用的模拟图像"""
    
    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    
    # 创建基础图像 (1024x1024)
    size = (1024, 1024)
    
    # 图像1：基础场景（绿色背景 + 一些白色建筑）
    img1 = np.ones((*size, 3), dtype=np.uint8) * 50  # 深绿背景
    
    # 添加一些原始建筑物
    cv2.rectangle(img1, (200, 200), (300, 300), (255, 255, 255), -1)  # 白色建筑1
    cv2.rectangle(img1, (400, 400), (500, 550), (255, 255, 255), -1)  # 白色建筑2
    cv2.rectangle(img1, (700, 100), (800, 200), (200, 200, 200), -1)  # 灰色建筑3
    
    # 添加道路
    cv2.rectangle(img1, (0, 512), (1024, 532), (128, 128, 128), -1)   # 水平道路
    cv2.rectangle(img1, (500, 0), (520, 1024), (128, 128, 128), -1)   # 垂直道路
    
    # 图像2：变化后的场景（新增建筑物和变化）
    img2 = img1.copy()
    
    # 新增建筑物（这些应该被检测为变化）
    cv2.rectangle(img2, (600, 200), (700, 350), (255, 255, 255), -1)  # 新建筑1
    cv2.rectangle(img2, (150, 600), (250, 700), (255, 255, 255), -1)  # 新建筑2
    cv2.rectangle(img2, (800, 400), (900, 500), (200, 200, 200), -1)  # 新建筑3
    
    # 修改原有建筑（扩建）
    cv2.rectangle(img2, (400, 400), (550, 600), (255, 255, 255), -1)  # 扩建原建筑2
    
    # 道路变化
    cv2.rectangle(img2, (700, 600), (720, 800), (128, 128, 128), -1)  # 新增小路
    
    # 保存图像
    img1_path = os.path.join(TEST_DATA_DIR, "test_image1.jpg")
    img2_path = os.path.join(TEST_DATA_DIR, "test_image2.jpg")
    
    cv2.imwrite(img1_path, img1)
    cv2.imwrite(img2_path, img2)
    
    print(f"✅ 测试图像已创建:")
    print(f"   - 基础图像: {img1_path}")
    print(f"   - 变化图像: {img2_path}")
    
    return img1_path, img2_path

def test_health_check():
    """测试健康检查接口"""
    print("\n🔍 测试健康检查...")
    
    try:
        response = requests.get(f"{API_BASE}/api/health")
        if response.status_code == 200:
            print("✅ 健康检查通过")
            print(f"   状态: {response.json()}")
            return True
        else:
            print(f"❌ 健康检查失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 健康检查异常: {e}")
        return False

def test_image_upload_and_compare(img1_path, img2_path):
    """测试图像上传和比较功能"""
    print("\n🔍 测试图像上传和变化检测...")
    
    try:
        # 准备文件
        files = {
            'image1': ('test_image1.jpg', open(img1_path, 'rb'), 'image/jpeg'),
            'image2': ('test_image2.jpg', open(img2_path, 'rb'), 'image/jpeg')
        }
        
        data = {
            'description': '测试用例：检测新增建筑物和道路变化'
        }
        
        print("   📤 上传图像并执行检测...")
        start_time = time.time()
        
        response = requests.post(
            f"{API_BASE}/api/upload-and-compare",
            files=files,
            data=data,
            timeout=60  # 60秒超时
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # 关闭文件
        files['image1'][1].close()
        files['image2'][1].close()
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 变化检测成功完成 (耗时: {processing_time:.2f}秒)")
            print(f"   任务ID: {result['task_id']}")
            
            # 分析检测结果
            detection = result['detection_results']
            print(f"\n📊 检测结果分析:")
            print(f"   - 变化区域数量: {detection['change_regions_count']}")
            print(f"   - 变化面积比例: {detection['change_percentage']}%")
            print(f"   - 总变化像素: {detection['total_change_area_pixels']}")
            print(f"   - 图像对齐状态: {'✅' if detection['transform_applied'] else '❌'}")
            print(f"   - 图像尺寸: {detection['image_size']['width']}x{detection['image_size']['height']}")
            
            # GPS信息
            gps_info = result.get('gps_info', {})
            if gps_info.get('image1') or gps_info.get('image2'):
                print(f"\n📍 GPS信息:")
                if gps_info.get('image1'):
                    print(f"   - 图像1: {gps_info['image1']}")
                if gps_info.get('image2'):
                    print(f"   - 图像2: {gps_info['image2']}")
            else:
                print(f"\n📍 GPS信息: 未找到GPS数据（测试图像）")
            
            # 评估检测质量
            print(f"\n🎯 检测质量评估:")
            if detection['change_regions_count'] > 0:
                print("   ✅ 成功检测到变化区域")
            else:
                print("   ⚠️  未检测到变化区域")
                
            if detection['change_percentage'] > 0.5:
                print("   ✅ 变化比例合理")
            else:
                print("   ⚠️  变化比例较低")
                
            if processing_time < 30:
                print("   ✅ 处理时间符合要求")
            else:
                print("   ⚠️  处理时间较长")
            
            return True, result
            
        else:
            print(f"❌ 变化检测失败: {response.status_code}")
            print(f"   错误信息: {response.text}")
            return False, None
            
    except Exception as e:
        print(f"❌ 变化检测异常: {e}")
        return False, None

def test_result_retrieval(task_id):
    """测试结果获取功能"""
    print(f"\n🔍 测试结果获取 (任务ID: {task_id})...")
    
    try:
        response = requests.get(f"{API_BASE}/api/result/{task_id}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 结果获取成功")
            print(f"   时间戳: {result.get('timestamp')}")
            print(f"   状态: {result.get('status')}")
            return True
        else:
            print(f"❌ 结果获取失败: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ 结果获取异常: {e}")
        return False

def test_result_image_access(result_image_url):
    """测试结果图像访问"""
    print(f"\n🔍 测试结果图像访问...")
    
    try:
        # 构建完整URL
        if result_image_url.startswith('/'):
            full_url = API_BASE + result_image_url
        else:
            full_url = result_image_url
            
        response = requests.get(full_url)
        
        if response.status_code == 200:
            print("✅ 结果图像访问成功")
            print(f"   图像大小: {len(response.content)} 字节")
            print(f"   URL: {full_url}")
            
            # 保存结果图像到本地查看
            result_path = os.path.join(TEST_DATA_DIR, "result_image.jpg")
            with open(result_path, 'wb') as f:
                f.write(response.content)
            print(f"   已保存到: {result_path}")
            
            return True
        else:
            print(f"❌ 结果图像访问失败: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ 结果图像访问异常: {e}")
        return False

def run_comprehensive_test():
    """运行完整的MVP测试"""
    print("🚀 开始地理测绘变化检测系统 MVP 功能测试")
    print("=" * 60)
    
    # 测试结果记录
    test_results = {
        'health_check': False,
        'image_creation': False,
        'change_detection': False,
        'result_retrieval': False,
        'image_access': False
    }
    
    # 1. 健康检查
    test_results['health_check'] = test_health_check()
    
    # 2. 创建测试图像
    try:
        img1_path, img2_path = create_test_images()
        test_results['image_creation'] = True
    except Exception as e:
        print(f"❌ 测试图像创建失败: {e}")
        test_results['image_creation'] = False
    
    # 3. 图像上传和变化检测
    if test_results['image_creation']:
        success, result = test_image_upload_and_compare(img1_path, img2_path)
        test_results['change_detection'] = success
        
        # 4. 结果获取测试
        if success and result:
            task_id = result.get('task_id')
            if task_id:
                test_results['result_retrieval'] = test_result_retrieval(task_id)
            
            # 5. 结果图像访问测试
            result_image_url = result.get('result_image_url')
            if result_image_url:
                test_results['image_access'] = test_result_image_access(result_image_url)
    
    # 测试总结
    print("\n" + "=" * 60)
    print("🎯 MVP功能测试总结")
    print("=" * 60)
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    for test_name, passed in test_results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"   {test_name:20} {status}")
    
    print(f"\n📊 测试通过率: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests == total_tests:
        print("\n🎉 恭喜！所有MVP功能测试通过")
        print("   系统已准备好用于演示和使用")
    elif passed_tests >= total_tests * 0.8:
        print(f"\n⚠️  大部分功能正常，有{total_tests - passed_tests}个测试失败")
        print("   建议检查失败的功能后再部署")
    else:
        print(f"\n❌ 多个关键功能存在问题，需要调试修复")
        print("   建议检查日志和错误信息")
    
    return test_results

if __name__ == "__main__":
    print("地理测绘变化检测系统 MVP 测试工具")
    print("请确保后端服务已启动 (python backend/main.py)")
    print("API地址:", API_BASE)
    print()
    
    input("按 Enter 键开始测试...")
    
    test_results = run_comprehensive_test()
    
    print(f"\n测试完成！结果文件保存在 {TEST_DATA_DIR} 目录中")