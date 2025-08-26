@echo off
echo ========================================
echo 地理测绘变化检测系统 MVP 启动脚本
echo ========================================
echo.

:: 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python，请先安装Python 3.8+
    pause
    exit /b 1
)

echo ✅ Python环境检查通过

:: 进入后端目录
cd /d "%~dp0backend"

:: 检查依赖是否安装
echo.
echo 🔍 检查Python依赖...
pip show fastapi >nul 2>&1
if errorlevel 1 (
    echo ⚠️  依赖未安装，正在安装...
    pip install -r requirements_mvp.txt
    if errorlevel 1 (
        echo ❌ 依赖安装失败
        pause
        exit /b 1
    )
    echo ✅ 依赖安装完成
) else (
    echo ✅ 依赖已安装
)

:: 创建必要目录
if not exist "uploads" mkdir uploads
if not exist "results" mkdir results
if not exist "static" mkdir static

echo.
echo 🚀 启动变化检测API服务...
echo 📍 访问地址: http://localhost:8000
echo 📖 API文档: http://localhost:8000/docs
echo 🖥️  前端界面: 请打开 frontend/index.html
echo.
echo 按 Ctrl+C 停止服务
echo ========================================

:: 启动FastAPI服务
python main.py

echo.
echo 服务已停止
pause