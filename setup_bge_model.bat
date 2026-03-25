@echo off
REM 一键下载并配置 BAAI/bge-large-zh-v1.5 模型

echo ============================================================
echo BAAI/bge-large-zh-v1.5 模型一键配置
echo ============================================================
echo.

REM 步骤1: 下载模型
echo [步骤 1/3] 下载模型...
echo.
call download_model.bat
if errorlevel 1 (
    echo.
    echo 模型下载失败！
    echo 请检查网络连接或手动下载
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo [步骤 2/3] 更新配置文件...
echo ============================================================
echo.
python update_config_for_bge.py
if errorlevel 1 (
    echo.
    echo 配置更新失败！
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo [步骤 3/3] 验证模型加载...
echo ============================================================
echo.
python test_model.py
if errorlevel 1 (
    echo.
    echo 模型验证失败！
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo ✓ 所有步骤完成！
echo ============================================================
echo.
echo BAAI/bge-large-zh-v1.5 模型已成功安装并配置！
echo.
echo 现在可以运行主程序了：
echo   python main.py
echo.
pause
