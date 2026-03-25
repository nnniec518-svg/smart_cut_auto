@echo off
REM 使用国内镜像下载 BAAI/bge-large-zh-v1.5 模型
REM 这个脚本会在后台运行，不会阻塞命令行

echo ============================================================
echo 开始下载 BAAI/bge-large-zh-v1.5 模型
echo ============================================================
echo.
echo 镜像源: https://hf-mirror.com
echo 预计时间: 5-10 分钟（取决于网络速度）
echo.

python download_model.py

echo.
echo ============================================================
echo 下载完成！
echo ============================================================
echo.
echo 下一步：
echo 1. 更新 config.yaml 中的 embedding_model 配置
echo 2. 重新运行主程序
echo.
pause
