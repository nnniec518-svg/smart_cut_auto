@echo off
chcp 65001 > nul
echo ========================================================================
echo BGE 模型文件检查和备份工具
echo ========================================================================
echo.

cd /d "%~dp0"

:: 检查模型目录
set MODEL_DIR=models\sentence_transformers\models--BAAI--bge-large-zh-v1.5

echo 正在检查模型目录...
if not exist "%MODEL_DIR%" (
    echo ❌ 模型目录不存在: %MODEL_DIR%
    echo 请确保模型已下载到本地
    pause
    exit /b 1
)

echo ✅ 模型目录存在
echo.

:: 检查关键文件
echo 正在检查关键文件...

set MISSING_FILES=0

if not exist "%MODEL_DIR%\snapshots\79e7739b6ab944e86d6171e44d24c997fc1e0116\config.json" (
    echo ❌ 缺少: config.json
    set /a MISSING_FILES+=1
) else (
    echo ✅ config.json
)

if not exist "%MODEL_DIR%\snapshots\79e7739b6ab944e86d6171e44d24c997fc1e0116\pytorch_model.bin" (
    echo ❌ 缺少: pytorch_model.bin
    set /a MISSING_FILES+=1
) else (
    echo ✅ pytorch_model.bin
)

if not exist "%MODEL_DIR%\snapshots\79e7739b6ab944e86d6171e44d24c997fc1e0116\tokenizer.json" (
    echo ❌ 缺少: tokenizer.json
    set /a MISSING_FILES+=1
) else (
    echo ✅ tokenizer.json
)

if not exist "%MODEL_DIR%\snapshots\79e7739b6ab944e86d6171e44d24c997fc1e0116\vocab.txt" (
    echo ❌ 缺少: vocab.txt
    set /a MISSING_FILES+=1
) else (
    echo ✅ vocab.txt
)

if not exist "%MODEL_DIR%\refs\main" (
    echo ❌ 缺少: refs\main
    set /a MISSING_FILES+=1
) else (
    echo ✅ refs\main
)

echo.

if %MISSING_FILES% GTR 0 (
    echo ⚠️  检测到 %MISSING_FILES% 个缺失文件
    echo 模型不完整,无法离线运行
    pause
    exit /b 1
)

echo ✅ 所有关键文件都存在
echo.

:: 计算模型大小
echo 正在计算模型大小...
for /f "tokens=3" %%a in ('dir "%MODEL_DIR%" /-c /s ^| find "个文件"') do set SIZE=%%a
set /a SIZE_MB=%SIZE% / 1048576

echo 模型总大小: %SIZE_MB% MB
echo.

:: 询问是否测试
set TEST_MODEL=0
set /p TEST_MODEL="是否测试模型加载? (1=是, 0=否): "

if %TEST_MODEL% == 1 (
    echo.
    echo ========================================================================
    echo 测试模型加载...
    echo ========================================================================
    echo.
    python offline_model_test.py
    if %errorlevel% == 0 (
        echo.
        echo ✅ 模型测试成功!
    ) else (
        echo.
        echo ❌ 模型测试失败
        pause
        exit /b 1
    )
)

:: 询问是否备份
set BACKUP_MODEL=0
set /p BACKUP_MODEL="是否创建模型备份? (1=是, 0=否): "

if %BACKUP_MODEL% == 1 (
    echo.
    echo ========================================================================
    echo 创建模型备份...
    echo ========================================================================
    echo.

    set BACKUP_DIR=models_backup
    if exist "%BACKUP_DIR%" (
        echo 删除旧备份...
        rmdir /s /q "%BACKUP_DIR%"
    )

    echo 正在创建备份: %BACKUP_DIR%\
    xcopy "%MODEL_DIR%" "%BACKUP_DIR%\models--BAAI--bge-large-zh-v1.5\" /E /I /H /Y

    if %errorlevel% == 0 (
        echo ✅ 备份成功!
        echo 备份位置: %BACKUP_DIR%\models--BAAI--bge-large-zh-v1.5\
    ) else (
        echo ❌ 备份失败
        pause
        exit /b 1
    )
)

echo.
echo ========================================================================
echo ✅ 模型文件检查完成!
echo ========================================================================
echo.
echo 总结:
echo   - 模型目录: 完整
echo   - 关键文件: 全部存在
echo   - 模型大小: %SIZE_MB% MB
echo   - 离线运行: ✅ 支持
echo.
echo 您现在可以运行主程序,无需网络连接:
echo   python main.py
echo.
pause
