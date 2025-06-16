@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: 设置源目录和目标目录
set "SOURCE_DIR=dist"
:: modToolPack 名字不能修改，在MCS代码中会用到
set "TARGET_DIR=..\..\..\lib\modToolPack"

echo 开始复制文件...
echo 源目录: %SOURCE_DIR%
echo 目标目录: %TARGET_DIR%
echo.

:: 检查源目录是否存在
if not exist "%SOURCE_DIR%" (
    echo 错误: 源目录 %SOURCE_DIR% 不存在！
    goto :error
)

:: 检查源目录是否为空
dir /b "%SOURCE_DIR%" >nul 2>&1
if errorlevel 1 (
    echo 警告: 源目录 %SOURCE_DIR% 为空！
    goto :end
)

:: 创建目标目录（如果不存在）
if not exist "%TARGET_DIR%" (
    echo 创建目标目录: %TARGET_DIR%
    mkdir "%TARGET_DIR%" 2>nul
    if errorlevel 1 (
        echo 错误: 无法创建目标目录 %TARGET_DIR%！
        goto :error
    )
)

:: 遍历各个项目文件夹
set "copied_count=0"
for /d %%i in ("%SOURCE_DIR%\*") do (
    set "project_name=%%~ni"
    echo 正在处理项目: !project_name!
    
    :: 检查项目文件夹是否包含文件
    dir /b "%%i\*" >nul 2>&1
    if not errorlevel 1 (
        :: 复制项目文件夹中的所有文件到目标目录
        xcopy "%%i\*" "%TARGET_DIR%\" /E /I /H /Y /Q
        if errorlevel 1 (
            echo 警告: 复制项目 !project_name! 时出现错误！
        ) else (
            echo   - 项目 !project_name! 复制完成
            set /a copied_count+=1
        )
    ) else (
        echo   - 项目 !project_name! 为空，跳过
    )
    echo.
)

:: 检查是否有项目被复制
if !copied_count! equ 0 (
    echo 警告: 没有找到任何项目文件或所有项目文件夹都为空！
    goto :end
)

echo.
echo 复制完成！
echo 已将 %SOURCE_DIR% 目录下的所有文件复制到 %TARGET_DIR%
goto :end

:error
echo.
echo 操作失败！
exit /b 1

:end
echo.
:: 保持窗口打开
pause