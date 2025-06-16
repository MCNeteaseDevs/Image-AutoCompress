::删除build下的内容
if exist build rmdir /s /q build
::删除dist下的指定项目的内容
if exist dist rmdir /s /q dist/compressImage

pyinstaller build_compressImage.spec

:: 保持窗口打开
pause