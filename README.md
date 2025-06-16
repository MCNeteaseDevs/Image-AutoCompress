# 贴图自动压缩工具

欢迎使用贴图自动压缩工具（beta版），详细请参考使用教程

该工具可一键对模组资源包的所有贴图进行自动化降低分辨率处理，同时生成html报告确认压缩情况，支持json文件调整压缩时的处理参数

# 使用教程

## 下载

在Releases中包含最新版的工具，可前往Releases中下载```imageAutoCompress.zip```文件，解压运行compressImage.exe程序即可开始使用

## 使用

### 准备工作

1. 找到需要处理的模组包体

2. 找到资源包路径

### 压缩贴图

1. 点击compressImage.exe运行工具

![Image text](https://nie.res.netease.com/r/pic/20250613/0f89b5e1-6110-46b1-9371-53a763ce2cf7.png)
	
2. 输入需要压缩贴图的资源包路径到命令行，回车确定

- 如资源包根目录： D:\\\*\*\*\mod\_abc\resource*\_*abc
- 如ui图目录： D:\\\*\*\*\mod\_abc\resource*\_*abc\textures\ui
	
![Image text](https://nie.res.netease.com/r/pic/20250613/2912b3cc-ce74-449d-b974-d8e4053c7f3d.png)

3. 输入导出优化后贴图文件的路径到命令行，回车确定

- 为防止文件覆盖，导出文件路径必须为空
- 输入的导出文件夹名不存在，则会自动生成同名文件夹
	
![Image text](https://nie.res.netease.com/r/pic/20250613/749783d9-ad79-43e7-ba80-056ebe929f36.png)

4. 等待压缩进度结束，查看报告

报告中会给出修改的图片、没有修改的图片以及没修改的原因

**生成报告位置**

![Image text](https://nie.res.netease.com/r/pic/20250613/4f6ed4d7-551c-4c2f-a132-4b317abc2ddc.png)

**优化效果展示**

![Image text](https://nie.res.netease.com/r/pic/20250613/5193923b-ecda-4622-b30b-311b8fea293c.png)

### 检查替换

1. 打开输出文件夹，按需替换降分辨率的贴图

2. 进入游戏查看效果

## 配置

配置文件在下图所示位置

![Image text](https://nie.res.netease.com/r/pic/20250613/d80a8e14-fadc-4d44-815b-cd3b4739d79b.png)

### 默认配置
```javascript
{
    "MIN_RESOLUTION": {
        "value": 32,
        "description": "低于该分辨率的贴图，不进行压缩"
    },
    "PARTICLE_MIN_RESOLUTION": {
        "value": 128,
        "description": "低于该分辨率的特效图，不进行压缩"
    },
    "ICON_MIN_RESOLUTION": {
        "value": 16,
        "description": "低于该分辨率的方块、物品icon图，不进行压缩"
    },
    "NORMAL_QUALITY": {
        "value": 0.9,
        "description": "评估质量阈值-普通贴图"
    },
    "TEXT_QUALITY": {
        "value": 0.95,
        "description": "评估质量阈值-文本贴图需略高于普通贴图"
    },
    "BLACK_PATHS": {
        "value": [
            "textures\\environment",
            "textures\\colormap",
            "textures\\gui",
            "textures\\map",
            "textures\\misc",
            "textures\\painting",
            "textures\\particle\\particles.png",
            "textures\\models",
            "textures\\entity"
        ],
        "description": "黑名单目录(相对路径)，该目录下的图片不处理"
    }
}
```

### 配置说明

- 不允许工具处理的目录，可在**BLACK_PATH**中添加需要输入相对路径，且**必须使用`\\`**

- 如果感觉处理过的贴图分辨率过低，调高**NORMAL\_QUALITY、TEXT\_QUALITY**的值，最高为1，代表不作处理

- 设置**MIN_RESOLUTION、PARTICLE_MIN_RESOLUTION、ICON_MIN_RESOLUTION**的值，可以调节需要处理的图片最低像素

# 注意事项

- 本工具**不对模型纹理图做处理**。如需要压缩模型纹理图，请使用[Blockbench 模型UV优化插件](https://github.com/MCNeteaseDevs/UV-Optimizer/tree/main)

- 本工具对**部分原版的资源存放路径下的内容不做处理**。如environment、colormap等。如需要处理，可自行修改配置内容

- 本工具没有做备份处理，建议使用**git、svn等版本管理工具**，方便回滚资源。或者在使用该工具前，对模组进行备份

# 贡献

如果您发现一些可以优化的点，或者想要补充一些新的功能，欢迎在本仓库提交分支，并留下你的大名，感谢您的贡献！