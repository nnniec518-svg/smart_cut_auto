# 智能剪辑 - 抖音团购短视频生成工具

自动完成素材筛选、音频处理、视频拼接以及添加实时字幕，生成适用于抖音团购的短视频。

## 功能特点

- **素材分析**: 自动分析视频素材中的语音内容
- **智能匹配**: 使用Sentence-Transformer进行语义相似度匹配
- **语音识别**: 采用OpenAI Whisper进行高精度语音转文字
- **VAD分割**: 使用Silero VAD进行语音活动检测
- **视频拼接**: 支持多素材智能拼接，统一输出格式
- **字幕生成**: 自动生成并嵌入高质量字幕

## 技术栈

- Python 3.9+
- PyQt5 - 图形界面
- FFmpeg - 视频处理
- MoviePy - 视频编辑辅助
- librosa + noisereduce - 音频处理
- Whisper - 语音识别
- Silero VAD - 语音活动检测
- Sentence-Transformers - 语义匹配

## 环境要求

1. Python 3.9 或更高版本
2. FFmpeg (需添加到系统PATH)

## 安装步骤

1. 克隆或下载本项目

2. 创建虚拟环境（推荐）：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

4. 下载必要的模型：
   - Whisper模型会在首次运行时自动下载
   - Silero VAD模型会自动加载
   - Sentence-Transformer模型会自动下载

## 使用方法

1. 运行程序：
```bash
python main.py
```

2. 导入素材：
   - 点击"导入素材"按钮选择视频文件
   - 或通过"素材文件夹"导入整个目录

3. 输入文案：
   - 在左侧文案编辑区输入口播文案
   - 或点击"加载文案"从文本文件导入

4. 设置参数：
   - 在右侧参数面板调整匹配阈值、降噪强度等

5. 生成视频：
   - 点击菜单"处理"->"分析素材"
   - 分析完成后点击"处理"->"生成视频"

## 项目结构

```
smart_cut_auto/
├── main.py                 # 程序入口
├── core/                   # 核心算法模块
│   ├── audio_processor.py  # 音频处理
│   ├── asr.py              # 语音识别
│   ├── matcher.py           # 文案匹配
│   ├── video_processor.py  # 视频处理
│   ├── subtitle.py         # 字幕生成
│   └── utils.py            # 工具函数
├── gui/                    # GUI模块
│   ├── main_window.py      # 主窗口
│   ├── panels/             # 面板组件
│   └── timeline/           # 时间线控件
├── config/                 # 配置文件
├── temp/                   # 临时文件
└── requirements.txt        # 依赖列表
```

## 配置说明

配置文件位于 `config/settings.json`，可调整以下参数：

- `similarity_threshold`: 句子匹配阈值 (默认0.6)
- `single_threshold`: 单素材选用阈值 (默认0.85)
- `silence_threshold`: 静音分割阈值 (默认1.5秒)
- `noise_reduce_strength`: 降噪强度 (默认0.3)
- `video_resolution`: 输出分辨率 (默认1080x1920)
- `video_fps`: 输出帧率 (默认30fps)

## 注意事项

1. 首次运行时会下载模型，请确保网络畅通
2. Whisper large模型较大(~3GB)，首次使用需等待下载
3. 视频处理需要较大磁盘空间，建议保留20GB以上
4. 建议使用SSD硬盘以提高处理速度

## 常见问题

**Q: 程序启动报错"找不到FFmpeg"**
A: 请确保FFmpeg已安装并添加到系统PATH环境变量

**Q: Whisper模型下载失败**
A: 可以手动下载模型，或使用较小的模型(tiny/base/small)

**Q: 处理速度很慢**
A: 建议使用GPU加速，或在参数中选择较小的Whisper模型

## 许可证

MIT License
