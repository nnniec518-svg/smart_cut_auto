"""
统一配置管理模块 - Config
负责从 config.yaml 加载所有配置，提供全局配置对象
"""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger("smart_cut")


class Config:
    """
    统一配置管理类
    
    单例模式，全局唯一配置对象
    从 config.yaml 加载所有配置参数
    """
    
    _instance: Optional['Config'] = None
    
    def __new__(cls, config_path: Optional[str] = None):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化配置"""
        if self._initialized:
            return
        
        self.config_path = Path(config_path) if config_path else Path(__file__).parent.parent / "config.yaml"
        self._config: Dict[str, Any] = {}
        self._load_config()
        self._initialized = True
        logger.info(f"Config loaded from: {self.config_path}")
    
    def _load_config(self) -> None:
        """从 config.yaml 加载配置"""
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}, using defaults")
            self._config = self._get_default_config()
            return
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)
            logger.info("Config loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load config: {e}, using defaults")
            self._config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "models": {
                "asr_model": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                "embedding_model": "BAAI/bge-m3",
                "fallback_embedding": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            },
            "vad": {
                "max_single_segment_time": 6000,
                "speech_noise_thres": 0.6,
                "merge_length": 1500,
                "chunk_size": 200
            },
            "filter": {
                "min_text_length": 3,
                "min_confidence": 0.8,
                "max_no_speech_prob": 0.5,
                "dedup_similarity_threshold": 0.8,
                "min_final_duration": 0.4,
                "score_weights": {
                    "text_length": 0.30,
                    "confidence": 0.40,
                    "audio_energy": 0.30
                },
                "a_roll_threshold": 70,
                "logic_version": "2.0"
            },
            "evaluator": {
                "score_weights": {
                    "vad_ratio": 0.30,
                    "energy": 0.25,
                    "length": 0.20,
                    "semantic_integrity": 0.25
                },
                "min_score": 0.65,
                "min_text_length": 3,
                "min_text_duration": 0.5,
                "min_final_duration": 0.4,
                "min_audio_db": -45,
                "dedup_similarity_threshold": 0.85,
                "dedup_min_score": 0.65,
                "energy_min_db": -50,
                "energy_max_db": -10,
                "config_version": "2.0"
            },
            "database": {
                "path": "storage/materials.db",
                "mtime_tolerance": 1.0
            },
            "paths": {
                "raw_folder": "storage/materials-all",
                "output_dir": "temp",
                "logs_dir": "logs"
            },
            "render": {
                "width": 1080,
                "height": 1920,
                "fps": 30,
                "sample_rate": 44100,
                "video_codec": "libx264",
                "crf": 18,
                "audio_bitrate": "128k",
                "crossfade_duration": 0.2
            },
            "classification": {
                "silence_threshold": -40,
                "a_roll_threshold": 0.3,
                "b_roll_threshold": 0.1
            },
            "matching": {
                "penalty_repeat": 0.8,
                "penalty_time_reverse": 1.0,
                "reward_sequence": 0.2,
                "threshold": 0.6,
                "order_bonus": 0.2,
                "max_time_gap": 5.0,
                "index_penalty_factor": 0.1,
                "search_window_size": 30,
                "window_backtrack": 5,
                "adaptive_window": True,
                "enable_number_constraint": True,
                "penalty_time_reverse_light": 0.5,
                "penalty_hard_time_reverse": 1.0,
                "penalty_number_mismatch": 0.4
            },
            "cue_words": {
                "patterns": [
                    "一", "二", "三", "四", "五",
                    "1", "2", "3", "4", "5",
                    "走", "开始", "准备", "action", "咔", "好的", "321", "三二一"
                ],
                "max_search_time": 3.0,
                "buffer_sec": 0.1
            },
            "b_roll_keywords": [
                "空镜头", "空镜", "风景", "景色", "环境",
                "画面", "场景", "外景", "室内", "办公室",
                "产品", "商品", "展示", "演示", "背景",
                "氛围", "街头", "道路", "建筑", "自然",
                "城市", "乡村", "海边", "山景", "花",
                "草", "树", "天空", "云", "日出", "日落",
                "夜景", "灯光", "特写", "远景", "全景"
            ],
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": "logs/smart_cut.log"
            },
            "overlay": {
                "position": "bottom-right",
                "scale": "0.3",
                "opacity": "1.0",
                "padding": "10"
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            key: 配置键（支持点分隔符，如 "filter.min_text_length"）
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_nested(self, *keys: str, default: Any = None) -> Any:
        """
        获取嵌套配置值（便捷方法）
        
        Args:
            *keys: 配置键序列
            default: 默认值
            
        Returns:
            配置值
        """
        return self.get('.'.join(keys), default)
    
    def set(self, key: str, value: Any) -> None:
        """
        设置配置值（运行时动态修改）
        
        Args:
            key: 配置键
            value: 配置值
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        logger.debug(f"Config updated: {key} = {value}")
    
    def reload(self) -> None:
        """重新加载配置文件"""
        self._load_config()
        logger.info("Config reloaded")
    
    def to_dict(self) -> Dict[str, Any]:
        """获取完整配置字典"""
        return self._config.copy()
    
    # ========== 便捷方法：获取常用配置 ==========
    
    @property
    def asr_model(self) -> str:
        """ASR 模型路径"""
        return self.get("models.asr_model")
    
    @property
    def embedding_model(self) -> str:
        """嵌入模型路径"""
        return self.get("models.embedding_model")
    
    @property
    def vad_params(self) -> Dict[str, Any]:
        """VAD 参数"""
        return self._config.get("vad", {})
    
    @property
    def filter_config(self) -> Dict[str, Any]:
        """过滤配置"""
        return self._config.get("filter", {})
    
    @property
    def evaluator_config(self) -> Dict[str, Any]:
        """评分器配置"""
        return self._config.get("evaluator", {})
    
    @property
    def matching_config(self) -> Dict[str, Any]:
        """匹配配置"""
        return self._config.get("matching", {})
    
    @property
    def render_config(self) -> Dict[str, Any]:
        """渲染配置"""
        return self._config.get("render", {})
    
    @property
    def database_path(self) -> str:
        """数据库路径"""
        return self.get("database.path", "storage/materials.db")
    
    @property
    def output_dir(self) -> str:
        """输出目录"""
        return self.get("paths.output_dir", "temp")
    
    @property
    def logs_dir(self) -> str:
        """日志目录"""
        return self.get("paths.logs_dir", "logs")


# ========== 全局配置实例 ==========
config = Config()
