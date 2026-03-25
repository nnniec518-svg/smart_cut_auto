"""
数据库模型 - 使用 SQLAlchemy 定义数据表结构
用于素材存储、段落管理和断点续传
"""
import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.sql import func

Base = declarative_base()


class Asset(Base):
    """
    素材表 - 存储视频素材的元数据
    """
    __tablename__ = "assets"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # 文件路径 (唯一索引)
    file_path = Column(String(500), unique=True, nullable=False, index=True)
    file_name = Column(String(200), nullable=False)
    
    # 素材类型: A_ROLL (主轨/口播) / B_ROLL (空镜头/静音)
    track_type = Column(String(20), nullable=False, default="B_ROLL")
    
    # 有效起始偏移量 (秒) - 用于跳过 321、开始、action 等提示词
    valid_start_offset = Column(Float, default=0.0)
    
    # 视频时长
    duration = Column(Float, default=0.0)
    
    # 音频信息
    has_audio = Column(Boolean, default=True)
    audio_db = Column(Float, default=-100.0)
    
    # 文件修改时间 (用于断点续传)
    mtime = Column(Float, nullable=False)
    
    # ASR 文本
    asr_text = Column(Text, default="")
    
    # 完整转录 JSON (包含时间戳)
    transcript_json = Column(Text, default="")
    
    # 向量嵌入 (存储为 JSON 字符串，用于快速相似度计算)
    embedding_vector = Column(Text, default="")
    
    # ASR 置信度 (0-1)
    asr_confidence = Column(Float, default=0.0)
    
    # A-Roll 打分 (0-100)
    a_roll_score = Column(Float, default=0.0)
    
    # 片段质量标记: valid, invalid_noise, invalid_short, duplicate
    quality_status = Column(String(20), default="valid")
    
    # 时间戳
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # 关联段落
    segments = relationship("Segment", back_populates="asset", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Asset(id={self.id}, name='{self.file_name}', type='{self.track_type}')>"


class Segment(Base):
    """
    段落表 - 存储素材的 ASR 片段信息
    用于语义匹配和精准裁剪
    """
    __tablename__ = "segments"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # 外键关联素材
    asset_id = Column(Integer, ForeignKey("assets.id"), nullable=False, index=True)
    
    # 视频ID (关联回 Asset 的 id)
    video_id = Column(Integer, nullable=False, index=True)
    
    # 时间范围 (相对于素材原始开始时间)
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    
    # 有效起始偏移量 (来自 Asset 的 valid_start_offset)
    valid_start_offset = Column(Float, default=0.0)
    
    # 段落文本
    asr_text = Column(Text, default="")
    
    # 词级时间戳 (JSON 格式: [[start_ms, end_ms], ...])
    timestamps_json = Column(Text, default="")
    
    # 向量嵌入 (JSON 字符串)
    embedding_vector = Column(Text, default="")
    
    # 关联素材
    asset = relationship("Asset", back_populates="segments")
    
    def __repr__(self):
        return f"<Segment(id={self.id}, video_id={self.video_id}, text='{self.asr_text[:20]}...')>"


class Database:
    """
    数据库管理器 - 封装常用操作
    """
    
    def __init__(self, db_path: str = "storage/materials.db"):
        """
        初始化数据库连接
        
        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        
        self.engine = create_engine(
            f"sqlite:///{db_path}",
            echo=False,
            connect_args={"check_same_thread": False}
        )
        
        # 创建表
        Base.metadata.create_all(self.engine)
        
        self.Session = sessionmaker(bind=self.engine)
        
    def get_session(self):
        """获取数据库会话"""
        return self.Session()
    
    def add_asset(self, asset: Asset) -> Asset:
        """
        添加素材
        
        Args:
            asset: Asset 对象
            
        Returns:
            添加后的 Asset 对象
        """
        session = self.get_session()
        try:
            # 检查是否已存在
            existing = session.query(Asset).filter_by(file_path=asset.file_path).first()
            if existing:
                # 更新现有记录
                existing.file_name = asset.file_name
                existing.track_type = asset.track_type
                existing.valid_start_offset = asset.valid_start_offset
                existing.duration = asset.duration
                existing.has_audio = asset.has_audio
                existing.audio_db = asset.audio_db
                existing.mtime = asset.mtime
                existing.asr_text = asset.asr_text
                existing.transcript_json = asset.transcript_json
                existing.embedding_vector = asset.embedding_vector
                existing.updated_at = datetime.now()
                session.commit()
                session.refresh(existing)
                return existing
            else:
                session.add(asset)
                session.commit()
                session.refresh(asset)
                return asset
        finally:
            session.close()
    
    def get_asset_by_path(self, file_path: str) -> Asset:
        """
        通过文件路径获取素材
        
        Args:
            file_path: 文件路径
            
        Returns:
            Asset 对象或 None
        """
        session = self.get_session()
        try:
            return session.query(Asset).filter_by(file_path=file_path).first()
        finally:
            session.close()
    
    def get_assets_by_type(self, track_type: str) -> list:
        """
        获取指定类型的素材
        
        Args:
            track_type: 素材类型 (A_ROLL / B_ROLL)
            
        Returns:
            Asset 对象列表
        """
        session = self.get_session()
        try:
            return session.query(Asset).filter_by(track_type=track_type).all()
        finally:
            session.close()
    
    def check_asset_fresh(self, file_path: str, current_mtime: float) -> bool:
        """
        检查素材是否需要更新 (断点续传)
        
        Args:
            file_path: 文件路径
            current_mtime: 当前文件的修改时间
            
        Returns:
            True 表示素材是最新的，不需要重新处理
        """
        asset = self.get_asset_by_path(file_path)
        if not asset:
            return False
        return asset.mtime == current_mtime
    
    def add_segment(self, segment: Segment) -> Segment:
        """
        添加段落
        
        Args:
            segment: Segment 对象
            
        Returns:
            添加后的 Segment 对象
        """
        session = self.get_session()
        try:
            session.add(segment)
            session.commit()
            session.refresh(segment)
            return segment
        finally:
            session.close()
    
    def get_segments_by_video_id(self, video_id: int) -> list:
        """
        获取指定视频的所有段落
        
        Args:
            video_id: 视频 ID
            
        Returns:
            Segment 对象列表
        """
        session = self.get_session()
        try:
            return session.query(Segment).filter_by(video_id=video_id).all()
        finally:
            session.close()
    
    def close(self):
        """关闭数据库连接"""
        self.engine.dispose()


if __name__ == "__main__":
    # 测试代码
    db = Database("storage/test.db")
    
    # 创建测试素材
    import time
    asset = Asset(
        file_path="test_video.MOV",
        file_name="test_video",
        track_type="A_ROLL",
        valid_start_offset=1.5,
        duration=30.0,
        has_audio=True,
        audio_db=-20.0,
        mtime=time.time(),
        asr_text="这是测试文本",
        transcript_json='{"text": "这是测试文本", "segments": []}'
    )
    
    saved_asset = db.add_asset(asset)
    logger.info(f"保存素材: {saved_asset}")

    # 查询
    assets = db.get_assets_by_type("A_ROLL")
    logger.info(f"A_ROLL 素材数量: {len(assets)}")
    
    db.close()
