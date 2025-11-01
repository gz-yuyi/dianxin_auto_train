"""文件存储服务"""

from pathlib import Path

from src.core.config import config
from src.utils.logger import logger


class StorageService:
    """文件存储服务"""

    def __init__(self):
        self.uploads_dir = Path(config.UPLOADS_DIR)
        self.models_dir = Path(config.MODELS_DIR)

        # 确保目录存在
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"存储服务初始化完成 - 上传目录: {self.uploads_dir}, 模型目录: {self.models_dir}"
        )

    def save_uploaded_file(self, filename: str, content: bytes) -> str:
        """保存上传的文件"""
        file_path = self.uploads_dir / filename

        # 如果文件已存在，删除旧文件
        if file_path.exists():
            file_path.unlink()
            logger.warning(f"文件已存在，删除旧文件: {filename}")

        # 保存文件
        with open(file_path, "wb") as f:
            f.write(content)

        logger.info(f"文件已保存: {filename} -> {file_path}")
        return str(file_path)

    def get_file_path(self, filename: str) -> str:
        """获取文件完整路径"""
        return str(self.uploads_dir / filename)

    def file_exists(self, filename: str) -> bool:
        """检查文件是否存在"""
        file_path = self.uploads_dir / filename
        return file_path.exists()

    def delete_file(self, filename: str) -> bool:
        """删除文件"""
        file_path = self.uploads_dir / filename
        if file_path.exists():
            file_path.unlink()
            logger.info(f"文件已删除: {filename}")
            return True
        return False

    def get_model_path(self, model_name: str) -> str:
        """获取模型保存路径"""
        return str(self.models_dir / f"{model_name}.pt")

    def save_model(self, model_name: str, model_data: bytes) -> str:
        """保存模型"""
        model_path = self.get_model_path(model_name)

        with open(model_path, "wb") as f:
            f.write(model_data)

        logger.info(f"模型已保存: {model_name} -> {model_path}")
        return model_path

    def model_exists(self, model_name: str) -> bool:
        """检查模型是否存在"""
        model_path = Path(self.get_model_path(model_name))
        return model_path.exists()

    def delete_model(self, model_name: str) -> bool:
        """删除模型"""
        model_path = Path(self.get_model_path(model_name))
        if model_path.exists():
            model_path.unlink()
            logger.info(f"模型已删除: {model_name}")
            return True
        return False

    def get_uploaded_files(self) -> list:
        """获取已上传的文件列表"""
        files = []
        for file_path in self.uploads_dir.iterdir():
            if file_path.is_file():
                files.append(
                    {
                        "filename": file_path.name,
                        "size": file_path.stat().st_size,
                        "modified_time": file_path.stat().st_mtime,
                    }
                )
        return files

    def cleanup_old_files(self, days: int = 30) -> int:
        """清理旧文件"""
        import time

        current_time = time.time()
        deleted_count = 0

        for file_path in self.uploads_dir.iterdir():
            if file_path.is_file():
                file_age_days = (current_time - file_path.stat().st_mtime) / (24 * 3600)
                if file_age_days > days:
                    file_path.unlink()
                    deleted_count += 1
                    logger.info(
                        f"删除旧文件: {file_path.name} (年龄: {file_age_days:.1f}天)"
                    )

        logger.info(f"清理完成，删除文件数: {deleted_count}")
        return deleted_count


# 全局存储服务实例
storage_service = StorageService()
