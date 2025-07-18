"""
設定管理の基底クラス
"""

import os
import yaml
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Type, Union
from dataclasses import dataclass, field
import logging

from ..utils.exceptions import ConfigurationError


@dataclass
class BaseConfig(ABC):
    """設定管理の基底クラス"""
    
    def __post_init__(self):
        """設定ファイルから値を読み込んで既定値を上書き"""
        self._load_configuration()
        self._validate_configuration()
    
    def _load_configuration(self):
        """設定ファイルから設定を読み込み"""
        config_data = self._load_yaml_config()
        if config_data:
            self._update_from_dict(config_data)
    
    def _load_yaml_config(self) -> Optional[Dict[str, Any]]:
        """YAML設定ファイルを読み込み"""
        yaml_path = self._get_config_file_path()
        if not yaml_path.exists():
            return None
        
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logging.warning(f"Failed to load YAML config from {yaml_path}: {e}")
            return None
    
    def _get_config_file_path(self) -> Path:
        """設定ファイルのパスを取得"""
        return Path(__file__).parent / "settings.yaml"
    
    @abstractmethod
    def _update_from_dict(self, config_dict: Dict[str, Any]):
        """辞書から設定値を更新（サブクラスで実装）"""
        pass
    
    @abstractmethod
    def _validate_configuration(self):
        """設定値の検証（サブクラスで実装）"""
        pass
    
    def get_environment_variable(self, key: str, default: Any = None) -> Any:
        """環境変数を取得"""
        return os.getenv(key, default)
    
    def update_setting(self, key: str, value: Any):
        """設定値を動的に更新"""
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            raise ConfigurationError(f"Setting '{key}' does not exist")
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """設定値を取得"""
        return getattr(self, key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """設定をディクショナリに変換"""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    def save_to_file(self, file_path: Union[str, Path]):
        """設定をファイルに保存"""
        file_path = Path(file_path)
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            raise ConfigurationError(f"Failed to save config to {file_path}: {e}")


class ConfigValidator:
    """設定値の検証を行うクラス"""
    
    @staticmethod
    def validate_positive_integer(value: int, name: str):
        """正の整数値の検証"""
        if not isinstance(value, int) or value <= 0:
            raise ConfigurationError(f"{name} must be a positive integer, got: {value}")
    
    @staticmethod
    def validate_positive_float(value: float, name: str):
        """正の浮動小数点値の検証"""
        if not isinstance(value, (int, float)) or value <= 0:
            raise ConfigurationError(f"{name} must be a positive number, got: {value}")
    
    @staticmethod
    def validate_string(value: str, name: str, min_length: int = 1):
        """文字列の検証"""
        if not isinstance(value, str) or len(value) < min_length:
            raise ConfigurationError(f"{name} must be a string with length >= {min_length}, got: {value}")
    
    @staticmethod
    def validate_url(value: str, name: str):
        """URL形式の検証"""
        if not isinstance(value, str) or not value.startswith(('http://', 'https://')):
            raise ConfigurationError(f"{name} must be a valid URL, got: {value}")
    
    @staticmethod
    def validate_file_path(value: Union[str, Path], name: str, must_exist: bool = False):
        """ファイルパスの検証"""
        path = Path(value)
        if must_exist and not path.exists():
            raise ConfigurationError(f"{name} path does not exist: {value}")
    
    @staticmethod
    def validate_dict(value: Dict[str, Any], name: str, min_items: int = 0):
        """辞書の検証"""
        if not isinstance(value, dict) or len(value) < min_items:
            raise ConfigurationError(f"{name} must be a dictionary with at least {min_items} items, got: {value}")
    
    @staticmethod
    def validate_list(value: list, name: str, min_items: int = 0):
        """リストの検証"""
        if not isinstance(value, list) or len(value) < min_items:
            raise ConfigurationError(f"{name} must be a list with at least {min_items} items, got: {value}")
    
    @staticmethod
    def validate_choice(value: Any, choices: list, name: str):
        """選択肢の検証"""
        if value not in choices:
            raise ConfigurationError(f"{name} must be one of {choices}, got: {value}")


class ConfigManager:
    """複数の設定クラスを管理するクラス"""
    
    def __init__(self):
        self._configs: Dict[str, BaseConfig] = {}
    
    def register_config(self, name: str, config: BaseConfig):
        """設定を登録"""
        self._configs[name] = config
    
    def get_config(self, name: str) -> BaseConfig:
        """設定を取得"""
        if name not in self._configs:
            raise ConfigurationError(f"Config '{name}' not found")
        return self._configs[name]
    
    def reload_all_configs(self):
        """すべての設定を再読み込み"""
        for config in self._configs.values():
            config._load_configuration()
    
    def validate_all_configs(self):
        """すべての設定を検証"""
        for name, config in self._configs.items():
            try:
                config._validate_configuration()
            except Exception as e:
                raise ConfigurationError(f"Validation failed for config '{name}': {e}")
    
    def get_all_configs(self) -> Dict[str, BaseConfig]:
        """すべての設定を取得"""
        return self._configs.copy()


# シングルトンパターンで設定マネージャーを提供
_config_manager = ConfigManager()

def get_config_manager() -> ConfigManager:
    """設定マネージャーのシングルトンインスタンスを取得"""
    return _config_manager