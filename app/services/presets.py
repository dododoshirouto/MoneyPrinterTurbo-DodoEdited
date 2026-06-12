import os
import json
from loguru import logger
from app.utils import utils

PRESETS_FILE = os.path.join(utils.storage_dir(), "presets.json")

def load_presets() -> dict:
    """
    プリセット一覧を読み込みます。
    """
    if not os.path.exists(PRESETS_FILE):
        return {}
    try:
        with open(PRESETS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"failed to load presets: {e}")
        return {}

def save_preset(name: str, params_dict: dict) -> bool:
    """
    指定した名前でプリセットを保存します。
    """
    presets = load_presets()
    presets[name] = params_dict
    try:
        # ディレクトリの存在を確認して保存
        os.makedirs(os.path.dirname(PRESETS_FILE), exist_ok=True)
        with open(PRESETS_FILE, "w", encoding="utf-8") as f:
            json.dump(presets, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"failed to save preset {name}: {e}")
        return False

def delete_preset(name: str) -> bool:
    """
    指定したプリセットを削除します。
    """
    presets = load_presets()
    if name in presets:
        del presets[name]
        try:
            with open(PRESETS_FILE, "w", encoding="utf-8") as f:
                json.dump(presets, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"failed to delete preset {name}: {e}")
            return False
    return False
