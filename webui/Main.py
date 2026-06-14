import json
import os
import sys
import webbrowser
import glob
from uuid import UUID, uuid4

import requests
import streamlit as st
from loguru import logger

# Add the root directory of the project to the system path to allow importing modules from the project
root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)
    print("******** sys.path ********")
    print(sys.path)
    print("")

from app.config import config
# Apply GPU settings as early as possible before loading other packages like torch/whisper
cuda_device = config.app.get("cuda_device", "")
if cuda_device:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device).strip()

from app.models.schema import (
    MaterialInfo,
    VideoAspect,
    VideoConcatMode,
    VideoParams,
    VideoTransitionMode,
)
from app.services import llm, voice, presets, research_agent
from app.services import task as tm
from app.utils import utils

st.set_page_config(
    page_title="MoneyPrinterTurbo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        "Report a bug": "https://github.com/harry0703/MoneyPrinterTurbo/issues",
        "About": "# MoneyPrinterTurbo\nSimply provide a topic or keyword for a video, and it will "
        "automatically generate the video copy, video materials, video subtitles, "
        "and video background music before synthesizing a high-definition short "
        "video.\n\nhttps://github.com/harry0703/MoneyPrinterTurbo",
    },
)


streamlit_style = """
<style>
h1 {
    padding-top: 0 !important;
}
</style>
"""
st.markdown(streamlit_style, unsafe_allow_html=True)

# 定义资源目录
font_dir = os.path.join(root_dir, "resource", "fonts")
song_dir = os.path.join(root_dir, "resource", "songs")
i18n_dir = os.path.join(root_dir, "webui", "i18n")
config_file = os.path.join(root_dir, "webui", ".streamlit", "webui.toml")
system_locale = utils.get_system_locale()

# ---------------------------------------------------------------------------
# Preset registry – single source of truth for all saveable GUI fields.
# Format: (config_key, config_store, session_state_key_or_None)
#   config_key   : key used in config.app / config.ui
#   config_store : "app" | "ui"
#   session_key  : st.session_state key when the widget uses key=, else None
#                  (None = keyless widget; controlled via config value/index)
# ---------------------------------------------------------------------------
_LLM_PROVIDERS_LIST = [
    "openai", "aihubmix", "aimlapi", "moonshot", "azure", "qwen",
    "deepseek", "modelscope", "gemini", "grok", "groq", "ollama",
    "llmcpp", "g4f", "oneapi", "cloudflare", "ernie", "minimax",
    "mimo", "pollinations", "litellm",
]

_PRESET_REGISTRY: list[tuple[str, str, str | None]] = [
    # Video
    ("llm_provider",                "app", None),
    ("video_source",                "app", None),
    ("video_concat_mode",           "app", "video_concat_mode"),
    ("video_clip_fit",              "app", None),
    ("video_clip_duration",         "app", "video_clip_duration"),
    ("match_materials_to_script",   "app", "match_materials_to_script"),
    ("video_count",                 "app", "video_count"),
    ("video_transition_mode",       "app", "video_transition_mode"),
    ("video_aspect",                "app", "video_aspect"),
    # Script
    ("paragraph_number",            "app", "paragraph_number_input"),
    ("video_script_prompt",         "app", "video_script_prompt"),
    ("custom_system_prompt",        "app", "custom_system_prompt"),
    ("use_custom_system_prompt",    "app", "use_custom_system_prompt"),
    ("task_folder_template",        "app", None),
    ("use_title_in_script",         "app", "use_title_in_script"),
    # Web search
    ("web_search_enabled",          "app", "web_search_enabled"),
    ("web_search_prompt",           "app", "web_search_prompt"),
    ("web_search_max_steps",        "app", "web_search_max_steps"),
    # YouTube
    ("youtube_enabled",             "app", "youtube_enabled"),
    ("youtube_selected_account",    "app", "youtube_selected_account"),
    ("youtube_privacy",             "app", None),
    ("youtube_schedule_hours",      "app", "youtube_schedule_hours"),
    ("youtube_auto_metadata",       "app", "youtube_auto_metadata"),
    # Audio
    ("voice_volume",                "app", "voice_volume"),
    ("voice_rate",                  "app", "voice_rate"),
    ("bgm_type",                    "app", "bgm_type"),
    ("bgm_file",                    "app", None),
    ("bgm_volume",                  "app", "bgm_volume"),
    # Subtitle
    ("subtitle_enabled",            "app", "subtitle_enabled"),
    ("subtitle_offset",             "app", "subtitle_offset"),
    ("n_threads",                   "app", None),
    # System
    ("voicevox_base_url",           "app", "voicevox_base_url_input"),
    ("aivisspeech_base_url",        "app", "aivisspeech_base_url_input"),
    ("timezone",                    "app", "timezone_input"),
    ("cuda_device",                 "app", "cuda_device_input"),
    # UI store – TTS / voice
    ("tts_server",                  "ui",  None),
    ("voice_name",                  "ui",  None),
    # UI store – background video
    # NOTE: bg_video_type is handled specially in _apply_preset_data because the
    # selectbox stores an integer index in session_state, not the string value.
    ("bg_video_type",               "ui",  None),
    ("bg_video_file",               "ui",  "custom_bg_video_file_selectbox"),
    # UI store – video display
    ("video_margin_ratio",          "ui",  None),
    # UI store – font / subtitle styling
    ("font_name",                   "ui",  None),
    ("text_fore_color",             "ui",  None),
    ("font_size",                   "ui",  None),
    ("stroke_color",                "ui",  "stroke_color"),
    ("stroke_width",                "ui",  "stroke_width"),
    ("subtitle_position",           "ui",  None),
    ("custom_position",             "ui",  "custom_position_input"),
    ("subtitle_background_enabled", "ui",  None),
    ("subtitle_background_color",   "ui",  None),
    ("rounded_subtitle_background", "ui",  None),
    ("text_margin_x",               "ui",  None),
    ("text_margin_y",               "ui",  None),
    # UI store – color highlight
    ("text_color_highlight_enabled","ui",  None),
    ("color1_fore",                 "ui",  "color1_fore_picker"),
    ("color1_stroke",               "ui",  "color1_stroke_picker"),
    ("color1_stroke_width",         "ui",  "color1_stroke_width_slider"),
    ("color2_fore",                 "ui",  "color2_fore_picker"),
    ("color2_stroke",               "ui",  "color2_stroke_picker"),
    ("color2_stroke_width",         "ui",  "color2_stroke_width_slider"),
    ("color3_fore",                 "ui",  "color3_fore_picker"),
    ("color3_stroke",               "ui",  "color3_stroke_picker"),
    ("color3_stroke_width",         "ui",  "color3_stroke_width_slider"),
]
for _p in _LLM_PROVIDERS_LIST:
    _PRESET_REGISTRY += [
        (f"{_p}_model_name", "app", None),
        (f"{_p}_api_key",    "app", None),
        (f"{_p}_base_url",   "app", None),
    ]
_PRESET_REGISTRY += [
    ("ernie_secret_key",      "app", None),
    ("cloudflare_account_id", "app", None),
]

# Session-state keys owned by the preset system (auto-derived).
_PRESET_SESSION_KEYS: set[str] = {skey for _, _, skey in _PRESET_REGISTRY if skey}

# Keys that must NEVER be deleted during a preset load.
_UI_PRESERVED_KEYS: set[str] = {
    "top_language_selector", "selected_preset_selector", "ui_language",
    "video_subject", "video_script", "video_terms", "video_title",
    "preset_loaded_message", "video_title_enabled",
}

_BG_VIDEO_OPTIONS = ["none", "random", "source", "custom"]


def _collect_preset_data() -> dict:
    """Collect current GUI state into a flat preset dict.

    Step 1 – sync keyed-widget session_state values into config so the
              collected data reflects the live GUI, not stale config values.
    Step 2 – read every registered field from its canonical config store.
    """
    for cfg_key, store, skey in _PRESET_REGISTRY:
        if skey is None or skey not in st.session_state:
            continue
        cfg = config.app if store == "app" else config.ui
        val = st.session_state[skey]
        if cfg_key == "custom_position":
            try:
                val = float(val)
            except (ValueError, TypeError):
                continue
        cfg[cfg_key] = val

    data: dict = {}
    for cfg_key, store, _ in _PRESET_REGISTRY:
        cfg = config.app if store == "app" else config.ui
        if cfg_key in cfg:
            data[cfg_key] = cfg[cfg_key]
    return data


def _apply_preset_data(raw_data: dict) -> None:
    """Apply a preset dict to both config stores and session_state.

    Handles both the new flat format and the legacy format that stored UI
    fields under a nested "ui" key and session fields under "session".
    """
    # Flatten legacy format
    data: dict = {}
    for k, v in raw_data.items():
        if k == "ui" and isinstance(v, dict):
            data.update(v)
        elif k == "session" and isinstance(v, dict):
            # Legacy session entries were only use_custom_system_prompt; skip
            pass
        else:
            data[k] = v

    for cfg_key, store, skey in _PRESET_REGISTRY:
        if cfg_key not in data:
            continue
        val = data[cfg_key]
        cfg = config.app if store == "app" else config.ui
        cfg[cfg_key] = val

        if skey is None:
            continue
        # custom_position: config is float, session_state is string (text_input)
        if cfg_key == "custom_position":
            st.session_state[skey] = str(val)
        else:
            st.session_state[skey] = val

    # Special: bg_video_type selectbox uses an integer index in session_state
    if "bg_video_type" in data:
        val = data["bg_video_type"]
        st.session_state["bg_video_type_selectbox"] = (
            _BG_VIDEO_OPTIONS.index(val) if val in _BG_VIDEO_OPTIONS else 0
        )


if "_one_click_pending" not in st.session_state:
    st.session_state["_one_click_pending"] = False
if "video_subject" not in st.session_state:
    st.session_state["video_subject"] = ""
if "video_script" not in st.session_state:
    st.session_state["video_script"] = ""
if "video_terms" not in st.session_state:
    st.session_state["video_terms"] = ""
if "video_script_prompt" not in st.session_state:
    st.session_state["video_script_prompt"] = ""
if "custom_system_prompt" not in st.session_state:
    st.session_state["custom_system_prompt"] = llm.DEFAULT_SCRIPT_SYSTEM_PROMPT
if "use_custom_system_prompt" not in st.session_state:
    st.session_state["use_custom_system_prompt"] = False
if "web_search_enabled" not in st.session_state:
    st.session_state["web_search_enabled"] = bool(config.app.get("web_search_enabled", True))
if "web_search_prompt" not in st.session_state:
    st.session_state["web_search_prompt"] = config.app.get("web_search_prompt", "")
if "web_search_max_steps" not in st.session_state:
    st.session_state["web_search_max_steps"] = int(config.app.get("web_search_max_steps", 3))
if "match_materials_to_script" not in st.session_state:
    st.session_state["match_materials_to_script"] = bool(
        config.app.get("match_materials_to_script", False)
    )
if "ui_language" not in st.session_state:
    st.session_state["ui_language"] = config.ui.get("language", system_locale)
if "local_video_materials" not in st.session_state:
    # 记住用户最近一次已经落盘的本地素材，避免仅修改文案后二次生成时丢失素材列表。
    st.session_state["local_video_materials"] = []

# 加载语言文件
locales = utils.load_locales(i18n_dir)

# 创建一个顶部栏，包含标题和语言选择
title_col, lang_col = st.columns([3, 1])

with title_col:
    st.title(f"MoneyPrinterTurbo v{config.project_version}")

with lang_col:
    display_languages = []
    selected_index = 0
    for i, code in enumerate(locales.keys()):
        display_languages.append(f"{code} - {locales[code].get('Language')}")
        if code == st.session_state.get("ui_language", ""):
            selected_index = i

    selected_language = st.selectbox(
        "Language / 语言",
        options=display_languages,
        index=selected_index,
        key="top_language_selector",
        label_visibility="collapsed",
    )
    if selected_language:
        code = selected_language.split(" - ")[0].strip()
        st.session_state["ui_language"] = code
        config.ui["language"] = code

support_locales = [
    "zh-CN",
    "zh-HK",
    "zh-TW",
    "de-DE",
    "en-US",
    "fr-FR",
    "ru-RU",
    "vi-VN",
    "th-TH",
    "tr-TR",
]


def get_all_fonts():
    fonts = []
    for root, dirs, files in os.walk(font_dir):
        for file in files:
            if file.endswith(".ttf") or file.endswith(".ttc") or file.endswith(".otf"):
                fonts.append(file)
    fonts.sort()
    return fonts


def get_all_songs():
    songs = []
    for root, dirs, files in os.walk(song_dir):
        for file in files:
            if file.endswith(".mp3"):
                songs.append(file)
    return songs


def open_task_folder(task_id):
    try:
        # Jinja2 によるカスタムフォルダ名に対応するため、UUID検証から
        # 禁止文字クレンジングとパス穿越防止に緩和します。
        import re
        safe_task_id = re.sub(r'[\\/*?:"<>|\s]', '_', str(task_id)).strip(' ._')
        if not safe_task_id:
            logger.warning("invalid empty task_id")
            return

        tasks_root = os.path.abspath(os.path.join(root_dir, "storage", "tasks"))
        path = os.path.abspath(os.path.join(tasks_root, safe_task_id))

        # パス穿越の脆弱性チェック
        if not path.startswith(tasks_root + os.sep):
            logger.warning(f"invalid task folder path: {path}")
            return

        if os.path.isdir(path):
            webbrowser.open(f"file://{path}")
    except Exception as e:
        logger.error(e)


def scroll_to_bottom():
    js = """
    <script>
        console.log("scroll_to_bottom");
        function scroll(dummy_var_to_force_repeat_execution){
            var sections = parent.document.querySelectorAll('section.main');
            console.log(sections);
            for(let index = 0; index<sections.length; index++) {
                sections[index].scrollTop = sections[index].scrollHeight;
            }
        }
        scroll(1);
    </script>
    """
    st.components.v1.html(js, height=0, width=0)


def init_log():
    logger.remove()
    _lvl = "DEBUG"

    def format_record(record):
        # 获取日志记录中的文件全路径
        file_path = record["file"].path
        # 将绝对路径转换为相对于项目根目录的路径
        relative_path = os.path.relpath(file_path, root_dir)
        # 更新记录中的文件路径
        record["file"].path = f"./{relative_path}"
        # 返回修改后的格式字符串
        # 您可以根据需要调整这里的格式
        record["message"] = record["message"].replace(root_dir, ".")

        # タイムゾーン変換: config に設定されたタイムゾーン（デフォルト Asia/Tokyo）を使用
        try:
            from zoneinfo import ZoneInfo
            tz_name = config.app.get("timezone", "Asia/Tokyo") or "Asia/Tokyo"
            tz = ZoneInfo(tz_name)
            local_time = record["time"].astimezone(tz)
        except Exception:
            local_time = record["time"]

        _format = (
            "<green>{time:%Y-%m-%d %H:%M:%S}</> | "
            + "<level>{level}</> | "
            + '"{file.path}:{line}":<blue> {function}</> '
            + "- <level>{message}</>"
            + "\n"
        )
        # loguru の {time} は record["time"] を使うため、ローカル時間を record に反映する
        record["time"] = local_time
        return _format

    logger.add(
        sys.stdout,
        level=_lvl,
        format=format_record,
        colorize=True,
    )


init_log()

locales = utils.load_locales(i18n_dir)


def tr(key):
    loc = locales.get(st.session_state["ui_language"], {})
    return loc.get("Translation", {}).get(key, key)

@st.cache_data(ttl=300, show_spinner=False)
def get_groq_model_ids(api_key: str, base_url: str) -> list[str]:
    if not api_key:
        return []

    normalized_base_url = (base_url or "https://api.groq.com/openai/v1").strip().rstrip("/")
    models_url = f"{normalized_base_url}/models"

    try:
        response = requests.get(
            models_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data", [])

        model_ids = []
        for item in data:
            if isinstance(item, dict):
                model_id = item.get("id")
                if isinstance(model_id, str) and model_id.strip():
                    model_ids.append(model_id.strip())

        return sorted(set(model_ids))
    except Exception as e:
        logger.warning(f"failed to fetch groq models: {e}")
        return []

def on_preset_selected(preset_dict):
    selected = st.session_state.get("selected_preset_selector")
    if not selected or selected not in preset_dict:
        return
    _apply_preset_data(preset_dict[selected])
    config.save_config()
    # selected_preset_selector はクリアしない → 読み込んだプリセット名のまま保持
    st.session_state["preset_loaded_message"] = f"{tr('Preset loaded successfully')}: {selected}"

# プリセット保存/読込機能のUI
st.write(tr("Preset Settings"))
if "preset_loaded_message" in st.session_state and st.session_state["preset_loaded_message"]:
    st.success(st.session_state["preset_loaded_message"])
    st.session_state["preset_loaded_message"] = ""

preset_cols = st.columns([3, 3, 2])
with preset_cols[0]:
    # プリセット一覧を取得
    preset_dict = presets.load_presets()
    preset_names = list(preset_dict.keys())
    
    selected_preset = st.selectbox(
        tr("Select Preset"),
        options=[""] + preset_names,
        key="selected_preset_selector",
        label_visibility="collapsed",
        on_change=on_preset_selected,
        args=(preset_dict,)
    )
with preset_cols[1]:
    new_preset_name = st.text_input(
        tr("Preset Name"),
        key="new_preset_name_input",
        placeholder=tr("Preset Name"),
        label_visibility="collapsed"
    )
with preset_cols[2]:
    save_col, delete_col = st.columns(2)
    # プリセット保存ボタン
    if save_col.button(tr("Save Preset"), use_container_width=True):
        if new_preset_name.strip():
            preset_data = _collect_preset_data()
            if presets.save_preset(new_preset_name.strip(), preset_data):
                st.success(tr("Preset saved successfully"))
                st.rerun()
        else:
            st.error("Please enter a preset name")
            
    # プリセット削除ボタン
    if delete_col.button(tr("Delete"), use_container_width=True):
        if selected_preset:
            if presets.delete_preset(selected_preset):
                st.success(f"Preset deleted: {selected_preset}")
                st.rerun()
        else:
            st.error("Please select a preset to delete")

# ロード処理はon_preset_selectedコールバックに移行されました。

# プリセット エクスポート / インポート
exp_col, imp_col = st.columns([1, 1])
with exp_col:
    _exp_data = preset_dict.get(selected_preset) if selected_preset else None
    if _exp_data:
        st.download_button(
            tr("Export Preset"),
            data=json.dumps(_exp_data, ensure_ascii=False, indent=2),
            file_name=f"{selected_preset}.json",
            mime="application/json",
            use_container_width=True,
        )
    else:
        st.button(tr("Export Preset"), disabled=True, use_container_width=True)
with imp_col:
    _imported_file = st.file_uploader(
        tr("Import Preset"),
        type=["json"],
        key="import_preset_uploader",
        label_visibility="collapsed",
    )
    if _imported_file is not None:
        try:
            _imp_data = json.loads(_imported_file.read().decode("utf-8"))
            _imp_name = os.path.splitext(_imported_file.name)[0]
            if presets.save_preset(_imp_name, _imp_data):
                st.success(f"{tr('Preset Imported')}: {_imp_name}")
                st.rerun()
        except Exception as _e:
            st.error(f"Import failed: {_e}")

st.write("---")

# ワンクリック生成エリア（GUIトップ）
with st.container(border=True):
    _oc_left, _oc_right = st.columns([5, 2])
    with _oc_left:
        _oc_subject = st.text_input(
            tr("Video Subject"),
            value=st.session_state.get("video_subject", ""),
            key="one_click_subject_input",
            placeholder=tr("One Click Subject Placeholder"),
            label_visibility="collapsed",
        )
    with _oc_right:
        _oc_clicked = st.button(
            tr("One Click Generate"),
            use_container_width=True,
            type="primary",
        )
    if _oc_clicked:
        if not _oc_subject.strip():
            st.error(tr("Please Enter the Video Subject"))
        else:
            st.session_state["video_subject"] = _oc_subject.strip()
            st.session_state["video_script"] = ""
            st.session_state["video_terms"] = ""
            st.session_state["video_title"] = ""
            st.session_state["_one_click_pending"] = True
            st.rerun()

st.write("---")

# 创建基础设置折叠框
if not config.app.get("hide_config", False):
    with st.expander(tr("Basic Settings"), expanded=False):
        config_panels = st.columns(3)
        left_config_panel = config_panels[0]
        middle_config_panel = config_panels[1]
        right_config_panel = config_panels[2]

        # 左侧面板 - 日志设置
        with left_config_panel:
            # 是否隐藏配置面板
            hide_config = st.checkbox(
                tr("Hide Basic Settings"), value=config.app.get("hide_config", False)
            )
            config.app["hide_config"] = hide_config

            # 是否禁用日志显示
            hide_log = st.checkbox(
                tr("Hide Log"), value=config.ui.get("hide_log", False)
            )
            config.ui["hide_log"] = hide_log

            # 任务文件夹命名模板
            task_folder_template = st.text_input(
                tr("Task Folder Template"),
                value=config.app.get("task_folder_template", "{{task_id}}"),
                help=tr("Template variables: {{task_id}}, {{datetime}}, {{date}}, {{time}}, {{model_name}}, {{video_subject}}")
            )
            config.app["task_folder_template"] = task_folder_template

            # タイムゾーン設定
            saved_timezone = config.app.get("timezone", "Asia/Tokyo")
            timezone_input = st.text_input(
                tr("Timezone"),
                value=saved_timezone,
                key="timezone_input",
                help=tr("Timezone for task folder name (e.g. Asia/Tokyo, UTC)")
            )
            config.app["timezone"] = timezone_input.strip()

            # GPU設定
            saved_cuda_device = config.app.get("cuda_device", "")
            cuda_device_input = st.text_input(
                tr("GPU Device (CUDA_VISIBLE_DEVICES)"),
                value=saved_cuda_device,
                key="cuda_device_input",
                help=tr("GPU Device Help")
            )
            config.app["cuda_device"] = cuda_device_input.strip()

            # YouTube OAuth2 認証情報（config.toml に保存 → git管理外）
            st.write(tr("YouTube API Credentials"))
            st.caption(tr("YouTube API Credentials Help"))
            _yt_cid = st.text_input(
                tr("YouTube Client ID"),
                value=config.app.get("youtube_client_id", ""),
                key="youtube_client_id_input",
            )
            config.app["youtube_client_id"] = _yt_cid.strip()
            _yt_csecret = st.text_input(
                tr("YouTube Client Secret"),
                value=config.app.get("youtube_client_secret", ""),
                key="youtube_client_secret_input",
                type="password",
            )
            config.app["youtube_client_secret"] = _yt_csecret.strip()


        # 中间面板 - LLM 设置

        with middle_config_panel:
            st.write(tr("LLM Settings"))
            # 下拉框需要展示“AIHubMix（推荐）”这类面向用户的文案，
            # 但配置文件和后端逻辑必须继续使用稳定的小写 provider id。
            # 因此这里显式维护 display label 和 provider id 的映射，避免
            # UI 文案变化污染 `config.app["llm_provider"]`。
            aihubmix_label = f"AIHubMix ({tr('Recommended')})"
            if config.ui.get("language") == "zh":
                aihubmix_label = "AIHubMix（推荐）"
            llm_provider_options = [
                ("OpenAI", "openai"),
                (aihubmix_label, "aihubmix"),
                ("AIML API", "aimlapi"),
                ("Moonshot", "moonshot"),
                ("Azure", "azure"),
                ("Qwen", "qwen"),
                ("DeepSeek", "deepseek"),
                ("ModelScope", "modelscope"),
                ("Gemini", "gemini"),
                ("Grok", "grok"),
                ("Groq", "groq"),
                ("Ollama", "ollama"),
                ("LLM.cpp", "llmcpp"),
                ("G4f", "g4f"),
                ("OneAPI", "oneapi"),
                ("Cloudflare", "cloudflare"),
                ("ERNIE", "ernie"),
                ("MiniMax", "minimax"),
                ("MiMo", "mimo"),
                ("Pollinations", "pollinations"),
                ("LiteLLM", "litellm"),
            ]
            llm_provider_labels = [label for label, _ in llm_provider_options]
            llm_provider_values = {
                label: provider_id for label, provider_id in llm_provider_options
            }
            saved_llm_provider = config.app.get("llm_provider", "openai").lower()
            saved_llm_provider_index = 0
            for i, (_, provider_id) in enumerate(llm_provider_options):
                if provider_id == saved_llm_provider:
                    saved_llm_provider_index = i
                    break

            llm_provider_label = st.selectbox(
                tr("LLM Provider"),
                options=llm_provider_labels,
                index=saved_llm_provider_index,
            )
            llm_helper = st.container()
            llm_provider = llm_provider_values[llm_provider_label]
            config.app["llm_provider"] = llm_provider

            llm_api_key = config.app.get(f"{llm_provider}_api_key", "")
            llm_secret_key = config.app.get(
                f"{llm_provider}_secret_key", ""
            )  # only for baidu ernie
            llm_base_url = config.app.get(f"{llm_provider}_base_url", "")
            llm_model_name = config.app.get(f"{llm_provider}_model_name", "")
            llm_account_id = config.app.get(f"{llm_provider}_account_id", "")

            tips = ""
            if llm_provider == "llmcpp":
                if not llm_model_name:
                    llm_model_name = "local-model"
                if not llm_base_url:
                    llm_base_url = "http://localhost:8080/v1"
                with llm_helper:
                    tips = """
                            ##### LLM.cpp Configuration
                            - **API Key**: Usually not required (you can leave it empty or any string)
                            - **Base Url**: Default is http://localhost:8080/v1 (adjust if needed)
                            - **Model Name**: Model name specified in llama.cpp server configuration
                            """

            if llm_provider == "ollama":
                if not llm_model_name:
                    llm_model_name = "qwen:7b"
                if not llm_base_url:
                    llm_base_url = config.get_default_ollama_base_url()

                with llm_helper:
                    docker_hint = ""
                    if config.is_running_in_container():
                        docker_hint = "\n                            > 检测到容器环境，未配置 Base Url 时会默认使用 `http://host.docker.internal:11434/v1`\n"
                    tips = f"""
                            ##### Ollama配置说明
                            - **API Key**: 随便填写，比如 123
                            - **Base Url**: 一般为 http://localhost:11434/v1
                                - 如果 `MoneyPrinterTurbo` 和 `Ollama` **不在同一台机器上**，需要填写 `Ollama` 机器的IP地址
                                - 如果 `MoneyPrinterTurbo` 是 `Docker` 部署，建议填写 `http://host.docker.internal:11434/v1`{docker_hint}
                            - **Model Name**: 使用 `ollama list` 查看，比如 `qwen:7b`
                            """

            if llm_provider == "openai":
                if not llm_model_name:
                    llm_model_name = "gpt-3.5-turbo"
                with llm_helper:
                    tips = """
                            ##### OpenAI 配置说明
                            > 需要VPN开启全局流量模式
                            - **API Key**: [点击到官网申请](https://platform.openai.com/api-keys)
                            - **Base Url**: 官方 OpenAI 可留空；如果使用 OpenAI 兼容供应商（例如 OpenRouter），请填写对应的兼容接口地址
                            - **Model Name**: 填写**有权限**的模型；如果使用兼容供应商，请填写该平台支持的模型 ID
                            """

            if llm_provider == "aihubmix":
                if not llm_model_name:
                    llm_model_name = "gpt-5.4-mini"
                if not llm_base_url:
                    llm_base_url = "https://aihubmix.com/v1"
                with llm_helper:
                    tips = """
                            ##### AIHubMix 配置说明
                            - **注册链接**: [点击注册 AIHubMix](https://aihubmix.com/?aff=CEve)
                            - **Base Url**: 预填 https://aihubmix.com/v1
                            - **推荐模型**: 默认 gpt-5.4-mini，也可以填写 AIHubMix 支持的免费模型或其它模型 ID

                            推荐理由：
                            - **模型全**: Claude、GPT、Gemini、Grok、DeepSeek、通义等 700+ 模型一站覆盖
                            - **稳定**: 无限并发，永远在线，集群部署于谷歌云，长期为众多知名应用提供高并发服务
                            - **能力完整**: 文本、图片生成、视频生成、TTS、STT、向量嵌入、Rerank，多模态场景全搞定
                            - **计费透明**: 按量付费，无会员无包月，免费模型可使用
                            """

            if llm_provider == "aimlapi":
                if not llm_model_name:
                    llm_model_name = "openai/gpt-4o-mini"
                if not llm_base_url:
                    llm_base_url = "https://api.aimlapi.com/v1"
                with llm_helper:
                    tips = """
                            ##### AIML API Configuration
                            - **API Key**: create one at https://aimlapi.com/app/keys
                            - **Base Url**: https://api.aimlapi.com/v1
                            - **Model Name**: for example `openai/gpt-4o-mini`, `openai/gpt-4o`, `anthropic/claude-sonnet-4.5`, or `google/gemini-3-flash-preview`
                            """

            if llm_provider == "moonshot":
                if not llm_model_name:
                    llm_model_name = "moonshot-v1-8k"
                with llm_helper:
                    tips = """
                            ##### Moonshot 配置说明
                            - **API Key**: [点击到官网申请](https://platform.moonshot.cn/console/api-keys)
                            - **Base Url**: 固定为 https://api.moonshot.cn/v1
                            - **Model Name**: 比如 moonshot-v1-8k，[点击查看模型列表](https://platform.moonshot.cn/docs/intro#%E6%A8%A1%E5%9E%8B%E5%88%97%E8%A1%A8)
                            """
            if llm_provider == "oneapi":
                if not llm_model_name:
                    llm_model_name = (
                        "claude-3-5-sonnet-20240620"  # 默认模型，可以根据需要调整
                    )
                with llm_helper:
                    tips = """
                        ##### OneAPI 配置说明
                        - **API Key**: 填写您的 OneAPI 密钥
                        - **Base Url**: 填写 OneAPI 的基础 URL
                        - **Model Name**: 填写您要使用的模型名称，例如 claude-3-5-sonnet-20240620
                        """

            if llm_provider == "qwen":
                if not llm_model_name:
                    llm_model_name = "qwen-max"
                with llm_helper:
                    tips = """
                            ##### 通义千问Qwen 配置说明
                            - **API Key**: [点击到官网申请](https://dashscope.console.aliyun.com/apiKey)
                            - **Base Url**: 留空
                            - **Model Name**: 比如 qwen-max，[点击查看模型列表](https://help.aliyun.com/zh/dashscope/developer-reference/model-introduction#3ef6d0bcf91wy)
                            """

            if llm_provider == "g4f":
                if not llm_model_name:
                    llm_model_name = "gpt-3.5-turbo"
                with llm_helper:
                    tips = """
                            ##### gpt4free 配置说明
                            > [GitHub开源项目](https://github.com/xtekky/gpt4free)，可以免费使用GPT模型，但是**稳定性较差**
                            - **API Key**: 随便填写，比如 123
                            - **Base Url**: 留空
                            - **Model Name**: 比如 gpt-3.5-turbo，[点击查看模型列表](https://github.com/xtekky/gpt4free/blob/main/g4f/models.py#L308)
                            """
            if llm_provider == "azure":
                with llm_helper:
                    tips = """
                            ##### Azure 配置说明
                            > [点击查看如何部署模型](https://learn.microsoft.com/zh-cn/azure/ai-services/openai/how-to/create-resource)
                            - **API Key**: [点击到Azure后台创建](https://portal.azure.com/#view/Microsoft_Azure_ProjectOxford/CognitiveServicesHub/~/OpenAI)
                            - **Base Url**: 留空
                            - **Model Name**: 填写你实际的部署名
                            """

            if llm_provider == "gemini":
                if not llm_model_name:
                    llm_model_name = "gemini-1.0-pro"

                with llm_helper:
                    tips = """
                            ##### Gemini 配置说明
                            > 需要VPN开启全局流量模式
                            - **API Key**: [点击到官网申请](https://ai.google.dev/)
                            - **Base Url**: 留空
                            - **Model Name**: 比如 gemini-1.0-pro
                            """

            if llm_provider == "grok":
                if not llm_model_name:
                    llm_model_name = "grok-4.3"
                if not llm_base_url:
                    llm_base_url = "https://api.x.ai/v1"

                with llm_helper:
                    tips = """
                            ##### Grok 配置说明
                            - **API Key**: 填写您的 GrokAPI 密钥
                            - **Base Url**: 填写 GrokAPI 的基础 URL
                            - **Model Name**: 比如 grok-4.3
                            """

            if llm_provider == "groq":
                if not llm_model_name:
                    llm_model_name = "llama-3.3-70b-versatile"
                if not llm_base_url:
                    llm_base_url = "https://api.groq.com/openai/v1"

                with llm_helper:
                    tips = """
                            ##### Groq 配置说明
                            - **API Key**: [点击到官网申请](https://console.groq.com/keys)
                            - **Base Url**: 固定为 https://api.groq.com/openai/v1
                            - **Model Name**: 比如 llama-3.3-70b-versatile
                            """

            if llm_provider == "deepseek":
                if not llm_model_name:
                    llm_model_name = "deepseek-chat"
                if not llm_base_url:
                    llm_base_url = "https://api.deepseek.com"
                with llm_helper:
                    tips = """
                            ##### DeepSeek 配置说明
                            - **API Key**: [点击到官网申请](https://platform.deepseek.com/api_keys)
                            - **Base Url**: 固定为 https://api.deepseek.com
                            - **Model Name**: 固定为 deepseek-chat
                            """

            if llm_provider == "mimo":
                if not llm_model_name:
                    llm_model_name = "mimo-v2.5-pro"
                if not llm_base_url:
                    llm_base_url = "https://api.xiaomimimo.com/v1"
                with llm_helper:
                    tips = """
                            ##### Xiaomi MiMo 配置说明
                            - **API Key**: [点击到官网申请](https://platform.xiaomimimo.com/docs/zh-CN/quick-start/first-api-call)
                            - **Base Url**: 固定为 https://api.xiaomimimo.com/v1
                            - **Model Name**: 默认 mimo-v2.5-pro，也可以按官方文档填写其它可用模型
                            """

            if llm_provider == "modelscope":
                if not llm_model_name:
                    llm_model_name = "Qwen/Qwen3-32B"
                if not llm_base_url:
                    llm_base_url = "https://api-inference.modelscope.cn/v1/"
                with llm_helper:
                    tips = """
                            ##### ModelScope 配置说明
                            - **API Key**: [点击到官网申请](https://modelscope.cn/docs/model-service/API-Inference/intro)
                            - **Base Url**: 固定为 https://api-inference.modelscope.cn/v1/
                            - **Model Name**: 比如 Qwen/Qwen3-32B，[点击查看模型列表](https://modelscope.cn/models?filter=inference_type&page=1)
                            """

            if llm_provider == "ernie":
                with llm_helper:
                    tips = """
                            ##### 百度文心一言 配置说明
                            - **API Key**: [点击到官网申请](https://console.bce.baidu.com/qianfan/ais/console/applicationConsole/application)
                            - **Secret Key**: [点击到官网申请](https://console.bce.baidu.com/qianfan/ais/console/applicationConsole/application)
                            - **Base Url**: 填写 **请求地址** [点击查看文档](https://cloud.baidu.com/doc/WENXINWORKSHOP/s/jlil56u11#%E8%AF%B7%E6%B1%82%E8%AF%B4%E6%98%8E)
                            """

            if llm_provider == "pollinations":
                if not llm_model_name:
                    llm_model_name = "default"
                with llm_helper:
                    tips = """
                            ##### Pollinations AI Configuration
                            - **API Key**: Optional - Leave empty for public access
                            - **Base Url**: Default is https://text.pollinations.ai/openai
                            - **Model Name**: Use 'openai-fast' or specify a model name
                            """

            if llm_provider == "litellm":
                if not llm_model_name:
                    llm_model_name = "openai/gpt-4o-mini"
                with llm_helper:
                    tips = """
                            ##### LiteLLM Configuration
                            > [LiteLLM](https://github.com/BerriAI/litellm) routes to 100+ LLM providers via a unified interface.
                            > Set your provider's API key as an env var: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `AWS_ACCESS_KEY_ID`, etc.
                            - **Model Name**: LiteLLM format — `openai/gpt-4o`, `anthropic/claude-sonnet-4-20250514`, `bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0`, `gemini/gemini-2.5-flash`. See [full provider list](https://docs.litellm.ai/docs/providers)
                            """

            if tips and config.ui["language"] == "zh":
                # AIHubMix 自身就是 OpenAI-compatible 聚合平台；用户主动选择
                # 该 provider 时，再显示 DeepSeek/Moonshot 的通用推荐会造成
                # 信息干扰，也不利于保持合作入口的轻量、清晰。
                if llm_provider != "aihubmix":
                    st.warning(
                        "中国用户建议使用 **DeepSeek** 或 **Moonshot** 作为大模型提供商\n- 国内可直接访问，不需要VPN \n- 注册就送额度，基本够用"
                    )
                st.info(tips)

            st_llm_api_key = st.text_input(
                tr("API Key"), value=llm_api_key, type="password"
            )
            st_llm_base_url = st.text_input(tr("Base Url"), value=llm_base_url)
            st_llm_model_name = ""
            if llm_provider != "ernie":
                if llm_provider == "groq":
                    effective_api_key = st_llm_api_key or llm_api_key
                    effective_base_url = st_llm_base_url or llm_base_url
                    groq_models = get_groq_model_ids(
                        api_key=effective_api_key,
                        base_url=effective_base_url,
                    )

                    if groq_models:
                        selected_index = 0
                        if llm_model_name in groq_models:
                            selected_index = groq_models.index(llm_model_name)

                        st_llm_model_name = st.selectbox(
                            tr("Model Name"),
                            options=groq_models,
                            index=selected_index,
                            key="groq_model_name_select",
                        )
                    else:
                        st_llm_model_name = st.text_input(
                            tr("Model Name"),
                            value=llm_model_name,
                            key="groq_model_name_input",
                        )
                        if effective_api_key:
                            st.caption(
                                "Unable to load Groq model list right now. You can still enter a model name manually — note it won't be validated until generation."
                            )
                        else:
                            st.caption(
                                "Add a Groq API key to load available models automatically."
                            )
                elif llm_provider == "llmcpp":
                    st_llm_model_name = st.text_input(
                        tr("Model Name"),
                        value=llm_model_name,
                        key="llmcpp_model_name_input",
                    )
                    llmcpp_history = list(config.app.get("llmcpp_model_history", []) or [])
                    if llmcpp_history:
                        selected_from_history = st.selectbox(
                            tr("接続履歴から選択") if config.ui.get("language") == "ja" else "Select from history",
                            options=[""] + llmcpp_history,
                            index=0,
                            key="llmcpp_model_history_select",
                        )
                        if selected_from_history:
                            st_llm_model_name = selected_from_history
                else:
                    st_llm_model_name = st.text_input(
                        tr("Model Name"),
                        value=llm_model_name,
                        key=f"{llm_provider}_model_name_input",
                    )
                if st_llm_model_name:
                    config.app[f"{llm_provider}_model_name"] = st_llm_model_name
            else:
                st_llm_model_name = None

            if st_llm_api_key:
                config.app[f"{llm_provider}_api_key"] = st_llm_api_key
            if st_llm_base_url:
                config.app[f"{llm_provider}_base_url"] = st_llm_base_url
            if st_llm_model_name:
                config.app[f"{llm_provider}_model_name"] = st_llm_model_name
            if llm_provider == "ernie":
                st_llm_secret_key = st.text_input(
                    tr("Secret Key"), value=llm_secret_key, type="password"
                )
                config.app[f"{llm_provider}_secret_key"] = st_llm_secret_key

            if llm_provider == "cloudflare":
                st_llm_account_id = st.text_input(
                    tr("Account ID"), value=llm_account_id
                )
                if st_llm_account_id:
                    config.app[f"{llm_provider}_account_id"] = st_llm_account_id

        # 右侧面板 - API 密钥设置
        with right_config_panel:

            def get_keys_from_config(cfg_key):
                api_keys = config.app.get(cfg_key, [])
                if isinstance(api_keys, str):
                    api_keys = [api_keys]
                api_key = ", ".join(api_keys)
                return api_key

            def save_keys_to_config(cfg_key, value):
                value = value.replace(" ", "")
                if value:
                    config.app[cfg_key] = value.split(",")

            st.write(tr("Video Source Settings"))

            pexels_api_key = get_keys_from_config("pexels_api_keys")
            pexels_api_key = st.text_input(
                tr("Pexels API Key"), value=pexels_api_key, type="password"
            )
            save_keys_to_config("pexels_api_keys", pexels_api_key)

            pixabay_api_key = get_keys_from_config("pixabay_api_keys")
            pixabay_api_key = st.text_input(
                tr("Pixabay API Key"), value=pixabay_api_key, type="password"
            )
            save_keys_to_config("pixabay_api_keys", pixabay_api_key)

            coverr_api_key = get_keys_from_config("coverr_api_keys")
            coverr_api_key = st.text_input(
                tr("Coverr API Key"), value=coverr_api_key, type="password"
            )
            save_keys_to_config("coverr_api_keys", coverr_api_key)

llm_provider = config.app.get("llm_provider", "").lower()
panel = st.columns(3)
left_panel = panel[0]
middle_panel = panel[1]
right_panel = panel[2]

params = VideoParams(video_subject="")
params.match_materials_to_script = bool(
    st.session_state.get("match_materials_to_script", False)
)
uploaded_files = []
uploaded_audio_file = None

with left_panel:
    with st.container(border=True):
        st.write(tr("Video Script Settings"))
        params.video_subject = st.text_input(
            tr("Video Subject"),
            key="video_subject",
        ).strip()

        video_languages = [
            (tr("Auto Detect"), ""),
        ]
        for code in support_locales:
            video_languages.append((code, code))

        selected_index = st.selectbox(
            tr("Script Language"),
            index=0,
            options=range(
                len(video_languages)
            ),  # Use the index as the internal option value
            format_func=lambda x: video_languages[x][
                0
            ],  # The label is displayed to the user
        )
        params.video_language = video_languages[selected_index][1]

        with st.expander(tr("Advanced Script Settings"), expanded=False):
            params.paragraph_number = st.slider(
                tr("Script Paragraph Number"),
                min_value=llm.MIN_SCRIPT_PARAGRAPH_NUMBER,
                max_value=llm.MAX_SCRIPT_PARAGRAPH_NUMBER,
                value=config.app.get("paragraph_number", 1),
                key="paragraph_number_input",
            )
            params.video_script_prompt = st.text_area(
                tr("Custom Script Requirements"),
                height=100,
                value=config.app.get("video_script_prompt", ""),
                max_chars=llm.MAX_SCRIPT_PROMPT_LENGTH,
                placeholder=tr("Custom Script Requirements Placeholder"),
                key="video_script_prompt",
            ).strip()

            use_custom_system_prompt = st.checkbox(
                tr("Use Custom System Prompt"),
                help=tr("Use Custom System Prompt Help"),
                key="use_custom_system_prompt",
            )

            if use_custom_system_prompt:
                custom_system_prompt = st.text_area(
                    tr("Custom System Prompt"),
                    height=240,
                    value=config.app.get("custom_system_prompt", ""),
                    max_chars=llm.MAX_SCRIPT_SYSTEM_PROMPT_LENGTH,
                    key="custom_system_prompt",
                ).strip()
                params.custom_system_prompt = custom_system_prompt
            else:
                params.custom_system_prompt = ""

        # Web検索設定
        with st.expander(tr("Web Search Settings"), expanded=False):
            params.web_search_enabled = st.checkbox(
                tr("Enable Web Search"),
                help=tr("Enable Web Search Help"),
                key="web_search_enabled",
            )

            if params.web_search_enabled:
                params.web_search_max_steps = st.slider(
                    tr("Web Search Max Steps"),
                    min_value=1,
                    max_value=10,
                    value=int(config.app.get("web_search_max_steps", 3)),
                    key="web_search_max_steps",
                )

                # Search instruction presets
                _search_presets: dict = config.app.get("web_search_presets", {}) or {}
                preset_names = list(_search_presets.keys())
                if preset_names:
                    preset_cols = st.columns([2, 1])
                    with preset_cols[0]:
                        selected_preset = st.selectbox(
                            tr("Web Search Preset Instructions"),
                            options=[""] + preset_names,
                            index=0,
                            label_visibility="collapsed",
                        )
                    with preset_cols[1]:
                        if st.button(tr("Load Search Preset"), key="load_search_preset"):
                            if selected_preset:
                                st.session_state["web_search_prompt"] = _search_presets[selected_preset]
                                st.rerun()

                params.web_search_prompt = st.text_area(
                    tr("Web Search Instructions"),
                    height=80,
                    value=st.session_state.get("web_search_prompt", ""),
                    max_chars=2000,
                    placeholder=tr("Web Search Instructions Placeholder"),
                    key="web_search_prompt",
                ).strip()
            else:
                params.web_search_prompt = ""
                params.web_search_max_steps = int(config.app.get("web_search_max_steps", 3))

        # 動画タイトル入力
        saved_video_title = st.session_state.get("video_title", "")
        params.video_title = st.text_input(
            tr("Video Title"),
            value=saved_video_title,
        ).strip()
        st.session_state["video_title"] = params.video_title

        if st.button(
            tr("Generate Video Script and Keywords"), key="auto_generate_script"
        ):
            with st.spinner(tr("Generating Video Script and Keywords")):
                _research_context = ""
                if getattr(params, "web_search_enabled", False):
                    _status_placeholder = st.empty()
                    _status_placeholder.info(tr("Researching"))

                    def _ui_progress(step, total, message):
                        _status_placeholder.info(
                            tr("Research Step").format(step=step, total=total, message=message)
                        )

                    _result = research_agent.run_research(
                        subject=params.video_subject,
                        instructions=getattr(params, "web_search_prompt", ""),
                        max_steps=getattr(params, "web_search_max_steps", 3),
                        progress_callback=_ui_progress,
                    )
                    _status_placeholder.empty()
                    if _result.success:
                        _research_context = _result.to_context_string()

                script = llm.generate_script(
                    video_subject=params.video_subject,
                    language=params.video_language,
                    paragraph_number=params.paragraph_number,
                    video_script_prompt=params.video_script_prompt,
                    custom_system_prompt=params.custom_system_prompt,
                    research_context=_research_context,
                )
                if "Error: " in script:
                    st.error(tr(script))
                else:
                    # タイトルの自動生成 (空の場合のみ)
                    if not params.video_title:
                        title = llm.generate_title(script)
                        st.session_state["video_title"] = title
                    
                    # 色強調が有効な場合は自動ハイライト
                    if getattr(params, "text_color_highlight_enabled", False):
                        script = llm.highlight_script_with_llm(script)
                        current_title = st.session_state.get("video_title", "")
                        if current_title:
                            st.session_state["video_title"] = llm.highlight_title_with_llm(current_title)
                        
                    terms = llm.generate_terms(
                        params.video_subject,
                        script,
                        amount=8 if params.match_materials_to_script else 5,
                        match_script_order=params.match_materials_to_script,
                    )
                    
                    if "Error: " in terms:
                        st.error(tr(terms))
                    else:
                        st.session_state["video_script"] = script
                        st.session_state["video_terms"] = ", ".join(terms)
                        st.rerun()

        params.video_script = st.text_area(
            tr("Video Script"), value=st.session_state["video_script"], height=280
        )
        
        # タイトル・ハイライトAI生成ボタン
        title_btn_cols = st.columns([0.5, 0.5])
        with title_btn_cols[0]:
            if st.button(tr("Generate Title from Script (AI)"), key="auto_generate_title"):
                if not params.video_script:
                    st.error(tr("Please Enter or Generate the Video Script First"))
                else:
                    with st.spinner(tr("Generating Video Title")):
                        title = llm.generate_title(params.video_script)
                        st.session_state["video_title"] = title
                        st.rerun()
        with title_btn_cols[1]:
            if st.button(tr("Highlight Script with AI Tags"), key="auto_highlight_script"):
                if not params.video_script:
                    st.error(tr("Please Enter or Generate the Video Script First"))
                else:
                    with st.spinner(tr("Highlighting Video Script and Title")):
                        highlighted = llm.highlight_script_with_llm(params.video_script)
                        st.session_state["video_script"] = highlighted
                        current_title = st.session_state.get("video_title", "")
                        if current_title:
                            st.session_state["video_title"] = llm.highlight_title_with_llm(current_title)
                        st.rerun()

        if st.button(tr("Generate Video Keywords"), key="auto_generate_terms"):
            if not params.video_script:
                st.error(tr("Please Enter the Video Subject"))
                st.stop()

            with st.spinner(tr("Generating Video Keywords")):
                terms = llm.generate_terms(
                    params.video_subject,
                    params.video_script,
                    amount=8 if params.match_materials_to_script else 5,
                    match_script_order=params.match_materials_to_script,
                )
                if "Error: " in terms:
                    st.error(tr(terms))
                else:
                    st.session_state["video_terms"] = ", ".join(terms)

        params.video_terms = st.text_area(
            tr("Video Keywords"), value=st.session_state["video_terms"]
        )

with middle_panel:
    with st.container(border=True):
        st.write(tr("Video Settings"))
        video_concat_modes = [
            (tr("Sequential"), "sequential"),
            (tr("Random"), "random"),
        ]
        video_sources = [
            (tr("Pexels"), "pexels"),
            (tr("Pixabay"), "pixabay"),
            (tr("Coverr"), "coverr"),
            (tr("Local file"), "local"),
            (tr("TikTok"), "douyin"),
            (tr("Bilibili"), "bilibili"),
            (tr("Xiaohongshu"), "xiaohongshu"),
        ]

        saved_video_source_name = config.app.get("video_source", "pexels")
        saved_video_source_index = [v[1] for v in video_sources].index(
            saved_video_source_name
        )

        selected_index = st.selectbox(
            tr("Video Source"),
            options=range(len(video_sources)),
            format_func=lambda x: video_sources[x][0],
            index=saved_video_source_index,
        )
        params.video_source = video_sources[selected_index][1]
        config.app["video_source"] = params.video_source

        if params.video_source == "local":
            # Streamlit 的文件类型校验对扩展名大小写敏感，这里同时放行大小写两种形式。
            local_file_types = ["mp4", "mov", "avi", "flv", "mkv", "jpg", "jpeg", "png"]
            uploaded_files = st.file_uploader(
                tr("Upload Local Files"),
                type=local_file_types + [file_type.upper() for file_type in local_file_types],
                accept_multiple_files=True,
            )

        saved_concat_mode = config.app.get("video_concat_mode", "random")
        saved_concat_index = 0 if saved_concat_mode == "sequential" else 1
        selected_index = st.selectbox(
            tr("Video Concat Mode"),
            index=saved_concat_index,
            options=range(len(video_concat_modes)),
            format_func=lambda x: video_concat_modes[x][0],
            key="video_concat_mode",
        )
        params.video_concat_mode = VideoConcatMode(
            video_concat_modes[selected_index][1]
        )
        config.app["video_concat_mode"] = params.video_concat_mode.value

        # 视频转场模式
        video_transition_modes = [
            (tr("None"), VideoTransitionMode.none.value),
            (tr("Shuffle"), VideoTransitionMode.shuffle.value),
            (tr("FadeIn"), VideoTransitionMode.fade_in.value),
            (tr("FadeOut"), VideoTransitionMode.fade_out.value),
            (tr("SlideIn"), VideoTransitionMode.slide_in.value),
            (tr("SlideOut"), VideoTransitionMode.slide_out.value),
        ]
        saved_transition_mode = config.app.get("video_transition_mode", None)
        saved_transition_index = 0
        for i, (_, mode_val) in enumerate(video_transition_modes):
            if mode_val == saved_transition_mode:
                saved_transition_index = i
                break
        selected_index = st.selectbox(
            tr("Video Transition Mode"),
            options=range(len(video_transition_modes)),
            format_func=lambda x: video_transition_modes[x][0],
            index=saved_transition_index,
            key="video_transition_mode",
        )
        params.video_transition_mode = VideoTransitionMode(
            video_transition_modes[selected_index][1]
        )
        config.app["video_transition_mode"] = params.video_transition_mode.value if params.video_transition_mode else None

        video_aspect_ratios = [
            (tr("Portrait"), VideoAspect.portrait.value),
            (tr("Landscape"), VideoAspect.landscape.value),
        ]
        # Coverr 库 99% 是 16:9 横屏,默认竖屏会让画面被大量黑边包围。
        # プリセットアスペクトを優先し、無ければデフォルトインデックスを計算
        saved_aspect = config.app.get("video_aspect", VideoAspect.portrait.value)
        aspect_values = [v[1] for v in video_aspect_ratios]
        if saved_aspect in aspect_values:
            default_aspect_index = aspect_values.index(saved_aspect)
        else:
            default_aspect_index = 1 if params.video_source == "coverr" else 0
        selected_index = st.selectbox(
            tr("Video Ratio"),
            options=range(len(video_aspect_ratios)),
            format_func=lambda x: video_aspect_ratios[x][0],
            index=default_aspect_index,
            key="video_aspect",
        )
        params.video_aspect = VideoAspect(video_aspect_ratios[selected_index][1])
        config.app["video_aspect"] = params.video_aspect.value

        # Video clip fit mode: contain or cover
        video_clip_fits = [
            (tr("Contain"), "contain"),
            (tr("Cover"), "cover"),
        ]
        saved_fit_mode = config.app.get("video_clip_fit", "contain").lower()
        saved_fit_index = 1 if saved_fit_mode == "cover" else 0
        selected_index = st.selectbox(
            tr("Fit Mode"),
            options=range(len(video_clip_fits)),
            format_func=lambda x: video_clip_fits[x][0],
            index=saved_fit_index,
            key=f"video_clip_fit_for_{params.video_source}",
        )
        params.video_clip_fit = video_clip_fits[selected_index][1]
        config.app["video_clip_fit"] = params.video_clip_fit

        saved_duration = config.app.get("video_clip_duration", 5)
        options = [2, 3, 4, 5, 6, 7, 8, 9, 10]
        saved_duration_index = options.index(saved_duration) if saved_duration in options else 1
        params.video_clip_duration = st.selectbox(
            tr("Clip Duration"),
            options=options,
            index=saved_duration_index,
            key="video_clip_duration",
        )
        config.app["video_clip_duration"] = params.video_clip_duration

        saved_count = config.app.get("video_count", 1)
        count_options = [1, 2, 3, 4, 5]
        saved_count_index = count_options.index(saved_count) if saved_count in count_options else 0
        params.video_count = st.selectbox(
            tr("Number of Videos Generated Simultaneously"),
            options=count_options,
            index=saved_count_index,
            key="video_count",
        )
        config.app["video_count"] = params.video_count

        with st.expander(tr("Advanced Video Settings"), expanded=False):
            # 默认关闭，避免影响老用户的随机素材体验。开启后只改变关键词和素材
            # 下载/拼接顺序，用于改善画面主题早于或晚于旁白的问题。
            params.match_materials_to_script = st.checkbox(
                tr("Match Materials to Script Order"),
                help=tr("Match Materials to Script Order Help"),
                key="match_materials_to_script",
            )
            config.app["match_materials_to_script"] = params.match_materials_to_script

            video_codec_options = [
                ("libx264 (CPU)", "libx264"),
                ("NVIDIA NVENC (h264_nvenc)", "h264_nvenc"),
                ("AMD AMF (h264_amf)", "h264_amf"),
                ("Intel QSV (h264_qsv)", "h264_qsv"),
                ("Windows MediaFoundation (h264_mf)", "h264_mf"),
                ("macOS VideoToolbox (h264_videotoolbox)", "h264_videotoolbox"),
            ]
            saved_video_codec = config.app.get("video_codec", "libx264")
            saved_video_codec_values = [item[1] for item in video_codec_options]
            if saved_video_codec not in saved_video_codec_values:
                saved_video_codec = "libx264"
            selected_codec_index = saved_video_codec_values.index(saved_video_codec)
            selected_codec_index = st.selectbox(
                tr("Video Encoder"),
                options=range(len(video_codec_options)),
                index=selected_codec_index,
                format_func=lambda x: video_codec_options[x][0],
                help=tr("Video Encoder Help"),
            )
            config.app["video_codec"] = video_codec_options[selected_codec_index][1]

            # 動画上下マージン
            st.write("---")
            saved_video_margin_ratio = config.ui.get("video_margin_ratio", 0.0)
            params.video_margin_ratio = st.slider(
                tr("Video Margin Ratio (Top/Bottom)"),
                0.0, 0.45, saved_video_margin_ratio, step=0.01,
                help=tr("Video Margin Ratio Help"),
            )
            config.ui["video_margin_ratio"] = params.video_margin_ratio

            # 背景動画（2レイヤー合成）設定
            st.write("---")
            st.write(tr("2-Layer Background Video Settings"))
            bg_video_options = [
                (tr("None"), "none"),
                (tr("Random Background Video"), "random"),
                (tr("Source Background Video"), "source"),
                (tr("Custom Background Video File"), "custom"),
            ]
            saved_bg_video_type = config.ui.get("bg_video_type", "none")
            saved_bg_index = 0
            for i, (_, type_val) in enumerate(bg_video_options):
                if type_val == saved_bg_video_type:
                    saved_bg_index = i
                    break
                    
            selected_bg_index = st.selectbox(
                tr("Background Video Type"),
                options=range(len(bg_video_options)),
                index=saved_bg_index,
                key="bg_video_type_selectbox",
                format_func=lambda x: bg_video_options[x][0],
            )
            params.bg_video_type = bg_video_options[selected_bg_index][1]
            config.ui["bg_video_type"] = params.bg_video_type
            
            if params.bg_video_type == "custom":
                bg_dir = utils.resource_dir("bg_videos")
                if not os.path.exists(bg_dir):
                    os.makedirs(bg_dir, exist_ok=True)
                bg_files = []
                for ext in ["*.mp4", "*.mkv", "*.mov", "*.avi"]:
                    bg_files.extend(glob.glob(os.path.join(bg_dir, ext)))
                bg_filenames = sorted([os.path.basename(f) for f in bg_files])
                
                if not bg_filenames:
                    st.warning(tr("No background videos found in resource/bg_videos"))
                    params.bg_video_file = ""
                else:
                    saved_bg_video_file = config.ui.get("bg_video_file", "")
                    try:
                        saved_bg_file_index = bg_filenames.index(saved_bg_video_file)
                    except ValueError:
                        saved_bg_file_index = 0
                        
                    selected_bg_file = st.selectbox(
                        tr("Custom Background Video File"),
                        options=bg_filenames,
                        index=saved_bg_file_index,
                        key="custom_bg_video_file_selectbox"
                    )
                    params.bg_video_file = selected_bg_file
                    config.ui["bg_video_file"] = params.bg_video_file
    with st.container(border=True):
        st.write(tr("Audio Settings"))

        # 添加TTS服务器选择下拉框
        tts_servers = [
            (voice.NO_VOICE_NAME, tr("No Voice")),
            ("azure-tts-v1", "Azure TTS V1"),
            ("azure-tts-v2", "Azure TTS V2"),
            ("siliconflow", "SiliconFlow TTS"),
            ("gemini-tts", "Google Gemini TTS"),
            ("mimo-tts", "Xiaomi MiMo TTS"),
            ("voicevox", "VOICEVOX"),
            ("aivisspeech", "AivisSpeech"),
        ]

        # 获取保存的TTS服务器，默认为v1
        saved_tts_server = config.ui.get("tts_server", "azure-tts-v1")
        saved_tts_server_index = 0
        for i, (server_value, _) in enumerate(tts_servers):
            if server_value == saved_tts_server:
                saved_tts_server_index = i
                break

        selected_tts_server_index = st.selectbox(
            tr("TTS Servers"),
            options=range(len(tts_servers)),
            format_func=lambda x: tts_servers[x][1],
            index=saved_tts_server_index,
        )

        selected_tts_server = tts_servers[selected_tts_server_index][0]
        config.ui["tts_server"] = selected_tts_server

        # 根据选择的TTS服务器获取声音列表
        filtered_voices = []

        if selected_tts_server == voice.NO_VOICE_NAME:
            # 无配音是显式模式，只提供一个稳定 sentinel。这样普通 TTS 的空配置
            # 不会被误判为静音，后端也能继续通过同一条音频/字幕流程生成视频。
            filtered_voices = [voice.NO_VOICE_NAME]
        elif selected_tts_server == "siliconflow":
            # 获取硅基流动的声音列表
            filtered_voices = voice.get_siliconflow_voices()
        elif selected_tts_server == "gemini-tts":
            # 获取Gemini TTS的声音列表
            filtered_voices = voice.get_gemini_voices()
        elif selected_tts_server == "mimo-tts":
            # 获取 Xiaomi MiMo TTS 的预置音色列表
            filtered_voices = voice.get_mimo_voices()
        elif selected_tts_server in ["voicevox", "aivisspeech"]:
            # VOICEVOX または AivisSpeech の話者リストを取得
            filtered_voices = voice.get_voicevox_voices(tts_type=selected_tts_server)
        else:
            # 获取Azure的声音列表
            all_voices = voice.get_all_azure_voices(filter_locals=None)

            # 根据选择的TTS服务器筛选声音
            for v in all_voices:
                if selected_tts_server == "azure-tts-v2":
                    # V2版本的声音名称中包含"v2"
                    if "V2" in v:
                        filtered_voices.append(v)
                else:
                    # V1版本的声音名称中不包含"v2"
                    if "V2" not in v:
                        filtered_voices.append(v)

        if selected_tts_server == voice.NO_VOICE_NAME:
            friendly_names = {voice.NO_VOICE_NAME: tr("No Voice")}
        elif selected_tts_server in ["voicevox", "aivisspeech"]:
            friendly_names = {}
            for v in filtered_voices:
                parts = v.split(":")
                if len(parts) >= 3:
                    full_name = parts[2]
                    name_style = full_name.split("-")
                    if len(name_style) >= 2:
                        friendly_names[v] = f"{name_style[0]} ({name_style[1]})"
                    else:
                        friendly_names[v] = full_name
                else:
                    friendly_names[v] = v
        else:
            friendly_names = {
                v: v.replace("Female", tr("Female"))
                .replace("Male", tr("Male"))
                .replace("Neural", "")
                for v in filtered_voices
            }

        saved_voice_name = config.ui.get("voice_name", "")
        saved_voice_name_index = 0

        # 检查保存的声音是否在当前筛选的声音列表中
        if saved_voice_name in friendly_names:
            saved_voice_name_index = list(friendly_names.keys()).index(saved_voice_name)
        else:
            # 如果不在，则根据当前UI语言选择一个默认声音
            for i, v in enumerate(filtered_voices):
                if v.lower().startswith(st.session_state["ui_language"].lower()):
                    saved_voice_name_index = i
                    break

        # 如果没有找到匹配的声音，使用第一个声音
        if saved_voice_name_index >= len(friendly_names) and friendly_names:
            saved_voice_name_index = 0

        # 确保有声音可选
        if friendly_names:
            selected_friendly_name = st.selectbox(
                tr("Speech Synthesis"),
                options=list(friendly_names.values()),
                index=min(saved_voice_name_index, len(friendly_names) - 1)
                if friendly_names
                else 0,
            )

            voice_name = list(friendly_names.keys())[
                list(friendly_names.values()).index(selected_friendly_name)
            ]
            params.voice_name = voice_name
            config.ui["voice_name"] = voice_name
        else:
            # 如果没有声音可选，显示提示信息
            st.warning(
                tr(
                    "No voices available for the selected TTS server. Please select another server."
                )
            )
            params.voice_name = ""
            config.ui["voice_name"] = ""

        # 无配音模式会生成静音占位音频，不展示试听按钮，避免用户误以为需要测试声音。
        if (
            friendly_names
            and selected_tts_server != voice.NO_VOICE_NAME
            and st.button(tr("Play Voice"))
        ):
            play_content = params.video_subject
            if not play_content:
                play_content = params.video_script
            if not play_content:
                play_content = tr("Voice Example")
            with st.spinner(tr("Synthesizing Voice")):
                temp_dir = utils.storage_dir("temp", create=True)
                audio_file = os.path.join(temp_dir, f"tmp-voice-{str(uuid4())}.mp3")
                sub_maker = voice.tts(
                    text=play_content,
                    voice_name=voice_name,
                    voice_rate=params.voice_rate,
                    voice_file=audio_file,
                    voice_volume=params.voice_volume,
                )
                # if the voice file generation failed, try again with a default content.
                if not sub_maker:
                    play_content = "This is a example voice. if you hear this, the voice synthesis failed with the original content."
                    sub_maker = voice.tts(
                        text=play_content,
                        voice_name=voice_name,
                        voice_rate=params.voice_rate,
                        voice_file=audio_file,
                        voice_volume=params.voice_volume,
                    )

                if sub_maker and os.path.exists(audio_file):
                    st.audio(audio_file, format="audio/mp3")
                    if os.path.exists(audio_file):
                        os.remove(audio_file)

        # 当选择V2版本或者声音是V2声音时，显示服务区域和API key输入框
        if selected_tts_server == "azure-tts-v2" or (
            voice_name and voice.is_azure_v2_voice(voice_name)
        ):
            saved_azure_speech_region = config.azure.get("speech_region", "")
            saved_azure_speech_key = config.azure.get("speech_key", "")
            azure_speech_region = st.text_input(
                tr("Speech Region"),
                value=saved_azure_speech_region,
                key="azure_speech_region_input",
            )
            azure_speech_key = st.text_input(
                tr("Speech Key"),
                value=saved_azure_speech_key,
                type="password",
                key="azure_speech_key_input",
            )
            config.azure["speech_region"] = azure_speech_region
            config.azure["speech_key"] = azure_speech_key

        # 当选择硅基流动时，显示API key输入框和说明信息
        if selected_tts_server == "siliconflow" or (
            voice_name and voice.is_siliconflow_voice(voice_name)
        ):
            saved_siliconflow_api_key = config.siliconflow.get("api_key", "")

            siliconflow_api_key = st.text_input(
                tr("SiliconFlow API Key"),
                value=saved_siliconflow_api_key,
                type="password",
                key="siliconflow_api_key_input",
            )

            # 显示硅基流动的说明信息
            st.info(
                tr("SiliconFlow TTS Settings")
                + ":\n"
                + "- "
                + tr("Speed: Range [0.25, 4.0], default is 1.0")
                + "\n"
                + "- "
                + tr("Volume: Uses Speech Volume setting, default 1.0 maps to gain 0")
            )

            config.siliconflow["api_key"] = siliconflow_api_key

        # 当选择 Xiaomi MiMo TTS 时，复用 MiMo LLM provider 的 API Key。
        # 这样用户如果同时使用 MiMo 生成文案和语音，只需要维护一份密钥。
        if selected_tts_server == "mimo-tts" or (
            voice_name and voice.is_mimo_voice(voice_name)
        ):
            saved_mimo_api_key = config.app.get("mimo_api_key", "")

            mimo_api_key = st.text_input(
                tr("MiMo API Key"),
                value=saved_mimo_api_key,
                type="password",
                key="mimo_tts_api_key_input",
            )

            st.info(
                tr("MiMo TTS Settings")
                + ":\n"
                + "- "
                + tr("Uses Xiaomi MiMo V2.5 TTS preset voices")
                + "\n"
                + "- "
                + tr("Speed and volume are currently handled by the provider defaults")
            )

            config.app["mimo_api_key"] = mimo_api_key

        saved_voice_volume = config.app.get("voice_volume", 1.0)
        voice_volume_options = [0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 4.0, 5.0]
        saved_volume_index = voice_volume_options.index(saved_voice_volume) if saved_voice_volume in voice_volume_options else 2
        params.voice_volume = st.selectbox(
            tr("Speech Volume"),
            options=voice_volume_options,
            index=saved_volume_index,
            key="voice_volume",
        )
        config.app["voice_volume"] = params.voice_volume

        saved_voice_rate = config.app.get("voice_rate", 1.0)
        voice_rate_options = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.8, 2.0]
        saved_rate_index = voice_rate_options.index(saved_voice_rate) if saved_voice_rate in voice_rate_options else 2
        params.voice_rate = st.selectbox(
            tr("Speech Rate"),
            options=voice_rate_options,
            index=saved_rate_index,
            key="voice_rate",
        )
        config.app["voice_rate"] = params.voice_rate

        custom_audio_file_types = ["mp3", "wav", "m4a", "aac", "flac", "ogg"]
        uploaded_audio_file = st.file_uploader(
            tr("Custom Audio File"),
            type=custom_audio_file_types
            + [file_type.upper() for file_type in custom_audio_file_types],
            accept_multiple_files=False,
            key="custom_audio_file_uploader",
        )
        if uploaded_audio_file:
            st.audio(uploaded_audio_file, format="audio/mp3")
            st.info(
                tr(
                    "Custom audio will be used directly. TTS synthesis will be skipped for this task."
                )
            )

        bgm_options = [
            (tr("No Background Music"), ""),
            (tr("Random Background Music"), "random"),
            (tr("Custom Background Music"), "custom"),
        ]
        saved_bgm_type = config.app.get("bgm_type", "random")
        bgm_values = [b[1] for b in bgm_options]
        saved_bgm_index = bgm_values.index(saved_bgm_type) if saved_bgm_type in bgm_values else 1
        selected_index = st.selectbox(
            tr("Background Music"),
            index=saved_bgm_index,
            options=range(len(bgm_options)),
            format_func=lambda x: bgm_options[x][0],
            key="bgm_type",
        )
        # Get the selected background music type
        params.bgm_type = bgm_options[selected_index][1]
        config.app["bgm_type"] = params.bgm_type

        # Show or hide components based on the selection
        if params.bgm_type == "custom":
            custom_bgm_file = st.text_input(
                tr("Custom Background Music File"), key="custom_bgm_file_input"
            )
            if custom_bgm_file:
                # 这里不直接用 os.path.exists 判断，因为用户常见输入是
                # output000.mp3，这个文件名需要由服务层映射 to resource/songs
                # 目录后再校验。服务层会统一限制目录和文件类型，避免任意路径读取。
                params.bgm_file = custom_bgm_file.strip()
                # st.write(f":red[已选择自定义背景音乐]：**{custom_bgm_file}**")
        saved_bgm_volume = config.app.get("bgm_volume", 0.2)
        bgm_vol_options = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        saved_bgm_vol_index = bgm_vol_options.index(saved_bgm_volume) if saved_bgm_volume in bgm_vol_options else 2
        params.bgm_volume = st.selectbox(
            tr("Background Music Volume"),
            options=bgm_vol_options,
            index=saved_bgm_vol_index,
            key="bgm_volume",
        )
        config.app["bgm_volume"] = params.bgm_volume

with right_panel:
    with st.container(border=True):
        st.write(tr("Subtitle Settings"))
        saved_subtitle_enabled = config.app.get("subtitle_enabled", True)
        params.subtitle_enabled = st.checkbox(
            tr("Enable Subtitles"),
            value=saved_subtitle_enabled,
            key="subtitle_enabled",
        )
        config.app["subtitle_enabled"] = params.subtitle_enabled
        font_names = get_all_fonts()
        saved_font_name = config.ui.get("font_name", "MicrosoftYaHeiBold.ttc")
        saved_font_name_index = 0
        if saved_font_name in font_names:
            saved_font_name_index = font_names.index(saved_font_name)
        params.font_name = st.selectbox(
            tr("Font"), font_names, index=saved_font_name_index
        )
        config.ui["font_name"] = params.font_name

        subtitle_positions = [
            (tr("Top"), "top"),
            (tr("Center"), "center"),
            (tr("Bottom"), "bottom"),
            (tr("Custom"), "custom"),
        ]
        saved_subtitle_position = config.ui.get("subtitle_position", "bottom")
        saved_position_index = 2
        for i, (_, pos_value) in enumerate(subtitle_positions):
            if pos_value == saved_subtitle_position:
                saved_position_index = i
                break
        selected_index = st.selectbox(
            tr("Position"),
            index=saved_position_index,
            options=range(len(subtitle_positions)),
            format_func=lambda x: subtitle_positions[x][0],
        )
        params.subtitle_position = subtitle_positions[selected_index][1]
        config.ui["subtitle_position"] = params.subtitle_position

        if params.subtitle_position == "custom":
            saved_custom_position = config.ui.get("custom_position", 70.0)
            custom_position = st.text_input(
                tr("Custom Position (% from top)"),
                value=str(saved_custom_position),
                key="custom_position_input",
            )
            try:
                params.custom_position = float(custom_position)
                if params.custom_position < 0 or params.custom_position > 100:
                    st.error(tr("Please enter a value between 0 and 100"))
                else:
                    config.ui["custom_position"] = params.custom_position
            except ValueError:
                st.error(tr("Please enter a valid number"))

        font_cols = st.columns([0.3, 0.7])
        with font_cols[0]:
            saved_text_fore_color = config.ui.get("text_fore_color", "#FFFFFF")
            params.text_fore_color = st.color_picker(
                tr("Font Color"), saved_text_fore_color
            )
            config.ui["text_fore_color"] = params.text_fore_color

        with font_cols[1]:
            saved_font_size = config.ui.get("font_size", 60)
            params.font_size = st.slider(tr("Font Size"), 30, 100, saved_font_size)
            config.ui["font_size"] = params.font_size

        stroke_cols = st.columns([0.3, 0.7])
        saved_stroke_color = config.ui.get("stroke_color", "#000000")
        saved_stroke_width = config.ui.get("stroke_width", 1.5)
        with stroke_cols[0]:
            params.stroke_color = st.color_picker(
                tr("Stroke Color"),
                saved_stroke_color,
                key="stroke_color",
            )
            config.ui["stroke_color"] = params.stroke_color
        with stroke_cols[1]:
            params.stroke_width = st.slider(
                tr("Stroke Width"),
                0.0,
                10.0,
                saved_stroke_width,
                key="stroke_width",
            )
            config.ui["stroke_width"] = params.stroke_width

        subtitle_bg_cols = st.columns([0.4, 0.6])
        saved_subtitle_background_enabled = config.ui.get(
            "subtitle_background_enabled", True
        )
        with subtitle_bg_cols[0]:
            subtitle_background_enabled = st.checkbox(
                tr("Enable Subtitle Background"),
                value=saved_subtitle_background_enabled,
            )
        config.ui["subtitle_background_enabled"] = subtitle_background_enabled
        if subtitle_background_enabled:
            with subtitle_bg_cols[1]:
                saved_subtitle_background_color = config.ui.get(
                    "subtitle_background_color", "#000000"
                )
                params.text_background_color = st.color_picker(
                    tr("Subtitle Background Color"),
                    saved_subtitle_background_color,
                )
                config.ui["subtitle_background_color"] = params.text_background_color
        else:
            params.text_background_color = False

        saved_rounded_subtitle_background = config.ui.get(
            "rounded_subtitle_background", False
        )
        # 背景关闭时，圆角背景没有可渲染的底色。这里禁用控件并保留原配置，
        # 用户下次重新开启字幕背景后，可以继续使用之前保存的圆角偏好。
        params.rounded_subtitle_background = st.checkbox(
            tr("Rounded Subtitle Background"),
            value=(
                saved_rounded_subtitle_background
                if subtitle_background_enabled
                else False
            ),
            help=tr("Rounded Subtitle Background Help"),
            disabled=not subtitle_background_enabled,
        )
        if subtitle_background_enabled:
            config.ui["rounded_subtitle_background"] = (
                params.rounded_subtitle_background
            )

        # 文字セーフマージン設定
        st.write("---")
        st.write(tr("Text Margin Settings"))
        saved_text_margin_x = config.ui.get("text_margin_x", 0.05)
        params.text_margin_x = st.slider(tr("Text Horizontal Margin Ratio"), 0.0, 0.45, saved_text_margin_x, step=0.01)
        config.ui["text_margin_x"] = params.text_margin_x

        saved_text_margin_y = config.ui.get("text_margin_y", 0.1)
        params.text_margin_y = st.slider(tr("Text Vertical Margin Ratio"), 0.0, 0.45, saved_text_margin_y, step=0.01)
        config.ui["text_margin_y"] = params.text_margin_y

        # 字幕タイムオフセット設定 (音声ズレ補正)
        st.write("---")
        saved_subtitle_offset = config.app.get("subtitle_offset", 0.0)
        params.subtitle_offset = st.number_input(
            tr("Subtitle Offset (seconds)"),
            min_value=-1.0,
            max_value=3.0,
            value=float(saved_subtitle_offset),
            step=0.05,
            format="%.2f",
            help=tr("Subtitle Offset Help"),
            key="subtitle_offset",
        )
        config.app["subtitle_offset"] = params.subtitle_offset

        # 色強調設定
        st.write("---")
        saved_text_color_highlight_enabled = config.ui.get("text_color_highlight_enabled", False)
        params.text_color_highlight_enabled = st.checkbox(
            tr("Enable Text Color Highlight"),
            value=saved_text_color_highlight_enabled
        )
        config.ui["text_color_highlight_enabled"] = params.text_color_highlight_enabled

        if params.text_color_highlight_enabled:
            st.info(tr("Highlight Tag Info"))
            
            # 強調色1
            st.write(tr("Highlight Color 1"))
            saved_color1_fore = config.ui.get("color1_fore", "#FF3B30")
            saved_color1_stroke = config.ui.get("color1_stroke", "#000000")
            saved_color1_stroke_width = config.ui.get("color1_stroke_width", 1.5)
            c1_cols = st.columns([0.3, 0.3, 0.4])
            with c1_cols[0]:
                params.color1_fore = st.color_picker(tr("Color 1 Text"), saved_color1_fore, key="color1_fore_picker")
                config.ui["color1_fore"] = params.color1_fore
            with c1_cols[1]:
                params.color1_stroke = st.color_picker(tr("Color 1 Border"), saved_color1_stroke, key="color1_stroke_picker")
                config.ui["color1_stroke"] = params.color1_stroke
            with c1_cols[2]:
                params.color1_stroke_width = st.slider(tr("Color 1 Border Width"), 0.0, 10.0, saved_color1_stroke_width, key="color1_stroke_width_slider")
                config.ui["color1_stroke_width"] = params.color1_stroke_width
                
            # 強調色2
            st.write(tr("Highlight Color 2"))
            saved_color2_fore = config.ui.get("color2_fore", "#007AFF")
            saved_color2_stroke = config.ui.get("color2_stroke", "#FFFFFF")
            saved_color2_stroke_width = config.ui.get("color2_stroke_width", 1.5)
            c2_cols = st.columns([0.3, 0.3, 0.4])
            with c2_cols[0]:
                params.color2_fore = st.color_picker(tr("Color 2 Text"), saved_color2_fore, key="color2_fore_picker")
                config.ui["color2_fore"] = params.color2_fore
            with c2_cols[1]:
                params.color2_stroke = st.color_picker(tr("Color 2 Border"), saved_color2_stroke, key="color2_stroke_picker")
                config.ui["color2_stroke"] = params.color2_stroke
            with c2_cols[2]:
                params.color2_stroke_width = st.slider(tr("Color 2 Border Width"), 0.0, 10.0, saved_color2_stroke_width, key="color2_stroke_width_slider")
                config.ui["color2_stroke_width"] = params.color2_stroke_width
                
            # 強調色3
            st.write(tr("Highlight Color 3"))
            saved_color3_fore = config.ui.get("color3_fore", "#FFCC00")
            saved_color3_stroke = config.ui.get("color3_stroke", "#000000")
            saved_color3_stroke_width = config.ui.get("color3_stroke_width", 1.5)
            c3_cols = st.columns([0.3, 0.3, 0.4])
            with c3_cols[0]:
                params.color3_fore = st.color_picker(tr("Color 3 Text"), saved_color3_fore, key="color3_fore_picker")
                config.ui["color3_fore"] = params.color3_fore
            with c3_cols[1]:
                params.color3_stroke = st.color_picker(tr("Color 3 Border"), saved_color3_stroke, key="color3_stroke_picker")
                config.ui["color3_stroke"] = params.color3_stroke
            with c3_cols[2]:
                params.color3_stroke_width = st.slider(tr("Color 3 Border Width"), 0.0, 10.0, saved_color3_stroke_width, key="color3_stroke_width_slider")
                config.ui["color3_stroke_width"] = params.color3_stroke_width
    with st.expander(tr("Click to show API Key management"), expanded=False):
        st.subheader(tr("Manage Pexels, Pixabay and Coverr API Keys"))

        col1, col2, col3 = st.tabs([
            tr("Pexels API Keys"),
            tr("Pixabay API Keys"),
            tr("Coverr API Keys"),
        ])

        with col1:
            st.subheader(tr("Pexels API Keys"))
            if config.app["pexels_api_keys"]:
                st.write(tr("Current Keys:"))
                for key in config.app["pexels_api_keys"]:
                    st.code(key)
            else:
                st.info(tr("No Pexels API Keys currently"))

            new_key = st.text_input(tr("Add Pexels API Key"), key="pexels_new_key")
            if st.button(tr("Add Pexels API Key")):
                if new_key and new_key not in config.app["pexels_api_keys"]:
                    config.app["pexels_api_keys"].append(new_key)
                    config.save_config()
                    st.success(tr("Pexels API Key added successfully"))
                elif new_key in config.app["pexels_api_keys"]:
                    st.warning(tr("This API Key already exists"))
                else:
                    st.error(tr("Please enter a valid API Key"))

            if config.app["pexels_api_keys"]:
                delete_key = st.selectbox(
                    tr("Select Pexels API Key to delete"), config.app["pexels_api_keys"], key="pexels_delete_key"
                )
                if st.button(tr("Delete Selected Pexels API Key")):
                    config.app["pexels_api_keys"].remove(delete_key)
                    config.save_config()
                    st.success(tr("Pexels API Key deleted successfully"))

        with col2:
            st.subheader(tr("Pixabay API Keys"))

            if config.app["pixabay_api_keys"]:
                st.write(tr("Current Keys:"))
                for key in config.app["pixabay_api_keys"]:
                    st.code(key)
            else:
                st.info(tr("No Pixabay API Keys currently"))

            new_key = st.text_input(tr("Add Pixabay API Key"), key="pixabay_new_key")
            if st.button(tr("Add Pixabay API Key")):
                if new_key and new_key not in config.app["pixabay_api_keys"]:
                    config.app["pixabay_api_keys"].append(new_key)
                    config.save_config()
                    st.success(tr("Pixabay API Key added successfully"))
                elif new_key in config.app["pixabay_api_keys"]:
                    st.warning(tr("This API Key already exists"))
                else:
                    st.error(tr("Please enter a valid API Key"))

            if config.app["pixabay_api_keys"]:
                delete_key = st.selectbox(
                    tr("Select Pixabay API Key to delete"), config.app["pixabay_api_keys"], key="pixabay_delete_key"
                )
                if st.button(tr("Delete Selected Pixabay API Key")):
                    config.app["pixabay_api_keys"].remove(delete_key)
                    config.save_config()
                    st.success(tr("Pixabay API Key deleted successfully"))

        with col3:
            st.subheader(tr("Coverr API Keys"))

            # 与 pexels/pixabay 不同,coverr_api_keys 是 PR 新增配置项,
            # 老用户的 config.toml 不一定包含,这里先兜底初始化为空列表,
            # 防止下面 .append / 索引访问触发 KeyError。
            if "coverr_api_keys" not in config.app or config.app["coverr_api_keys"] is None:
                config.app["coverr_api_keys"] = []

            if config.app["coverr_api_keys"]:
                st.write(tr("Current Keys:"))
                for key in config.app["coverr_api_keys"]:
                    st.code(key)
            else:
                st.info(tr("No Coverr API Keys currently"))

            new_key = st.text_input(tr("Add Coverr API Key"), key="coverr_new_key")
            if st.button(tr("Add Coverr API Key")):
                if new_key and new_key not in config.app["coverr_api_keys"]:
                    config.app["coverr_api_keys"].append(new_key)
                    config.save_config()
                    st.success(tr("Coverr API Key added successfully"))
                elif new_key in config.app["coverr_api_keys"]:
                    st.warning(tr("This API Key already exists"))
                else:
                    st.error(tr("Please enter a valid API Key"))

            if config.app["coverr_api_keys"]:
                delete_key = st.selectbox(
                    tr("Select Coverr API Key to delete"), config.app["coverr_api_keys"], key="coverr_delete_key"
                )
                if st.button(tr("Delete Selected Coverr API Key")):
                    config.app["coverr_api_keys"].remove(delete_key)
                    config.save_config()
                    st.success(tr("Coverr API Key deleted successfully"))

# ---------------------------------------------------------------------------
# YouTube Shorts 自動投稿設定
# ---------------------------------------------------------------------------
from app.services import youtube as _yt

with st.expander(tr("YouTube Settings"), expanded=False):
    params.youtube_enabled = st.checkbox(
        tr("Enable YouTube Upload"),
        value=config.app.get("youtube_enabled", False),
        key="youtube_enabled",
    )
    config.app["youtube_enabled"] = params.youtube_enabled

    if not _yt.is_app_configured():
        st.warning(tr("YouTube Not Configured"))
    else:
        st.divider()
        # ── 連携済みアカウント一覧 ──────────────────────────────────────────
        _yt_accounts = _yt.list_accounts()
        st.write(tr("YouTube Accounts"))
        if _yt_accounts:
            for _acct in _yt_accounts:
                _acct_cols = st.columns([3, 1])
                _acct_cols[0].write(f"**{_acct}**")
                if _acct_cols[1].button(tr("YouTube Disconnect"), key=f"yt_del_{_acct}"):
                    _yt.delete_account(_acct)
                    if config.app.get("youtube_selected_account") == _acct:
                        config.app["youtube_selected_account"] = ""
                    st.rerun()
        else:
            st.info(tr("YouTube No Accounts"))

        # ── 新規アカウント追加（Googleでログイン）──────────────────────────
        st.write(tr("YouTube Add Account"))
        _yt_add_cols = st.columns([3, 2])
        _yt_new_name = _yt_add_cols[0].text_input(
            tr("YouTube Account Nickname"),
            key="yt_new_nickname",
            placeholder=tr("YouTube Account Nickname Placeholder"),
            label_visibility="collapsed",
        )
        _yt_connect_clicked = _yt_add_cols[1].button(
            tr("YouTube Connect"), key="yt_connect", use_container_width=True
        )

        _yt_waiting = st.session_state.get("_yt_login_waiting", False)

        if _yt_connect_clicked:
            if not _yt_new_name.strip():
                st.error(tr("YouTube Nickname Required"))
            else:
                _auth_url = _yt.start_login(_yt_new_name.strip())
                if _auth_url:
                    st.session_state["_yt_login_waiting"] = True
                    st.session_state["_yt_auth_url"] = _auth_url
                    st.rerun()
                else:
                    _res = _yt.oauth_result()
                    st.error(f"{tr('YouTube Auth Failed')}: {_res.get('error', '')}")

        if _yt_waiting:
            _yt_res = _yt.oauth_result()
            if _yt_res.get("success"):
                st.session_state.pop("_yt_login_waiting", None)
                st.session_state.pop("_yt_auth_url", None)
                st.success(f"{tr('YouTube Auth Complete')}: {_yt_res.get('nickname', '')}")
                st.rerun()
            elif _yt_res.get("error"):
                st.session_state.pop("_yt_login_waiting", None)
                st.session_state.pop("_yt_auth_url", None)
                st.error(f"{tr('YouTube Auth Failed')}: {_yt_res['error']}")
            else:
                _yt_saved_url = st.session_state.get("_yt_auth_url", "")
                if _yt_saved_url:
                    st.link_button(tr("YouTube Auth URL"), _yt_saved_url, use_container_width=True)
                st.info(tr("YouTube Login Waiting"))
                _yt_btn_cols = st.columns([1, 1])
                if _yt_btn_cols[0].button(tr("YouTube Auth Check"), key="yt_auth_check"):
                    st.rerun()
                if _yt_btn_cols[1].button(tr("Cancel"), key="yt_auth_cancel"):
                    st.session_state.pop("_yt_login_waiting", None)
                    st.session_state.pop("_yt_auth_url", None)
                    st.rerun()

        # ── 投稿設定 ────────────────────────────────────────────────────────
        if _yt_accounts and params.youtube_enabled:
            st.divider()

            _yt_saved_acct = config.app.get("youtube_selected_account", "")
            _yt_acct_idx = _yt_accounts.index(_yt_saved_acct) if _yt_saved_acct in _yt_accounts else 0
            _yt_selected = st.selectbox(
                tr("YouTube Selected Account"),
                options=_yt_accounts,
                index=_yt_acct_idx,
                key="youtube_selected_account",
            )
            params.youtube_selected_account = _yt_selected
            config.app["youtube_selected_account"] = _yt_selected

            _yt_settings_cols = st.columns([1, 1])
            with _yt_settings_cols[0]:
                _yt_privacy_options = ["private", "unlisted", "public"]
                _yt_privacy_labels = [tr("YouTube Private"), tr("YouTube Unlisted"), tr("YouTube Public")]
                _yt_current_privacy = config.app.get("youtube_privacy", "private")
                _yt_privacy_idx = _yt_privacy_options.index(_yt_current_privacy) if _yt_current_privacy in _yt_privacy_options else 0
                _yt_privacy = st.selectbox(
                    tr("YouTube Privacy"),
                    options=_yt_privacy_labels,
                    index=_yt_privacy_idx,
                    key="youtube_privacy_select",
                )
                params.youtube_privacy = _yt_privacy_options[_yt_privacy_labels.index(_yt_privacy)]
                config.app["youtube_privacy"] = params.youtube_privacy

            with _yt_settings_cols[1]:
                params.youtube_schedule_hours = st.number_input(
                    tr("YouTube Schedule Hours"),
                    min_value=0,
                    max_value=8760,
                    value=int(config.app.get("youtube_schedule_hours", 0)),
                    step=1,
                    key="youtube_schedule_hours",
                    help=tr("YouTube Schedule Hours Help"),
                )
                config.app["youtube_schedule_hours"] = params.youtube_schedule_hours

            params.youtube_auto_metadata = st.checkbox(
                tr("YouTube Auto Metadata"),
                value=config.app.get("youtube_auto_metadata", True),
                key="youtube_auto_metadata",
                help=tr("YouTube Auto Metadata Help"),
            )
            config.app["youtube_auto_metadata"] = params.youtube_auto_metadata

start_button = st.button(tr("Generate Video"), use_container_width=True, type="primary")
_one_click_fire = st.session_state.pop("_one_click_pending", False)
if start_button or _one_click_fire:
    if _one_click_fire:
        # ワンクリック生成: スクリプト・キーワードを強制クリアして自動生成させる
        params.video_script = ""
        params.video_terms = None
    config.save_config()
    raw_uuid = str(uuid4())
    task_id = utils.render_task_folder_name(params, raw_uuid)
    if not params.video_subject and not params.video_script:
        st.error(tr("Video Script and Subject Cannot Both Be Empty"))
        scroll_to_bottom()
        st.stop()

    if params.video_source not in ["pexels", "pixabay", "coverr", "local"]:
        st.error(tr("Please Select a Valid Video Source"))
        scroll_to_bottom()
        st.stop()

    if params.video_source == "pexels" and not config.app.get("pexels_api_keys", ""):
        st.error(tr("Please Enter the Pexels API Key"))
        scroll_to_bottom()
        st.stop()

    if params.video_source == "pixabay" and not config.app.get("pixabay_api_keys", ""):
        st.error(tr("Please Enter the Pixabay API Key"))
        scroll_to_bottom()
        st.stop()

    if params.video_source == "coverr" and not config.app.get("coverr_api_keys", ""):
        st.error(tr("Please Enter the Coverr API Key"))
        scroll_to_bottom()
        st.stop()

    if uploaded_audio_file:
        task_dir = utils.task_dir(task_id)
        # 上传文件名来自浏览器，不能直接拼到磁盘路径里；这里只保留扩展名，
        # 并使用固定文件名保存到当前任务目录，避免路径穿越或特殊字符问题。
        _, audio_ext = os.path.splitext(os.path.basename(uploaded_audio_file.name))
        audio_ext = audio_ext.lower() or ".mp3"
        custom_audio_path = os.path.join(task_dir, f"custom-audio{audio_ext}")
        with open(custom_audio_path, "wb") as f:
            f.write(uploaded_audio_file.getbuffer())
        params.custom_audio_file = custom_audio_path

    if uploaded_files:
        local_videos_dir = utils.storage_dir("local_videos", create=True)
        # 每次重新上传时都以本次选择的素材为准，避免旧素材不断重复追加。
        params.video_materials = []
        persisted_local_materials = []
        for file in uploaded_files:
            file_path = os.path.join(local_videos_dir, f"{file.file_id}_{file.name}")
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
                m = MaterialInfo()
                m.provider = "local"
                m.url = file_path
                params.video_materials.append(m)
                persisted_local_materials.append(
                    {
                        "provider": m.provider,
                        "url": m.url,
                        "duration": m.duration,
                    }
                )
        # 将已上传并保存到本地的视频素材写入会话，供后续只改文案时直接复用。
        st.session_state["local_video_materials"] = persisted_local_materials
    elif params.video_source == "local" and st.session_state["local_video_materials"]:
        # 当用户没有重新上传文件时，复用最近一次已经保存到磁盘的本地素材列表。
        params.video_materials = []
        for material in st.session_state["local_video_materials"]:
            m = MaterialInfo()
            m.provider = material.get("provider", "local")
            m.url = material.get("url", "")
            m.duration = material.get("duration", 0)
            if m.url:
                params.video_materials.append(m)

    log_container = st.empty()
    log_records = []

    def log_received(msg):
        if config.ui["hide_log"]:
            return
        with log_container:
            log_records.append(msg)
            st.code("\n".join(log_records))

    logger.add(log_received)

    st.toast(tr("Generating Video"))
    logger.info(tr("Start Generating Video"))
    logger.info(utils.to_json(params))
    scroll_to_bottom()

    result = tm.start(task_id=task_id, params=params)
    if not result or "videos" not in result:
        st.error(tr("Video Generation Failed"))
        logger.error(tr("Video Generation Failed"))
        scroll_to_bottom()
        st.stop()

    video_files = result.get("videos", [])
    st.success(tr("Video Generation Completed"))
    try:
        if video_files:
            player_cols = st.columns(len(video_files) * 2 + 1)
            for i, url in enumerate(video_files):
                player_cols[i * 2 + 1].video(url)
    except Exception:
        pass

    open_task_folder(task_id)
    logger.info(tr("Video Generation Completed"))
    scroll_to_bottom()

config.save_config()
