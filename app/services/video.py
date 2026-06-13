import glob
import itertools
import io
import os
import random
import re
import gc
import shutil
import subprocess
from contextlib import redirect_stdout
from functools import lru_cache
from typing import List
from loguru import logger
import numpy as np
from moviepy import (
    AudioFileClip,
    ColorClip,
    CompositeAudioClip,
    CompositeVideoClip,
    ImageClip,
    TextClip,
    VideoFileClip,
    afx,
    vfx,
)
from moviepy.video.tools.subtitles import SubtitlesClip
from PIL import Image, ImageDraw, ImageFont

from app.config import config
from app.models import const
from app.models.schema import (
    MaterialInfo,
    VideoAspect,
    VideoConcatMode,
    VideoParams,
    VideoTransitionMode,
)
from app.services.utils import video_effects
from app.utils import file_security, utils

class SubClippedVideoClip:
    def __init__(
        self,
        file_path,
        start_time=None,
        end_time=None,
        width=None,
        height=None,
        duration=None,
        source_file_path=None,
    ):
        self.file_path = file_path
        self.start_time = start_time
        self.end_time = end_time
        self.width = width
        self.height = height
        self.source_file_path = source_file_path or file_path
        if duration is None:
            self.duration = end_time - start_time
        else:
            self.duration = duration

    def __str__(self):
        return f"SubClippedVideoClip(file_path={self.file_path}, start_time={self.start_time}, end_time={self.end_time}, duration={self.duration}, width={self.width}, height={self.height})"


audio_codec = "aac"
# Docker 里的 ffmpeg/AAC 组合在默认配置下更容易出现音频质量波动，
# 这里显式抬高音频码率，避免成片阶段因为默认值过低而引入明显失真。
audio_bitrate = "192k"
fps = 30
_BGM_EXTENSIONS = (".mp3",)
_DEFAULT_VIDEO_CODEC = "libx264"
_SUPPORTED_VIDEO_CODECS = (
    "libx264",
    "h264_nvenc",
    "h264_amf",
    "h264_qsv",
    "h264_mf",
    "h264_videotoolbox",
)
_runtime_disabled_video_codecs = set()


def _prioritize_unique_source_clips(
    subclipped_items: List[SubClippedVideoClip],
    concat_mode: VideoConcatMode,
) -> List[SubClippedVideoClip]:
    """
    优先让每个源素材只出现一次，降低成片里同一素材反复出现的概率。

    线上素材经常会遇到“一个长视频被切成多个短片段”的情况。旧逻辑在
    random 模式下直接打乱所有短片段，导致同一个源视频的多个切片可能
    分布在开头和中间，用户会感知为素材重复。本函数只调整片段顺序：
    先放每个源文件里最长的一个片段，剩余片段作为兜底；当素材总时长不足时，
    仍然允许后续片段补齐音频长度，避免破坏视频生成成功率。优先选择最长
    片段是为了避免随机选中视频尾部的零碎短片段，导致明明有足够素材却过早复用。
    """
    if not subclipped_items:
        return []

    concat_mode_value = getattr(concat_mode, "value", concat_mode)
    if concat_mode_value != VideoConcatMode.random.value:
        return subclipped_items

    grouped_items: dict[str, list[SubClippedVideoClip]] = {}
    for item in subclipped_items:
        grouped_items.setdefault(item.source_file_path, []).append(item)

    primary_items = []
    overflow_items = []
    for items in grouped_items.values():
        primary_item = max(items, key=lambda item: item.duration)
        primary_items.append(primary_item)
        overflow_items.extend(item for item in items if item is not primary_item)

    random.shuffle(primary_items)
    random.shuffle(overflow_items)
    logger.info(
        "prioritized unique video materials, "
        f"sources: {len(grouped_items)}, "
        f"primary clips: {len(primary_items)}, "
        f"fallback clips: {len(overflow_items)}"
    )
    return primary_items + overflow_items


def get_ffmpeg_binary():
    """
    兼容历史上直接从 video 服务读取 FFmpeg 路径的调用方。

    真正的解析逻辑已经抽到 `app.utils.utils.get_ffmpeg_binary()`，视频、语音
    和后续新增链路都应复用同一套优先级；这里保留薄包装，避免外部脚本或
    旧测试直接导入 `app.services.video.get_ffmpeg_binary` 时出现 AttributeError。
    """
    return utils.get_ffmpeg_binary()


def _get_configured_video_codec() -> str:
    """
    读取用户配置的视频编码器。

    该配置面向高级用户，用于尝试启用 NVENC/AMF/QSV/VideoToolbox 等硬件
    编码。这里刻意只允许固定白名单，避免开放任意 FFmpeg 参数后，用户填错
    参数导致输出格式不可控，甚至让生成任务在后续阶段才失败。
    """
    configured_codec = str(
        config.app.get("video_codec", _DEFAULT_VIDEO_CODEC) or _DEFAULT_VIDEO_CODEC
    ).strip()
    if configured_codec not in _SUPPORTED_VIDEO_CODECS:
        logger.warning(
            f"unsupported video codec configured: {configured_codec}, "
            f"fallback to {_DEFAULT_VIDEO_CODEC}"
        )
        return _DEFAULT_VIDEO_CODEC
    return configured_codec


@lru_cache(maxsize=16)
def _ffmpeg_encoder_exists(ffmpeg_binary: str, codec: str) -> bool:
    """
    检查当前 FFmpeg 是否声明支持指定编码器。

    这只能证明 FFmpeg 编译时包含该 encoder，不能证明当前机器硬件和驱动
    一定可用。因此实际编码失败时仍会再回退到 libx264。
    """
    try:
        result = subprocess.run(
            [ffmpeg_binary, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning(
            "failed to inspect ffmpeg encoders, "
            f"fallback to {_DEFAULT_VIDEO_CODEC}: {str(exc)}"
        )
        return False

    if result.returncode != 0:
        logger.warning(
            "failed to inspect ffmpeg encoders, "
            f"fallback to {_DEFAULT_VIDEO_CODEC}: {(result.stderr or result.stdout or '').strip()}"
        )
        return False
    return codec in result.stdout


def _get_effective_video_codec(preferred_codec: str | None = None) -> str:
    """
    返回本次实际使用的视频编码器。

    用户选择硬件编码器时，先做 FFmpeg encoder 列表检测；如果本进程里已经
    实际编码失败过，也直接回退，避免一个任务里每个片段都重复失败。
    """
    selected_codec = preferred_codec or _get_configured_video_codec()
    if selected_codec == _DEFAULT_VIDEO_CODEC:
        return _DEFAULT_VIDEO_CODEC

    if selected_codec in _runtime_disabled_video_codecs:
        logger.warning(
            f"video codec {selected_codec} was disabled after a runtime failure, "
            f"fallback to {_DEFAULT_VIDEO_CODEC}"
        )
        return _DEFAULT_VIDEO_CODEC

    ffmpeg_binary = utils.get_ffmpeg_binary()
    if not _ffmpeg_encoder_exists(ffmpeg_binary, selected_codec):
        logger.warning(
            f"ffmpeg encoder {selected_codec} is not available, "
            f"fallback to {_DEFAULT_VIDEO_CODEC}"
        )
        return _DEFAULT_VIDEO_CODEC

    return selected_codec


def _disable_runtime_video_codec(codec: str, reason: str):
    if codec == _DEFAULT_VIDEO_CODEC:
        return
    _runtime_disabled_video_codecs.add(codec)
    logger.warning(
        f"video codec {codec} failed, fallback to {_DEFAULT_VIDEO_CODEC}. "
        f"reason: {reason}"
    )


def _fallback_write_videofile(clip, output_file: str, failed_codec: str, reason: str, **kwargs):
    """
    硬件编码失败后用 libx264 重试，只有重试成功才禁用该硬件编码器。

    Windows 上 FFmpeg 失败原因比较复杂：可能是显卡/驱动不支持，也可能是输出
    文件被占用、目录权限、杀软拦截等通用 IO 问题。只有 libx264 能成功写出时，
    才能判断原始失败大概率来自硬件编码器本身，避免误伤后续任务。
    """
    clip.write_videofile(output_file, codec=_DEFAULT_VIDEO_CODEC, **kwargs)
    _disable_runtime_video_codec(failed_codec, reason)
    return _DEFAULT_VIDEO_CODEC


def _write_videofile_with_codec_fallback(clip, output_file: str, codec: str, **kwargs):
    """
    使用指定编码器写出视频，失败时自动用 libx264 重试一次。

    硬件编码器是否可用不仅取决于 FFmpeg，还取决于显卡、驱动和当前运行环境。
    生成任务不能因为高级编码器不可用而整体失败，所以这里把回退集中处理。
    """
    effective_codec = _get_effective_video_codec(codec)
    try:
        clip.write_videofile(output_file, codec=effective_codec, **kwargs)
        return effective_codec
    except Exception as exc:
        if effective_codec == _DEFAULT_VIDEO_CODEC:
            raise
        return _fallback_write_videofile(
            clip,
            output_file,
            failed_codec=effective_codec,
            reason=str(exc),
            **kwargs,
        )


def _escape_ffmpeg_concat_path(file_path: str) -> str:
    # concat demuxer 使用单引号包裹路径，路径中的单引号需要先转义。
    return file_path.replace("'", "'\\''")


def _format_ffmpeg_concat_path(file_path: str) -> str:
    """
    生成 concat demuxer 文件列表中的路径。

    FFmpeg 官方文档要求 concat list 中的特殊字符和空格需要转义；Windows
    绝对路径里的反斜杠也容易被解析成转义字符。这里统一转成正斜杠形式，
    让 `C:\\Users\\...` 变成 `C:/Users/...`，再处理单引号，兼容 macOS/Linux。
    """
    absolute_path = os.path.abspath(file_path)
    return _escape_ffmpeg_concat_path(absolute_path.replace("\\", "/"))


def concat_video_clips_with_ffmpeg(
    clip_files: List[str], output_file: str, threads: int, output_dir: str
):
    concat_list_file = os.path.join(output_dir, "ffmpeg-concat-list.txt")
    with open(concat_list_file, "w", encoding="utf-8") as fp:
        for clip_file in clip_files:
            fp.write(f"file '{_format_ffmpeg_concat_path(clip_file)}'\n")

    def build_command(codec: str) -> list[str]:
        return [
            utils.get_ffmpeg_binary(),
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_list_file,
            "-c:v",
            codec,
            "-threads",
            str(threads or 2),
            "-pix_fmt",
            "yuv420p",
            output_file,
        ]

    def run_concat(codec: str):
        command = build_command(codec)
        # 使用 ffmpeg 只做一次串联与编码，避免 MoviePy 逐段合并时反复重编码，
        # 从而降低画质劣化与颜色偏移风险。
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            error_message = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(error_message or "ffmpeg concat failed")
        return codec

    try:
        effective_codec = _get_effective_video_codec()
        try:
            return run_concat(effective_codec)
        except Exception as exc:
            if effective_codec == _DEFAULT_VIDEO_CODEC:
                raise
            result_codec = run_concat(_DEFAULT_VIDEO_CODEC)
            _disable_runtime_video_codec(effective_codec, str(exc))
            return result_codec
    finally:
        delete_files(concat_list_file)


def _sanitize_image_file(image_path: str) -> str:
    # 某些本地图片虽然能被 Pillow 打开，但会因为损坏的 EXIF/eXIf 元数据导致
    # ImageClip 在解析阶段直接抛异常。这里重新导出一份“干净图片”，把坏元数据剥离掉。
    image_root, _ = os.path.splitext(image_path)
    sanitized_path = f"{image_root}.sanitized.png"

    with Image.open(image_path) as image:
        image.load()
        # 统一导出为 PNG，避免 JPEG/PNG 不同元数据路径继续把坏块带过去。
        cleaned_image = Image.new(image.mode, image.size)
        cleaned_image.putdata(list(image.getdata()))
        cleaned_image.save(sanitized_path)

    return sanitized_path


def _open_image_clip_with_fallback(image_path: str):
    # 优先直接打开原始图片；如果因为损坏元数据失败，再尝试生成无元数据副本。
    try:
        return ImageClip(image_path), image_path
    except Exception as exc:
        logger.warning(
            f"failed to open image directly, trying sanitized copy: {image_path}, error: {str(exc)}"
        )
        sanitized_path = _sanitize_image_file(image_path)
        return ImageClip(sanitized_path), sanitized_path


def _open_video_clip_quietly(video_path: str, audio: bool = False) -> VideoFileClip:
    """
    安静地打开视频文件，避免 MoviePy 2.1.x 把 ffmpeg 探测信息直接打印到 stdout。

    背景：
    当前依赖版本的 `FFMPEG_VideoReader` 内部存在 `print(self.infos)` 和
    `print(ffmpeg command)`，读取无音轨的中间视频时会输出
    `audio_found: False`。这只是输入素材 metadata，不代表最终成片没有音频，
    但会误导 WebUI/终端用户以为生成失败。

    实现：
    1. 只在打开 VideoFileClip 的短窗口内重定向 stdout；
    2. 默认 `audio=False`，因为项目视频素材阶段不需要保留素材原声，
       最终音频会在 `generate_video()` 阶段统一挂载；
    3. 如果依赖库确实输出了内容，降级为 debug 日志，便于必要时排查。
    """
    captured_stdout = io.StringIO()
    with redirect_stdout(captured_stdout):
        clip = VideoFileClip(video_path, audio=audio)

    moviepy_stdout = captured_stdout.getvalue().strip()
    if moviepy_stdout:
        logger.debug(
            "suppressed MoviePy video reader stdout for "
            f"{video_path}, chars: {len(moviepy_stdout)}"
        )

    return clip


def close_clip(clip):
    if clip is None:
        return
        
    try:
        # close main resources
        if hasattr(clip, 'reader') and clip.reader is not None:
            clip.reader.close()
            
        # close audio resources
        if hasattr(clip, 'audio') and clip.audio is not None:
            if hasattr(clip.audio, 'reader') and clip.audio.reader is not None:
                clip.audio.reader.close()
            del clip.audio
            
        # close mask resources
        if hasattr(clip, 'mask') and clip.mask is not None:
            if hasattr(clip.mask, 'reader') and clip.mask.reader is not None:
                clip.mask.reader.close()
            del clip.mask
            
        # handle child clips in composite clips
        if hasattr(clip, 'clips') and clip.clips:
            for child_clip in clip.clips:
                if child_clip is not clip:  # avoid possible circular references
                    close_clip(child_clip)
            
        # clear clip list
        if hasattr(clip, 'clips'):
            clip.clips = []
            
    except Exception as e:
        logger.error(f"failed to close clip: {str(e)}")
    
    del clip
    gc.collect()

def delete_files(files: List[str] | str):
    if isinstance(files, str):
        files = [files]

    for file in files:
        try:
            os.remove(file)
        except Exception as e:
            logger.debug(f"failed to delete file {file}: {str(e)}")


def _resolve_bgm_file_path(song_dir: str, bgm_file: str) -> str:
    # 背景音乐只允许读取 resource/songs 目录内的文件，避免用户输入任意路径后
    # 被 MoviePy 打开。这里兼容两种常见输入：
    # 1. output000.mp3：来自 BGM 列表或用户只填写文件名
    # 2. ./resource/songs/output000.mp3：用户按项目目录结构填写的相对路径
    # 两种写法最终都会再次通过 resource/songs 白名单校验，不能绕过目录限制。
    try:
        return file_security.resolve_path_within_directory(song_dir, bgm_file)
    except ValueError as song_dir_exc:
        if os.path.isabs(bgm_file):
            raise song_dir_exc

        project_relative_file = os.path.join(utils.root_dir(), bgm_file)
        try:
            return file_security.resolve_path_within_directory(
                song_dir, project_relative_file
            )
        except ValueError as root_dir_exc:
            raise ValueError(str(root_dir_exc)) from song_dir_exc


def get_bgm_file(bgm_type: str = "random", bgm_file: str = ""):
    if not bgm_type:
        return ""

    if bgm_file:
        song_dir = utils.song_dir()
        try:
            resolved_bgm_file = _resolve_bgm_file_path(song_dir, bgm_file)
        except ValueError as exc:
            # API 请求里的 bgm_file 来自用户输入，不能直接把任意绝对路径交给
            # MoviePy 打开。这里强制限制到 resource/songs 目录，阻止读取
            # /etc/passwd、配置文件、密钥等非背景音乐文件。
            logger.warning(
                f"reject unsafe bgm file: {bgm_file}, song_dir: {song_dir}, error: {str(exc)}"
            )
            return ""

        if not resolved_bgm_file.lower().endswith(_BGM_EXTENSIONS):
            logger.warning(f"reject unsupported bgm file extension: {resolved_bgm_file}")
            return ""

        return resolved_bgm_file

    if bgm_type == "random":
        suffix = "*.mp3"
        song_dir = utils.song_dir()
        files = glob.glob(os.path.join(song_dir, suffix))
        # 当背景音乐目录为空时，直接回退为“不使用 BGM”，避免 random.choice([]) 抛异常。
        if not files:
            logger.warning(f"no bgm files found in song directory: {song_dir}")
            return ""
        return random.choice(files)

    return ""


def combine_videos(
    combined_video_path: str,
    video_paths: List[str],
    audio_file: str,
    video_aspect: VideoAspect = VideoAspect.portrait,
    video_concat_mode: VideoConcatMode = VideoConcatMode.random,
    video_transition_mode: VideoTransitionMode = None,
    max_clip_duration: int = 5,
    threads: int = 2,
    video_clip_fit: str = "contain",
    video_margin_ratio: float = 0.0,
    bg_video_file: str = "",
) -> str:
    audio_clip = AudioFileClip(audio_file)
    try:
        # 这里只需要读取旁白音频时长来决定素材视频拼接长度；后续不会再使用
        # audio_clip。读取完成后立即关闭，避免早退或异常路径泄漏文件句柄。
        audio_duration = audio_clip.duration
    finally:
        close_clip(audio_clip)
    logger.info(f"audio duration: {audio_duration} seconds")
    logger.info(f"maximum clip duration: {max_clip_duration} seconds")

    # 兼容 API 直接调用时未传转场模式的情况，避免后续访问 .value 时崩溃。
    transition_value = getattr(video_transition_mode, "value", video_transition_mode)
    output_dir = os.path.dirname(combined_video_path)

    aspect = VideoAspect(video_aspect)
    video_width, video_height = aspect.to_resolution()

    # 背景動画の事前ロード（containモード時に黒帯の代わりに使う）
    bg_clip_for_combine = None
    if bg_video_file and os.path.exists(bg_video_file):
        try:
            _bg = _open_video_clip_quietly(bg_video_file)
            _bg = _bg.with_effects([vfx.Loop(duration=audio_duration + 10)])
            _bg_w, _bg_h = _bg.size
            _bg_ratio = _bg_w / _bg_h
            _canvas_ratio = video_width / video_height
            if _bg_ratio > _canvas_ratio:
                _bg_scale = video_height / _bg_h
            else:
                _bg_scale = video_width / _bg_w
            _bg_nw = int(_bg_w * _bg_scale)
            _bg_nh = int(_bg_h * _bg_scale)
            _bg = _bg.resized(new_size=(_bg_nw, _bg_nh))
            bg_clip_for_combine = _bg.cropped(
                width=video_width, height=video_height,
                x_center=_bg_nw / 2, y_center=_bg_nh / 2,
            )
            logger.info(f"background video loaded for combine_videos: {bg_video_file}")
        except Exception as e:
            logger.warning(f"failed to load bg_video_file for combine: {e}")
            bg_clip_for_combine = None

    processed_clips = []
    subclipped_items = []
    video_duration = 0
    for video_path in video_paths:
        clip = _open_video_clip_quietly(video_path)
        clip_duration = clip.duration
        clip_w, clip_h = clip.size
        close_clip(clip)
        
        start_time = 0

        while start_time < clip_duration:
            end_time = min(start_time + max_clip_duration, clip_duration)

            # 保留所有有效分段。
            # 这样既不会丢掉“整段视频本身就短于 max_clip_duration”的素材，
            # 也不会吞掉长视频最后剩下的一小段尾部内容。
            if end_time > start_time:
                subclipped_items.append(
                    SubClippedVideoClip(
                        file_path=video_path,
                        start_time=start_time,
                        end_time=end_time,
                        width=clip_w,
                        height=clip_h,
                        source_file_path=video_path,
                    )
                )

            start_time = end_time
            if video_concat_mode.value == VideoConcatMode.sequential.value:
                break

    subclipped_items = _prioritize_unique_source_clips(
        subclipped_items=subclipped_items,
        concat_mode=video_concat_mode,
    )
        
    logger.debug(f"total subclipped items: {len(subclipped_items)}")
    
    # Add downloaded clips over and over until the duration of the audio (max_duration) has been reached
    for i, subclipped_item in enumerate(subclipped_items):
        if video_duration >= audio_duration:
            break
        
        logger.debug(
            f"processing clip {i+1}: {subclipped_item.width}x{subclipped_item.height}, "
            f"source: {os.path.basename(subclipped_item.source_file_path)}, "
            f"current duration: {video_duration:.2f}s, "
            f"remaining: {audio_duration - video_duration:.2f}s"
        )
        
        try:
            clip = _open_video_clip_quietly(subclipped_item.file_path).subclipped(
                subclipped_item.start_time, subclipped_item.end_time
            )
            clip_duration = clip.duration
            # Not all videos are same size, so we need to resize them
            clip_w, clip_h = clip.size
            
            target_w = video_width
            target_h = video_height
            if video_clip_fit == "cover" and video_margin_ratio > 0.0:
                target_h = int(video_height * (1.0 - 2 * video_margin_ratio))
                
            if clip_w != video_width or clip_h != video_height or video_margin_ratio > 0.0:
                clip_ratio = clip.w / clip.h
                target_ratio = target_w / target_h
                logger.debug(f"resizing clip ({video_clip_fit}), source: {clip_w}x{clip_h}, ratio: {clip_ratio:.2f}, target: {target_w}x{target_h}, ratio: {target_ratio:.2f}")
                
                if clip_ratio == target_ratio and video_margin_ratio == 0.0:
                    clip = clip.resized(new_size=(video_width, video_height))
                else:
                    if video_clip_fit == "cover":
                        if clip_ratio > target_ratio:
                            scale_factor = target_h / clip_h
                        else:
                            scale_factor = target_w / clip_w

                        new_width = int(clip_w * scale_factor)
                        new_height = int(clip_h * scale_factor)
                        clip_resized = clip.resized(new_size=(new_width, new_height))

                        # crop the center region
                        x_center = new_width / 2
                        y_center = new_height / 2
                        clip_cropped = clip_resized.cropped(width=target_w, height=target_h, x_center=x_center, y_center=y_center)

                        background = ColorClip(size=(video_width, video_height), color=(0, 0, 0)).with_duration(clip_duration)
                        clip = CompositeVideoClip([background, clip_cropped.with_position("center")])
                    else:  # contain
                        if clip_ratio > target_ratio:
                            scale_factor = target_w / clip_w
                        else:
                            scale_factor = target_h / clip_h

                        new_width = int(clip_w * scale_factor)
                        new_height = int(clip_h * scale_factor)

                        if bg_clip_for_combine is not None:
                            # bgクリップの該当タイムスライスをバックグラウンドとして使う
                            background = bg_clip_for_combine.subclipped(
                                video_duration, video_duration + clip_duration
                            ).with_duration(clip_duration)
                        else:
                            background = ColorClip(size=(video_width, video_height), color=(0, 0, 0)).with_duration(clip_duration)
                        clip_resized = clip.resized(new_size=(new_width, new_height)).with_position("center")
                        clip = CompositeVideoClip([background, clip_resized])
                    
            shuffle_side = random.choice(["left", "right", "top", "bottom"])
            if transition_value in (None, VideoTransitionMode.none.value):
                clip = clip
            elif transition_value == VideoTransitionMode.fade_in.value:
                clip = video_effects.fadein_transition(clip, 1)
            elif transition_value == VideoTransitionMode.fade_out.value:
                clip = video_effects.fadeout_transition(clip, 1)
            elif transition_value == VideoTransitionMode.slide_in.value:
                clip = video_effects.slidein_transition(clip, 1, shuffle_side)
            elif transition_value == VideoTransitionMode.slide_out.value:
                clip = video_effects.slideout_transition(clip, 1, shuffle_side)
            elif transition_value == VideoTransitionMode.shuffle.value:
                transition_funcs = [
                    lambda c: video_effects.fadein_transition(c, 1),
                    lambda c: video_effects.fadeout_transition(c, 1),
                    lambda c: video_effects.slidein_transition(c, 1, shuffle_side),
                    lambda c: video_effects.slideout_transition(c, 1, shuffle_side),
                ]
                shuffle_transition = random.choice(transition_funcs)
                clip = shuffle_transition(clip)

            if clip.duration > max_clip_duration:
                clip = clip.subclipped(0, max_clip_duration)
                
            # wirte clip to temp file
            clip_file = f"{output_dir}/temp-clip-{i+1}.mp4"
            _write_videofile_with_codec_fallback(
                clip,
                clip_file,
                codec=_get_configured_video_codec(),
                logger=None,
                fps=fps,
            )

            # Store clip duration before closing
            clip_duration_saved = clip.duration
            close_clip(clip)

            processed_clips.append(
                SubClippedVideoClip(
                    file_path=clip_file,
                    duration=clip_duration_saved,
                    width=clip_w,
                    height=clip_h,
                    source_file_path=subclipped_item.source_file_path,
                )
            )
            video_duration += clip_duration_saved
            
        except Exception as e:
            logger.error(f"failed to process clip: {str(e)}")
    
    # loop processed clips until the video duration matches or exceeds the audio duration.
    if video_duration < audio_duration:
        logger.warning(f"video duration ({video_duration:.2f}s) is shorter than audio duration ({audio_duration:.2f}s), looping clips to match audio length.")
        base_clips = processed_clips.copy()
        for clip in itertools.cycle(base_clips):
            if video_duration >= audio_duration:
                break
            processed_clips.append(clip)
            video_duration += clip.duration
        logger.info(f"video duration: {video_duration:.2f}s, audio duration: {audio_duration:.2f}s, looped {len(processed_clips)-len(base_clips)} clips")
     
    # merge video clips progressively, avoid loading all videos at once to avoid memory overflow
    logger.info("starting clip merging process")
    if not processed_clips:
        logger.warning("no clips available for merging")
        return combined_video_path
    
    # if there is only one clip, use it directly
    if len(processed_clips) == 1:
        logger.info("using single clip directly")
        shutil.copy(processed_clips[0].file_path, combined_video_path)
        delete_files([processed_clips[0].file_path])
        logger.info("video combining completed")
        return combined_video_path

    clip_files = [clip.file_path for clip in processed_clips]
    logger.info(f"concatenating {len(clip_files)} clips with ffmpeg")
    concat_video_clips_with_ffmpeg(
        clip_files=clip_files,
        output_file=combined_video_path,
        threads=threads,
        output_dir=output_dir,
    )
    
    # clean temp files
    delete_files(clip_files)
            
    logger.info("video combining completed")
    return combined_video_path


def wrap_text(text, max_width, font="Arial", fontsize=60):
    # 字幕换行必须在真正创建 TextClip 前完成，否则 MoviePy 只会按原始文本
    # 计算渲染区域。这里用 PIL 按当前字体和字号测量宽度，确保每一行都尽量
    # 控制在视频可用宽度内，避免大字号或中文长句直接溢出画面。
    font = ImageFont.truetype(font, fontsize)
    max_width = int(max_width)

    def get_text_size(inner_text):
        inner_text = inner_text.strip()
        if not inner_text:
            return 0, fontsize
        left, top, right, bottom = font.getbbox(inner_text)
        return right - left, bottom - top

    width, height = get_text_size(text)
    if width <= max_width:
        return text, height

    def split_long_token(token):
        # 当一个 token 本身就超宽时（常见于中文无空格长句，或英文超长单词），
        # 退化为字符级拆分。关键点是：检测到 candidate 超宽时，先提交上一个
        # 仍然合法的 current，再把当前字符放入下一行，不能把超宽字符塞回上一行。
        lines = []
        current = ""
        for char in token:
            candidate = f"{current}{char}"
            candidate_width, _ = get_text_size(candidate)
            if candidate_width <= max_width or not current:
                current = candidate
                continue
            lines.append(current)
            current = char
        if current:
            lines.append(current)
        return lines

    lines = []
    current = ""
    words = text.split(" ")
    for word in words:
        candidate = f"{current} {word}".strip() if current else word
        candidate_width, _ = get_text_size(candidate)
        if candidate_width <= max_width:
            current = candidate
            continue

        if current:
            lines.append(current)

        word_width, _ = get_text_size(word)
        if word_width <= max_width:
            current = word
        else:
            lines.extend(split_long_token(word))
            current = ""

    if current:
        lines.append(current)

    line_start_punctuation = "，。！？；：、,.!?;:)]}）】》」』”’"
    for index in range(1, len(lines)):
        if not lines[index] or lines[index][0] not in line_start_punctuation:
            continue
        if len(lines[index - 1]) <= 1:
            continue

        candidate = f"{lines[index - 1][-1]}{lines[index]}"
        candidate_width, _ = get_text_size(candidate)
        if candidate_width <= max_width:
            lines[index] = candidate
            lines[index - 1] = lines[index - 1][:-1]

    result = "\n".join(line.strip() for line in lines if line.strip()).strip()
    height = len(lines) * height
    return result, height


def get_bg_video_file(bg_video_type: str, bg_video_file: str) -> str:
    if bg_video_type in ["custom", "source"]:
        if bg_video_file and os.path.exists(bg_video_file):
            return bg_video_file
        logger.warning(f"{bg_video_type} bg video file not found: {bg_video_file}")
        return ""

    if bg_video_type == "random":
        bg_video_dir = utils.resource_dir("bg_videos")
        if not os.path.exists(bg_video_dir):
            os.makedirs(bg_video_dir)
        suffix = "*.mp4"
        files = glob.glob(os.path.join(bg_video_dir, suffix))
        if not files:
            logger.warning(f"no bg video files found in directory: {bg_video_dir}")
            return ""
        return random.choice(files)

    return ""


def generate_video(
    video_path: str,
    audio_path: str,
    subtitle_path: str,
    output_file: str,
    params: VideoParams,
):
    aspect = VideoAspect(params.video_aspect)
    video_width, video_height = aspect.to_resolution()

    logger.info(f"generating video: {video_width} x {video_height}")
    logger.info(f"  ① video: {video_path}")
    logger.info(f"  ② audio: {audio_path}")
    logger.info(f"  ③ subtitle: {subtitle_path}")
    logger.info(f"  ④ output: {output_file}")

    output_dir = os.path.dirname(output_file)

    font_path = ""
    if params.subtitle_enabled or getattr(params, "video_title", ""):
        if not params.font_name:
            params.font_name = "STHeitiMedium.ttc"
        font_path = os.path.join(utils.font_dir(), params.font_name)
        if os.name == "nt":
            font_path = font_path.replace("\\", "/")

        logger.info(f"  ⑤ font: {font_path}")

    def resolve_subtitle_background_color():
        if isinstance(params.text_background_color, bool):
            return "#000000" if params.text_background_color else None
        return params.text_background_color

    def parse_markup(text):
        pattern = re.compile(r'<(color[1-3])>(.*?)</\1>', re.DOTALL)
        segments = []
        last_idx = 0
        for match in pattern.finditer(text):
            start, end = match.span()
            if start > last_idx:
                segments.append((text[last_idx:start], None))
            tag = match.group(1)
            content = match.group(2)
            segments.append((content, tag))
            last_idx = end
        if last_idx < len(text):
            segments.append((text[last_idx:], None))
        return segments

    def split_segments_by_lines(segments, wrapped_plain_text):
        lines = []
        wrapped_lines = wrapped_plain_text.split('\n')
        seg_idx = 0
        char_in_seg_idx = 0
        
        for line in wrapped_lines:
            line_len = len(line)
            chars_needed = line_len
            line_segs = []
            
            while chars_needed > 0 and seg_idx < len(segments):
                seg_text, style = segments[seg_idx]
                seg_left = len(seg_text) - char_in_seg_idx
                
                if seg_left <= chars_needed:
                    line_segs.append((seg_text[char_in_seg_idx:], style))
                    chars_needed -= seg_left
                    seg_idx += 1
                    char_in_seg_idx = 0
                else:
                    line_segs.append((seg_text[char_in_seg_idx:char_in_seg_idx + chars_needed], style))
                    char_in_seg_idx += chars_needed
                    chars_needed = 0
            
            if not line_segs and line_len == 0:
                line_segs.append(("", None))
            lines.append(line_segs)
        return lines

    def create_text_clip(subtitle_item):
        params.font_size = int(params.font_size)
        params.stroke_width = int(params.stroke_width)
        phrase = subtitle_item[1]
        
        margin_x_ratio = getattr(params, "text_margin_x", 0.05)
        max_width = int(video_width * (1.0 - 2 * margin_x_ratio))
        
        bg_color = resolve_subtitle_background_color()
        rounded_bg_enabled = bool(
            getattr(params, "rounded_subtitle_background", False) and bg_color
        )
        has_subtitle_background = bool(bg_color)
        
        pad_x = int(params.font_size * 0.6) if has_subtitle_background else 0
        text_max_width = max(1, max_width - 2 * pad_x)
        
        segments = parse_markup(phrase)
        plain_text = "".join(text for text, _ in segments)
        
        wrapped_plain, txt_height = wrap_text(
            plain_text,
            max_width=text_max_width,
            font=font_path,
            fontsize=params.font_size,
        )
        
        lines_segs = split_segments_by_lines(segments, wrapped_plain)
        
        interline = int(params.font_size * 0.25)
        vertical_padding = int(params.font_size * 0.35)
        image_height = int(txt_height + 2 * vertical_padding + (interline * (len(lines_segs) - 1)))
        if image_height <= 0:
            image_height = params.font_size + 2 * vertical_padding
            
        pil_img = Image.new("RGBA", (max_width, image_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(pil_img)
        font = ImageFont.truetype(font_path, params.font_size)
        
        line_widths = []
        for line_segs in lines_segs:
            w = 0
            for text, _ in line_segs:
                w += font.getlength(text)
            line_widths.append(w)
            
        if bg_color:
            h = bg_color.lstrip('#')
            rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
            alpha = 140 if rounded_bg_enabled else 255
            fill_color = rgb + (alpha,)
            
            radius = max(8, int(params.font_size * 0.4)) if rounded_bg_enabled else 0
            
            for idx, (line_segs, line_w) in enumerate(zip(lines_segs, line_widths)):
                line_x = (max_width - line_w) / 2
                line_y = vertical_padding + idx * (params.font_size + interline)
                
                pad_x_bg = int(params.font_size * 0.6) if radius > 0 else int(params.font_size * 0.3)
                pad_y_bg = int(params.font_size * 0.15)
                
                left = max(0, line_x - pad_x_bg)
                top = max(0, line_y - pad_y_bg)
                right = min(max_width, line_x + line_w + pad_x_bg)
                bottom = min(image_height, line_y + params.font_size + pad_y_bg)
                
                if radius > 0:
                    draw.rounded_rectangle([left, top, right, bottom], radius=radius, fill=fill_color)
                else:
                    draw.rectangle([left, top, right, bottom], fill=fill_color)
                    
        for idx, line_segs in enumerate(lines_segs):
            line_w = line_widths[idx]
            x = (max_width - line_w) / 2
            y = vertical_padding + idx * (params.font_size + interline)
            
            for text, style in line_segs:
                fore_color = params.text_fore_color or "#FFFFFF"
                stroke_color = params.stroke_color or "#000000"
                stroke_width = int(params.stroke_width) if params.stroke_width is not None else 1
                
                if getattr(params, "text_color_highlight_enabled", False) and style:
                    if style == "color1":
                        fore_color = getattr(params, "color1_fore", "#FF3B30")
                        stroke_color = getattr(params, "color1_stroke", "#000000")
                        stroke_width = int(getattr(params, "color1_stroke_width", 1.5))
                    elif style == "color2":
                        fore_color = getattr(params, "color2_fore", "#007AFF")
                        stroke_color = getattr(params, "color2_stroke", "#FFFFFF")
                        stroke_width = int(getattr(params, "color2_stroke_width", 1.5))
                    elif style == "color3":
                        fore_color = getattr(params, "color3_fore", "#FFCC00")
                        stroke_color = getattr(params, "color3_stroke", "#000000")
                        stroke_width = int(getattr(params, "color3_stroke_width", 1.5))
                        
                draw.text((x, y), text, font=font, fill=fore_color, stroke_width=stroke_width, stroke_fill=stroke_color)
                x += font.getlength(text)
                
        img_np = np.array(pil_img)
        _clip = ImageClip(img_np, transparent=True)
        
        duration = subtitle_item[0][1] - subtitle_item[0][0]
        _clip = _clip.with_start(subtitle_item[0][0])
        _clip = _clip.with_end(subtitle_item[0][1])
        _clip = _clip.with_duration(duration)
        
        margin_y_ratio = getattr(params, "text_margin_y", 0.1)
        
        if params.subtitle_position == "bottom":
            _clip = _clip.with_position(("center", video_height * (1.0 - margin_y_ratio) - _clip.h))
        elif params.subtitle_position == "top":
            _clip = _clip.with_position(("center", video_height * margin_y_ratio))
        elif params.subtitle_position == "custom":
            margin = 10
            max_y = video_height - _clip.h - margin
            min_y = margin
            custom_y = (video_height - _clip.h) * (params.custom_position / 100)
            custom_y = max(min_y, min(custom_y, max_y))
            _clip = _clip.with_position(("center", custom_y))
        else:
            _clip = _clip.with_position(("center", "center"))
            
        return _clip

    video_clip = _open_video_clip_quietly(video_path)
    
    # 2レイヤー背景合成処理
    bg_video_type = getattr(params, "bg_video_type", "none")
    bg_video_file = getattr(params, "bg_video_file", "")
    bg_file = get_bg_video_file(bg_video_type, bg_video_file)
    
    if bg_file:
        try:
            logger.info(f"applying 2-layer background synthesis: {bg_file}")
            bg_clip = _open_video_clip_quietly(bg_file)
            bg_clip = bg_clip.with_effects([vfx.Loop(duration=video_clip.duration)])
            
            bg_w, bg_h = bg_clip.size
            if bg_w != video_width or bg_h != video_height:
                bg_ratio = bg_w / bg_h
                video_ratio = video_width / video_height
                if bg_ratio > video_ratio:
                    scale_factor = video_height / bg_h
                else:
                    scale_factor = video_width / bg_w
                new_width = int(bg_w * scale_factor)
                new_height = int(bg_h * scale_factor)
                bg_clip = bg_clip.resized(new_size=(new_width, new_height))
                bg_clip = bg_clip.cropped(
                    width=video_width,
                    height=video_height,
                    x_center=new_width / 2,
                    y_center=new_height / 2
                )
            
            main_w, main_h = video_clip.size
            main_ratio = main_w / main_h
            video_ratio = video_width / video_height
            if main_w != video_width or main_h != video_height:
                if main_ratio > video_ratio:
                    scale_factor = video_width / main_w
                else:
                    scale_factor = video_height / main_h
                new_width = int(main_w * scale_factor)
                new_height = int(main_h * scale_factor)
                video_clip_resized = video_clip.resized(new_size=(new_width, new_height))
            else:
                video_clip_resized = video_clip
                
            video_clip = CompositeVideoClip([bg_clip, video_clip_resized.with_position("center")])
        except Exception as e:
            logger.error(f"failed to apply 2-layer background synthesis: {str(e)}")

    audio_clip = AudioFileClip(audio_path).with_effects(
        [afx.MultiplyVolume(params.voice_volume)]
    )

    def make_textclip(text):
        return TextClip(
            text=text,
            font=font_path,
            font_size=params.font_size,
        )

    text_clips = []
    
    # タイトル表示
    video_title = getattr(params, "video_title", "")
    if video_title:
        try:
            margin_x_ratio = getattr(params, "text_margin_x", 0.05)
            margin_y_ratio = getattr(params, "text_margin_y", 0.1)
            title_max_width = int(video_width * (1.0 - 2 * margin_x_ratio))
            title_font_size = int(params.font_size * 1.2)
            
            segments = parse_markup(video_title)
            plain_title = "".join(text for text, _ in segments)
            
            wrapped_plain, title_height = wrap_text(
                plain_title,
                max_width=title_max_width,
                font=font_path,
                fontsize=title_font_size,
            )
            
            title_lines_segs = split_segments_by_lines(segments, wrapped_plain)
            
            title_interline = int(title_font_size * 0.25)
            title_vertical_padding = 10
            title_img_height = int(title_height + 2 * title_vertical_padding + (title_interline * (len(title_lines_segs) - 1)))
            if title_img_height <= 0:
                title_img_height = title_font_size + 2 * title_vertical_padding
                
            title_pil = Image.new("RGBA", (title_max_width, title_img_height), (0, 0, 0, 0))
            title_draw = ImageDraw.Draw(title_pil)
            title_font = ImageFont.truetype(font_path, title_font_size)
            
            line_widths = []
            for line_segs in title_lines_segs:
                w = 0
                for text, _ in line_segs:
                    w += title_font.getlength(text)
                line_widths.append(w)
                
            for idx, line_segs in enumerate(title_lines_segs):
                line_w = line_widths[idx]
                x = (title_max_width - line_w) / 2
                y = title_vertical_padding + idx * (title_font_size + title_interline)
                
                for text, style in line_segs:
                    fore_color = params.text_fore_color or "#FFFFFF"
                    stroke_color = params.stroke_color or "#000000"
                    stroke_width = int(params.stroke_width) if params.stroke_width is not None else 1
                    
                    if getattr(params, "text_color_highlight_enabled", False) and style:
                        if style == "color1":
                            fore_color = getattr(params, "color1_fore", "#FF3B30")
                            stroke_color = getattr(params, "color1_stroke", "#000000")
                            stroke_width = int(getattr(params, "color1_stroke_width", 1.5))
                        elif style == "color2":
                            fore_color = getattr(params, "color2_fore", "#007AFF")
                            stroke_color = getattr(params, "color2_stroke", "#FFFFFF")
                            stroke_width = int(getattr(params, "color2_stroke_width", 1.5))
                        elif style == "color3":
                            fore_color = getattr(params, "color3_fore", "#FFCC00")
                            stroke_color = getattr(params, "color3_stroke", "#000000")
                            stroke_width = int(getattr(params, "color3_stroke_width", 1.5))
                            
                    title_draw.text((x, y), text, font=title_font, fill=fore_color, stroke_width=stroke_width, stroke_fill=stroke_color)
                    x += title_font.getlength(text)
                    
            title_clip = ImageClip(np.array(title_pil), transparent=True)
            title_clip = title_clip.with_duration(video_clip.duration).with_position(("center", video_height * margin_y_ratio))
            text_clips.append(title_clip)
            logger.info(f"added title clip: {video_title}")
        except Exception as e:
            logger.error(f"failed to add title clip: {str(e)}")

    if subtitle_path and os.path.exists(subtitle_path):
        sub = SubtitlesClip(
            subtitles=subtitle_path, encoding="utf-8", make_textclip=make_textclip
        )
        for item in sub.subtitles:
            clip = create_text_clip(subtitle_item=item)
            text_clips.append(clip)
            
    if text_clips:
        video_clip = CompositeVideoClip([video_clip, *text_clips])

    bgm_file = get_bgm_file(bgm_type=params.bgm_type, bgm_file=params.bgm_file)
    if bgm_file:
        try:
            bgm_clip = AudioFileClip(bgm_file).with_effects(
                [
                    afx.MultiplyVolume(params.bgm_volume),
                    afx.AudioFadeOut(3),
                    afx.AudioLoop(duration=video_clip.duration),
                ]
            )
            audio_clip = CompositeAudioClip([audio_clip, bgm_clip])
        except Exception as e:
            logger.error(f"failed to add bgm: {str(e)}")

    video_clip = video_clip.with_audio(audio_clip)
    output_audio_fps = int(getattr(audio_clip, "fps", 0) or 44100)



    # 显式沿用输入音频的采样率；如果取不到，再回退到 MoviePy 默认的 44100Hz。
    # 这样可以减少不同运行环境，尤其是 Docker 环境中再次重采样带来的音质波动。
    output_audio_fps = int(getattr(audio_clip, "fps", 0) or 44100)
    _write_videofile_with_codec_fallback(
        video_clip,
        output_file=output_file,
        codec=_get_configured_video_codec(),
        audio_codec=audio_codec,
        audio_fps=output_audio_fps,
        audio_bitrate=audio_bitrate,
        temp_audiofile_path=output_dir,
        threads=params.n_threads or 2,
        logger=None,
        fps=fps,
    )
    video_clip.close()
    del video_clip


def preprocess_video(materials: List[MaterialInfo], clip_duration=4):
    # WebUI 在某些二次生成场景下可能传入空素材列表，这里直接返回空结果，避免抛出 NoneType 异常。
    if not materials:
        return []

    # 仅返回通过预处理校验的素材，避免低分辨率图片继续进入后续的视频合成流程。
    valid_materials = []
    local_videos_dir = utils.storage_dir("local_videos", create=True)

    for material in materials:
        if not material.url:
            continue

        try:
            material_source_path = file_security.resolve_path_within_directory(
                local_videos_dir, material.url
            )
        except ValueError as exc:
            # local video_source 的素材路径来自 API 参数，必须限制在专用素材目录。
            # 允许用户传文件名，也兼容历史返回的绝对路径，但不允许逃逸到系统
            # 其他目录，避免任意文件读取或通过 MoviePy 探测本地敏感文件。
            logger.warning(
                f"skip unsafe local material: {material.url}, "
                f"local_videos_dir: {local_videos_dir}, error: {str(exc)}"
            )
            continue

        ext = utils.parse_extension(material_source_path)
        try:
            # 图片素材直接按图片方式读取，避免先走 VideoFileClip 误判后触发不稳定的回退分支。
            if ext in const.FILE_TYPE_IMAGES:
                clip, material_source_path = _open_image_clip_with_fallback(
                    material_source_path
                )
            else:
                clip = _open_video_clip_quietly(material_source_path)
        except Exception:
            # 非标准扩展名或探测失败时再回退到图片模式，兼容历史上直接传本地图片路径的情况。
            try:
                clip, material_source_path = _open_image_clip_with_fallback(
                    material_source_path
                )
            except Exception as exc:
                logger.warning(
                    f"skip unreadable local material: {material.url}, error: {str(exc)}"
                )
                continue
        try:
            width = clip.size[0]
            height = clip.size[1]
            if width < 480 or height < 480:
                logger.warning(f"low resolution material: {width}x{height}, minimum 480x480 required")
                # 探测到低分辨率素材后立即关闭资源，并且不要把该素材返回给后续流程。
                close_clip(clip)
                continue

            if ext in const.FILE_TYPE_IMAGES:
                logger.info(f"processing image: {material_source_path}")
                # 探测尺寸时已经打开过一次素材，这里先释放探测句柄，再重新创建用于导出的图片 clip。
                close_clip(clip)
                # Create an image clip and set its duration to 3 seconds
                clip = (
                    ImageClip(material_source_path)
                    .with_duration(clip_duration)
                    .with_position("center")
                )
                # Apply a zoom effect using the resize method.
                # A lambda function is used to make the zoom effect dynamic over time.
                # The zoom effect starts from the original size and gradually scales up to 120%.
                # t represents the current time, and clip.duration is the total duration of the clip (3 seconds).
                # Note: 1 represents 100% size, so 1.2 represents 120% size.
                zoom_clip = clip.resized(
                    lambda t: 1 + (clip_duration * 0.03) * (t / clip.duration)
                )

                # Optionally, create a composite video clip containing the zoomed clip.
                # This is useful when you want to add other elements to the video.
                final_clip = CompositeVideoClip([zoom_clip])

                # Output the video to a file.
                video_file = f"{material_source_path}.mp4"
                final_clip.write_videofile(video_file, fps=30, logger=None)
                close_clip(clip)
                close_clip(final_clip)
                material.url = video_file
                logger.success(f"image processed: {video_file}")
            else:
                # 普通视频素材只需要读取尺寸做校验，校验完成后立即释放句柄即可。
                close_clip(clip)
                # Update url to the resolved absolute path so that downstream
                # stages (combine_videos) can open the file without re-resolving.
                material.url = material_source_path
        except Exception:
            close_clip(clip)
            raise

        valid_materials.append(material)

    return valid_materials
