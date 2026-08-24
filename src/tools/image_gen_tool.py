"""华师新生形象生成工具 - 调用图片生成API生成角色形象"""
import os
import logging
import requests
from langchain.tools import tool
from coze_coding_dev_sdk import ImageGenerationClient
from coze_coding_utils.log.write_log import request_context
from coze_coding_utils.runtime_ctx.context import new_context

logger = logging.getLogger(__name__)


@tool
def generate_character_image(prompt: str, style: str = "3d_pixar") -> str:
    """根据描述生成华师新生卡通形象图片。

    Args:
        prompt: 生图提示词，描述角色形象、专业特征、性格爱好等。
                例如: "a cute college freshman, major in Computer Science, wearing glasses, holding a book, cheerful expression"
        style: 画风预设，可选值: 3d_pixar(3D皮克斯), anime(日系动漫), watercolor(水彩), flat_design(扁平插画), ghibli(吉卜力)

    Returns:
        生成的图片URL地址
    """
    ctx = request_context.get() or new_context(method="generate_character_image")

    # 加载风格配置
    import json
    workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
    config_path = os.path.join(workspace_path, "assets", "postcard_config.json")

    style_suffix = ""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        presets = cfg.get("style_presets", {})
        if style in presets:
            style_suffix = presets[style].get("prompt_suffix", "")
            logger.info(f"Using style preset: {style} - {presets[style].get('name', '')}")
        else:
            logger.warning(f"Style '{style}' not found, using default")
            default_style = cfg.get("default_style", "3d_pixar")
            if default_style in presets:
                style_suffix = presets[default_style].get("prompt_suffix", "")
    except Exception as e:
        logger.error(f"Failed to load postcard config: {e}")

    # 组装完整prompt
    full_prompt = f"{prompt}, {style_suffix}" if style_suffix else prompt
    logger.info(f"Final image generation prompt: {full_prompt}")

    try:
        client = ImageGenerationClient(ctx=ctx)
        response = client.generate(
            prompt=full_prompt,
            size="2K",
            watermark=False,
        )

        if response.success and response.image_urls:
            image_url = response.image_urls[0]
            logger.info(f"Image generated successfully: {image_url}")
            return f"图片生成成功！图片URL: {image_url}"
        else:
            error_msg = response.error_messages if hasattr(response, 'error_messages') else "Unknown error"
            logger.error(f"Image generation failed: {error_msg}")
            return f"图片生成失败: {error_msg}"
    except Exception as e:
        logger.error(f"Image generation error: {e}")
        return f"图片生成异常: {str(e)}"


@tool
def generate_from_selfie(image_url: str, prompt: str) -> str:
    """根据上传的自拍照片生成卡通形象（图生图）。

    Args:
        image_url: 自拍图片的URL地址
        prompt: 对角色的补充描述，如专业、性格等。
                例如: "as a Computer Science major student, cheerful and energetic"

    Returns:
        生成的卡通形象图片URL地址
    """
    ctx = request_context.get() or new_context(method="generate_from_selfie")

    import json
    workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
    config_path = os.path.join(workspace_path, "assets", "postcard_config.json")

    style_suffix = ""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        default_style = cfg.get("default_style", "3d_pixar")
        presets = cfg.get("style_presets", {})
        if default_style in presets:
            style_suffix = presets[default_style].get("prompt_suffix", "")
    except Exception as e:
        logger.error(f"Failed to load postcard config: {e}")

    full_prompt = f"{prompt}, {style_suffix}" if style_suffix else prompt
    logger.info(f"Selfie-to-image prompt: {full_prompt}")

    try:
        client = ImageGenerationClient(ctx=ctx)
        response = client.generate(
            prompt=full_prompt,
            image=image_url,
            size="2K",
            watermark=False,
        )

        if response.success and response.image_urls:
            image_url_out = response.image_urls[0]
            logger.info(f"Selfie image generated successfully: {image_url_out}")
            return f"卡通形象生成成功！图片URL: {image_url_out}"
        else:
            error_msg = response.error_messages if hasattr(response, 'error_messages') else "Unknown error"
            logger.error(f"Selfie image generation failed: {error_msg}")
            return f"卡通形象生成失败: {error_msg}"
    except Exception as e:
        logger.error(f"Selfie image generation error: {e}")
        return f"卡通形象生成异常: {str(e)}"
