"""明信片合成工具 - 将生成的角色图片与华师明信片模板合成"""
import os
import io
import json
import logging
import requests
from PIL import Image, ImageDraw, ImageFont
from langchain.tools import tool
from coze_coding_dev_sdk.s3 import S3SyncStorage
from coze_coding_utils.log.write_log import request_context
from coze_coding_utils.runtime_ctx.context import new_context

logger = logging.getLogger(__name__)


def _load_postcard_config():
    """加载明信片配置"""
    workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
    config_path = os.path.join(workspace_path, "assets", "postcard_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _hex_to_rgb(hex_color: str) -> tuple:
    """将十六进制颜色转为RGB"""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def _create_gradient_background(width: int, height: int, color_start: str, color_end: str) -> Image.Image:
    """创建渐变背景"""
    r1, g1, b1 = _hex_to_rgb(color_start)
    r2, g2, b2 = _hex_to_rgb(color_end)
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)
    for y in range(height):
        ratio = y / height
        r = int(r1 + (r2 - r1) * ratio)
        g = int(g1 + (g2 - g1) * ratio)
        b = int(b1 + (b2 - b1) * ratio)
        draw.line([(0, y), (width, y)], fill=(r, g, b))
    return img


def _draw_rounded_rect(draw, xy: tuple, radius: int, fill: str):
    """绘制圆角矩形"""
    x1, y1, x2, y2 = xy
    draw.rectangle([x1 + radius, y1, x2 - radius, y2], fill=fill)
    draw.rectangle([x1, y1 + radius, x2, y2 - radius], fill=fill)
    draw.pieslice([x1, y1, x1 + 2*radius, y1 + 2*radius], 180, 270, fill=fill)
    draw.pieslice([x2 - 2*radius, y1, x2, y1 + 2*radius], 270, 360, fill=fill)
    draw.pieslice([x1, y2 - 2*radius, x1 + 2*radius, y2], 90, 180, fill=fill)
    draw.pieslice([x2 - 2*radius, y2 - 2*radius, x2, y2], 0, 90, fill=fill)


def _wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list:
    """文本自动换行"""
    lines = []
    words = list(text)  # 中文按字符分割
    current_line = ""
    for char in words:
        test_line = current_line + char
        bbox = font.getbbox(test_line)
        if bbox[2] - bbox[0] > max_width:
            if current_line:
                lines.append(current_line)
            current_line = char
        else:
            current_line = test_line
    if current_line:
        lines.append(current_line)
    return lines


def _compose_postcard_image(character_img: Image.Image, name: str, major: str,
                             wish: str, config: dict) -> Image.Image:
    """合成明信片"""
    pc = config["postcard"]
    width = pc["width"]
    height = pc["height"]
    margin = pc["card_margin"]
    font_path = pc["font_path"]

    # 创建渐变背景
    bg = _create_gradient_background(width, height,
                                      pc["background_color_start"],
                                      pc["background_color_end"])
    draw = ImageDraw.Draw(bg)

    # 绘制白色卡片区域
    _draw_rounded_rect(draw, (margin, margin, width - margin, height - margin),
                        pc["card_radius"], pc["card_color"])

    # 加载字体
    try:
        title_font = ImageFont.truetype(font_path, pc["title_font_size"])
        body_font = ImageFont.truetype(font_path, pc["body_font_size"])
        wish_font = ImageFont.truetype(font_path, pc["wish_font_size"])
        small_font = ImageFont.truetype(font_path, 24)
    except Exception as e:
        logger.error(f"Font load error: {e}")
        title_font = ImageFont.load_default()
        body_font = title_font
        wish_font = title_font
        small_font = title_font

    text_color = _hex_to_rgb(pc["text_color"])
    accent_color = _hex_to_rgb(pc["accent_color"])

    # 绘制顶部标题
    header = pc["header_text"]
    sub_header = pc["sub_header_text"]

    header_bbox = title_font.getbbox(header)
    header_w = header_bbox[2] - header_bbox[0]
    draw.text(((width - header_w) // 2, margin + 40), header,
              fill=accent_color, font=title_font)

    sub_bbox = small_font.getbbox(sub_header)
    sub_w = sub_bbox[2] - sub_bbox[0]
    draw.text(((width - sub_w) // 2, margin + 110), sub_header,
              fill=(*text_color[:3], 180), font=small_font)

    # 放置角色图片（居中，占卡片宽度的70%）
    card_inner_width = width - 2 * margin - 80
    img_target_w = int(card_inner_width * 0.7)
    img_ratio = character_img.height / character_img.width
    img_target_h = int(img_target_w * img_ratio)

    # 限制图片高度
    max_img_h = int(height * 0.4)
    if img_target_h > max_img_h:
        img_target_h = max_img_h
        img_target_w = int(img_target_h / img_ratio)

    character_img_resized = character_img.resize((img_target_w, img_target_h), Image.Resampling.LANCZOS)
    img_x = (width - img_target_w) // 2
    img_y = margin + 170
    bg.paste(character_img_resized, (img_x, img_y),
             character_img_resized if character_img_resized.mode == "RGBA" else None)

    # 绘制姓名
    content_y = img_y + img_target_h + 30
    if name:
        name_text = f"🎓 {name}"
        name_bbox = body_font.getbbox(name_text)
        name_w = name_bbox[2] - name_bbox[0]
        draw.text(((width - name_w) // 2, content_y), name_text,
                  fill=text_color, font=body_font)
        content_y += 55

    # 绘制专业
    if major:
        major_text = f"📚 {major}"
        major_bbox = body_font.getbbox(major_text)
        major_w = major_bbox[2] - major_bbox[0]
        draw.text(((width - major_w) // 2, content_y), major_text,
                  fill=(*text_color[:3], 200), font=body_font)
        content_y += 55

    # 绘制开学愿望
    if wish:
        content_y += 15
        # 绘制引号装饰
        wish_label = "✨ 开学愿望"
        label_bbox = small_font.getbbox(wish_label)
        label_w = label_bbox[2] - label_bbox[0]
        draw.text(((width - label_w) // 2, content_y), wish_label,
                  fill=accent_color, font=small_font)
        content_y += 40

        # 愿望文本（自动换行）
        inner_width = width - 2 * margin - 120
        wish_lines = _wrap_text(f"「{wish}」", wish_font, inner_width)
        for line in wish_lines:
            line_bbox = wish_font.getbbox(line)
            line_w = line_bbox[2] - line_bbox[0]
            draw.text(((width - line_w) // 2, content_y), line,
                      fill=text_color, font=wish_font)
            content_y += 45

    # 绘制底部标语
    footer = pc["footer_text"]
    footer_bbox = small_font.getbbox(footer)
    footer_w = footer_bbox[2] - footer_bbox[0]
    draw.text(((width - footer_w) // 2, height - margin - 50), footer,
              fill=(255, 255, 255, 200), font=small_font)

    return bg


@tool
def compose_postcard(image_url: str, name: str, major: str, wish: str) -> str:
    """将生成的角色图片与华师明信片模板合成，输出高清明信片图片。

    Args:
        image_url: 已生成的角色形象图片URL
        name: 学生姓名或昵称
        major: 专业名称，如"计算机科学与技术"
        wish: 开学愿望，如"希望在桂子山遇到志同道合的朋友"

    Returns:
        合成后的明信片图片URL
    """
    ctx = request_context.get() or new_context(method="compose_postcard")
    logger.info(f"Composing postcard for {name}, major: {major}")

    try:
        # 下载角色图片
        resp = requests.get(image_url, timeout=30)
        resp.raise_for_status()
        character_img = Image.open(io.BytesIO(resp.content)).convert("RGBA")
        logger.info(f"Character image downloaded: {character_img.size}")
    except Exception as e:
        logger.error(f"Failed to download character image: {e}")
        return f"下载角色图片失败: {str(e)}"

    try:
        # 加载配置并合成
        config = _load_postcard_config()
        postcard_img = _compose_postcard_image(character_img, name, major, wish, config)
        logger.info("Postcard composed successfully")
    except Exception as e:
        logger.error(f"Failed to compose postcard: {e}")
        return f"明信片合成失败: {str(e)}"

    try:
        # 上传到对象存储
        storage = S3SyncStorage(
            endpoint_url=os.getenv("COZE_BUCKET_ENDPOINT_URL"),
            access_key="",
            secret_key="",
            bucket_name=os.getenv("COZE_BUCKET_NAME"),
            region="cn-beijing",
        )

        buffer = io.BytesIO()
        postcard_img.save(buffer, format="PNG", quality=95)
        buffer.seek(0)

        safe_name = name.replace(" ", "_") if name else "student"
        file_name = f"ccnu_postcard_{safe_name}.png"

        key = storage.upload_file(
            file_content=buffer.read(),
            file_name=file_name,
            content_type="image/png",
        )
        logger.info(f"Postcard uploaded with key: {key}")

        # 生成签名URL（有效期7天）
        signed_url = storage.generate_presigned_url(key=key, expire_time=604800)
        logger.info(f"Postcard URL generated: {signed_url}")

        return f"明信片合成成功！下载链接: {signed_url}"
    except Exception as e:
        logger.error(f"Failed to upload postcard: {e}")
        return f"明信片上传失败: {str(e)}"
