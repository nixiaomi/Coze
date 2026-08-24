"""学生卡合成工具 - 将生成的角色形象与华南师范大学学生卡模板合成"""
import os
import json
import logging
import requests
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from langchain.tools import tool
from coze_coding_dev_sdk.s3 import S3SyncStorage
from coze_coding_utils.log.write_log import request_context
from coze_coding_utils.runtime_ctx.context import new_context

logger = logging.getLogger(__name__)

# 常见中文字体候选路径（兼容 Linux 部署环境与 Windows 本地环境）
_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
    "/System/Library/Fonts/PingFang.ttc",
]


def _load_student_card_config():
    """加载学生卡配置"""
    workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
    config_path = os.path.join(workspace_path, "assets", "postcard_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _hex_to_rgb(hex_color: str) -> tuple:
    """将十六进制颜色转为 RGB 三元组"""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def _find_font() -> str:
    """查找可用的中文字体路径"""
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    """加载指定字号的中文字体，找不到时回退到默认字体"""
    path = _find_font()
    if path:
        try:
            return ImageFont.truetype(path, size)
        except Exception as e:
            logger.error(f"Font load error for {path}: {e}")
    return ImageFont.load_default()


def _create_gradient_background(width, height, color_start, color_end):
    """创建垂直渐变背景"""
    r1, g1, b1 = _hex_to_rgb(color_start)
    r2, g2, b2 = _hex_to_rgb(color_end)
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)
    for y in range(height):
        ratio = y / max(height - 1, 1)
        r = int(r1 + (r2 - r1) * ratio)
        g = int(g1 + (g2 - g1) * ratio)
        b = int(b1 + (b2 - b1) * ratio)
        draw.line([(0, y), (width, y)], fill=(r, g, b))
    return img


def _wrap_text(text: str, font, max_width: int) -> list:
    """按字符自动换行"""
    lines = []
    current_line = ""
    for char in text:
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


def _paste_logo(bg, logo_path: str, max_width: int):
    """将学院 logo 贴到背景顶部，返回 logo 底部 y 坐标"""
    if not logo_path or not os.path.exists(logo_path):
        return 40
    try:
        logo = Image.open(logo_path).convert("RGBA")
        # 缩放到指定最大宽度
        if logo.width > max_width:
            new_h = int(logo.height * max_width / logo.width)
            logo = logo.resize((max_width, new_h), Image.Resampling.LANCZOS)
        x = (bg.width - logo.width) // 2
        y = 36
        bg.paste(logo, (x, y), logo)
        return y + logo.height
    except Exception as e:
        logger.error(f"Logo paste error: {e}")
        return 40


def _paste_circular_photo(bg, photo, center_x, center_y, size):
    """将角色照片以圆形（带白边）方式贴到背景"""
    photo = photo.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    mask = Image.new("L", (size, size), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.ellipse((0, 0, size, size), fill=255)
    top_left = (center_x - size // 2, center_y - size // 2)
    bg.paste(photo, top_left, mask)
    # 白色描边圆环
    ring = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ring_draw = ImageDraw.Draw(ring)
    ring_draw.ellipse((3, 3, size - 3, size - 3), outline=(255, 255, 255, 220), width=6)
    bg.paste(ring, top_left, ring)


def _compose_student_card(character_img, name, major, gender, wish, config) -> Image.Image:
    """合成学生卡"""
    school = config["school"]
    sc = config["student_card"]
    width = sc["width"]
    height = sc["height"]

    # 深蓝渐变背景
    bg = _create_gradient_background(width, height, sc["bg_color_start"], sc["bg_color_end"])
    draw = ImageDraw.Draw(bg)

    accent = _hex_to_rgb(sc["accent_color"])
    gold = _hex_to_rgb(sc["gold_color"])
    white = _hex_to_rgb(sc["text_color"])
    sub = _hex_to_rgb(sc["sub_text_color"])

    # 金色描边外框
    draw.rounded_rectangle((12, 12, width - 12, height - 12), radius=24,
                           outline=gold, width=3)

    # 顶部学院 logo
    workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
    logo_path = os.path.join(workspace_path, "assets", "school_logo.png")
    logo_bottom = _paste_logo(bg, logo_path, sc["logo_max_width"])

    # 英文校名
    en_font = _load_font(22)
    en_text = school["name_en"]
    en_bbox = en_font.getbbox(en_text)
    en_w = en_bbox[2] - en_bbox[0]
    draw.text(((width - en_w) // 2, logo_bottom + 10), en_text, fill=sub, font=en_font)

    # 分隔装饰线
    line_y = logo_bottom + 52
    line_margin = 80
    draw.line([(line_margin, line_y), (width // 2 - 60, line_y)], fill=gold, width=2)
    draw.ellipse((width // 2 - 8, line_y - 8, width // 2 + 8, line_y + 8), fill=gold)
    draw.line([(width // 2 + 60, line_y), (width - line_margin, line_y)], fill=gold, width=2)

    # 左侧圆形照片
    photo_size = sc["photo_size"]
    photo_center_x = 250
    photo_center_y = (line_y + height) // 2
    _paste_circular_photo(bg, character_img, photo_center_x, photo_center_y, photo_size)

    # 右侧信息区
    info_x = 460
    info_right = width - 60

    name_font = _load_font(sc["name_font_size"])
    body_font = _load_font(sc["body_font_size"])
    small_font = _load_font(sc["small_font_size"])

    # 姓名
    y = line_y + 46
    if name:
        draw.text((info_x, y), name, fill=white, font=name_font)
        y += name_font.getbbox(name)[3] - name_font.getbbox(name)[1] + 24

    # 专业
    if major:
        draw.text((info_x, y), "专业", fill=gold, font=small_font)
        y += 30
        draw.text((info_x, y), major, fill=white, font=body_font)
        y += body_font.getbbox(major)[3] - body_font.getbbox(major)[1] + 26

    # 学院 + 性别
    college = school["college"]
    gender_text = {"male": "男", "female": "女"}.get(gender, "")
    tag = f"{college}" + (f"  ·  {gender_text}" if gender_text else "")
    draw.text((info_x, y), tag, fill=sub, font=small_font)
    y += small_font.getbbox(tag)[3] - small_font.getbbox(tag)[1] + 30

    # 寄语
    if wish:
        draw.text((info_x, y), "开学寄语", fill=gold, font=small_font)
        y += 34
        wish_lines = _wrap_text(wish, body_font, info_right - info_x)
        for line in wish_lines[:3]:
            draw.text((info_x, y), line, fill=white, font=body_font)
            y += body_font.getbbox(line)[3] - body_font.getbbox(line)[1] + 14

    # 底部校训
    motto_font = _load_font(22)
    motto = school["motto"]
    motto_bbox = motto_font.getbbox(motto)
    motto_w = motto_bbox[2] - motto_bbox[0]
    draw.text(((width - motto_w) // 2, height - 52), motto, fill=sub, font=motto_font)

    return bg


@tool
def compose_postcard(image_url: str, name: str, major: str, gender: str, wish: str) -> str:
    """将生成的角色形象与华南师范大学学生卡模板合成，输出高清学生卡图片。

    Args:
        image_url: 已生成的角色形象图片 URL
        name: 学生姓名或昵称
        major: 专业名称，如"软件工程"
        gender: 性别，可选值 "male" / "female" / ""（空字符串表示未知）
        wish: 开学寄语，如"希望付出总有回报"

    Returns:
        合成的学生卡图片下载 URL
    """
    ctx = request_context.get() or new_context(method="compose_postcard")

    try:
        # 下载角色图片
        logger.info(f"Downloading character image from: {image_url[:80]}...")
        resp = requests.get(image_url, timeout=60)
        resp.raise_for_status()
        character_img = Image.open(BytesIO(resp.content)).convert("RGB")

        # 加载配置并合成学生卡
        config = _load_student_card_config()
        card = _compose_student_card(character_img, name, major, gender, wish, config)

        # 保存到临时文件
        os.makedirs("/tmp", exist_ok=True)
        local_path = f"/tmp/scnu_card_{os.getpid()}.png"
        card.save(local_path, "PNG")
        logger.info(f"Student card saved locally: {local_path}")

        # 上传到对象存储
        storage = S3SyncStorage()
        with open(local_path, "rb") as f:
            key = storage.upload_file(
                file_content=f.read(),
                file_name="scnu_student_card.png",
                content_type="image/png",
            )
        file_path = storage.generate_presigned_url(key=key)
        logger.info(f"Student card uploaded: {file_path}")

        return f"学生卡合成成功！下载链接: {file_path}"

    except Exception as e:
        logger.error(f"compose_postcard error: {e}", exc_info=True)
        return f"学生卡合成失败: {str(e)}"