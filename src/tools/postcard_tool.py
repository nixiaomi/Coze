"""
SCNU 华南师范大学人工智能学院 - 学生卡合成工具
基于官方宣传图底图 + 抠图人物 + 寄语排版
"""
import os
import io
import math
import requests
import re
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from langchain.tools import tool
from coze_coding_dev_sdk.s3 import S3SyncStorage
import logging
logger = logging.getLogger(__name__)

# ============== 配置 ==============
WORKSPACE = os.getenv("COZE_WORKSPACE_PATH", os.getcwd())
BG_PATH = os.path.join(WORKSPACE, "assets", "scnu_card_bg.png")
CONFIG_PATH = os.path.join(WORKSPACE, "assets", "postcard_config.json")

# 底图椭圆照片区坐标（基于 1748x1240 底图精确测量）
ELLIPSE_BOX = (200, 175, 1548, 920)  # left, top, right, bottom

# 寄语区域（椭圆下方空白）
WISH_AREA = (120, 940, 1628, 1180)

CARD_SIZE = (1748, 1240)


def _find_font(preferred_names=None, size=30):
    """查找可用中文字体"""
    if preferred_names is None:
        preferred_names = ["wqy-zenhei.ttc", "wqy-microhei.ttc"]
    search_dirs = [
        "/usr/share/fonts/truetype/wqy",
        "/usr/share/fonts/opentype/noto",
        "/usr/share/fonts/truetype/noto",
        "/usr/share/fonts",
        "C:/Windows/Fonts",
        "/System/Library/Fonts",
        "/Library/Fonts",
    ]
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for name in preferred_names:
            p = os.path.join(d, name)
            if os.path.exists(p):
                return ImageFont.truetype(p, size)
    # fallback 搜索
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for f in files:
                if f.lower().endswith((".ttf", ".ttc", ".otf")) and (
                    "wqy" in f.lower() or "noto" in f.lower() or "cjk" in f.lower()
                    or "hei" in f.lower() or "song" in f.lower() or "yahei" in f.lower()
                    or "simsun" in f.lower() or "simhei" in f.lower()
                ):
                    try:
                        return ImageFont.truetype(os.path.join(root, f), size)
                    except Exception:
                        pass
    return ImageFont.load_default()


def _load_image_from_url(url: str) -> Image.Image:
    """从 URL 下载图片"""
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGBA")


def _remove_white_bg(img: Image.Image, threshold: int = 240, edge_feather: int = 8) -> Image.Image:
    """
    去除纯色（白/近白）背景，保留人物，返回 RGBA 图。
    适用于生图时指定 "solid white background" 后再做抠图。
    - threshold: 像素亮度大于此值视为背景 (0-255)
    - edge_feather: 边缘羽化像素
    """
    img = img.convert("RGBA")
    w, h = img.size
    data = img.load()
    mask = Image.new("L", (w, h), 0)
    md = mask.load()

    for y in range(h):
        for x in range(w):
            px = data[x, y]
            if isinstance(px, (int, float)):
                # 灰度图兜底
                r = g = b = int(px)
                a = 255
            else:
                r, g, b, a = (list(px) + [255])[:4]
            # 判断是否为白色/近白背景（R/G/B 都高且接近）
            brightness = (r + g + b) / 3
            is_white = brightness > threshold and abs(r - g) < 25 and abs(g - b) < 25 and abs(r - b) < 25
            # 边缘稍微放松：浅蓝/浅灰背景也去
            is_light = brightness > threshold + 10
            if not (is_white or is_light):
                # 距离白色越远，alpha 越高
                if brightness < threshold - 20:
                    alpha_val = 255
                else:
                    alpha_val = max(0, min(255, int(255 - (brightness - (threshold-20))*3)))
                md[x, y] = alpha_val
            else:
                md[x, y] = 0

    # 羽化边缘
    if edge_feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=edge_feather/2))

    # 把 mask 贴到 alpha 通道
    result = img.copy()
    result.putalpha(mask)
    return result


def _fit_person_into_ellipse(person_img: Image.Image) -> Image.Image:
    """
    将抠好的人物图放入椭圆照片区，自动缩放适配，
    人物贴底部（类似证件照站位），超出椭圆部分裁掉。
    返回贴到全画布 RGBA 的图层。
    """
    el, et, er, eb = ELLIPSE_BOX
    ew, eh = er - el, eb - et
    canvas = Image.new("RGBA", CARD_SIZE, (0,0,0,0))

    pw, ph = person_img.size
    # 目标：人物宽度占椭圆宽度的 65-80%（避免撑满椭圆），高度自适应
    target_w = int(ew * 0.72)
    scale = target_w / pw
    target_h = int(ph * scale)
    # 如果人物高度超过椭圆高度的 90%，以高度为准缩放
    if target_h > eh * 0.92:
        scale = (eh * 0.92) / ph
        target_w = int(pw * scale)
        target_h = int(ph * scale)

    person_resized = person_img.resize((target_w, target_h), Image.Resampling.LANCZOS)

    # 居中偏右放置（和底图里建筑花树在右的视觉平衡），贴底部
    x = el + int((ew - target_w) * 0.55)  # 稍微偏右
    y = eb - target_h - int(eh * 0.04)   # 距离椭圆底边留 4% 空间

    # 用椭圆做蒙版，超出椭圆部分裁掉
    ellipse_mask = Image.new("L", CARD_SIZE, 0)
    emd = ImageDraw.Draw(ellipse_mask)
    emd.ellipse(ELLIPSE_BOX, fill=255)

    canvas.paste(person_resized, (x, y), person_resized)
    # 应用椭圆蒙版
    alpha = canvas.split()[-1]
    alpha = Image.composite(alpha, Image.new("L", CARD_SIZE, 0), ellipse_mask)
    canvas.putalpha(alpha)
    return canvas


def _draw_wish(base: Image.Image, wish: str, name: str):
    """在椭圆下方空白处绘制寄语 + 署名，竖排居中，带半透明底衬。"""
    draw = ImageDraw.Draw(base)
    wl, wt, wr, wb = WISH_AREA
    ww, wh = wr - wl, wb - wt

    # 字体大小自适应
    font_size = 44
    if len(wish) > 30:
        font_size = 38
    if len(wish) > 45:
        font_size = 34

    font_wish = _find_font(size=font_size)
    font_sign = _find_font(size=int(font_size * 0.75))

    # 寄语内容加引号
    wish_text = f"\u201c{wish}\u201d"
    sign_text = f"\u2014\u2014{name}"

    # 计算文字宽度，自动折行（寄语最多两行）
    def wrap_text(text, font, max_w):
        lines = []
        line = ""
        for ch in text:
            if draw.textlength(line + ch, font=font) <= max_w:
                line += ch
            else:
                lines.append(line)
                line = ch
        if line:
            lines.append(line)
        return lines

    lines = wrap_text(wish_text, font_wish, ww - 80)
    if len(lines) > 2:
        # 超过两行就再缩小
        font_size = int(font_size * 0.85)
        font_wish = _find_font(size=font_size)
        font_sign = _find_font(size=int(font_size * 0.75))
        lines = wrap_text(wish_text, font_wish, ww - 80)
    lines = lines[:2]

    # 半透明浅色圆角底衬（让文字在蓝底上更清晰）
    line_h = font_size + 18
    total_h = line_h * len(lines) + int(font_size * 0.75) + 30
    pad = 30
    rect_y0 = wt + (wh - total_h)//2 - pad
    rect_y1 = rect_y0 + total_h + pad*2
    rect_x0 = wl + 60
    rect_x1 = wr - 60
    overlay = Image.new("RGBA", CARD_SIZE, (0,0,0,0))
    od = ImageDraw.Draw(overlay)
    od.rounded_rectangle([rect_x0, rect_y0, rect_x1, rect_y1], radius=24,
                         fill=(255,255,255,90))
    base.alpha_composite(overlay)
    draw = ImageDraw.Draw(base)

    # 居中绘制文字
    cur_y = rect_y0 + pad
    text_color = (20, 55, 100, 240)
    for line in lines:
        tw = draw.textlength(line, font=font_wish)
        tx = wl + (ww - tw) // 2
        # 轻微阴影增加可读性
        draw.text((tx+1, cur_y+1), line, fill=(255,255,255,120), font=font_wish)
        draw.text((tx, cur_y), line, fill=text_color, font=font_wish)
        cur_y += line_h

    # 署名右对齐
    sw = draw.textlength(sign_text, font=font_sign)
    sx = rect_x1 - pad - sw
    sy = cur_y + 6
    draw.text((sx+1, sy+1), sign_text, fill=(255,255,255,100), font=font_sign)
    draw.text((sx, sy), sign_text, fill=(40, 80, 140, 230), font=font_sign)


@tool
def compose_postcard(image_url: str, name: str, major: str,
                     gender: str = "", wish: str = "", style: str = "") -> str:
    """
    将生成的卡通人物图抠去白底，贴入华南师大蓝白渐变宣传底图的椭圆区域，
    并在下方空白处写入寄语「"寄语内容"——姓名」。

    Args:
        image_url: 卡通人物图 URL（应为纯色白底，便于抠图）
        name: 学生姓名/昵称
        major: 学生专业（保留字段，当前底图已有学院信息，预留扩展用）
        gender: "male"/"female"/"" （保留字段）
        wish: 开学寄语（将在图片下方空白处以引号+署名格式展示）
        style: 画风（保留字段，不影响合成）

    Returns:
        合成后的学生卡下载 URL
    """
    try:
        if not os.path.exists(BG_PATH):
            return "学生卡底图缺失，请联系管理员。"

        # 1. 加载底图
        bg = Image.open(BG_PATH).convert("RGBA")

        # 2. 加载人物图并抠白底
        person = _load_image_from_url(image_url)
        person_nobg = _remove_white_bg(person, threshold=235, edge_feather=6)

        # 3. 把人物贴入椭圆区域
        person_layer = _fit_person_into_ellipse(person_nobg)
        bg.alpha_composite(person_layer)

        # 4. 写寄语（椭圆下方空白）
        if wish:
            _draw_wish(bg, wish, name)

        # 5. 保存并上传
        out_path = f"/tmp/scnu_student_card_{name}_{int(datetime.now().timestamp())}.png"
        bg.convert("RGB").save(out_path, "PNG")

        storage = S3SyncStorage()
        with open(out_path, "rb") as f:
            key = storage.upload_file(
                file_content=f.read(),
                file_name=os.path.basename(out_path),
                content_type="image/png"
            )
        url = storage.generate_presigned_url(key=key)

        logger.info(f"学生卡合成成功: name={name}, url={url[:60]}...")
        return f"学生卡合成成功！下载链接: {url}"

    except Exception as e:
        import traceback
        logger.error(f"学生卡合成失败: {e}\n{traceback.format_exc()}")
        return f"学生卡合成失败: {str(e)}"
