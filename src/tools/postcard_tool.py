"""
华南师范大学人工智能学院 - 开学电子学生卡合成工具
使用真实宣传图作为底图：
- 将生成的虚拟学生形象抠除白底后，贴在椭圆照片区域中间偏右位置（遮住建筑部分，保留左侧异木棉花枝）
- 在椭圆下方的蓝白留白区写入开学寄语："内容"——姓名
- 底图自带校徽、校名、椭圆蓝边、渐变背景、星芒装饰，代码不再绘制这些元素
"""
import os
import io
import re
import random
import textwrap
import requests
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from langchain.tools import tool
from coze_coding_dev_sdk.s3 import S3SyncStorage
from coze_coding_utils.log.write_log import request_context
from coze_coding_utils.runtime_ctx.context import new_context

# ============== 路径与布局常量 ==============
WORKSPACE = os.getenv("COZE_WORKSPACE_PATH", os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
ASSETS_DIR = os.path.join(WORKSPACE, "assets")
# ==== 底图配置 ====
# 底图已上传到对象存储（365天有效），优先从 URL 下载，不依赖本地 assets 文件
BG_PUBLIC_URL = (
    "https://coze-coding-project.tos.coze.site/coze_storage_7677382662650134580/"
    "scnu_card_bg_1d673199.png?sign=1819185009-f2e97e60d9-0-"
    "deab7f88b45dcfd05e7205e58087a249a0cb0c7644d17c42f3181b8131b95b58"
)
BG_PATH = os.path.join(ASSETS_DIR, "scnu_card_bg.png")  # 本地 fallback
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

# 椭圆精确位置（基于真实底图测量，蓝色边框）
ELLIPSE_BOX = (86, 67, 1661, 1029)  # left, top, right, bottom
ECX = (ELLIPSE_BOX[0] + ELLIPSE_BOX[2]) // 2  # 873
ECY = (ELLIPSE_BOX[1] + ELLIPSE_BOX[3]) // 2  # 548
ERX = (ELLIPSE_BOX[2] - ELLIPSE_BOX[0]) // 2  # 787
ERY = (ELLIPSE_BOX[3] - ELLIPSE_BOX[1]) // 2  # 481

# 人物放置位置：椭圆中心偏右下（遮住右侧建筑和树干，保留左侧异木棉花枝）
PERSON_CENTER_X_RATIO = 0.62  # 椭圆中心往右偏移（中间偏右）
PERSON_BOTTOM_PADDING = 6     # 人物脚离椭圆底部的距离
PERSON_WIDTH_RATIO = 0.52     # 人物宽度占椭圆宽度的比例

# 寄语区域（椭圆底部 ~ 图片底部的留白）
WISH_AREA_TOP = ELLIPSE_BOX[3] + 18
WISH_AREA_BOTTOM = 1240
WISH_TEXT_COLOR = (25, 55, 100)        # 深墨蓝
WISH_QUOTE_COLOR = (40, 110, 190)      # 引号浅蓝
WISH_SHADOW = (255, 255, 255, 180)     # 白色阴影/底衬
WISH_FONT_SIZE = 36
WISH_LINE_SPACING = 12
WISH_MAX_WIDTH_RATIO = 0.82  # 文字宽度占画面宽度比例


# ============== 工具函数 ==============
def _find_font(size: int) -> ImageFont.FreeTypeFont:
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _remove_white_bg(img: Image.Image, threshold: int = 245, edge_feather: int = 5) -> Image.Image:
    """去除图片白底，返回带透明通道的RGBA图。针对纯白/近白背景优化。"""
    import numpy as np
    src = img.convert("RGBA")
    src_arr = np.array(src).astype(int)
    h, w = src_arr.shape[:2]

    # 采样四角，如果都是浅色则为白底
    corners = [tuple(src_arr[0,0,:3]), tuple(src_arr[0,w-1,:3]), tuple(src_arr[h-1,0,:3]), tuple(src_arr[h-1,w-1,:3])]
    if not all(all(int(c) > 230 for c in corner) for corner in corners):
        threshold = 253

    r_ch, g_ch, b_ch = src_arr[:,:,0], src_arr[:,:,1], src_arr[:,:,2]
    mx = np.maximum(np.maximum(r_ch, g_ch), b_ch)
    mn = np.minimum(np.minimum(r_ch, g_ch), b_ch)
    is_white = (r_ch > threshold) & (g_ch > threshold) & (b_ch > threshold) & ((mx - mn) < 18)
    alpha_arr = np.where(is_white, 0, 255).astype(np.uint8)
    alpha = Image.fromarray(alpha_arr, mode="L")

    # 羽化边缘
    if edge_feather > 0:
        alpha = alpha.filter(ImageFilter.GaussianBlur(radius=edge_feather))

    src.putalpha(alpha)
    return src


def _fit_person_into_ellipse(person: Image.Image) -> Image.Image:
    """
    将抠好背景的人物图贴到椭圆中间偏右下位置，贴底站立。
    返回一个与底图同尺寸的透明图层。
    """
    canvas = Image.new("RGBA", (1748, 1240), (0, 0, 0, 0))

    pw = int(ERX * 2 * PERSON_WIDTH_RATIO)  # 目标宽度（约819px）
    # 按比例缩放
    w, h = person.size
    scale = pw / w
    new_w, new_h = pw, int(h * scale)
    person = person.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # 如果人物高度超出椭圆，按高度限制缩放
    max_h = int(ERY * 2 * 0.97)
    if new_h > max_h:
        scale = max_h / new_h
        new_w, new_h = int(new_w * scale), max_h
        person = person.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # 位置：右对齐（人物右边缘在椭圆约78%处），底部贴椭圆底
    pos_x = ELLIPSE_BOX[0] + int(ERX * 2 * 0.78) - new_w
    pos_y = ELLIPSE_BOX[3] - PERSON_BOTTOM_PADDING - new_h

    # 用椭圆蒙版裁切超出椭圆的部分（人物不要超出椭圆蓝框）
    mask = Image.new("L", (1748, 1240), 0)
    md = ImageDraw.Draw(mask)
    md.ellipse(ELLIPSE_BOX, fill=255)
    canvas.paste(person, (pos_x, pos_y), person)
    # 用椭圆蒙版裁掉超出椭圆的部分
    canvas.putalpha(Image.eval(canvas.getchannel("A"), lambda a: a))
    final_alpha = Image.new("L", (1748, 1240), 0)
    final_alpha.paste(canvas.getchannel("A"), (0, 0), mask)
    canvas.putalpha(final_alpha)

    return canvas


def _draw_wish(bg: Image.Image, wish: str, name: str) -> None:
    """在椭圆下方的留白区写寄语，格式："内容"——姓名。居中排版，深墨蓝文字。"""
    draw = ImageDraw.Draw(bg)

    # 清理文字：去掉多余引号和空白，保证格式统一
    quote_chars = '"\u201c\u201d\u2018\u2019\u300c\u300d\u300e\u300f\u3010\u3011[]'
    wish = wish.strip().strip(quote_chars)
    wish = wish.rstrip("。.！!？?")
    formatted_text = f"\u201c{wish}\u3002\u201d"  # 中文左引号+内容+句号+右引号
    signature = f"\u2014\u2014{name}"  # 中文破折号+姓名

    area_w = int(1748 * WISH_MAX_WIDTH_RATIO)
    area_x = (1748 - area_w) // 2

    # 自动换行（按宽度，中文按字符断行）
    font = _find_font(WISH_FONT_SIZE)
    sig_font = _find_font(int(WISH_FONT_SIZE * 0.85))

    # 断行：估算每字符宽度（中文约等于字体大小）
    char_w = WISH_FONT_SIZE * 0.95
    max_chars_per_line = max(8, int(area_w / char_w))
    lines = textwrap.wrap(formatted_text, width=max_chars_per_line)

    # 计算总高度
    line_h = WISH_FONT_SIZE + WISH_LINE_SPACING
    sig_h = int(WISH_FONT_SIZE * 0.85) + WISH_LINE_SPACING
    total_h = line_h * len(lines) + sig_h + 10

    # 垂直居中于寄语区
    area_h = WISH_AREA_BOTTOM - WISH_AREA_TOP
    start_y = WISH_AREA_TOP + (area_h - total_h) // 2

    # 画正文
    for i, line in enumerate(lines):
        bbox = font.getbbox(line)
        line_w = bbox[2] - bbox[0]
        x = (1748 - line_w) // 2
        y = start_y + i * line_h
        # 轻阴影（左下偏移半透明白色，提高在渐变背景上的可读性）
        draw.text((x + 1, y + 1), line, font=font, fill=WISH_SHADOW)
        draw.text((x, y), line, font=font, fill=WISH_TEXT_COLOR)

    # 画署名（右对齐正文块）
    last_line_bbox = font.getbbox(lines[-1])
    last_line_w = last_line_bbox[2] - last_line_bbox[0]
    sig_y = start_y + len(lines) * line_h + 8
    sig_bbox = sig_font.getbbox(signature)
    sig_w = sig_bbox[2] - sig_bbox[0]
    # 署名右对齐在正文右侧一点
    sig_x = (1748 - last_line_w) // 2 + last_line_w - sig_w - 10
    draw.text((sig_x + 1, sig_y + 1), signature, font=sig_font, fill=WISH_SHADOW)
    draw.text((sig_x, sig_y), signature, font=sig_font, fill=WISH_QUOTE_COLOR)


def _upload_image(local_path: str, filename: str) -> str:
    """上传图片到对象存储并返回公网 URL"""
    ctx = request_context.get() or new_context(method="scnu_upload")
    storage = S3SyncStorage()
    with open(local_path, "rb") as f:
        content = f.read()
    key = storage.upload_file(
        file_content=content,
        file_name=filename,
        content_type="image/png"
    )
    return storage.generate_presigned_url(key=key)


# ============== 对外工具 ==============
@tool
def compose_postcard(image_url: str, name: str, major: str, gender: str = "", wish: str = "") -> str:
    """
    将生成好的学生形象抠除白底后贴到华南师范大学人工智能学院的官方宣传底图上，并写入开学寄语，生成一张完整的电子学生卡。
    使用真实宣传图作为底图（自带校徽、校名、渐变背景、椭圆蓝边），代码只做两件事：
    1) 抠除人物白底，贴入椭圆照片区中间偏右下位置（遮住右侧建筑和树干，保留左侧异木棉花枝）
    2) 在椭圆下方蓝白留白区写寄语，格式："内容"——姓名

    参数:
        image_url: 生图工具返回的学生形象图片 URL（必须为带纯白背景的人像）
        name: 学生姓名或昵称（用于署名）
        major: 学生专业（此版本底图自带专业信息区，此参数仅保留兼容，可不传）
        gender: 学生性别（此版本不影响排版，保留兼容）
        wish: 学生的开学寄语/愿望（一句话，30字以内效果最佳）
    返回:
        合成好的学生卡图片公网下载链接
    """
    try:
        # 1. 下载人物图
        resp = requests.get(image_url, timeout=30)
        resp.raise_for_status()
        person_img = Image.open(io.BytesIO(resp.content)).convert("RGB")

        # 2. 加载真实底图：优先从公网URL下载（不依赖本地部署是否带assets）
        bg = None
        # 2a. 优先用公网URL下载（365天有效的签名URL）
        try:
            bg_resp = requests.get(BG_PUBLIC_URL, timeout=30)
            bg_resp.raise_for_status()
            bg = Image.open(io.BytesIO(bg_resp.content)).convert("RGBA")
            print(f"[postcard] 底图从公网URL加载成功，大小: {len(bg_resp.content)/1024:.0f}KB")
        except Exception as e:
            print(f"[postcard] 公网URL加载底图失败: {e}，尝试本地文件")
            # 2b. 降级到本地文件
            if os.path.exists(BG_PATH):
                bg = Image.open(BG_PATH).convert("RGBA")
                print(f"[postcard] 使用本地底图: {BG_PATH}")
            else:
                return f"错误：底图加载失败，公网URL和本地文件均不可用"
        if bg.size != (1748, 1240):
            bg = bg.resize((1748, 1240), Image.Resampling.LANCZOS)

        # 3. 抠除人物白底
        person_nobg = _remove_white_bg(person_img, threshold=245, edge_feather=5)

        # 4. 贴人物到椭圆中间偏右
        person_layer = _fit_person_into_ellipse(person_nobg)
        bg.alpha_composite(person_layer)

        # 5. 在下方留白区写寄语
        _draw_wish(bg, wish or "希望在华师度过充实美好的四年", name)

        # 6. 保存并上传
        safe_name = re.sub(r'[^\w\u4e00-\u9fa5]', '_', name)[:20]
        out_path = f"/tmp/scnu_student_card_{safe_name}_{random.randint(10**8, 10**9-1)}.png"
        bg.convert("RGB").save(out_path, "PNG")

        url = _upload_image(out_path, os.path.basename(out_path))
        return f"学生卡合成成功！下载链接: {url}"

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[postcard_tool] 合成失败: {e}\n{tb}")
        return f"学生卡合成失败: {str(e)}"
