"""华师新生卡 Web 服务 - 前端页面 + API 代理"""
import os
import json
import uuid
import logging
import requests
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_env_file():
    """从 .env 文件加载环境变量"""
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    if value and not value.startswith("<"):
                        os.environ.setdefault(key, value)
        logger.info("Loaded .env file")


load_env_file()

app = FastAPI(title="华师新生卡 Web 服务")

# 挂载静态文件
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


class GenerateRequest(BaseModel):
    query: str
    style: Optional[str] = "3d_pixar"
    name: Optional[str] = ""
    major: Optional[str] = ""
    personality: Optional[str] = ""
    wish: Optional[str] = ""
    session_id: Optional[str] = ""


@app.get("/", response_class=HTMLResponse)
async def index():
    """返回前端页面"""
    html_path = os.path.join(static_dir, "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/api/generate")
async def generate_postcard(req: GenerateRequest):
    """
    代理调用 Agent API，生成明信片。
    前端发送用户信息，后端组装请求调用 Agent 的 stream_run 接口，
    解析流式响应提取图片 URL 返回给前端。
    """
    # Agent API 配置
    agent_api_url = os.getenv("AGENT_API_URL", "https://rgmx4tkpcs.coze.site/stream_run")
    agent_token = os.getenv("AGENT_API_TOKEN", "")
    project_id = int(os.getenv("AGENT_PROJECT_ID", "7677355551965216803"))

    if not agent_token:
        logger.error("AGENT_API_TOKEN not configured")
        raise HTTPException(status_code=500, detail="服务端未配置 API Token，请联系管理员")

    session_id = req.session_id or str(uuid.uuid4())

    # 构造 Agent API 请求体
    payload = {
        "content": {
            "query": {
                "prompt": [
                    {
                        "type": "text",
                        "content": {
                            "text": req.query
                        }
                    }
                ]
            }
        },
        "type": "query",
        "session_id": session_id,
        "project_id": project_id
    }

    headers = {
        "Authorization": f"Bearer {agent_token}",
        "Content-Type": "application/json"
    }

    logger.info(f"Calling Agent API, session: {session_id}, query: {req.query[:50]}...")

    try:
        # 调用 Agent API（流式响应）
        response = requests.post(
            agent_api_url,
            headers=headers,
            json=payload,
            stream=True,
            timeout=300
        )
        response.raise_for_status()

        # 解析流式响应，提取图片 URL
        character_image_url = ""
        postcard_url = ""
        full_content = ""

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue

            # 处理 SSE 格式数据
            line_str = line.strip()
            if line_str.startswith("data:"):
                data_str = line_str[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    # 提取文本内容
                    if isinstance(data, dict):
                        content = data.get("content", "")
                        if content:
                            full_content += content
                except json.JSONDecodeError:
                    # 可能是纯文本片段
                    full_content += line_str
            else:
                # 非 SSE 格式，尝试直接解析
                try:
                    data = json.loads(line_str)
                    if isinstance(data, dict):
                        content = data.get("content", "")
                        if content:
                            full_content += content
                except json.JSONDecodeError:
                    full_content += line_str

        logger.info(f"Agent response length: {len(full_content)} chars")

        # 从响应中提取图片 URL
        character_image_url, postcard_url = _extract_image_urls(full_content)

        if character_image_url and postcard_url:
            return JSONResponse(content={
                "success": True,
                "character_image_url": character_image_url,
                "postcard_url": postcard_url,
                "message": "明信片生成成功！"
            })
        elif character_image_url:
            # 只生成了角色图片，没合成明信片
            return JSONResponse(content={
                "success": True,
                "character_image_url": character_image_url,
                "postcard_url": character_image_url,
                "message": "卡通形象已生成，明信片合成中..."
            })
        else:
            logger.warning(f"No image URLs found in response. Content preview: {full_content[:200]}")
            return JSONResponse(content={
                "success": False,
                "error": "未能从Agent响应中提取到图片，请重试"
            })

    except requests.exceptions.Timeout:
        logger.error("Agent API timeout")
        raise HTTPException(status_code=504, detail="Agent 响应超时，请稍后重试")
    except requests.exceptions.RequestException as e:
        logger.error(f"Agent API error: {e}")
        raise HTTPException(status_code=502, detail=f"调用 Agent 失败: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"服务内部错误: {str(e)}")


def _extract_image_urls(content: str) -> tuple:
    """
    从 Agent 响应文本中提取图片 URL。
    Agent 会返回两张图片：角色形象 + 明信片。
    按出现顺序，第一张是角色形象，第二张是明信片。
    """
    import re

    # 匹配所有 URL（包括带签名参数的）
    url_pattern = r'https?://[^\s\)\"\'>\]]+\.(?:png|jpg|jpeg|webp)(?:\?[^\s\)\"\'>\]]+)?'
    urls = re.findall(url_pattern, content)

    # 去重保持顺序
    seen = set()
    unique_urls = []
    for url in urls:
        # 去除末尾的标点
        url = url.rstrip('.,;:!?)')
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)

    logger.info(f"Extracted {len(unique_urls)} unique image URLs")

    character_url = ""
    postcard_url = ""

    if len(unique_urls) >= 2:
        # 第一张是角色形象，第二张（通常是最后一张）是明信片
        character_url = unique_urls[0]
        postcard_url = unique_urls[-1]  # 最后一张通常是明信片
    elif len(unique_urls) == 1:
        character_url = unique_urls[0]

    return character_url, postcard_url


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("WEB_PORT", "8080"))
    logger.info(f"Starting web server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
