"""华师新生卡 Web 服务 - 前端页面 + API 代理"""
import os
import re
import json
import uuid
import logging
import requests
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
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
    else:
        logger.warning(f".env file not found at: {env_path}")


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
    """代理调用 Agent API，生成明信片。"""
    agent_api_url = os.getenv("AGENT_API_URL", "https://rgmx4tkpcs.coze.site/stream_run")
    agent_token = os.getenv("AGENT_API_TOKEN", "")
    project_id = int(os.getenv("AGENT_PROJECT_ID", "7677355551965216803"))

    logger.info(f"Token configured: {'Yes' if agent_token else 'No'} (length: {len(agent_token)})")
    logger.info(f"Agent API URL: {agent_api_url}")
    logger.info(f"Project ID: {project_id}")

    if not agent_token:
        logger.error("AGENT_API_TOKEN not configured!")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": "服务端未配置 API Token，请检查 web/.env 文件"}
        )

    session_id = req.session_id or str(uuid.uuid4())

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

    logger.info(f"Calling Agent API, session: {session_id}")
    logger.info(f"Query: {req.query[:80]}...")

    try:
        response = requests.post(
            agent_api_url,
            headers=headers,
            json=payload,
            stream=True,
            timeout=300
        )

        logger.info(f"Agent API response status: {response.status_code}")

        if response.status_code != 200:
            error_body = response.text[:500]
            logger.error(f"Agent API returned {response.status_code}: {error_body}")
            return JSONResponse(
                status_code=502,
                content={"success": False, "error": f"Agent API 返回 {response.status_code}: {error_body[:200]}"}
            )

        # 解析流式响应
        full_content = ""
        raw_lines = []

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue

            line_str = line.strip()
            raw_lines.append(line_str)

            # SSE 格式: data: {...}
            if line_str.startswith("data:"):
                data_str = line_str[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    _extract_content_from_data(data, full_content)
                    full_content += _get_text_from_data(data)
                except json.JSONDecodeError:
                    full_content += data_str
            else:
                # 尝试直接解析 JSON
                try:
                    data = json.loads(line_str)
                    full_content += _get_text_from_data(data)
                except json.JSONDecodeError:
                    # 纯文本行
                    if not line_str.startswith("event:") and not line_str.startswith("id:"):
                        full_content += line_str

        logger.info(f"Parsed content length: {len(full_content)} chars")

        # 如果内容为空，打印原始响应前几行用于调试
        if not full_content.strip():
            logger.warning("No content parsed. Raw lines sample:")
            for rl in raw_lines[:10]:
                logger.warning(f"  RAW: {rl[:200]}")

        # 提取图片 URL
        character_url, postcard_url = _extract_image_urls(full_content)

        if character_url and postcard_url:
            return JSONResponse(content={
                "success": True,
                "character_image_url": character_url,
                "postcard_url": postcard_url,
                "message": "明信片生成成功！"
            })
        elif character_url:
            return JSONResponse(content={
                "success": True,
                "character_image_url": character_url,
                "postcard_url": character_url,
                "message": "卡通形象已生成"
            })
        else:
            # 尝试从原始行中提取（兜底）
            raw_text = "\n".join(raw_lines)
            character_url, postcard_url = _extract_image_urls(raw_text)
            if character_url and postcard_url:
                return JSONResponse(content={
                    "success": True,
                    "character_image_url": character_url,
                    "postcard_url": postcard_url,
                    "message": "明信片生成成功！"
                })

            logger.warning(f"No image URLs found. Content preview: {full_content[:300]}")
            return JSONResponse(content={
                "success": False,
                "error": "未能从Agent响应中提取到图片URL，请查看服务端日志排查"
            })

    except requests.exceptions.Timeout:
        logger.error("Agent API timeout")
        return JSONResponse(status_code=504, content={"success": False, "error": "Agent 响应超时(>5分钟)，请稍后重试"})
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error: {e}")
        return JSONResponse(status_code=502, content={"success": False, "error": f"无法连接 Agent 服务: {str(e)[:100]}"})
    except Exception as e:
        logger.error(f"Unexpected error: {type(e).__name__}: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"success": False, "error": f"服务内部错误: {str(e)}"})


def _get_text_from_data(data: dict) -> str:
    """从不同格式的响应数据中提取文本内容"""
    if not isinstance(data, dict):
        return ""

    text = ""

    # 格式1: {"content": "text"}
    if "content" in data and isinstance(data["content"], str):
        text += data["content"]

    # 格式2: {"choices": [{"delta": {"content": "text"}}]}
    choices = data.get("choices", [])
    if choices and isinstance(choices, list):
        for choice in choices:
            if isinstance(choice, dict):
                delta = choice.get("delta", {})
                if isinstance(delta, dict):
                    text += delta.get("content", "")
                msg = choice.get("message", {})
                if isinstance(msg, dict):
                    text += msg.get("content", "")

    # 格式3: {"data": {"content": "text"}}
    inner_data = data.get("data", {})
    if isinstance(inner_data, dict):
        text += inner_data.get("content", "")
        # 格式4: {"data": {"messages": [{"content": "text"}]}}
        messages = inner_data.get("messages", [])
        if messages and isinstance(messages, list):
            for msg in messages:
                if isinstance(msg, dict):
                    text += msg.get("content", "")

    # 格式5: {"text": "text"}
    text += data.get("text", "")

    # 格式6: {"output": "text"}
    text += data.get("output", "")

    return text


def _extract_content_from_data(data, content):
    """Debug helper - not used directly"""
    pass


def _extract_image_urls(content: str) -> tuple:
    """从响应文本中提取图片 URL"""
    # 匹配所有图片 URL（包括带签名参数的长 URL）
    url_pattern = r'https?://[^\s\)\"\'>\]\}\\]+(?:\.(?:png|jpg|jpeg|webp|gif))(?:\?[^\s\)\"\'>\]\}\\]+)?'
    urls = re.findall(url_pattern, content, re.IGNORECASE)

    # 去重保持顺序
    seen = set()
    unique_urls = []
    for url in urls:
        url = url.rstrip('.,;:!?)')
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)

    logger.info(f"Extracted {len(unique_urls)} unique image URLs from content")
    for i, u in enumerate(unique_urls):
        logger.info(f"  URL[{i}]: {u[:80]}...")

    character_url = ""
    postcard_url = ""

    if len(unique_urls) >= 2:
        character_url = unique_urls[0]
        postcard_url = unique_urls[-1]
    elif len(unique_urls) == 1:
        character_url = unique_urls[0]

    return character_url, postcard_url


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("WEB_PORT", "8080"))
    logger.info(f"Starting web server on port {port}")
    logger.info(f"Agent API: {os.getenv('AGENT_API_URL', 'not set')}")
    logger.info(f"Token set: {'Yes' if os.getenv('AGENT_API_TOKEN') else 'No'}")
    uvicorn.run(app, host="0.0.0.0", port=port)
