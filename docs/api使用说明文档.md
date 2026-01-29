# VoxCPM TTS API 使用指南

本文档介绍如何从其他 Python 项目调用 VoxCPM TTS API 服务。

## 目录

- [快速开始](#快速开始)
- [HTTP API 接口](#http-api-接口)
- [WebSocket API 接口](#websocket-api-接口)
- [Python SDK](#python-sdk)
- [WebSocket Python 客户端](#websocket-python-客户端)
- [FastAPI 集成](#fastapi-集成)
- [常见问题](#常见问题)

---

## 快速开始

### 1. 启动 TTS 服务

```bash
# 在 VoxCPM 项目目录下启动服务
cd /home/zju/VoxCPM
python api.py

# 服务默认运行在 ws://0.0.0.0:8080
# HTTP 接口: http://localhost:8080
# WebSocket 接口: ws://localhost:8080
```

### 2. 验证服务

```bash
# 健康检查
curl http://localhost:8080/health
# 返回: {"status":"ok","message":"TTS service is running"}

# 获取模型信息
curl http://localhost:8080/models
```

### 3. 第一次语音合成（HTTP）

```bash
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "你好，这是语音合成测试。"}' \
  -o output.wav
```

### 4. 第一次语音合成（WebSocket，推荐）

```python
# 详见 WebSocket Python 客户端章节
```

---

## HTTP API 接口

### 健康检查

**GET** `/health`

检查服务是否正常运行。

**响应示例:**
```json
{
  "status": "ok",
  "message": "TTS service is running"
}
```

### 获取模型信息

**GET** `/models`

获取当前加载的模型信息。

**响应示例:**
```json
{
  "model_path": "/home/zju/VoxCPM/models/openbmb__VoxCPM1.5",
  "sample_rate": 44100,
  "device": "cuda",
  "dtype": "bfloat16"
}
```

### 语音合成

**POST** `/generate`

生成语音音频文件。

**请求体 (JSON):**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `text` | string | 是 | - | 目标文本 |
| `prompt_wav_path` | string | 否 | null | 参考音频路径（用于声音克隆） |
| `prompt_text` | string | 否 | null | 参考文本（需与 prompt_wav_path 同时提供） |
| `cfg_value` | number | 否 | 2.0 | CFG 值 (1.0-3.0)，值越大越贴近参考音色 |
| `inference_timesteps` | integer | 否 | 10 | 推理步数 (4-30)，值越大质量越好 |
| `normalize` | boolean | 否 | false | 是否文本正则化 |
| `denoise` | boolean | 否 | false | 是否对参考音频降噪 |

**响应:** audio/wav 文件

**请求示例:**

```bash
# 基础调用
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "你好，这是语音合成测试。"}' \
  -o output.wav

# 带参数调用
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "这是高质量语音合成。",
    "cfg_value": 2.5,
    "inference_timesteps": 20
  }' \
  -o output.wav

# 声音克隆
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "使用参考声音合成。",
    "prompt_wav_path": "/path/to/reference.wav",
    "prompt_text": "参考音频对应的文本内容。",
    "cfg_value": 2.0
  }' \
  -o output.wav
```

### 构建 Prompt Cache

**POST** `/build_cache`

构建 prompt cache，后续生成可复用，无需每次传入参考音频。

**请求体 (JSON):**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `prompt_wav_path` | string | 是 | 参考音频路径 |
| `prompt_text` | string | 是 | 参考文本 |

**响应示例:**
```json
{
  "cache_id": "abc12345",
  "message": "Prompt cache built successfully",
  "prompt_text": "参考音频对应的文本内容。"
}
```

### 使用 Cache 生成

**POST** `/generate_with_cache`

使用预构建的 prompt cache 生成语音，比普通 generate 更快。

**请求体 (JSON):**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `text` | string | 是 | - | 目标文本 |
| `prompt_cache_id` | string | 是 | - | 通过 `/build_cache` 获取的 cache_id |
| `cfg_value` | number | 否 | 2.0 | CFG 值 |
| `inference_timesteps` | integer | 否 | 10 | 推理步数 |

**请求示例:**

```bash
# 先构建 cache
curl -X POST http://localhost:8080/build_cache \
  -H "Content-Type: application/json" \
  -d '{
    "prompt_wav_path": "/home/zju/VoxCPM/examples/hailan02.mp3",
    "prompt_text": "Just by listening a few minutes a day..."
  }'

# 返回: {"cache_id": "abc12345", ...}

# 使用 cache 生成
curl -X POST http://localhost:8080/generate_with_cache \
  -H "Content-Type: application/json" \
  -d '{
    "text": "使用缓存音色生成。",
    "prompt_cache_id": "abc12345"
  }' \
  -o output.wav
```

### 管理 Cache

```bash
# 列出所有 active 的 cache
curl http://localhost:8080/caches

# 删除指定 cache
curl -X DELETE http://localhost:8080/cache/abc12345
```

---

## WebSocket API 接口

WebSocket 接口支持双向实时通信，适合需要流式响应或长连接的应用场景。

### WebSocket 连接地址

```
ws://localhost:8080/ws/generate          # 语音合成
```

### 1. 语音合成

**ws://localhost:8080/ws/generate**

客户端发送 JSON 参数，服务器返回二进制音频数据。

**发送格式 (JSON):**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| text | string | 是 | - | 目标文本 |
| prompt_wav_path | string | 否 | null | 参考音频路径（用于声音克隆） |
| prompt_text | string | 否 | null | 参考文本（需与 prompt_wav_path 同时提供） |
| cfg_value | number | 否 | 2.0 | CFG 值 (1.0-3.0) |
| inference_timesteps | integer | 否 | 10 | 推理步数 (4-30) |
| normalize | boolean | 否 | false | 是否文本正则化 |
| denoise | boolean | 否 | false | 是否对参考音频降噪 |

**接收格式:**

- 成功：先发送音频 bytes，然后发送 `{"status": "success", "audio_size": N, "sample_rate": 44100}`
- 失败：发送 `{"status": "error", "message": "错误信息"}`

**发送示例:**

```json
{
  "text": "你好，这是 WebSocket 测试。",
  "cfg_value": 2.0,
  "inference_timesteps": 10
}
```

> **注意**: Prompt Cache 相关操作（构建和使用 Cache）目前仅支持 HTTP 接口，不支持 WebSocket。请使用 HTTP API `/build_cache` 和 `/generate_with_cache`。

---

---

## Python SDK

在调用项目中安装依赖并使用：

### 1. 安装依赖

```bash
pip install requests
```

### 2. 创建 TTS 客户端

```python
# tts_client.py
import requests
import os
from typing import Optional, Dict, Any

class VoxCPMTTSClient:
    """VoxCPM TTS API 客户端"""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url.rstrip('/')
        self.cache_id: Optional[str] = None
    
    def health_check(self) -> bool:
        """健康检查"""
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        resp = requests.get(f"{self.base_url}/models")
        resp.raise_for_status()
        return resp.json()
    
    def generate(
        self,
        text: str,
        prompt_wav_path: Optional[str] = None,
        prompt_text: Optional[str] = None,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        output_path: Optional[str] = None,
    ) -> bytes:
        """
        生成语音
        
        Args:
            text: 目标文本
            prompt_wav_path: 参考音频路径（可选，用于声音克隆）
            prompt_text: 参考文本（可选）
            cfg_value: CFG 值
            inference_timesteps: 推理步数
            output_path: 输出文件路径（可选）
        
        Returns:
            音频数据 (bytes)
        """
        payload = {
            "text": text,
            "cfg_value": cfg_value,
            "inference_timesteps": inference_timesteps,
        }
        
        if prompt_wav_path:
            payload["prompt_wav_path"] = prompt_wav_path
            payload["prompt_text"] = prompt_text
        
        resp = requests.post(
            f"{self.base_url}/generate",
            json=payload,
            timeout=300  # 5分钟超时
        )
        resp.raise_for_status()
        
        if output_path:
            with open(output_path, "wb") as f:
                f.write(resp.content)
        
        return resp.content
    
    def build_cache(
        self,
        prompt_wav_path: str,
        prompt_text: str,
    ) -> str:
        """
        构建 prompt cache
        
        Returns:
            cache_id
        """
        resp = requests.post(
            f"{self.base_url}/build_cache",
            json={
                "prompt_wav_path": prompt_wav_path,
                "prompt_text": prompt_text,
            },
            timeout=60
        )
        resp.raise_for_status()
        self.cache_id = resp.json()["cache_id"]
        return self.cache_id
    
    def generate_with_cache(
        self,
        text: str,
        cache_id: Optional[str] = None,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        output_path: Optional[str] = None,
    ) -> bytes:
        """
        使用 cache 生成语音
        
        Args:
            text: 目标文本
            cache_id: cache_id (可选，使用上一次 build 的 cache)
            cfg_value: CFG 值
            inference_timesteps: 推理步数
            output_path: 输出文件路径
        
        Returns:
            音频数据 (bytes)
        """
        if cache_id is None:
            cache_id = self.cache_id
        
        if cache_id is None:
            raise ValueError("No cache_id provided. Please call build_cache first.")
        
        payload = {
            "text": text,
            "prompt_cache_id": cache_id,
            "cfg_value": cfg_value,
            "inference_timesteps": inference_timesteps,
        }
        
        resp = requests.post(
            f"{self.base_url}/generate_with_cache",
            json=payload,
            timeout=300
        )
        resp.raise_for_status()
        
        if output_path:
            with open(output_path, "wb") as f:
                f.write(resp.content)
        
        return resp.content
    
    def list_caches(self) -> list:
        """列出所有 cache"""
        resp = requests.get(f"{self.base_url}/caches")
        resp.raise_for_status()
        return resp.json()["cache_ids"]
    
    def delete_cache(self, cache_id: str) -> bool:
        """删除 cache"""
        resp = requests.delete(f"{self.base_url}/cache/{cache_id}")
        return resp.status_code == 200
```

### 3. 使用示例

```python
# usage_example.py
from tts_client import VoxCPMTTSClient

client = VoxCPMTTSClient(base_url="http://localhost:8080")

# ===== 方式1：基础调用 =====
print("=== 基础调用 ===")
audio = client.generate(
    text="你好，这是测试。",
    output_path="test1.wav"
)
print(f"音频已保存: test1.wav")

# ===== 方式2：带参数调用 =====
print("\n=== 带参数调用 ===")
audio = client.generate(
    text="这是高质量语音合成测试。",
    cfg_value=2.5,
    inference_timesteps=20,
    output_path="test2.wav"
)
print(f"高质量音频已保存: test2.wav")

# ===== 方式3：声音克隆（每次传入参考音频） =====
print("\n=== 声音克隆 ===")
audio = client.generate(
    text="使用参考声音合成。",
    prompt_wav_path="/home/zju/VoxCPM/examples/hailan02.mp3",
    prompt_text="Just by listening a few minutes a day...",
    output_path="test3.wav"
)
print(f"克隆音频已保存: test3.wav")

# ===== 方式4：使用 Prompt Cache（推荐，用于固定音色） =====
print("\n=== Prompt Cache 优化 ===")

# 构建一次 cache，后续无需再传参考音频
cache_id = client.build_cache(
    prompt_wav_path="/home/zju/VoxCPM/examples/hailan02.mp3",
    prompt_text="Just by listening a few minutes a day..."
)
print(f"Cache ID: {cache_id}")

# 多次生成，速度更快
audio = client.generate_with_cache(
    text="第一次使用 cache 生成。",
    output_path="test4.wav"
)
print(f"第一次生成: test4.wav")

audio = client.generate_with_cache(
    text="第二次使用 cache 生成。",
    output_path="test5.wav"
)
print(f"第二次生成: test5.wav")

audio = client.generate_with_cache(
    text="第三次使用 cache 生成。",
    output_path="test6.wav"
)
print(f"第三次生成: test6.wav")
```

---

## WebSocket Python 客户端

WebSocket 客户端支持实时双向通信，推荐用于生产环境。

### 1. 安装依赖

```bash
pip install websocket-client
```

### 2. 同步客户端

```python
# ws_tts_client.py
import websocket
import json
import threading
from typing import Optional, Callable

class VoxCPMTTSWebSocketClient:
    """VoxCPM TTS WebSocket 客户端（同步版）"""
    
    def __init__(self, host: str = "localhost", port: int = 8080):
        self.url = f"ws://{host}:{port}"
        self.ws = None
        self.audio_data = None
        self.response_received = threading.Event()
    
    def _on_message(self, ws, message):
        """接收消息回调"""
        if isinstance(message, bytes):
            # 音频数据
            self.audio_data = message
        else:
            # JSON 状态消息
            data = json.loads(message)
            if data.get("status") == "success":
                print(f"生成成功，音频大小: {data['audio_size']} bytes")
            else:
                print(f"错误: {data.get('message')}")
        self.response_received.set()
    
    def _on_error(self, ws, error):
        print(f"WebSocket 错误: {error}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        print("WebSocket 连接关闭")
    
    def _on_open(self, ws):
        print("WebSocket 连接已建立")
    
    def generate(
        self,
        text: str,
        prompt_wav_path: Optional[str] = None,
        prompt_text: Optional[str] = None,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        output_path: Optional[str] = None,
    ) -> bytes:
        """
        生成语音
        
        Args:
            text: 目标文本
            prompt_wav_path: 参考音频路径（可选）
            prompt_text: 参考文本（可选）
            cfg_value: CFG 值
            inference_timesteps: 推理步数
            output_path: 输出文件路径（可选）
        
        Returns:
            音频数据 (bytes)
        """
        self.audio_data = None
        self.response_received.clear()
        
        # 创建 WebSocket 连接
        self.ws = websocket.WebSocketApp(
            self.url + "/ws/generate",
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        
        # 在单独线程中运行
        def run_ws():
            self.ws.run_forever()
        
        ws_thread = threading.Thread(target=run_ws)
        ws_thread.start()
        
        # 等待连接建立
        import time
        time.sleep(0.5)
        
        # 发送请求
        request = {
            "text": text,
            "cfg_value": cfg_value,
            "inference_timesteps": inference_timesteps,
        }
        if prompt_wav_path:
            request["prompt_wav_path"] = prompt_wav_path
            request["prompt_text"] = prompt_text
        
        self.ws.send(json.dumps(request))
        
        # 等待响应
        self.response_received.wait(timeout=300)
        self.ws.close()
        
        if self.audio_data is None:
            raise RuntimeError("Failed to generate audio")
        
        if output_path:
            with open(output_path, "wb") as f:
                f.write(self.audio_data)
        
        return self.audio_data
```

### 3. 异步客户端（推荐）

```python
# async_ws_tts_client.py
import asyncio
import json
import aiohttp
from typing import Optional

class AsyncVoxCPMTTSClient:
    """VoxCPM TTS 异步客户端（支持 WebSocket）"""
    
    def __init__(self, host: str = "localhost", port: int = 8080):
        self.http_url = f"http://{host}:{port}"
        self.ws_url = f"ws://{host}:{port}"
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def connect(self):
        """建立连接"""
        self.session = aiohttp.ClientSession()
    
    async def close(self):
        """关闭连接"""
        if self.session:
            await self.session.close()
    
    async def generate(
        self,
        text: str,
        prompt_wav_path: Optional[str] = None,
        prompt_text: Optional[str] = None,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        output_path: Optional[str] = None,
    ) -> bytes:
        """
        生成语音（使用 HTTP）
        
        Returns:
            音频数据 (bytes)
        """
        async with self.session.post(
            f"{self.http_url}/generate",
            json={
                "text": text,
                "prompt_wav_path": prompt_wav_path,
                "prompt_text": prompt_text,
                "cfg_value": cfg_value,
                "inference_timesteps": inference_timesteps,
            },
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            audio_data = await resp.read()
        
        if output_path:
            with open(output_path, "wb") as f:
                f.write(audio_data)
        
        return audio_data
    
    async def ws_generate(
        self,
        text: str,
        prompt_wav_path: Optional[str] = None,
        prompt_text: Optional[str] = None,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        output_path: Optional[str] = None,
    ) -> bytes:
        """
        生成语音（使用 WebSocket，推荐）
        
        Returns:
            音频数据 (bytes)
        """
        async with self.session.ws_connect(f"{self.ws_url}/ws/generate") as ws:
            # 发送请求
            await ws.send_json({
                "text": text,
                "cfg_value": cfg_value,
                "inference_timesteps": inference_timesteps,
            })
            if prompt_wav_path:
                await ws.send_json({
                    "prompt_wav_path": prompt_wav_path,
                    "prompt_text": prompt_text,
                })
            
            # 接收音频数据
            audio_data = None
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.BINARY:
                    audio_data = msg.data
                elif msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    if data.get("status") == "error":
                        raise RuntimeError(data.get("message"))
                    break
        
        if audio_data is None:
            raise RuntimeError("Failed to generate audio")
        
        if output_path:
            with open(output_path, "wb") as f:
                f.write(audio_data)
        
        return audio_data
    
    async def build_cache(
        self,
        prompt_wav_path: str,
        prompt_text: str,
    ) -> str:
        """构建 prompt cache"""
        async with self.session.post(
            f"{self.http_url}/build_cache",
            json={
                "prompt_wav_path": prompt_wav_path,
                "prompt_text": prompt_text,
            },
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            data = await resp.json()
            return data["cache_id"]
    
    async def generate_with_cache(
        self,
        text: str,
        cache_id: str,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        output_path: Optional[str] = None,
    ) -> bytes:
        """使用 cache 生成语音"""
        async with self.session.post(
            f"{self.http_url}/generate_with_cache",
            json={
                "text": text,
                "prompt_cache_id": cache_id,
                "cfg_value": cfg_value,
                "inference_timesteps": inference_timesteps,
            },
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            audio_data = await resp.read()
        
        if output_path:
            with open(output_path, "wb") as f:
                f.write(audio_data)
        
        return audio_data
```

### 4. 使用示例

```python
# ws_usage_example.py
import asyncio
from ws_tts_client import VoxCPMTTSWebSocketClient
from async_ws_tts_client import AsyncVoxCPMTTSClient

# ===== 方式1：同步客户端 =====
print("=== 同步客户端测试 ===")
client = VoxCPMTTSWebSocketClient()

# 基础调用
audio = client.generate(
    text="你好，这是 WebSocket 测试。",
    output_path="ws_test.wav"
)
print(f"音频已保存: ws_test.wav")

# ===== 方式2：异步客户端（推荐）=====
async def main():
    print("\n=== 异步客户端测试 ===")
    client = AsyncVoxCPMTTSClient()
    await client.connect()
    
    try:
        # WebSocket 生成
        audio = await client.ws_generate(
            text="使用 WebSocket 异步生成。",
            output_path="async_ws_test.wav"
        )
        print(f"异步 WebSocket 音频已保存: async_ws_test.wav")
        
        # 构建 cache (使用 HTTP)
        cache_id = await client.build_cache(
            prompt_wav_path="/home/zju/VoxCPM/examples/hailan02.mp3",
            prompt_text="Just by listening a few minutes a day..."
        )
        print(f"Cache ID: {cache_id}")
        
        # 使用 cache 生成
        audio = await client.generate_with_cache(
            text="使用 cache 异步生成。",
            cache_id=cache_id,
            output_path="cache_test.wav"
        )
        print(f"Cache 音频已保存: cache_test.wav")
        
    finally:
        await client.close()

asyncio.run(main())
```

---

## FastAPI 集成

如果你的项目也是 FastAPI，可以直接调用：

```python
# main.py
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import tempfile

app = FastAPI()

TTS_API_URL = "http://localhost:8080"

@app.post("/tts")
async def text_to_speech(text: str, cfg_value: float = 2.0):
    """
    文字转语音接口
    
    Args:
        text: 要转换的文本
        cfg_value: CFG 值
    
    Returns:
        WAV 音频文件
    """
    try:
        # 调用 TTS API
        resp = requests.post(
            f"{TTS_API_URL}/generate",
            json={
                "text": text,
                "cfg_value": cfg_value,
                "inference_timesteps": 15
            },
            timeout=300
        )
        resp.raise_for_status()
        
        # 保存临时文件并返回
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(resp.content)
            temp_path = f.name
        
        return FileResponse(
            temp_path,
            media_type="audio/wav",
            filename="tts_output.wav"
        )
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"TTS API error: {str(e)}")


@app.post("/tts-with-cache")
async def tts_with_cache(text: str, cache_id: str):
    """使用 prompt cache 的 TTS 接口"""
    try:
        resp = requests.post(
            f"{TTS_API_URL}/generate_with_cache",
            json={
                "text": text,
                "prompt_cache_id": cache_id,
                "inference_timesteps": 15
            },
            timeout=300
        )
        resp.raise_for_status()
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(resp.content)
            temp_path = f.name
        
        return FileResponse(
            temp_path,
            media_type="audio/wav",
            filename="tts_output.wav"
        )
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"TTS API error: {str(e)}")
```

---

## 常见问题

### Q1: 服务启动失败，提示端口被占用

```bash
# 查看占用端口的进程
lsof -i :8080

# 或使用其他端口
export SERVER_PORT=8081
python api.py
```

### Q2: 如何修改模型路径？

```bash
export VOCXPM_MODEL_PATH=/path/to/your/model
python api.py
```

### Q3: 内存不足/显存不足

```bash
# 使用 CPU（慢但稳定）
export CUDA_VISIBLE_DEVICES=
python api.py
```

### Q4: 生成的音频质量不佳

增加 `inference_timesteps` 参数：

```python
audio = client.generate(
    text="测试文本",
    inference_timesteps=20,  # 提高到 20
    output_path="output.wav"
)
```

### Q5: 如何处理长文本？

API 会自动处理长文本，但如果文本特别长（>1000字），建议分段调用：

```python
# 分段生成示例
texts = [
    "第一段文本。",
    "第二段文本。",
    "第三段文本。",
]

all_audio = []
for text in texts:
    audio = client.generate(
        text=text,
        cfg_value=2.0,
        inference_timesteps=10
    )
    all_audio.append(audio)

# all_audio 是 bytes 列表，可按需合并
```

### Q6: 如何在生产环境部署？

建议使用 systemd：

```ini
# /etc/systemd/system/tts.service
[Unit]
Description=VoxCPM TTS Service
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/zju/VoxCPM
ExecStart=/home/zju/miniconda3/envs/voxcpm/bin/python api.py
Restart=always
RestartSec=5
Environment=VOXCPM_MODEL_PATH=/home/zju/VoxCPM/models/openbmb__VoxCPM1.5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable tts
sudo systemctl start tts
```

### Q7: 交互式 API 文档

访问 `http://localhost:8080/docs` 可以查看 Swagger UI 交互式文档。

---

## 联系

如有问题，请查看 [VoxCPM 项目主页](https://github.com/OpenBMB/VoxCPM)。
