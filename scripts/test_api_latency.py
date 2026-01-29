import asyncio
import websockets
import json
import time
import os
import sys
import argparse

# 默认配置
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8080
DEFAULT_AUDIO_PATH = "/home/zju/VoxCPM/examples/hailan07_short.wav"

async def test_health(uri):
    """测试健康检查接口"""
    print(f"\n[Testing Health] Connecting to {uri}...")
    try:
        start_time = time.time()
        async with websockets.connect(uri) as websocket:
            connect_time = time.time()
            print(f"  Connected in {(connect_time - start_time)*1000:.2f} ms")
            
            resp = await websocket.recv()
            end_time = time.time()
            data = json.loads(resp)
            
            print(f"  Response: {data}")
            print(f"  Total Latency: {(end_time - start_time)*1000:.2f} ms")
    except Exception as e:
        print(f"  Error: {e}")

async def test_models(uri):
    """测试模型信息接口"""
    print(f"\n[Testing Models] Connecting to {uri}...")
    try:
        start_time = time.time()
        async with websockets.connect(uri) as websocket:
            connect_time = time.time()
            print(f"  Connected in {(connect_time - start_time)*1000:.2f} ms")
            
            resp = await websocket.recv()
            end_time = time.time()
            data = json.loads(resp)
            
            print(f"  Response: {json.dumps(data, indent=2)}")
            print(f"  Total Latency: {(end_time - start_time)*1000:.2f} ms")
    except Exception as e:
        print(f"  Error: {e}")

async def test_vad(uri, audio_path):
    """测试 VAD 接口"""
    print(f"\n[Testing VAD] Connecting to {uri}...")
    if not os.path.exists(audio_path):
        print(f"  Error: Audio file not found at {audio_path}")
        return

    try:
        with open(audio_path, "rb") as f:
            audio_data = f.read()
        print(f"  Audio size: {len(audio_data)} bytes")

        start_time = time.time()
        async with websockets.connect(uri) as websocket:
            connect_time = time.time()
            print(f"  Connected in {(connect_time - start_time)*1000:.2f} ms")
            
            # 发送音频
            send_start = time.time()
            await websocket.send(audio_data)
            print(f"  Audio sent in {(time.time() - send_start)*1000:.2f} ms")
            
            # 接收结果
            resp = await websocket.recv()
            end_time = time.time()
            data = json.loads(resp)
            
            print(f"  Response: {json.dumps(data, indent=2)}")
            print(f"  Processing Latency (send+recv): {(end_time - send_start)*1000:.2f} ms")
            print(f"  Total Latency: {(end_time - start_time)*1000:.2f} ms")
    except Exception as e:
        print(f"  Error: {e}")

async def test_asr(uri, audio_path):
    """测试 ASR 接口"""
    print(f"\n[Testing ASR] Connecting to {uri}...")
    if not os.path.exists(audio_path):
        print(f"  Error: Audio file not found at {audio_path}")
        return

    try:
        with open(audio_path, "rb") as f:
            audio_data = f.read()
        print(f"  Audio size: {len(audio_data)} bytes")

        start_time = time.time()
        async with websockets.connect(uri) as websocket:
            connect_time = time.time()
            print(f"  Connected in {(connect_time - start_time)*1000:.2f} ms")
            
            # 发送音频
            send_start = time.time()
            await websocket.send(audio_data)
            print(f"  Audio sent in {(time.time() - send_start)*1000:.2f} ms")
            
            # 接收结果
            resp = await websocket.recv()
            end_time = time.time()
            data = json.loads(resp)
            
            print(f"  Response: {json.dumps(data, indent=2)}")
            print(f"  Processing Latency (send+recv): {(end_time - send_start)*1000:.2f} ms")
            print(f"  Total Latency: {(end_time - start_time)*1000:.2f} ms")
    except Exception as e:
        print(f"  Error: {e}")

async def test_generate(uri, text="用电量或扣费明细我这就为您调取～单一制用户的峰谷电价浮动比例，是根据容量来定的。", prompt_wav=None, prompt_text="感谢您的耐心，我这就去核实一下，在江苏电力现货市场里，费用分摊主要涉及几类。", stream=True):
    """测试 TTS 生成接口"""
    mode_str = "Streaming" if stream else "Non-streaming"
    print(f"\n[Testing TTS Generate ({mode_str})] Connecting to {uri}...")
    
    req_data = {
        "text": text,
        "cfg_value": 2.0,
        "inference_timesteps": 25,
        "normalize": False,
        "denoise": False,
        "stream": stream
    }
    if prompt_wav:
        req_data["prompt_wav_path"] = prompt_wav
    if prompt_text:
        req_data["prompt_text"] = prompt_text
        
    print(f"  Request: {json.dumps(req_data, indent=2, ensure_ascii=False)}")

    try:
        start_time = time.time()
        async with websockets.connect(uri) as websocket:
            connect_time = time.time()
            print(f"  Connected in {(connect_time - start_time)*1000:.2f} ms")
            
            # 发送请求
            await websocket.send(json.dumps(req_data))
            send_time = time.time()
            
            # 接收音频
            first_chunk_time = None
            total_audio_bytes = 0
            
            # 协议：先发音频 bytes，最后发 JSON 状态
            while True:
                msg = await websocket.recv()
                if isinstance(msg, bytes):
                    if first_chunk_time is None:
                        first_chunk_time = time.time()
                        # print(f"  First chunk received: {len(msg)} bytes")
                    total_audio_bytes += len(msg)
                else:
                    # JSON 消息
                    end_time = time.time()
                    try:
                        res_json = json.loads(msg)
                        if res_json.get("status") == "error":
                            print(f"  Error response: {res_json}")
                            return
                        # print(f"  Final Response: {res_json}")
                    except:
                        print(f"  Received non-JSON text: {msg}")
                    break
            
            # 统计计算
            first_latency_ms = (first_chunk_time - send_time) * 1000 if first_chunk_time else 0
            total_latency_ms = (end_time - send_time) * 1000
            total_latency_s = total_latency_ms / 1000.0
            
            # 计算音频时长 (44100Hz, 16bit, Mono)
            # 16bit = 2 bytes per sample
            sample_rate = 44100
            bytes_per_sample = 2
            audio_duration_s = total_audio_bytes / (sample_rate * bytes_per_sample)
            
            # 计算 RTF (Real Time Factor)
            rtf = total_latency_s / audio_duration_s if audio_duration_s > 0 else 0
            
            print("-" * 40)
            print(f"  Mode: {mode_str}")
            print(f"  First Byte Latency (TTFB): {first_latency_ms:.2f} ms")
            print(f"  Total Generation Time:     {total_latency_ms:.2f} ms")
            print(f"  Audio Duration:            {audio_duration_s:.2f} s")
            print(f"  Real Time Factor (RTF):    {rtf:.4f}")
            print(f"  Audio Size:                {total_audio_bytes} bytes")
            print("-" * 40)
            
    except Exception as e:
        print(f"  Error: {e}")

async def main():
    parser = argparse.ArgumentParser(description="VoxCPM API Latency Test Script")
    parser.add_argument("--host", default=DEFAULT_HOST, help="API Server Host")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="API Server Port")
    parser.add_argument("--audio", default=DEFAULT_AUDIO_PATH, help="Path to audio file for VAD/ASR test")
    parser.add_argument("--text", default="用电量或扣费明细我这就为您调取～单一制用户的峰谷电价浮动比例，是根据容量来定的。", help="Text for TTS test")
    args = parser.parse_args()

    base_uri = f"ws://{args.host}:{args.port}"
    print(f"Target Server: {base_uri}")
    print(f"Test Audio: {args.audio}")

    # 1. Health
    await test_health(f"{base_uri}/ws/health")
    
    # 2. Models
    await test_models(f"{base_uri}/ws/models")
    
    # 3. VAD
    await test_vad(f"{base_uri}/ws/vad", args.audio)
    
    # 4. ASR
    await test_asr(f"{base_uri}/ws/asr", args.audio)
    
    # 5. TTS
    # 纯文本 - Streaming
    await test_generate(f"{base_uri}/ws/generate", text=args.text, prompt_text=None, stream=True)

    # 纯文本 - Non-streaming
    await test_generate(f"{base_uri}/ws/generate", text=args.text, prompt_text=None, stream=False)
    
    # 带参考音频（声音克隆）- Streaming
    if os.path.exists(args.audio):
        print("\n[Testing TTS with Cloning]...")
        # 假设 audio 对应的文本未知，让后端自动识别
        # 流式声音克隆
        await test_generate(f"{base_uri}/ws/generate", text=args.text, prompt_wav=os.path.abspath(args.audio), stream=True)
    # 带参考音频（声音克隆）- Non-streaming
    if os.path.exists(args.audio):
        print("\n[Testing TTS with Cloning]...")
        # 假设 audio 对应的文本未知，让后端自动识别
        # 流式声音克隆
        await test_generate(f"{base_uri}/ws/generate", text=args.text, prompt_wav=os.path.abspath(args.audio), stream=False)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTest interrupted.")
    except ConnectionRefusedError:
        print("\nError: Connection refused. Is the server running? (python api.py)")
