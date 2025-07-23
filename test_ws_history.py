"""
Test for WebSocket 'history' action (historical replay)

Usage:
    python test_ws_history.py --token <TOKEN> --timeframe <TIMEFRAME> [--count N] [--url ws://localhost:8000/ws/stream]

Example:
    python test_ws_history.py --token 123456 --timeframe 1m --count 10
"""
import asyncio
import websockets
import json
import argparse

async def test_history(url, token, timeframe, count):
    async with websockets.connect(url) as ws:
        # Send history request
        req = {
            "action": "history",
            "token": token,
            "timeframe": timeframe,
            "count": count
        }
        await ws.send(json.dumps(req))
        response = await ws.recv()
        try:
            data = json.loads(response)
        except Exception:
            print("Raw response:", response)
            return
        print("Received response:")
        print(json.dumps(data, indent=2))
        if data.get("type") == "history":
            print(f"Received {data.get('count')} candles for token {token} ({timeframe})")
        else:
            print("Unexpected response type:", data.get("type"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://localhost:8000/ws/stream", help="WebSocket URL")
    parser.add_argument("--token", required=True, help="Instrument token")
    parser.add_argument("--timeframe", required=True, help="Timeframe (e.g., 1m, 5m, 1d)")
    parser.add_argument("--count", type=int, default=10, help="Number of candles")
    args = parser.parse_args()
    asyncio.run(test_history(args.url, args.token, args.timeframe, args.count)) 