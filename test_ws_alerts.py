"""
Test for WebSocket custom alerts (price_cross)

Usage:
    python test_ws_alerts.py --token <TOKEN> --threshold <PRICE> [--url ws://localhost:8000/ws/stream]

Example:
    python test_ws_alerts.py --token 123456 --threshold 2500
"""
import asyncio
import websockets
import json
import argparse

async def test_alerts(url, token, threshold):
    async with websockets.connect(url) as ws:
        # Register price_cross alert
        req = {
            "action": "register_alert",
            "alert": {
                "type": "price_cross",
                "token": token,
                "params": {
                    "threshold": threshold,
                    "direction": "above"
                }
            }
        }
        await ws.send(json.dumps(req))
        response = await ws.recv()
        print("Register alert response:", response)
        # Listen for alert events
        print("Waiting for alert events...")
        while True:
            msg = await ws.recv()
            try:
                data = json.loads(msg)
            except Exception:
                print("Raw message:", msg)
                continue
            if data.get("type") == "alert":
                print("ALERT TRIGGERED:")
                print(json.dumps(data, indent=2))
            else:
                print("Other message:", data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://localhost:8000/ws/stream", help="WebSocket URL")
    parser.add_argument("--token", required=True, help="Instrument token")
    parser.add_argument("--threshold", type=float, required=True, help="Price threshold for alert")
    args = parser.parse_args()
    asyncio.run(test_alerts(args.url, args.token, args.threshold)) 