import asyncio
import websockets
import json
import httpx

async def test_ws():
    uri = 'ws://localhost:8000/ws/stream'
    async with websockets.connect(uri) as ws:
        # Subscribe with batching enabled
        await ws.send(json.dumps({
            'action': 'subscribe',
            'tokens': [123],
            'timeframes': ['1m'],
            'throttle_ms': 1000,
            'batch': True,
            'batch_size': 3
        }))
        resp = await ws.recv()
        print('Subscribe response:', resp)

        # Simulate or wait for batched data
        print('Waiting for batched messages (timeout 10s)...')
        try:
            for _ in range(2):  # Expecting 2 batches if 5 messages are sent
                batch = await asyncio.wait_for(ws.recv(), timeout=10)
                print('Received batch:', batch)
        except asyncio.TimeoutError:
            print('Timeout waiting for batch messages.')

async def test_realtime_and_fallback():
    # Test REST API for /analyze with a likely real-time symbol
    async with httpx.AsyncClient() as client:
        payload = {
            "stock": "RELIANCE",
            "exchange": "NSE",
            "period": 5,
            "interval": "1m"
        }
        response = await client.post("http://localhost:8000/analyze", json=payload)
        assert response.status_code == 200, f"API returned {response.status_code}"
        data = response.json()
        print('Analyze response:', data.get('message'))
        freshness = data.get('results', {}).get('metadata', {}).get('data_freshness')
        if freshness != 'real_time':
            print(f"WARNING: Data freshness is {freshness}, expected 'real_time' during market hours.")
        else:
            print("Data is real-time as expected.")

        # Now test fallback by using a symbol or interval unlikely to have real-time data
        payload_fallback = {
            "stock": "RELIANCE",
            "exchange": "NSE",
            "period": 365,
            "interval": "day"
        }
        response_fallback = await client.post("http://localhost:8000/analyze", json=payload_fallback)
        assert response_fallback.status_code == 200, f"API returned {response_fallback.status_code}"
        data_fallback = response_fallback.json()
        freshness_fallback = data_fallback.get('results', {}).get('metadata', {}).get('data_freshness')
        print('Fallback analyze response:', data_fallback.get('message'))
        if freshness_fallback != 'historical':
            print(f"WARNING: Data freshness is {freshness_fallback}, expected 'historical' for fallback.")
        else:
            print("Fallback to historical data works as expected.")

if __name__ == '__main__':
    asyncio.run(test_ws())
    asyncio.run(test_realtime_and_fallback()) 