import asyncio
import ngrok

async def test():
    listener = await ngrok.forward(3000)
    print(f"Type: {type(listener)}")
    print(f"Attributes: {[attr for attr in dir(listener) if not attr.startswith('_')]}")
    await listener.close()

asyncio.run(test())