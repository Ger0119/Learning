# 协程
## monkey_patch
```python
from gevent import monkey
monkey.patch_all()         # 整个程序补丁
monkey.patch_socket()      # 单个程序补丁
```
## async
### asyncio
```python
import asyncio

async def asytest(n):
    await asyncio.sleep(n)

loop = asyncio.get_event_loop()
task = [asytest(x) for x in range(1,4)]
loop.run_until_complete(asyncio.wait(task))

# =========================================
async def job():
    tasks = [
        asyncio.create_task(asytest(1)),
        asyncio.create_task(asytest(2)),
        asyncio.create_task(asytest(3)),
    ]
    await asyncio.wait(tasks)

loop = asyncio.get_event_loop()
loop.run_until_complete(job())
loop.close()
```
### aiohttp
```python
import aiohttp

async def test(name=0, url="http://httpbin.org/user-agent"):
    print(f"{name} : Start")
    await get_url(url)
    print(f"{name} : Over")


async def get_url(url):
    async with aiohttp.ClientSession() as session:
        res = await session.get(url)
        txt = await res.text()    # res.content.read()
        print(len(txt))
```
### aiofiles
```python
import aiofiles

async with aiofiles.open("file","wb") as f:
    await f.write(data)
```