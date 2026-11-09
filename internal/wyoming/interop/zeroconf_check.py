"""Browse for _wyoming._tcp.local. with python-zeroconf, the library Home Assistant uses."""

import asyncio
import sys

from zeroconf import ServiceStateChange
from zeroconf.asyncio import AsyncServiceBrowser, AsyncServiceInfo, AsyncZeroconf

SERVICE = "_wyoming._tcp.local."


async def main(expected_ip):
    aiozc = AsyncZeroconf()
    found = asyncio.Queue()

    def on_change(zeroconf, service_type, name, state_change):
        if state_change is ServiceStateChange.Added:
            found.put_nowait(name)

    browser = AsyncServiceBrowser(aiozc.zeroconf, [SERVICE], handlers=[on_change])
    try:
        name = await asyncio.wait_for(found.get(), 10)
        info = AsyncServiceInfo(SERVICE, name)
        ok = await info.async_request(aiozc.zeroconf, 3000)
        addrs = info.parsed_addresses()
        print(f"found {name} port={info.port} addresses={addrs} server={info.server}")
        good = ok and info.port == 10300 and addrs == [expected_ip]
        print("ok   zeroconf discovery" if good else "FAIL zeroconf discovery")
        return 0 if good else 1
    except TimeoutError:
        print("FAIL zeroconf discovery: nothing found")
        return 1
    finally:
        await browser.async_cancel()
        await aiozc.async_close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1])))
