from tqdm import tqdm
import argparse
import aiohttp
import asyncio
import string
import random
random.seed(2021)  # deterministic random numbers


async def upload_follow(session, addr, user_0, user_1):
    payload = {'user_id': user_0, 'followee_id': user_1}
    async with session.post(addr + "/wrk2-api/user/follow", data=payload) as resp:
        return await resp.text()


async def upload_register(session, addr, user):
    payload = {'first_name': 'first_name_' + user, 'last_name': 'last_name_' + user,
               'username': 'username_' + user, 'password': 'password_' + user, 'user_id': user}
    async with session.post(addr + "/wrk2-api/user/register", data=payload) as resp:
        return await resp.text()


def getNodes(file):
    line = file.readline()
    word = line.split()[0]
    return int(word)


def getEdges(file):
    edges = []
    lines = file.readlines()
    for line in lines:
        edges.append(line.split())
    return edges


def printResults(results):
    result_type_count = {}
    for result in results:
        try:
            result_type_count[result] += 1
        except KeyError:
            result_type_count[result] = 1
    for result_type, count in result_type_count.items():
        if result_type == '' or result_type.startswith("Success"):
            print("Succeeded:", count)
        elif "500 Internal Server Error" in result_type:
            print("Failed:", count, "Error:", "Internal Server Error")
        else:
            print("Failed:", count, "Error:", result_type.strip())


async def register(addr, nodes):
    idx = 0
    tasks = []
    conn = aiohttp.TCPConnector(limit=200)
    async with aiohttp.ClientSession(connector=conn) as session:
        print("Registering %d users..." % nodes)
        for i in tqdm(range(1, nodes + 1)):
            task = asyncio.ensure_future(upload_register(session, addr, str(i)))
            tasks.append(task)
            idx += 1
            if idx % 200 == 0:
                _ = await asyncio.gather(*tasks)
        results = await asyncio.gather(*tasks)
        printResults(results)


async def follow(addr, edges):
    idx = 0
    tasks = []
    conn = aiohttp.TCPConnector(limit=200)
    async with aiohttp.ClientSession(connector=conn) as session:
        print("Adding %d follows..." % len(edges))
        for edge in tqdm(edges):
            task = asyncio.ensure_future(upload_follow(session, addr, edge[0], edge[1]))
            tasks.append(task)
            task = asyncio.ensure_future(upload_follow(session, addr, edge[1], edge[0]))
            tasks.append(task)
            idx += 1
            if idx % 200 == 0:
                _ = await asyncio.gather(*tasks)
        results = await asyncio.gather(*tasks)
        printResults(results)


if __name__ == '__main__':
    filename_default = './datasets/social-graph/socfb-Reed98.mtx'

    parser = argparse.ArgumentParser("Social network initializer")
    parser.add_argument("--graph", help="Path to graph file", default=filename_default)
    parser.add_argument("--addr", help="Address of the social network NGINX web server")
    args = parser.parse_args()

    with open(args.graph, 'r') as f:
        nodes = getNodes(f)
        edges = getEdges(f)

    loop = asyncio.get_event_loop()

    # Registration
    future = asyncio.ensure_future(register(args.addr, nodes))
    loop.run_until_complete(future)

    # Follow
    future = asyncio.ensure_future(follow(args.addr, edges))
    loop.run_until_complete(future)

