# -*- coding: utf8

from collections import defaultdict

import gzip

def get_graph_stamps(path, top=None):
    count = defaultdict(int)
    srcs = set()
    with gzip.open(path, 'r') as in_file:
        for line in in_file:
            if b',' in line:
                spl = line.split(b',')
            else:
                spl = line.split()
            src, dst = spl[:2]
            count[dst] += 1
            srcs.add(src)

    if top is None:
        valid = srcs
    else:
        valid = set()
        for v, k in sorted(((v, k) for k, v in count.items()), reverse=True):
            if k in srcs:
                valid.add(k)
                if len(valid) == top:
                    break

    graph = {}
    ids = {}
    with gzip.open(path, 'r') as in_file:
        timestamps = []
        for line in in_file:
            if b',' in line:
                spl = line.split(b',')
            else:
                spl = line.split()
            src, dst = spl[:2]
            stamp = float(spl[-1])
            if src not in valid:
                continue
            if dst not in valid:
                continue

            if src not in graph:
                graph[src] = {}
            if dst not in graph[src]:
                graph[src][dst] = 0
            graph[src][dst] += 1

            if dst in ids:
                timestamps[ids[dst]].append(stamp)
            else:
                ids[dst] = len(timestamps)
                timestamps.append([stamp])

    for id_ in list(graph.keys()):
        if id_ not in ids:
            del graph[id_]
    for id_ in ids:
        if id_ not in graph:
            graph[id_] = {}

    return timestamps, graph, ids
