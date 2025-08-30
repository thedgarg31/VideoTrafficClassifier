# capture_scapy_live.py
# Usage: sudo python capture_scapy_live.py --iface wlp3s0 --out live_flows.jsonl
import argparse, time, json
from scapy.all import sniff, IP, TCP, UDP
import numpy as np
from collections import defaultdict
parser = argparse.ArgumentParser()
parser.add_argument("--iface", required=True)
parser.add_argument("--out", default="live_flows.jsonl")
parser.add_argument("--min-pkts", type=int, default=8)
parser.add_argument("--idle-timeout", type=float, default=2.0)
args = parser.parse_args()

flows = {}  # key -> {'times':[], 'sizes':[], 'last':ts, 'dports':[]}

def canonical_key(pkt):
    if not IP in pkt:
        return None
    ip = pkt[IP]
    proto = pkt.payload.name
    sport = pkt.sport if hasattr(pkt, 'sport') else 0
    dport = pkt.dport if hasattr(pkt, 'dport') else 0
    a = (ip.src, sport)
    b = (ip.dst, dport)
    if a <= b:
        return (a[0], b[0], int(a[1]), int(b[1]), proto)
    else:
        return (b[0], a[0], int(b[1]), int(a[1]), proto)

def flush_flow(key, data, outfh):
    times = np.array(data['times'])
    sizes = np.array(data['sizes'])
    if len(sizes)==0: return
    dur = float(times.max() - times.min()) if len(times)>1 else 0.0
    pkts = int(len(sizes))
    total_bytes = int(sizes.sum())
    avg_pkt = float(sizes.mean())
    std_pkt = float(sizes.std()) if len(sizes)>1 else 0.0
    iat = np.diff(np.sort(times)) if len(times)>1 else np.array([0.0])
    mean_iat = float(iat.mean()) if len(iat)>0 else 0.0
    std_iat = float(iat.std()) if len(iat)>0 else 0.0
    first20 = [int(x) for x in list(sizes[:20]) + [0]*(20-min(20,len(sizes)))]
    rec = {
        "flow_key": key,
        "duration": dur,
        "pkt_count": pkts,
        "total_bytes": total_bytes,
        "avg_pkt": avg_pkt,
        "std_pkt": std_pkt,
        "mean_iat": mean_iat,
        "std_iat": std_iat,
        "first20": first20,
        "dports": data.get('dports',[]),
        "last_seen": data.get('last', None),
        "first_seen": data['times'][0] if len(data['times'])>0 else None
    }
    outfh.write(json.dumps(rec) + "\n")
    outfh.flush()

def packet_handler(pkt):
    key = canonical_key(pkt)
    if key is None:
        return
    ts = float(pkt.time)
    size = len(pkt)
    entry = flows.setdefault(key, {'times':[], 'sizes':[], 'last':ts, 'dports':[]})
    entry['times'].append(ts)
    entry['sizes'].append(size)
    entry['last'] = ts
    # record dport if TCP/UDP present
    if TCP in pkt or UDP in pkt:
        try:
            entry['dports'].append(int(pkt.dport))
        except:
            pass
    # flush if enough pkts
    if len(entry['sizes']) >= args.min_pkts:
        flush_flow(key, entry, outfh)
        del flows[key]

def idle_flusher():
    while True:
        now = time.time()
        to_del = []
        for k,v in list(flows.items()):
            if now - v['last'] > args.idle_timeout:
                flush_flow(k, v, outfh)
                to_del.append(k)
        for k in to_del:
            del flows[k]
        time.sleep(0.5)

if __name__ == "__main__":
    print("Starting scapy sniff on", args.iface)
    outfh = open(args.out, "a", buffering=1)
    import threading
    t = threading.Thread(target=idle_flusher, daemon=True)
    t.start()
    sniff(iface=args.iface, prn=packet_handler, store=False)
