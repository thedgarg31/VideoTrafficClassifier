import argparse, json, csv
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--infile", required=True)
parser.add_argument("--outfile", default="flow_features.csv")
args = parser.parse_args()

def extract_features(rec):
    src, dst, sport, dport, proto = rec["flow_key"]

    # basic
    duration = rec.get("duration", 0.0)
    pkts = rec.get("pkt_count", 0)
    total_bytes = rec.get("total_bytes", 0)

    # throughput
    throughput = (total_bytes * 8 / duration) if duration > 0 else 0.0

    # packet size stats
    sizes = np.array(rec.get("first20", []))
    avg_pkt = rec.get("avg_pkt", 0.0)
    min_pkt = int(sizes.min()) if len(sizes) > 0 else 0
    max_pkt = int(sizes.max()) if len(sizes) > 0 else 0
    median_pkt = float(np.median(sizes)) if len(sizes) > 0 else 0.0

    # timing
    mean_iat = rec.get("mean_iat", 0.0)
    std_iat = rec.get("std_iat", 0.0)

    # directionality (rough: small port → client, big port → server)
    client, server = (src, dst)
    if sport < 1024 and dport > 1024:
        client, server = dst, src
    upload_bytes = total_bytes / 2.0   # crude split (no dir info in JSON)
    download_bytes = total_bytes / 2.0
    up_down_ratio = (upload_bytes / download_bytes) if download_bytes > 0 else 0.0

    # app protocol guess
    service_map = {
        80: "HTTP", 443: "HTTPS", 53: "DNS", 22: "SSH",
        25: "SMTP", 110: "POP3", 143: "IMAP", 1935: "RTMP",
        3478: "STUN/TURN", 123: "NTP"
    }
    app_proto = service_map.get(int(dport), service_map.get(int(sport), "UNKNOWN"))

    return {
        "src": src,
        "dst": dst,
        "sport": sport,
        "dport": dport,
        "proto": proto,
        "duration": duration,
        "pkt_count": pkts,
        "total_bytes": total_bytes,
        "throughput_bps": throughput,
        "avg_pkt": avg_pkt,
        "min_pkt": min_pkt,
        "max_pkt": max_pkt,
        "median_pkt": median_pkt,
        "mean_iat": mean_iat,
        "std_iat": std_iat,
        "up_down_ratio": up_down_ratio,
        "app_proto": app_proto
    }

if __name__ == "__main__":
    features = []
    with open(args.infile) as f:
        for line in f:
            rec = json.loads(line)
            feat = extract_features(rec)
            features.append(feat)

    # write CSV
    if features:
        keys = list(features[0].keys())
        with open(args.outfile, "w", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=keys)
            writer.writeheader()
            writer.writerows(features)
    print(f"Wrote {len(features)} flows to {args.outfile}")
