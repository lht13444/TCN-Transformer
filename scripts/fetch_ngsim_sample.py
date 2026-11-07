"""Fetch the 12 NGSIM US-101 example trajectories with source verification."""
from pathlib import Path
import csv
import hashlib
import json
import ssl
import urllib.parse
import urllib.request

OUT = Path(__file__).resolve().parents[1] / 'data' / 'raw_samples'
BASE = 'https://data.transportation.gov/resource/8ect-6jqj.json'
METADATA_URL = 'https://data.transportation.gov/api/views/8ect-6jqj.json'
# vehicle_id, total_frames, first global_time, last global_time. Vehicle IDs
# repeat between recording periods, so the time bounds are essential.
TRAJECTORIES = (
    (2,437,1118846980200,1118847023800),
    (6,577,1118847871800,1118847929400),
    (9,714,1118848766000,1118848837300),
    (11,654,1118847866400,1118847931700),
    (12,658,1118848770000,1118848835700),
    (20,636,1118847870500,1118847934000),
    (31,465,1118846985200,1118847031600),
    (37,408,1118846990900,1118847031600),
    (39,756,1118848782800,1118848858300),
    (40,391,1118846989300,1118847028300),
    (43,708,1118848778800,1118848849500),
    (44,379,1118846991700,1118847029500),
)

def fetch(url):
    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = ssl.create_default_context()
    request = urllib.request.Request(url, headers={
        'User-Agent':'NGSIM-fixed-small-example/1.0','Accept':'application/json'})
    with urllib.request.urlopen(request, context=context, timeout=60) as response:
        return response.read()

def sample_api_url():
    clauses = [f'(vehicle_id={v} AND total_frames={n} AND global_time between {a} and {b})'
               for v,n,a,b in TRAJECTORIES]
    return BASE + '?' + urllib.parse.urlencode({
        '$where':"location='us-101' AND ("+' OR '.join(clauses)+')',
        '$order':'vehicle_id,global_time,frame_id', '$limit':6784})

def main():
    metadata_raw = fetch(METADATA_URL)
    metadata = json.loads(metadata_raw)
    if metadata.get('licenseId') != 'CC_30_BY_SA':
        raise RuntimeError('The official data-license metadata changed. Review it before redistribution.')
    rows = json.loads(fetch(sample_api_url()))
    if len(rows) != 6783:
        raise RuntimeError(f'Expected 6783 rows from the fixed selection, received {len(rows)}.')
    for vehicle,frames,start,end in TRAJECTORIES:
        trajectory = [r for r in rows if int(r['vehicle_id'])==vehicle]
        times = [int(r['global_time']) for r in trajectory]
        frame_ids = [int(r['frame_id']) for r in trajectory]
        if (len(trajectory)!=frames or times[0]!=start or times[-1]!=end
                or any(b-a!=100 for a,b in zip(times,times[1:]))
                or any(b-a!=1 for a,b in zip(frame_ids,frame_ids[1:]))):
            raise RuntimeError(f'Trajectory {vehicle} differs from the pinned continuous selection.')
    OUT.mkdir(parents=True,exist_ok=True)
    json_path = OUT/'ngsim_us101_12vehicles.json'
    json_path.write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    csv_path = OUT/'ngsim_us101_12vehicles.csv'
    with csv_path.open('w',encoding='utf-8',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Keep the original collection provenance. Record the re-fetch separately.
    receipt = {'data_api_url':sample_api_url(),'metadata_api_url':METADATA_URL,
               'row_count':len(rows),'vehicle_count':12,'license_id':metadata['licenseId'],
               'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [csv_path,json_path]}}
    (OUT/'last_fetch_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(f'Downloaded {len(rows)} original 10 Hz rows for 12 vehicles to {csv_path}')

if __name__ == '__main__':
    main()
