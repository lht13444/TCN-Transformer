"""Fetch six real highD trajectories from TUM's CommonRoad tutorial."""
from pathlib import Path
import csv
import hashlib
import io
import json
import ssl
import urllib.parse
import urllib.request

COMMIT = '78cbeeadda8ed19abcc7fefd9d0950d5a79515a7'
PROJECT = 'https://gitlab.lrz.de/tum-cps/commonroad-rl'
API = 'https://gitlab.lrz.de/api/v4/projects/75284/repository/files/'
SOURCE_DIR = 'commonroad_rl/tutorials/data/highD/raw/data/'
OUT = Path(__file__).resolve().parents[1] / 'data' / 'highd_sample'
UPSTREAM = {
    '03_tracks.csv': {'bytes':34929994,'sha256':'28fb1938287f03975f38bea5c6f8d2fabe2a964f11287a48c38fbdc758524958'},
    '03_tracksMeta.csv': {'bytes':73281,'sha256':'0051fb82678076a8ba511cc0e13b3e2785ffdd3a4d8ead645af99aba20583f24'},
    '03_recordingMeta.csv': {'bytes':280,'sha256':'9250b0d3c1623c9a86c51bdca760c0e34e64ed2e98df189eec0b07dc8a0b8f20'},
}

def source_url(name):
    return API + urllib.parse.quote(SOURCE_DIR+name,safe='') + '/raw?ref=' + COMMIT

def open_source(name):
    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = ssl.create_default_context()
    request = urllib.request.Request(source_url(name),headers={'User-Agent':'highD-public-tutorial-example/1.0'})
    return urllib.request.urlopen(request,context=context,timeout=60)

def checked_small_file(name):
    with open_source(name) as response:
        raw = response.read(200001)
    if len(raw)!=UPSTREAM[name]['bytes'] or hashlib.sha256(raw).hexdigest()!=UPSTREAM[name]['sha256']:
        raise RuntimeError('Pinned source metadata does not match: '+name)
    return raw

def main():
    metadata_raw = checked_small_file('03_tracksMeta.csv')
    recording_raw = checked_small_file('03_recordingMeta.csv')
    meta_rows = list(csv.DictReader(io.StringIO(metadata_raw.decode('utf-8'))))
    candidates = [r for r in meta_rows if int(r['numLaneChanges'])>0 and int(r['numFrames'])>=120]
    selected = sorted(candidates,key=lambda r:int(r['id']))[:6]
    if len(selected)!=6:
        raise RuntimeError('Expected six lane-changing trajectories in the pinned tutorial data.')
    ids = {int(r['id']) for r in selected}
    last_id = max(ids)
    kept_lines = []
    source_bytes_read = 0
    previous_id = -1
    with open_source('03_tracks.csv') as response:
        header = response.readline()
        source_bytes_read += len(header)
        fields = next(csv.reader([header.decode('utf-8').strip()]))
        if fields[:4]!=['frame','id','x','y'] or 'laneId' not in fields:
            raise RuntimeError('Unexpected highD trajectory schema.')
        for raw in response:
            source_bytes_read += len(raw)
            if not raw.strip():
                continue
            row = next(csv.reader([raw.decode('utf-8').strip()]))
            vehicle = int(row[1])
            if vehicle < previous_id:
                raise RuntimeError('The pinned source is no longer ordered by vehicle.')
            previous_id = vehicle
            if vehicle>last_id:
                break
            if vehicle in ids:
                kept_lines.append(raw)
    tracks_raw = header+b''.join(kept_lines)
    track_rows = list(csv.DictReader(io.StringIO(tracks_raw.decode('utf-8'))))
    trajectories=[]
    for meta in selected:
        trajectory=[r for r in track_rows if r['id']==meta['id']]
        frames=[int(r['frame']) for r in trajectory]
        lanes=[int(r['laneId']) for r in trajectory]
        if len(trajectory)!=int(meta['numFrames']) or any(b-a!=1 for a,b in zip(frames,frames[1:])):
            raise RuntimeError('Incomplete trajectory '+meta['id'])
        changes=[{'frame':frames[i],'from_lane':lanes[i-1],'to_lane':lanes[i],
                  'has_40_before_and_40_from_change':40<=i<=len(trajectory)-40}
                 for i in range(1,len(trajectory)) if lanes[i]!=lanes[i-1]]
        trajectories.append({'id':int(meta['id']),'row_count':len(trajectory),'first_frame':frames[0],
                             'last_frame':frames[-1],'lane_changes':changes})
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'03_tracks.csv').write_bytes(tracks_raw)
    # Preserve selected metadata rows byte-for-byte, with the original header.
    meta_lines=metadata_raw.splitlines(keepends=True)
    selected_meta=meta_lines[0]+b''.join(line for line in meta_lines[1:] if line.strip() and int(line.split(b',',1)[0]) in ids)
    (OUT/'03_tracksMeta.csv').write_bytes(selected_meta)
    (OUT/'03_recordingMeta.csv').write_bytes(recording_raw)
    source_files={name:{'api_url':source_url(name),'web_url':PROJECT+'/-/blob/'+COMMIT+'/'+SOURCE_DIR+name,
                        'upstream_bytes':v['bytes'],'upstream_sha256_from_gitlab':v['sha256']}
                  for name,v in UPSTREAM.items()}
    provenance={
        'dataset':'highD recording 03 real-trajectory subset',
        'dataset_owner':'RWTH Aachen University highD dataset / leveLXData',
        'tutorial_publisher':'Technical University of Munich CommonRoad project',
        'tutorial_page':'https://commonroad.in.tum.de/tutorials/static_docs_tutorials_data_preprocessing.html',
        'tutorial_repository':PROJECT,'commit':COMMIT,'access_date':'2026-11-07',
        'highd_official_page':'https://levelxdata.com/highd-dataset/',
        'source_files':source_files,'source_track_bytes_streamed':source_bytes_read,
        'source_recording_vehicle_count':914,'local_vehicle_count':len(ids),'local_trajectory_rows':len(track_rows),
        'sample_rate_hz':25,'timestep_seconds':0.04,'units':{'x':'m longitudinal','y':'m lateral',
            'xVelocity':'m/s longitudinal','yVelocity':'m/s lateral','xAcceleration':'m/s^2','yAcceleration':'m/s^2'},
        'selection':'First six vehicle IDs in the pinned metadata with numLaneChanges>0 and numFrames>=120; complete trajectories retained.',
        'changes':'Removed all other vehicle rows from tracks and tracksMeta. Selected source rows remain byte-for-byte unchanged. No smoothing, resampling, interpolation or synthesized values. Recording metadata remains original and describes the source recording, not the six-vehicle subset.',
        'data_terms':'Use of highD data follows the original dataset terms at highd_official_page.',
        'software_license_url':PROJECT+'/-/blob/'+COMMIT+'/LICENSE',
        'highd_paper_doi':'https://doi.org/10.1109/ITSC.2018.8569552',
        'trajectories':trajectories,
        'local_files':{p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
                       for p in [OUT/'03_tracks.csv',OUT/'03_tracksMeta.csv',OUT/'03_recordingMeta.csv']}
    }
    (OUT/'PROVENANCE.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'rows':len(track_rows),'vehicle_ids':sorted(ids),'lane_changes':sum(len(t['lane_changes']) for t in trajectories),
                      'source_track_bytes_streamed':source_bytes_read,'local_track_bytes':len(tracks_raw)},indent=2))

if __name__=='__main__':
    main()
