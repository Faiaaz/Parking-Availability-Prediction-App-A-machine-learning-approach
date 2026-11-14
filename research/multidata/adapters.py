"""Independent adapters with explicit measurement and availability provenance."""
import io
from pathlib import Path
import numpy as np
import pandas as pd
from research.run_experiments import clean_data


def common(frame, dataset):
    frame = frame.sort_values(['zone', 'time']).drop_duplicates(['zone','time'])
    frame['dataset_id'] = dataset
    frame['site_id'] = frame.zone
    frame['observed_at'] = frame.time
    frame['available_at'] = frame.time
    frame['availability_assumption'] = 'instantaneous; receipt timestamp unavailable'
    frame['occupancy_ratio'] = frame.rate
    frame['occupied_count'] = frame.occupancy
    frame['available_count'] = frame.capacity - frame.occupancy
    frame['observation_status'] = 'measured_sensor_reference'
    return frame


def birmingham(path):
    frame, audit = clean_data(path)
    return common(frame, 'birmingham'), audit


def harvard(path):
    data = Path(path).read_bytes()
    # A byte-range prefix ends mid-record. Retain complete lines only.
    raw = pd.read_csv(io.BytesIO(data[:data.rfind(b'\n')+1]), sep=';')
    frame = raw[['timestamp','segmentid','capacity','occupied']].rename(
        columns={'timestamp':'time','segmentid':'zone','occupied':'occupancy'})
    frame.time = pd.to_datetime(frame.time)
    frame.zone = frame.zone.astype(str)
    valid = (frame.capacity > 0) & frame.occupancy.between(0,frame.capacity)
    audit = {'raw_rows':len(frame),'invalid_rows':int((~valid).sum()),
             'raw_sites':frame.zone.nunique(),'simulation_columns_excluded':list(raw.columns[4:]),
             'timezone':'unspecified in downloaded PDF; naive local clock retained'}
    frame = frame.loc[valid].copy()
    frame['rate'] = frame.occupancy/frame.capacity
    return common(frame,'harvard_sfpark'), audit


def melbourne_intervals(path):
    """Retrospective event labels. Availability conservatively set to interval end."""
    raw = pd.read_csv(path)
    out = pd.DataFrame({'site_id':raw.BayId.astype(str),
                       'observed_at':pd.to_datetime(raw.ArrivalTime,format='%m/%d/%Y %I:%M:%S %p'),
                       'end':pd.to_datetime(raw.DepartureTime,format='%m/%d/%Y %I:%M:%S %p')})
    out['occupied_count'] = raw.VehiclePresent.map({True:1,False:0})
    out['capacity'] = 1
    out['available_at'] = out.end
    out['dataset_id'] = 'melbourne_2019'
    out['observation_status'] = 'reconstructed_event_interval'
    return out[out.end > out.observed_at]


def event_state(intervals, site, timestamp):
    """Unknown for gaps or conflicting overlap; never fill a gap as vacancy."""
    rows = intervals[(intervals.site_id==site)&(intervals.observed_at<=timestamp)&(intervals.end>timestamp)]
    states = rows.occupied_count.dropna().unique()
    return float(states[0]) if len(states)==1 else np.nan
