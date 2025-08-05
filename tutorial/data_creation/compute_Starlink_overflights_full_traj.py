import pyproj
from skyfield.api import load, wgs84, EarthSatellite, Star
from skyfield.data import hipparcos
from pathlib import Path
from urllib.request import urlopen
from sgp4 import omm
from sgp4.api import Satrec
from datetime import datetime, timedelta
import math as mt
from operator import itemgetter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# convert time to string
def safe_time_str(dt):
    return dt.strftime('%Y-%m-%dT%H_%M_%S.%f')[:-3]  # up to milliseconds


### TELSCOPES POSITIONS ###

# Westford coordinates
WESTFORD_X = 1492206.5970
WESTFORD_Y = -4458130.5170
WESTFORD_Z = 4296015.5320
# Observed offset from WGS84 ellipsoid location and Westford STK file
# Not sure which is really right. Need to check with Chet / Diman
WESTFORD_Z_OFFSET = 0.1582435

transformer = pyproj.Transformer.from_crs(
    {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
    {"proj":'latlong', "ellps":'WGS84', "datum":'WGS84'},
    )
lon1, lat1, alt1 = transformer.transform(WESTFORD_X,WESTFORD_Y,WESTFORD_Z,radians=False)
print (lat1, lon1, alt1 )

WESTFORD_LAT = lat1
WESTFORD_LON = lon1
WESTFORD_ALT = alt1 + WESTFORD_Z_OFFSET

# observers location
westford = wgs84.latlon(WESTFORD_LAT,WESTFORD_LON, WESTFORD_ALT)


### LOAD OBJECTS ###

# time scale
ts = load.timescale()

# load starlink library
starlink_url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=csv"#"http://celestrak.org/NORAD/elements/supplemental/sup-gp.php?FILE=starlink&FORMAT=csv"#"https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=csv"
active = Path("traj_files/Starlink_active.csv")
data = urlopen(starlink_url)
active.write_bytes(data.read())

starlink_sats = []
with open(active) as f:
    for fields in omm.parse_csv(f):
        sat = Satrec()
        omm.initialize(sat, fields)
        e = EarthSatellite.from_satrec(sat, ts)
        e.name = fields.get('OBJECT_NAME')
        starlink_sats.append(e)

print('Loaded', len(starlink_sats), 'satellites')



### OBSERVATION ###

# time of observation
fmt = '%Y-%m-%dT%H:%M:%S.%f'
t0 = ts.utc(2025,4,1,12,30,00)
t1 = ts.utc(2025,4,1,13,30,00)
# bug fix
dt0 = t0.utc_datetime()
dt1 = t1.utc_datetime()

# time resolution of trajectories
time_step = timedelta(milliseconds=1000)
time_round = '1000ms'
# storage of sats passing at Westford
traj_sat_w = []

# overflight trajectories of the dtc satellites seen by Westford
for sat in starlink_sats:
    # distance from sat to Westford over time
    diff_w = sat - westford
    # each time sat is seen by telescope (stored in t as rise - culm - set
    # times cycles and in events as 0 1 2)
    t_w, events_w = sat.find_events(westford, t0, t1, altitude_degrees=5.0)
    # for each time sat rises as seen by Westford
    for ind_t_w in range(0, len(t_w)-2, 3):
        # time of rise of sat in Westford
        t_w_r = t_w[ind_t_w]
        # bug fix
        dt_w_r = t_w_r.utc_datetime()
        # time of set of sat in Westford
        t_w_s = t_w[ind_t_w+2]
        # bug fix
        dt_w_s = t_w_s.utc_datetime()
        # get the beginning of the sample in sync with a rounded time_step
        dt_beg_sync = pd.Timestamp(dt_w_r).round(freq=time_round).to_pydatetime()
        # temporal grid of sat
        rise_to_set_range = pd.date_range(dt_beg_sync, dt_w_s, freq=time_step,
                                          tz='UTC')
        # compute positions of sat during range
        for t_rs in rise_to_set_range:
            # sat position relative to westford at t_rs
            diff_t_rs = diff_w.at(ts.from_datetime(t_rs))
            # translate in angles and distance
            ang_t_rs = diff_t_rs.altaz()
            els = ang_t_rs[0].degrees
            azs = ang_t_rs[1].degrees
            dis_w = ang_t_rs[2].m
            traj_sat_w.append((t_rs.strftime(fmt)[:-3], sat.name, els, azs, dis_w))

# # plot satellites trajectory
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1, polar=True)
# for s in traj_sat_w:
#     # this code is not working
#     # ax.plot(np.deg2rad(s[4]), [90 - s3 for s3 in s[3]], color='blue')
#     # s[3] is azimuth (degrees), s[2] is elevation (degrees)
#     # Convert azimuth to radians for polar plot, use 90 - elevation for y-axis
#     ax.plot(np.deg2rad(s[3]), 90 - s[2], 'o', color='blue', markersize=2) # this code is working

# ax.set_yticks(range(0, 90, 10))
# ax.set_yticklabels(map(str, range(90, 0, -10)))
# ax.set_theta_zero_location("N")
# plt.show()

# sort by time
traj_sat_w.sort(key=itemgetter(0))

pass_df_w_traj = pd.DataFrame(traj_sat_w, columns=['timestamp', 'sat', 'elevations',
                                                   'azimuths', 'ranges_westford'])

# bug fix
start_str = safe_time_str(dt0)
end_str = safe_time_str(dt1)
filename = f"traj_files/Starlink_trajectory_Westford_{start_str}_{end_str}.arrow"
pass_df_w_traj.to_feather(filename)


###############################################################################3

# ANIMATION
