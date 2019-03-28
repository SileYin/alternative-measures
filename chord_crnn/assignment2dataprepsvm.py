from music21 import *
import sqlite3
import plotly
#import numpy as np
import plotly.graph_objs as go
import operator
import numpy as np
plotly.tools.set_credentials_file(username='beachc', api_key='iCs0yeRySFPZ2pEyfQvn')
conn = sqlite3.connect('../../musi7100/sp_data.db')
c = conn.cursor()
c1 = conn.cursor()
c2 = conn.cursor()
track_id = '4iEuVk3bKkZcu8vcv7gR4I'
trks = []
i = 0
#pcvector = [[0] * 13 for i in range(1)]

# labelmatrix = np.empty([3000,1])
# pcalltrackmatrix = np.empty([3000,1000,12])
# tballtrackmatrix = np.empty([3000,1000,12])
labelmatrix = []
pcalltrackmatrix = []
tballtrackmatrix = []
for trk in (c.execute("select track_id,key,key_confidence from track_analysis group by track_id,key,key_confidence order by key_confidence desc")):
    track_id = trk[0]
    pckeylist = [track_id]
    tbkeylist = [track_id]
    key = int(trk[1])
    labelvector = np.empty([1])
    #labelvector[0,0] = track_id
    labelvector[0] = key
    #pcvector[0][0] = row[0]
    #pcvector[0] = row[1]
    # kv = np.chararray([1])
    # kv = track_id
    # print("kv ",kv)
    # pctrkmatrix[0] = kv
    # print("pctm ",pctrkmatrix[0])
    j = 0
    #pctrkmatrix = np.empty([1000,12])
    pctrkmatrix = []
    #pctrkmatrix = []
    for row in c1.execute("SELECT start,duration,pitch1,pitch2,pitch3,pitch4,pitch5,pitch6,pitch7,pitch8,pitch9,pitch10,pitch11,pitch12 from track_segment_pitch where track_id = '" + track_id + "' group by start,duration, pitch1,pitch2,pitch3,pitch4,pitch5,pitch6,pitch7,pitch8,pitch9,pitch10,pitch11,pitch12 order by start"):
        pcvector = np.zeros(12)
        #pcvector[0] = row[0]
        #pcvector[1] = key
        pcvector = row[2:]
        #print(pcvector[2])
        #pctrkmatrix[j] = pcvector
        pctrkmatrix.append(pcvector) 
        j += 1
    #     if j > 999:
    #         break
    # for k in range(j,1000):
    #     pctrkmatrix[j] = np.zeros([12])
    # if len(pctrkmatrix) > 0:
    if len(pctrkmatrix) > 0:
        pcalltrackmatrix.append(pctrkmatrix)
    # else:
    #     continue
    # tbtrkmatrix[0][i] = kv
    j = 0
    tbtrkmatrix = []
    #tbtrkmatrix = np.empty([1000,12])
    for row in c2.execute("SELECT start,duration,timbre1,timbre2,timbre3,timbre4,timbre5,timbre6,timbre7,timbre8,timbre9,timbre10,timbre11,timbre12 from track_segment_timbre where track_id = '" + track_id + "' group by start,duration,timbre1,timbre2,timbre3,timbre4,timbre5,timbre6,timbre7,timbre8,timbre9,timbre10,timbre11,timbre12 order by start"):
        tbvector = np.zeros(12)
        #tbvector[0] = row[0]
        #tbvector[1] = key
        tbvector = row[2:]
        #print(pcvector[2])
        tbtrkmatrix.append(tbvector)
    #     j += 1
    #     if j > 999:
    #         break
    # for k in range(j,1000):
    #     tbtrkmatrix[j] = np.zeros([12])
    if len(pctrkmatrix) > 0:
        tballtrackmatrix.append(tbtrkmatrix)
        labelmatrix.append(labelvector)
        i += 1
    #
    if i > 9999:
        break
#print("nptimbre shape ",pcalltrackmatrix) #,nptimbre.shape)
# pcfinalmatrix = np.ndarray(pcalltrackmatrix)
# print("pcfinalmatrix ",pcfinalmatrix.shape)
# tbfinalmatrix = np.ndarray(tballtrackmatrix)
# labelfinalmatrix = np.ndarray(labelmatrix)
# print("labelmatrix ",labelfinalmatrix.shape)
print("val shape ",len(pcalltrackmatrix),",",len(pcalltrackmatrix[1])) #,",",len(pcalltrackmatrix[1][0]))
print("val ",pcalltrackmatrix[0][1][0])
# x_a=np.array([np.array(xi) for xi in pcalltrackmatrix])
# print('xa ',x_a[0][0])
# print("pitch shape ", x_a.shape)
# x_b=np.array([np.array(xi) for xi in tballtrackmatrix])
# print('xb ',x_b[0][0])
# print("timbre shape ", x_b.shape)
# y=np.array([np.array(xi) for xi in labelmatrix])
print("tb shape ",len(tballtrackmatrix[0]),",",len(tballtrackmatrix[0][0]))
print("tb ",tballtrackmatrix[0][0][0])
print("label shape ",len(labelmatrix))
np.save('chroma_train_key.npy',pcalltrackmatrix)
np.save('timbre_train_key.npy',tballtrackmatrix)
np.save('label_train_key.npy',labelmatrix) 
    
