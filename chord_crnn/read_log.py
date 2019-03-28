import argparse
import os
import numpy as np
import pickle
import plotly
import plotly.graph_objs as go
fp = open('data/13.log','rb')
chorddict = pickle.load(fp)
oldsec = 0
quality_dict = {'maj':0,'min':1}
conf_matrix = np.zeros((27,27))
for sec in range(len(chorddict)):
    # print("c ",chorddict[sec][0],oldsec)
    # while chorddict[sec][0] == oldsec:
    #     pass
        #print('sec ',chorddict[sec][0],'pred ',chorddict[sec][1],'lab ',chorddict[sec][2])
    #print('sec ',chorddict[sec][0],'pred ',chorddict[sec][1],'lab ',chorddict[sec][2])
    
    predicted_key = chorddict[sec][1]
    if predicted_key == 'N':
        predicted_index = 24
    elif predicted_key == 'X':
        predicted_index = 25
    elif predicted_key == '':
        predicted_index = 26
    else:
        predicted_key = int(chorddict[sec][1].split(":")[0])
        predicted_quality = chorddict[sec][1].split(":")[1]
    correct_key = chorddict[sec][2]
    if correct_key == 'N':
        correct_index = 24
    elif correct_key == 'X':
        correct_index = 25
    elif correct_key == '':
        correct_index = 26
    else:
        correct_key = int(chorddict[sec][2].split(":")[0])
        correct_quality = chorddict[sec][2].split(":")[1]
    if isinstance(predicted_key,int):
        predicted_index = predicted_key * 2 + quality_dict[predicted_quality]
    if isinstance(correct_key,int):
        correct_index = correct_key * 2 + quality_dict[correct_quality]
    conf_matrix[predicted_index][correct_index] += 1
root_correct = 0
for x in range(12):
    ix = x * 2
    print('ix ',ix,'ix +2',ix + 2)
    root_correct += np.sum(conf_matrix[ix:ix+2,ix:ix+2]) 
    print("rc x ",root_correct)
    print("rc m ",conf_matrix[ix:ix+2,ix:ix+2])
print("rc ",root_correct,"all ",np.sum(conf_matrix))
    
xlegend = []
ylegend = []
qual_list = ['maj','min']
xtick = ['C','C#','D','Eb','E','F','F#','G','G#','A','Bb','B']
ytick = ['C','C#','D','Eb','E','F','F#','G','G#','A','Bb','B']
for i in range(12):
    yindices = []
    xindices = []
    for qual in range(len(qual_list)):
        xlegend.append(xtick[i] + qual_list[qual] )
        ylegend.append(ytick[i] + qual_list[qual] )
xlegend.append('N')
xlegend.append('X')
xlegend.append(' ')
ylegend.append('N')
ylegend.append('X')
ylegend.append(' ')
print(conf_matrix)
mtrace = go.Heatmap(z = conf_matrix,name='Confustion Matrix ',legendgroup='None',
                x = xlegend,
                y = ylegend)
# trace = go.Table(
#     header=dict(values=xlegend,
#                 fill = dict(color='#C2D4FF'),
#                 align = ['left'] * 5),
#     cells=dict(values=[df.Rank, conf_matrix[0,:], conf_matrix[1,:], conf_matrix[2,:],conf_matrix[3,:],conf_matrix[4,:],
#                         conf_matrix[5,:],conf_matrix[6,:],conf_matrix[7,:],conf_matrix[8,:],conf_matrix[9,:],conf_matrix[10,:],
#                         conf_matrix[11,:],conf_matrix[12,:],conf_matrix[13,:],conf_matrix[14,:],conf_matrix[15,:],conf_matrix[16,:],
#                         conf_matrix[17,:],conf_matrix[18,:],conf_matrix[19,:],conf_matrix[20,:],conf_matrix[21,:],conf_matrix[22,:],
#                         conf_matrix[23,:],conf_matrix[24,:],conf_matrix[25,:],conf_matrix[26,:],conf_matrix[21,:],conf_matrix[22,:],
#                fill = dict(color='#F5F8FF'),
#                align = ['left'] * 5))

data1 = [mtrace]
layout = go.Layout(
    title= 'Confusion Matrix',
    yaxis=dict(
        title='Correct'
    ),
    xaxis=dict(
        title='Predicted'
    )
)
fig1 = go.Figure(data=data1, layout=layout)



plotly.offline.plot(fig1,auto_open=True)
    #oldsec = chorddict[sec][0]