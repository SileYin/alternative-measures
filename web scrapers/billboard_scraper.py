import json
import re
import requests
from bs4 import BeautifulSoup
import sqlite3
import datetime

def get_songs_and_artists(soup):       
    
    #print(html)
    resultstable = soup.find_all("table")

    #print('type ',type(resultstable))
    rs2 = soup.table
    #print(rs2.children)
    billboard_tracks = []
    h = 0
    for row in rs2.find_all("tr"):
        print("r ",row)
        if h == 0:
            h += 1
            continue
        i = 0
        track = []
        for column in row.find_all("td"):
            if i == 1:
                artist = None
                for a in column.find_all('a'):
                    print("artist",a.string)
                    artist = a.string
                if artist is None:
                    artist = column.string.strip()
                #print("artist ",column.find_all("a").string," ",type(column))
            elif i == 2:
                print("title ",column.a.string)
                title = column.a.string
            elif i == 3:
                #print("date ",column.a.string)
                cht_date = column.a.string
            elif i == 4:
                #print("rank ",column.string.strip())
                rank = column.string.strip()
            elif i == 5:
                #print("weeks on ",column.string.strip())
                wks_on = column.string.strip()
            elif i == 6:
                #print("label ",column.string.strip())
                label = column.string.strip()
            i += 1
        print("h ",h,"ar ",artist)
        c.execute('INSERT INTO billboard_tracks VALUES (?,?,?,?,?,?)', (artist,title,cht_date,rank,wks_on,label))
        conn.commit()
        h += 1
            #billboard_tracks.append(track)
            
    

    # for a in child.a:
    #     print("a ",a)

#resultslines = soup2.find_all("a")
# for line in resultslines:
#     print("line ",line)
#print("XXXXXXXXXXXXXXXXXXXXXXXXXXX",resultstable)
def advance_week(fromdate):
    #fromdate = '1980-01-03'
    #todate = '1980-01-09'
    fmdt = datetime.datetime.strptime(fromdate,'%Y-%m-%d') 
    nxtdt = fmdt + datetime.timedelta(days = 7)
    nxttodate = nxtdt + datetime.timedelta(days = 6)
    fromdate = datetime.datetime.strftime(nxtdt,'%Y-%m-%d')
    todate = datetime.datetime.strftime(nxttodate,'%Y-%m-%d')
    print("fm ",fromdate,"nxt ",todate)
    return fromdate,todate
conn = sqlite3.connect('../../sp_data.db')
c = conn.cursor()
fromdate = '1979-12-27'
todate = '1980-01-02'
while datetime.datetime.strptime(fromdate,'%Y-%m-%d').year < 1990:
    fromdate,todate = advance_week(fromdate)
    print('f ',fromdate,'t ',todate)

    #url = 'https://www.billboard.com/biz/search/charts?f[0]=itm_field_chart_id%3AHot%20100&f[1]=ss_bb_type%3Achart&f[2]=ds_chart_date%3A%5B1980-01-01T05%3A00%3A00Z%20TO%201980-01-09T05%3A00%3A00Z%5D&type=4&date=1980-01-05'
    url = 'https://www.billboard.com/biz/search/charts?f[0]=itm_field_chart_id%3AHot%20100&f[1]=ss_bb_type%3Achart&f[2]=ds_chart_date%3A%5B' + fromdate + 'T05%3A00%3A00Z%20TO%20' + todate + 'T05%3A00%3A00Z%5D&type=4&date=' + fromdate

    html = requests.get(url).content.decode("utf-8") 
    soup = BeautifulSoup(html, "html.parser")
    get_songs_and_artists(soup)

    rs3 = soup.find_all("nav",class_="paginator")

    for row in rs3:
        for li in row.find_all("li",class_="pager-item"):
            print("href ",li.a.get('href'))
            url = 'https://www.billboard.com' + li.a.get('href')
            
            html = requests.get(url).content.decode("utf-8") 
            soup = BeautifulSoup(html, "html.parser")
            get_songs_and_artists(soup)




