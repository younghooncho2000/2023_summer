#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#download youtube VIDs 
url_txt = 'vidURL.txt' # txt file with youtube URLs . PLZ check path
with open(url_txt) as f:
    urls = f.readlines()
    urls = list(map(lambda s: s.strip(), urls))

url_list = []

i = 0 # 000000.mp4, 000001.mp4, 000002.mp4 ...
for x in urls:
    list1 = []
    list2 = []
    list2.append(x)
    list1.append(list2)
    list1.append('%06d'%i)
    url_list.append(list1)
    i += 1

import yt_dlp as y
ydl_opts = {
    'format':'135',
    'outtmpl':'video\%(title)s.%(ext)s',
    'writeautomaticsub' : False
}

cnt = 0
for u in url_list:
    ydl_opts['outtmpl'] = 'video\{}.%(ext)s'.format(u[1])
    print(ydl_opts['outtmpl'])
    with y.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download(u[0])
        except:
            continue

