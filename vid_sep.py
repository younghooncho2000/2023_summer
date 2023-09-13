video_path = "F:/python_projects/2023_summer_teamproject/StoryGen/video"
txt_path =  'F:/python_projects/2023_summer_teamproject/StoryGen/youtube_datas_younghoon/txts/'

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import glob
import os

files = glob.glob('./video/*.vtt')
for file in files기
    os.remove(file) # mp4 아닌거 날리기
os.listdir("F:/python_projects/2023_summer_teamproject/StoryGen/youtube_datas_younghoon/txts")
#Replace the filename below.
vid_files = os.listdir(video_path)
i = 2000 #   분할된 비디오 번호 임의로 설정
for required_video_file in vid_files:
    print(required_video_file, required_video_file[:-4])
    
    timestamp_path = required_video_file[:-4] + '.txt'
    if timestamp_path in os.listdir("F:/python_projects/2023_summer_teamproject/StoryGen/youtube_datas_younghoon/txts"):
#         os.remove(video_path+'/'+required_video_file)
#         print('존재합니까')
        with open(txt_path + timestamp_path) as f:
            times = f.readlines()
            print(times)
        times = [x.strip() for x in times] 
        for time in times:
            starttime = int(time.split("-")[0])
            endtime = int(time.split("-")[1])
            
            print(starttime, endtime, required_video_file)
            ffmpeg_extract_subclip(video_path+'/'+required_video_file, starttime, endtime, targetname = str(("%06d" % i)+".mp4"))
            i += 1
