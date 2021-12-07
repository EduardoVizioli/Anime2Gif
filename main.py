import os
import cv2
import math
import time
import random
import numpy as np
from PIL import Image
from threading import Thread

K_VIDEOS_DIR = './videos'
K_TEMP_DIR = './temp'
K_GIF_FILENAME = 'random.gif'
K_GIF_WIDTH = 640
K_GIF_HEIGHT = 360

class Exceptions():
    class E_VIDEOS_FOLDER_EMPTY(Exception):
        pass

class Video():
    def __init__(self,dir,file,name):
        self.name         = name
        self.capture      = cv2.VideoCapture(dir+'/'+file)
        self.frames_count = self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
        self.fps          = self.capture.get(cv2.CAP_PROP_FPS)

    def go_to_frame(self,frame):
        self.capture.set(cv2.CAP_PROP_POS_FRAMES,frame)
    
    def get_current_frame_matrix(self,resolution):
        success, frame = self.capture.read()
        
        if success:
            frame = cv2.resize(frame, resolution, interpolation = cv2.INTER_AREA)
        else:
            frame = []
        
        return success,frame

    def get_current_timestamp(self):
        miliseconds = self.capture.get(cv2.CAP_PROP_POS_MSEC)
        seconds=str(math.floor((miliseconds/1000)%60)).zfill(2)
        minutes=str(math.floor((miliseconds/(1000*60))%60)).zfill(2)
        return minutes+'-'+seconds

class Utils():
    #Converts minutes to seconds
    def min_to_sec(minutes):
        return minutes*60

class GifGenerator():
    #Selects a random video from the videos folder.
    def get_random_video(self):
        filenames  = []
        filetypes  = ['mkv','mp4']
        video_name = ''
        video_file = ''

        for filename in os.listdir(K_VIDEOS_DIR):
            if any(filetype in filename for filetype in filetypes):
                filenames.append(filename)

        video_file = random.choice(filenames)

        video_name = video_file
        for filetype in filetypes:
            video_name = video_name.replace('.'+filetype,'')

        try:
            return Video(K_VIDEOS_DIR,video_file,video_name)
        except IndexError:
            raise Exceptions.E_VIDEOS_FOLDER_EMPTY

    #Calculates the difference between two frames.
    def calc_frames_difference_percentual(self,frame,previous_frame):
        absolute_difference = cv2.absdiff(frame, previous_frame).astype(np.uint8)
        percentual = np.sum(absolute_difference)/previous_frame.shape[0]/previous_frame.shape[1]/3.
        return percentual

    #Detects scene transitions between frames.
    def detect_transition(self,frame,previous_frame,threshold):
        if previous_frame != []:
            return self.calc_frames_difference_percentual(frame,previous_frame) >= threshold
        else:
            return False

    def analyze_and_get_frames(self,min_frames,max_frames,transition_threshold):
        video = self.get_random_video()
        start_at = random.randint(0,video.frames_count)
        frames = []
        dynamicness_arr = []
        frame = []
        previous_frame = []
        recording = False
        success = True
        start_timestamp = ''
        end_timestamp = ''

        video.go_to_frame(start_at)
        while True:
            success,frame = video.get_current_frame_matrix((K_GIF_WIDTH,K_GIF_HEIGHT))
            
            #if it reaches the end of the video, starts again in a random position.
            while not success:
                print('Reached the end of the video, reseting...')
                start_at = random.randint(0,video.frames_count)
                video.go_to_frame(start_at)
                frames = []
                previous_frame = []
                dynamicness_arr = []
                recording = False
                success,frame = video.get_current_frame_matrix((K_GIF_WIDTH,K_GIF_HEIGHT))
                start_timestamp = ''
                end_timestamp = ''
            
            if self.detect_transition(frame,previous_frame,transition_threshold):
                print('Detected transition')
                if (recording and len(frames) >= min_frames):
                    break
                elif (recording and len(frames) < min_frames):
                    frames = []
                    previous_frame = []
                    dynamicness_arr = []
                    start_timestamp = video.get_current_timestamp()
                elif not recording:
                    start_timestamp = video.get_current_timestamp()
                    recording = True
            
            if recording:
                frames.append(frame)
                frame_difference = 0
                if previous_frame != []:
                    frame_difference = self.calc_frames_difference_percentual(frame,previous_frame)
                    if len(frames) > 1:
                        dynamicness_arr.append(frame_difference)

                print('Frame added '+str(len(frames))+' ('+str(frame_difference)+'%)')

                if len(frames) >= max_frames:
                    break
                
            previous_frame = frame

        dynamicness = sum(dynamicness_arr)/len(dynamicness_arr)
        end_timestamp = video.get_current_timestamp()
        text = video.name+' from '+start_timestamp+' to '+end_timestamp
        return frames,dynamicness,text

    #Multithreaded converter, converts the numpy arrays into PIL images.
    def convert_frames_to_pil(self,frames,total_threads):
        pil_frames = {}
        def worker(number,total_threads):
            for i in range(number,len(frames),total_threads):
                print('Worker '+str(number)+' converting frame '+str(i))
                try:
                    frame = frames[i]
                    pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    pil_frames[i] = pil_frame.convert(mode='P',dither=Image.NONE,palette=Image.ADAPTIVE,colors=256)
                except IndexError:
                    break
        threads = []

        for number in range(0,total_threads):
            threads.append(Thread(target=worker, args=(number,total_threads)))

        for number in range(0,total_threads):
            threads[number].start()

        for number in range(0,total_threads):
            threads[number].join()

        for i in range(0,len(frames)):
            frames[i] = pil_frames[i]

        return frames

    def save_gif(self,frames,gif_path):
        frames[0].save(gif_path, format='GIF', append_images=frames[1:], save_all=True, subrectangles=True, duration=40, loop=0)

    
    #Selects a video from the videos folder and generates a gif from a random scene.
    def generate_random_gif(self,min_frames,max_frames,transition_threshold,min_dynamicness,threads,giffolder):
        searching = True
        gif_path = giffolder+'/'+K_GIF_FILENAME
        while searching:
            print('Searching for a scene')
            frames,dynamicness,text = self.analyze_and_get_frames(min_frames,max_frames,transition_threshold)
            print('The scene is '+str(dynamicness)+'% dynamic')

            if dynamicness < min_dynamicness:
                print('The scene does not have enough action, searching for another one')
            else:
                searching = False

        print('Converting frames')    
        frames = self.convert_frames_to_pil(frames,threads)

        print('Saving gif to disk')

        try:
            self.save_gif(frames,gif_path)
        except FileNotFoundError:
            os.mkdir(giffolder)
            self.save_gif(frames,gif_path)

        os.rename(gif_path, giffolder+'/'+text+'.gif')
        print('Done')


gif_generator = GifGenerator()
gif_generator.generate_random_gif(30,160,45,0.34,4,K_TEMP_DIR)
