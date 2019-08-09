# coding=utf-8
"""Performs face detection in realtime.

Based on code from https://github.com/shanren7/real_time_face_recognition
"""
# MIT License
#
# Copyright (c) 2017 François Gervais
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import sys
import time
import cv2
import face
import operator
from collections import namedtuple
import numpy as np
from random import randint

debug = True
MARGIN = 20
FR_MAX_TEST  = 100
FR_THRESHOLD = 60
FR_NO_FACE_THRESHOLD = 60
DISTANCE_THRESHOLD = 1
MAX_TRACKER  = 4
confirmed_names = []
confidence = []

def add_overlays(frame, frame_count, frame_rate):
    height=35
    startx = 10
    starty = 30

    text = "frame: " + str(frame_count) + " @ " + str(frame_rate) + " fps"
    cv2.putText(frame, text, (startx, starty),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                thickness=2, lineType=2)

    for i, name in enumerate(confirmed_names):
        text = name + ", " + str(confidence[i]) + "%"
        cv2.putText(frame, text, (startx, starty + (i+1) * height),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                    thickness=2, lineType=2)

def next_faces_to_track(faces):
    #faces.sort(key=operator.attrgetter('distance'))
    faces.sort(key=lambda x: x.distance, reverse=False)
    tracked = []
    for f in faces:
        #if debug:
            #print("name: {}, distance: {}".format(f.name, f.distance))
        if f.name in confirmed_names:
            continue
        else:
            tracked.append(f)
            if(len(tracked) >= MAX_TRACKER):
                break
    return tracked

class GotoNextFrame(Exception): 
    pass

def main(args):
    frame_interval = 3  # Number of frames after which to run face detection
    fps_display_interval = 5  # seconds
    frame_rate = 0
    frame_count = 0

    video_capture = cv2.VideoCapture(args.video_file)
    if video_capture is None:
        print("Cannot open video file")
        return

    #cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    frame_width = int(video_capture.get(3))
    frame_height = int(video_capture.get(4))
    if args.output is not None:
        out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

    face_recognition = face.Recognition(args.model, args.known_embeddings, args.known_names)

    start_time = time.time()

    if args.debug:
        print("Debug enabled")
        face.debug = True

    stopped = False 
    total_frame = 0

    colors = []
    for i in range(MAX_TRACKER):
        colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))

    while video_capture.isOpened() and not stopped:
        # step 1: find a face to track
        ret, frame = video_capture.read()
        total_frame += 1

        faces = face_recognition.identify(frame)
        tracked_faces = next_faces_to_track(faces)
        multiTracker = cv2.MultiTracker_create()
    
        nrof_tracker = 0
        for tracked in tracked_faces:
            print("[DEBUG] tracked: {}, bbox: {}".format(tracked.name, tracked.bounding_box))
            bb = tracked.bounding_box
            bb[0] = max(0, tracked.bounding_box[0] - MARGIN)
            bb[1] = max(0, tracked.bounding_box[1] - MARGIN)
            bb[2] = tracked.bounding_box[2] - tracked.bounding_box[0] + MARGIN
            bb[3] = tracked.bounding_box[3] - tracked.bounding_box[1] + MARGIN

            tracker = cv2.TrackerKCF_create()
            multiTracker.add(tracker, frame, tuple(bb))
            nrof_tracker += 1
        
        print("[DEBUG] totally {} tracker is created".format(nrof_tracker))

        # step 2: start track and recognize face
        fr_total = [0 for i in range(nrof_tracker)]
        fr_no_faces = [0 for i in range(nrof_tracker)]
        fr_names = [{} for i in range(nrof_tracker)]
        while video_capture.isOpened() and not stopped:
            ret, frame = video_capture.read()
            frame_count += 1
            total_frame += 1

            # track the person frame by frame
            ret, bboxes = multiTracker.update(frame)

            all_done = True
            for t_id, bbox in enumerate(bboxes):
                # this face is already identified, or too many times cannot detect face, 
                # stop this tracker
                if fr_total[t_id] >= FR_MAX_TEST or fr_no_faces[t_id] >= FR_NO_FACE_THRESHOLD:
                    continue

                all_done = False
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                w  = int(bbox[2])
                h  = int(bbox[3])
                p1 = (x1, y1)
                p2 = (x1 + w, y1 + h)
                cv2.rectangle(frame, p1, p2, colors[t_id], 2, 1)

                # recognize the face in the tracked window every X frames
                if (total_frame % frame_interval) == 0:
                    tracked_img = frame[y1:y1+h, x1:x1+w]
                    new_faces = face_recognition.identify(tracked_img)
                    try:
                        if(len(new_faces) == 0):
                            # no face detected
                            #print("[DEBUG] no face detected")
                            fr_no_faces[t_id] += 1
                            raise GotoNextFrame

                        if (len(new_faces) > 1):
                            # more than one face detected in the tracking window
                            new_faces.sort(key=operator.attrgetter('distance'))
                        
                        # only pick one and the most likely one
                        new_face = new_faces[0]
                        if new_face.distance > DISTANCE_THRESHOLD:
                            # no face detected
                            #print("[DEBUG] face detected but very weak")
                            fr_no_faces[t_id] += 1
                            raise GotoNextFrame

                        fr_total[t_id] += 1

                        if new_face.name in fr_names[t_id]:
                            fr_names[t_id][new_face.name] += 1
                        else:
                            fr_names[t_id][new_face.name] = 1

                    except GotoNextFrame:
                        #print("[DEBUG] captured exception here.")
                        pass

            # Check our current fps
            end_time = time.time()
            if (end_time - start_time) > fps_display_interval:
                frame_rate = int(frame_count / (end_time - start_time))
                start_time = time.time()
                frame_count = 0

            add_overlays(frame, total_frame, frame_rate)
            cv2.imshow('Video', frame)
            if (total_frame % frame_interval) == 0:
                if out is not None:
                    out.write(frame)
                for i in range(nrof_tracker):
                    print("[DEBUG] # {} | {} | {} | {} | {}"
                        .format(total_frame, i, fr_total[i], fr_no_faces[i], fr_names[i]))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                stopped = True
                break
                    
            if all_done:
                break

        # done with the identification of the tracked person
        multiTracker.clear()
        for i in range(nrof_tracker):
            max_count = 0
            final = ""
            for name in fr_names[i]:
                #print("[DEBUG] name: {}, count: {}".format(name, fr_names[i][name]))
                if max_count < fr_names[i][name]:
                    max_count = fr_names[i][name]
                    final = name

            if max_count > FR_THRESHOLD:
                print("[DEBUG] finally identified name is {}, with {}/100 confidence."
                    .format(final, max_count))
                tracked.name = final
                tracked.confidence = max_count
                if final not in confirmed_names:
                    confirmed_names.append(final)
                    confidence.append(max_count)
                    print("[DEBUG] Now total {} faces identified.".format(len(confirmed_names)))
            else:
                print("[DEBUG] confidence too low, attempt failed this tracker.")

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('video_file', type=str,
                        help='input video filename.')
    parser.add_argument('-e', '--known_embeddings', type=str,
        help='The pre-generated embeddings for known/registered faces.', default="embeddings.npy")
    parser.add_argument('-n', '--known_names', type=str,
        help='The name associated with registered faces.', default="label_strings.npy")
    parser.add_argument('--output', type=str,
                        help='optional output video filename (in .avi format)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable some debug outputs.')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
