# Face-Recognition Based Automatic Meeting Attendance Tracking System (自动点名系统)

This demo shows how to use Google's Facenet and other open-source software to build a meeting attendance tracking system. Given a video feed or a video file, and a set of photos of known people, this application will scan frames from the video input and check who attend the meeting and who are absent.

One of the technical problems is the false rate is pretty high for a single frame, for example, A was recognized as B in a single frame. To overcome this problem, a simple voting method is used. Basically, this application will lock few faces by using a face-tracking algorithm, and in the next N frames, only locked faces are recognized by Facenet. Out of N frames, if in M times, a face is recognized as person A, then A is believed to attend the meeting. 

# Requirements

* python
* argparse
* opencv (>=3.0)
* opencv-contrib-python
* numpy
* scipy
* tensorflow

## Facenet

* copied from `https://github.com/davidsandberg/facenet`
  * already included in this repo.

# Register Known Faces

* first, copy `data/known_faces/` to `facenet/src/align/tmp`
```
$ cd facenet
$ ./copy_known_faces.sh
```

* run `register.py`
```
$ python register.py ../model/facenet/20180402-114759 src/align/tmp/known_faces
```

* two files will be created which include the generated embeddings of each known faces.
```
$ ls *.npy`
embeddings.npy	label_strings.npy
```

# Figure out who attended and who didn't

* run `attend.py`

`$ python attend.py ../model/facenet/20180402-114759 ../data/test.mp4 --output out.avi --known_embeddings embeddings.npy --known_names label_strings.npy`

* debug information read as this:
```
[DEBUG] # 156 | 1 | 50 | 2 | {'A': 48, 'B': 2}
```
* `# 156`: frame seq. #
* `1`: Tracker ID. We use multiple trackers, each tracker track one person.
* `50`: total number of frames a face is successfully identified within the bbox tracked by tracker `#1`.
  * If this number reaches `FR_MAX_TEST (default=100)`, tracker `#1` will be stopped.
* `2`: total number of frames `mtcnn` failed to identify a face within the bbox tracked by tracker `#1`.
  * If this number reaches `FR_NO_FACE_THRESHOLD (default=60)`, tracker `#1` will be stopped.
* `{'A': 48, 'B': 2}`: out of `50` successfully identified faces, `48` was identified as `A`, `2` was identified as `B`.

* when all trackers are stopped, if a name appeared more than `FR_THRESHOLD (default is 60)` times, then this person is considered as `present` (in a class / meeting).

```
[DEBUG] # 438 | 0 | 86 | 60 | {'AAA': 86}
[DEBUG] # 438 | 1 | 100 | 2 | {'BBB': 98, 'CCC': 2}
[DEBUG] # 438 | 2 | 15 | 60 | {'DDD': 15}
[DEBUG] # 438 | 3 | 59 | 60 | {'EEE': 54, 'FFF': 5}
[DEBUG] finally identified name is AAA, with 86/100 confidence.
[DEBUG] finally identified name is BBB, with 98/100 confidence.
```
