# strabismus-identification

actively modified code is 
1. run_with_webcam.py <- Main gaze-estimation software
   (currently, data is shown in terminal and (start/stop) is controlled by (running the software/typing "control+c" in the terminal)
3. webcam.py <- running the streamlit page
4. test_run_with_webcam.py <- combining the two

Other files are meant to be tests for combining frontend service and gaze-estimation software so that non-coders can use it.
this code aims to combine david-wb's gaze-estimation software with a streamlit webcam to create a strabismus-identifying service
output data is in pitch and yew, and degrees. 

todo:
more tests needed on real-life samples to increase credibility
