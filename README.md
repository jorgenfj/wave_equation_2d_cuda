* make       : builds the executable 'wave_2d'
* ./wave\_2d : fills the 'data/' directory with stored time steps
* make plot  : converts saved time steps to png files under 'images/', using gnuplot. Runs faster if launched with e.g. 4 threads (make -j4 plot).
* make movie : converts collection of png files under 'images' into an mp4 movie file, using ffmpeg
* make check : builds both executeables and compares their output
