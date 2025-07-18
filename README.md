#AIRCAT
My project, Artificial Intelligence Real-time Collision Avoidance Technology, or AIRCAT, uses NVIDIA’s detectnet software on their Jetson Orin Nano to identify birds (hence the name)  while piloting an aircraft. With a webcam hooked up, a warning message will be printed to the terminal when a bird is detected on the webcam feed, and could be easily modified to connect to a speaker in the future. This was made in an attempt to assist pilots avoid birdstrikes, a very frequent but hard to combat issue in aviation. The FAA estimates that in 2023, there were over 19,600 reported wildlife/birdstrikes, and it is estimated that 50-80% of these strikes go unreported, making it a very real problem.

#The Algorithm
My algorithm works by using the aforementioned detectnet model trained on a large amount of images of birds taken from Google’s Open Images Dataset. This dataset greatly improves functionality, as Google Open Images is renowned for high quality annotations, complexity, and diversity of the images used. This is paired with a simple python script that opens the paired webcam and continuously checks if there are any birds present, and prints a warning message in the circumstance that it detects one.

#Running this project
1. Have processing hardware and corresponding software for AI models and training (This was ran on a Jetson Orin Nano)
2. Detectnet trained on Google Open Images Dataset involving Birds, additional training optional
3. Plug webcam into processor
4. Run AIRCAT.py
5. Look at the terminal and observe if the model detects any birds in the webcam feed

