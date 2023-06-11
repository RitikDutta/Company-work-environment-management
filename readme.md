<h2>Employee Productivity Monitoring System</h2>
<p>This project aims to use facial recognition technology to monitor employee productivity and track specific activities in the workplace.</p>

![demo GIF](https://i.imgur.com/byI1SLS.gif)


<h3>Features</h3>
<ul>
  <li>Detects whether the right person is sitting in front of the camera</li>
  <li>Tracks key points on the face (such as eyes, mouth) to identify specific activities</li>
  <li>Classifies the type of activity the user is performing (e.g. taking a phone call, looking away from the screen, sleeping, looking tired)</li>
  <li>Lightweight and runs in the browser</li>
  <li>Sends summary data about activities to a central server in JSON format, without transmitting any images or videos</li>
  <li>Automatic Database Records: The system automatically adds records of employee activities to the database, eliminating the need for manual data entry. This ensures accurate and up-to-date tracking of employee actions and enables comprehensive reporting and analysis of productivity metrics.
</li>
</ul>
<h3>Requirements</h3>
<ul>
  <li>A computer with a webcam</li>
  <li>A modern web browser (such as Chrome or Firefox)</li>
</ul>
<h3>Installation</h3>
<h4>
Option 1: Local Installation
</h4>
<ol>
  <li>Clone this repository to your local machine</li>
  <li>Install the necessary dependencies by running <code>pip install -r requirements.txt</code></li>
  <li>Run the app by executing <code>python app.py</code></li>
  <li>Open your web browser and navigate to <code>http://localhost:8080</code></li>
</ol>
<h4>
Option 2: GitHub Codespaces
</h4>
<ol>
<li>Open the project in GitHub Codespaces.</li>
<li>Update the apt package manager and install FFmpeg:</li>
  <code>sudo apt update
sudo apt-get install ffmpeg</code>
<li>Install the necessary dependencies by running <code>pip install -r requirements.txt</code></li>
<li>Run the application by executing <code>python app.py</code></li>
<li>Open your web browser and navigate to http://localhost:8080</li>
</ol>
<h3>Technologies Used</h3>
<ul>
  <li><a href="http://dlib.net/">dlib</a> or <a href="https://github.com/ipazc/mtcnn">MTCNN</a> for facial recognition and keypoint detection</li>
  <li><a href="https://en.wikipedia.org/wiki/Machine_learning">Machine learning</a> or <a href="https://en.wikipedia.org/wiki/Deep_learning">deep learning</a> for activity classification</li>
  <li><a href="https://flask.palletsprojects.com/">Flask</a> for the web server</li>
</ul>
<h3>Privacy</h3>
<p>We take the privacy of our employees seriously. No images or videos of users are transmitted to the central server - only summary data about their activities is sent. All data is handled in accordance with relevant privacy laws and regulations.</p> 
