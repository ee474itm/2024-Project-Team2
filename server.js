const express = require('express');
const cors = require('cors');
const { spawn } = require('child_process');

const app = express();
const port = 8000;

app.use(cors());
app.use(express.json());
app.use(express.static('public'));

app.post('/generate-ssml', async (req, res) => {
    const { story, gender, speed, imageStyle } = req.body;

    const pythonProcess = spawn('python3', ['generating_video.py', story, gender, speed, imageStyle]);

    pythonProcess.stdout.on('data', (data) => {
        console.log(`stdout: ${data}`);
    });

    pythonProcess.stderr.on('data', (data) => {
        console.error(`stderr: ${data}`);
    });

    pythonProcess.on('close', (code) => {
        console.log(`child process exited with code ${code}`);
        res.json({ videoUrl: 'http://localhost:8000/output/output.mp4' });
    });
});

app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});
