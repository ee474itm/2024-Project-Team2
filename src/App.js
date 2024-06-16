import React, { useState } from 'react';
import './App.css';
import axios from 'axios';

function App() {
  const [story, setStory] = useState('');
  const [gender, setGender] = useState('');
  const [speed, setSpeed] = useState('');
  const [imageStyle, setImageStyle] = useState('');

  const generateVideo = async () => {
    try {
      const response = await axios.post('http://localhost:8000/generate-ssml', {
        story,
        gender,
        speed,
        imageStyle
      });

      const videoUrl = response.data.videoUrl;

      const videoOutput = document.getElementById('video-output');
      videoOutput.innerHTML = `
        <video width="640" height="480" controls>
          <source src="${videoUrl}" type="video/mp4"/>
        </video>
      `;

    } catch (error) {
      console.error('Error generating video', error);
    }
  };

  return (
    <div className="container">
      <div className="logo">
        <div className="image-class">
          <img src="/logo.png" alt='logo' width={25} height={25}/>
        </div>
        <h1>BEYOND THE PAGE</h1>
      </div>
      <div className="content">
        <div className="input-section">
          <div className="story-input">
            <h2>Input your Story</h2>
            <textarea
              value={story}
              onChange={(e) => setStory(e.target.value)}
              placeholder="Please input your story that you want to create as a multimedia book."
            ></textarea>
          </div>
          <div className="narration-style">
            <h2>Narration Style</h2>
            <div className="gender">
              <h3>Gender</h3>
              <label className={gender === 'male' ? 'selected' : ''}>
                <input
                  type="radio"
                  name="gender"
                  value="male"
                  checked={gender === 'male'}
                  onChange={() => setGender('male')}
                /> Male
              </label>
              <label className={gender === 'female' ? 'selected' : ''}>
                <input
                  type="radio"
                  name="gender"
                  value="female"
                  checked={gender === 'female'}
                  onChange={() => setGender('female')}
                /> Female
              </label>
            </div>
            <div className="speed">
              <h3>Speed</h3>
              <label className={speed === 'slow' ? 'selected' : ''}>
                <input
                  type="radio"
                  name="speed"
                  value="slow"
                  checked={speed === 'slow'}
                  onChange={() => setSpeed('slow')}
                /> slow
              </label>
              <label className={speed === 'medium' ? 'selected' : ''}>
                <input
                  type="radio"
                  name="speed"
                  value="medium"
                  checked={speed === 'medium'}
                  onChange={() => setSpeed('medium')}
                /> medium
              </label>
              <label className={speed === 'fast' ? 'selected' : ''}>
                <input
                  type="radio"
                  name="speed"
                  value="fast"
                  checked={speed === 'fast'}
                  onChange={() => setSpeed('fast')}
                /> fast
              </label>
            </div>
          </div>
          <div className="image-style">
            <h2>Image Style</h2>
            <div className="styles">
              <label className={imageStyle === 'Oil Painting' ? 'selected' : ''}>
                <input
                  type="radio"
                  name="imageStyle"
                  value="Oil Painting"
                  checked={imageStyle === 'Oil Painting'}
                  onChange={() => setImageStyle('Oil Painting')}
                /> Oil Painting
              </label>
              <label className={imageStyle === 'Watercolor' ? 'selected' : ''}>
                <input
                  type="radio"
                  name="imageStyle"
                  value="Watercolor"
                  checked={imageStyle === 'Watercolor'}
                  onChange={() => setImageStyle('Watercolor')}
                /> Watercolor
              </label>
              <label className={imageStyle === 'Gray-scale Pencil Sketch' ? 'selected' : ''}>
                <input
                  type="radio"
                  name="imageStyle"
                  value="Gray-scale Pencil Sketch"
                  checked={imageStyle === 'Gray-scale Pencil Sketch'}
                  onChange={() => setImageStyle('Gray-scale Pencil Sketch')}
                /> Gray-scale Pencil Sketch
              </label>
            </div>
          </div>
          <button onClick={generateVideo}>Generate Video</button>
        </div>
        <div className="image-class">
          <img src='/arrow_right.png' alt="arrow" width={50} height={50}/>
        </div>
        <div className="output-section">
          <h2>Generated Video</h2>
          <div id="video-output">
            <video width="640" height="480" controls>
              <source src="http://localhost:8000/output/output.mp4" type="video/mp4"/>
            </video>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;