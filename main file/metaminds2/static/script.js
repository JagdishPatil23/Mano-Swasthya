// Include face-api.js library
const script = document.createElement('script');
script.src = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.0.2/dist/face-api.min.js';
script.onload = init;
document.head.appendChild(script);

async function init() {
    await faceapi.nets.tinyFaceDetector.loadFromUri('/models');
    await faceapi.nets.faceLandmark68Net.loadFromUri('/models');
    await faceapi.nets.faceRecognitionNet.loadFromUri('/models');
    await faceapi.nets.faceExpressionNet.loadFromUri('/models');
    
    const video = document.getElementById('video');

    navigator.mediaDevices.getUserMedia({ video: {} })
        .then(stream => {
            video.srcObject = stream;
        });

    video.addEventListener('play', () => {
        const canvas = faceapi.createCanvasFromMedia(video);
        document.getElementById('camera-container').append(canvas);
        const displaySize = { width: video.width, height: video.height };
        faceapi.matchDimensions(canvas, displaySize);
        
        setInterval(async () => {
            const detections = await faceapi.detectAllFaces(video, new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks().withFaceExpressions();
            const resizedDetections = faceapi.resizeResults(detections, displaySize);
            canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
            faceapi.draw.drawDetections(canvas, resizedDetections);
            faceapi.draw.drawFaceLandmarks(canvas, resizedDetections);
            faceapi.draw.drawFaceExpressions(canvas, resizedDetections);

            if (detections[0]) {
                document.getElementById('emotion').innerText = `Detected Emotion: ${getDominantExpression(detections[0].expressions)}`;
            }
        }, 100);
    });
}

function getDominantExpression(expressions) {
    const maxConfidence = Math.max(...Object.values(expressions));
    return Object.keys(expressions).find(expression => expressions[expression] === maxConfidence);
}
