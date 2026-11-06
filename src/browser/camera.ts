export async function startCamera(video: HTMLVideoElement, onEnded: () => void): Promise<void> {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 1280, height: 720 },
    audio: false,
  });
  for (const track of stream.getVideoTracks()) {
    track.addEventListener('ended', onEnded);
  }
  video.srcObject = stream;
  await video.play();
}
