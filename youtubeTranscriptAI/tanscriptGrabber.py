from youtube_transcript_api import YouTubeTranscriptApi

# Replace 'YOUR_VIDEO_ID' with the actual video ID
video_id = 'LCPXgoIt5nE'
transcript = YouTubeTranscriptApi.get_transcript(video_id)

for entry in transcript:
    print(entry['text'])
