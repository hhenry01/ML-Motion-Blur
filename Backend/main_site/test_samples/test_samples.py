import ffmpeg
vid = ffmpeg.probe("man_running.mp4")
print(vid["streams"])
metadata = vid["streams"][0]
print(metadata)
print(metadata["r_frame_rate"])