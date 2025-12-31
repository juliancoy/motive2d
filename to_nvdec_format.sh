ffmpeg -i ../P1090533.MOV \
  -c:v libx265 \
  -preset fast \
  -pix_fmt yuv420p \
  -crf 16 \
  -c:a copy \
  P1090533_main8_hevc_fast.mkv

# Pose coordinates are stored in P1090533_pose_coords.txt for the overlay pipeline.
