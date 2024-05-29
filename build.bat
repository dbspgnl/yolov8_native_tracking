@echo off
:: pyinstaller -w 절대 금지
START /wait /b cmd /c "pyinstaller -n=wiz_yolo --icon=setup/logo.ico main.py"
START /wait /b cmd /c mkdir "dist/wiz_yolo/_internal/ultralytics"
START /wait /b cmd /c xcopy "env/Lib/site-packages/ultralytics" "dist/wiz_yolo/_internal/ultralytics" /E
START /wait /b cmd /c copy best_total2.pt "dist/wiz_yolo/best_total.pt"
START /wait /b cmd /c copy "setup\start.bat" "dist\wiz_yolo\start.bat"
START /wait /b cmd /c echo D:/yolo/build/wiz_yolo/wiz_yolo.exe --source_weights_path "D:/yolo/build/wiz_yolo/best_total.pt" --source_video_path "rtmp://localhost:1935/live/mist1" --target_video_path "rtmp://localhost:1935/live/mist5" --zone_in_polygons "[]" --count_line "[]" --count_show "false" --show "false" --file "false" --pts "10.0" --threads "4" --xml_output_path "result.xml" > dist/wiz_yolo/mist5.bat
START /wait /b cmd /c copy "setup\check.exe" "dist\wiz_yolo\check.exe"