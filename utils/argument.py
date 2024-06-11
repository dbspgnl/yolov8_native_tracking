'''
인자값 처리 
'''
import argparse


def get_argment():
    parser = argparse.ArgumentParser(description="Traffic Flow Analysis with YOLOv8")
    parser.add_argument("--source_weights_path", required=True, type=str, help="Path to the source weights file")
    parser.add_argument("--source_video_path", required=True, type=str, help="Path to the source video file")
    parser.add_argument("--target_video_path", required=True, type=str, help="Path to the target video file (output)")
    parser.add_argument("--detect_zone_areas", default=None, type=str, help="Coordinate array for detection area")
    parser.add_argument("--detect_none_areas", default=None, type=str, help="Coordinate array for non-detection area")
    parser.add_argument("--detect_zone_show", default=False, type=str, help="detection area Show")
    parser.add_argument("--detect_none_show", default=False, type=str, help="non-detection Show")
    parser.add_argument("--count_line", default=None, type=str, help="Line zone coordinates for entry/exit counting processing")
    parser.add_argument("--count_show", default=False, type=str, help="Line zone Count Label Show")
    parser.add_argument("--show", default=False, type=str, help="OpenCV Show")
    parser.add_argument("--file", default=False, type=str, help="Make File")
    parser.add_argument("--pts", default="10.0", type=str, help="FFmpeg PTS set value")
    parser.add_argument("--threads", default="0", type=str, help="FFmpeg threads set value")
    parser.add_argument("--keep_frame", default="3", type=str, help="How long will the frame be maintained?")
    parser.add_argument("--xml_output_path", default="result.xml", type=str, help="Where to save the xml file")
    parser.add_argument("--tracker_yaml", default ="bytetrack.yaml", type=str, help="Where tracker setting yaml file")
    parser.add_argument("--padding", default ="50", type=int, help="Video padding size")
    args = parser.parse_args()
    detect_zone_show_bool_true = (args.detect_zone_show == 'true')
    detect_none_show_bool_true = (args.detect_none_show == 'true')
    count_show_bool_true = (args.count_show == 'true')
    show_bool_true = (args.show == 'true')
    file_bool_true = (args.file == 'true')
    
    return {
        "source_weights_path": args.source_weights_path,
        "source_video_path": args.source_video_path,
        "target_video_path": args.target_video_path,
        "detect_zone_areas": args.detect_zone_areas,
        "detect_none_areas": args.detect_none_areas,
        "detect_zone_show": detect_zone_show_bool_true,
        "detect_none_show": detect_none_show_bool_true,
        "count_line": args.count_line,
        "is_count_show": count_show_bool_true,
        "is_show": show_bool_true,
        "is_file": file_bool_true,
        "pts": args.pts,
        "threads": args.threads,
        "keep_frame": args.keep_frame,
        "xml_output_path": args.xml_output_path,
        "tracker_yaml": args.tracker_yaml,
        "padding": args.padding,
    }
    