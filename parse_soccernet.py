#!/usr/bin/env python3

import os
import argparse
import json
from collections import defaultdict

from SoccerNet.utils import getListGames

from util.io import load_json
from util.dataset import read_fps, get_num_frames

def read_fps(video_frame_dir):
    # Append the suffix if necessary
    if not video_frame_dir.endswith('_224p'):
        video_frame_dir += '_224p'

    fps_file = os.path.join(video_frame_dir, 'fps.txt')
    if not os.path.exists(fps_file):
        raise FileNotFoundError(f"fps.txt not found in {video_frame_dir}")
   
    with open(fps_file) as fp:
        fps = float(fp.read().strip())
   
    return fps

def get_num_frames(video_frame_dir):
    # Append the suffix if necessary
    if not video_frame_dir.endswith('_224p'):
        video_frame_dir += '_224p'

    max_frame = -1
    for img_file in os.listdir(video_frame_dir.replace('\\', '/')):
        if img_file.endswith('.jpg'):
            frame = int(os.path.splitext(img_file)[0])
            max_frame = max(frame, max_frame)
    return max_frame + 1

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('label_dir', type=str,
                        help='Path to the SoccerNetV2 labels')
    parser.add_argument('frame_dir', type=str,
                        help='Path to extracted video frames')
    parser.add_argument('-o', '--out_dir', type=str,
                        help='Path to output parsed dataset')
    return parser.parse_args()

def load_split(split):
    if split == 'val':
        split = 'valid'

    videos = []
    for entry in getListGames(split):
        entry = entry.replace('\\', '/')  # Replace backslashes with forward slashes
        print(f"Entry: {entry}")  # Debugging statement
        try:
            league, season, game = entry.split('/')
            videos.append((league, season, game))
        except ValueError:
            print(f"Skipping entry with unexpected format: {entry}")
    return videos

def get_label_names(labels):
    return {e['label'] for v in labels for e in v['events']}

def main(label_dir, frame_dir, out_dir):
    print("Label directory:", label_dir)
    print("Frame directory:", frame_dir)
    print("Output directory:", out_dir)

    labels_by_split = defaultdict(list)
    for split in ['train', 'val', 'test', 'challenge']:
        videos = load_split(split)
        for video in videos:
            league, season, game = video

            video_label_path = os.path.join(
                label_dir, league, season, game, 'Labels-v2.json')

            if split != 'challenge':
                try:
                    video_labels = load_json(video_label_path)
                except FileNotFoundError:
                    print(f"Warning: Labels-v2.json not found for {video_label_path}")
                    continue  # Skip to the next video

            else:
                video_labels = {'annotations': []}

            num_events = 0
            for half in (1, 2):
                for suffix in ['', '_224p']:  # Check both without suffix and with _224p suffix
                    video_frame_dir = os.path.join(
                        frame_dir, league, season, game, f"{half}{suffix}")

                    try:
                        sample_fps = read_fps(video_frame_dir)
                        num_frames = get_num_frames(video_frame_dir)
                    except FileNotFoundError:
                        print(f"Warning: Directory not found for {video_frame_dir}")
                        continue  # Skip to the next half

                    video_id = f'{league}/{season}/{game}/{half}'

                    half_events = []
                    for label in video_labels['annotations']:
                        lhalf = int(label['gameTime'].split(' - ')[0])
                        if half == lhalf:
                            adj_frame = float(label['position']) / 1000 * sample_fps
                            half_events.append({
                                'frame': int(adj_frame),
                                'label': label['label'],
                                'comment': f"{label['team']}; {label['visibility']}"
                            })

                            if adj_frame >= num_frames:
                                print(f'Warning: Label past end: {video_id} -- {num_frames} < {int(adj_frame)} -- {label["label"]}')
                    num_events += len(half_events)
                    half_events.sort(key=lambda x: x['frame'])

                    labels_by_split[split].append({
                        'video': video_id,
                        'num_frames': num_frames,
                        'num_events': len(half_events),
                        'events': half_events,
                        'fps': sample_fps,
                        'width': 398,
                        'height': 224
                    })

    train_classes = get_label_names(labels_by_split['train'])
    assert train_classes == get_label_names(labels_by_split['test'])
    assert train_classes == get_label_names(labels_by_split['val'])

    print('Classes:', sorted(train_classes))

    for split, labels in labels_by_split.items():
        print(f'{split} : {len(labels)} videos : {sum(len(l["events"]) for l in labels)} events')
        labels.sort(key=lambda x: x['video'])

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        class_path = os.path.join(out_dir, 'class.txt')
        with open(class_path, 'w') as fp:
            fp.write('\n'.join(sorted(train_classes)))

        for split, labels in labels_by_split.items():
            out_path = os.path.join(out_dir, f'{split}.json')
            with open(out_path, 'w') as fp:
                json.dump(labels, fp, indent=2, sort_keys=True)

    print('Done!')

if __name__ == '__main__':
    main(**vars(get_args()))
