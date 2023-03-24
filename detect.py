import json
from pathlib import Path
from typing import Dict

import click
import cv2
from tqdm import tqdm


def detect(img_path: str) -> Dict[str, int]:

    red_count = 0;
    yellow_count = 0;
    green_count  = 0;
    violet_count  = 0;
    #
    img1 = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img1, (1200,700), interpolation=cv2.INTER_AREA)
    # Zdefiniowanie zakresów wartości kolorów w modelu HSV
    frame_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    red_range = cv2.inRange(frame_HSV, (0, 140, 60), (5, 255, 255))
    yellow_range = cv2.inRange(frame_HSV, (20, 233, 129), (25, 255, 255))
    green_range = cv2.inRange(frame_HSV, (38, 80, 10), (90, 255, 255))
    violet_range = cv2.inRange(frame_HSV, (130, 70, 70), (166, 255, 255))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated_mask = cv2.dilate(red_range, kernel, iterations=2)
    # Wykryj kontury cukierków na masce
    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Zlicz  red cukierki
    for contour in contours:
        # Jeśli kontur ma wystarczająco duży obszar, to uznaj go za cukierek
        if cv2.contourArea(contour) > 150:
            red_count += 1

    dilated_mask = cv2.dilate(yellow_range, kernel, iterations=2)
    # Wykryj kontury cukierków na masce
    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Zlicz  yellow cukierki
    for contour in contours:
        # Jeśli kontur ma wystarczająco duży obszar, to uznaj go za cukierek
        if cv2.contourArea(contour) > 150:
            yellow_count += 1
    dilated_mask = cv2.dilate(green_range, kernel, iterations=2)
    # Wykryj kontury cukierków na masce
    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Zlicz  green cukierki
    for contour in contours:
        # Jeśli kontur ma wystarczająco duży obszar, to uznaj go za cukierek
        if cv2.contourArea(contour) > 150:
            green_count += 1
    dilated_mask = cv2.dilate(violet_range, kernel, iterations=2)
    # Wykryj kontury cukierków na masce
    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Zlicz  violet cukierki
    for contour in contours:
        # Jeśli kontur ma wystarczająco duży obszar, to uznaj go za cukierek
        if cv2.contourArea(contour) > 150:
            violet_count += 1

    red = red_count
    yellow = yellow_count
    green = green_count
    purple = violet_count

    return {'red': red, 'yellow': yellow, 'green': green, 'purple': purple}


@click.command()
@click.option('-p', '--data_path', help='Path to data directory', type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option('-o', '--output_file_path', help='Path to output file', type=click.Path(dir_okay=False, path_type=Path), required=True)
def main(data_path: Path, output_file_path: Path):
    img_list = data_path.glob('*.jpg')

    results = {}

    for img_path in tqdm(sorted(img_list)):
        fruits = detect(str(img_path))
        results[img_path.name] = fruits

    with open(output_file_path, 'w') as ofp:
        json.dump(results, ofp)


if __name__ == '__main__':
    main()
