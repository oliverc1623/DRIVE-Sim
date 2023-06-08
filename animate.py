from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np
import os
import argparse
import cv2


def main(args):
    frames = os.listdir(args.frames_folder)
    frames.sort()
    image_frames = []
    for f in frames:
        img_name = args.frames_folder + '/' + f
        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_frames.append(img)

    fig, ax = plt.subplots(figsize=(image_frames[0].shape[1] / 100, image_frames[0].shape[0] / 100))
    ln = plt.imshow(image_frames[0])
    plt.axis('off')
    def init():
        ln.set_data(image_frames[0])
        return [ln]

    def update(frame):
        ln.set_array(frame)
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.margins(0, 0)
        return [ln]

    ani = FuncAnimation(fig, update, image_frames, init_func=init, interval=1000, blit=True)
    ani.save(f"{args.fname}.mp4", fps=30)

   
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Animator to visualize VISTA frames")
    parser.add_argument('--frames_folder')
    parser.add_argument('--fname', type=str)
    args = parser.parse_args()
    main(args)