import sys
from Static_Remover import Remover as ImageRemover
from Dinamic_Remover import Remover as VideoRemover

if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise TypeError("Bad input (enter something like python3 main.py test.jpeg image or python3 main.py test.mp4 video 30)")

    path = sys.argv[1]

    if sys.argv[2] == 'image':
        remover = ImageRemover(path)
        remover.remove_watermark()

    elif sys.argv[2] == 'video': 
        if len(sys.argv) == 3:
            try:
                framerate = int(input('Enter the frame rate (example 30, 60): '))
            except:
                raise TypeError("Enter an integer value")
        else: framerate = int(sys.argv[3])

        print(framerate)
        remover = VideoRemover(path, framerate)
        remover.remove_watermark()
    
    else:
        raise TypeError('Type video or image (python3 main.py test.jpeg image or python3 main.py test.mp4 video')