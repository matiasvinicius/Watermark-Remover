import sys
from Static_Remover import Remover as StaticRemover

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise TypeError("Wrong number of arguments")

    path = sys.argv[1]
    remover = StaticRemover(path)
    remover.watermark_remover()