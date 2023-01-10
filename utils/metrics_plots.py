import matplotlib.pyplot as plt
import math

def fps_to_ms(fps: int) -> int:
    '''
    Convert FPS to a millisecond interval.
    Args:
        fps: Input FPS as integer.
    Returns:
        Interval in milliseconds as integer number.
    '''
    return math.floor((1 / fps) * 1000)


def plot_mult_metrics():
    '''
        Draw Metrics plots
        Args:

        Returns:

        '''
    # yolov5
    x1_1 = [6.3, 6.4, 8.2, 10.1, 12.1]
    y1_1 = [28.0, 37.4, 45.4, 49.0, 50.7]
    l1_1 = ['YOLOv5n', 'YOLOv5s', 'YOLOv5m', 'YOLOv5l', 'YOLOv5x']

    ###############################################
    # yolov6
    # 1fps=0.3048 m/s
    x2_1 = [fps_to_ms(779), fps_to_ms(339), fps_to_ms(175), fps_to_ms(98)]
    y2_1 = [37.5, 45.0, 50.0, 52.8]
    l2_1 = ['YOLOv6-N', 'YOLOv6-S', 'YOLOv6-M', 'YOLOv6-L']

    x2_2 = [fps_to_ms(228), fps_to_ms(98), fps_to_ms(47), fps_to_ms(26)]
    y2_2 = [44.9, 50.3, 55.2, 57.2]
    l2_2 = ['YOLOv6-N6', 'YOLOv6-S6', 'YOLOv6-M6', 'YOLOv6-L6']
    ###############################################

    # yolov7
    x3 = [fps_to_ms(161), fps_to_ms(114), fps_to_ms(84), fps_to_ms(56), fps_to_ms(44), fps_to_ms(36)]
    y3 = [51.4, 53.1, 54.9, 56.0, 56.6, 56.8]
    l3 = ['YOLOv7', 'YOLOv7-X', 'YOLOv7-W6', 'YOLOv7-E6', 'YOLOv7-D6', 'YOLOv7-E6E']

    ###############################################
    # yolov8
    x4_1 = [5.6, 5.7, 8.3, 13.1, 20.4]
    y4_1 = [37.3, 44.9, 50.2, 52.9, 53.9]
    l4_1 = ['YOLOv8n', 'YOLOv8s', 'YOLOv8m', 'YOLOv8l', 'YOLOv8x']

    x4_2 = [11.3, 11.4, 15.3, 16.8, 23.8]
    y4_2 = [30.7, 37.0, 40.6, 42.5, 43.2]
    l4_2 = ['YOLOv8n-seg', 'YOLOv8s-seg', 'YOLOv8m-seg', 'YOLOv8l-seg', 'YOLOv8x-seg']

    ###############################################
    # plot
    # yolov5
    plt.plot(x1_1, y1_1, marker='.', markersize=10)

    # yolov6
    plt.plot(x2_1, y2_1, marker='.', markersize=10)
    plt.plot(x2_2, y2_2, marker='.', markersize=10)

    # yolov7
    plt.plot(x3, y3, marker='.', markersize=10)

    # yolov8
    plt.plot(x4_1, y4_1, marker='.', markersize=10)
    plt.plot(x4_2, y4_2, marker='.', markersize=10)

    plt.title("yolov8 metrics")
    plt.xlabel('PyTorch FP16 RTX3080(ms/img)')  # x轴标题
    plt.ylabel('COCO Mask AP val')  # y轴标题

    for fps, map, label in zip(x1_1, y1_1, l1_1):
        plt.text(fps, map, label, ha='center', va='bottom', fontsize=6)
    for fps, map, label in zip(x2_1, y2_1, l2_1):
        plt.text(fps, map, label, ha='center', va='bottom', fontsize=6)
    for fps, map, label in zip(x2_2, y2_2, l2_2):
        plt.text(fps, map, label, ha='center', va='bottom', fontsize=6)
    for fps, map, label in zip(x3, y3, l3):
        plt.text(fps, map, label, ha='center', va='bottom', fontsize=6)
    for fps, map, label in zip(x4_1, y4_1, l4_1):
        plt.text(fps, map, label, ha='center', va='bottom', fontsize=6)
    for fps, map, label in zip(x4_2, y4_2, l4_2):
        plt.text(fps, map, label, ha='center', va='bottom', fontsize=6)

    # legend
    plt.legend(['YOLOv5', 'YOLOv6', 'YOLOv6-6', 'YOLOv7', 'YOLOv8', 'YOLOv8-seg'], loc='lower right')

    # save
    plt.savefig("plot_metrics.jpg",dpi=640)
    # show
    plt.show()

if __name__ == '__main__':
    plot_mult_metrics()
