
import math
import matplotlib.pyplot as plt



def is_nan(x):
    return type(x) is float and math.isnan(float(x))

import pandas
def pd_read_csv(csv_path):

    df = pandas.read_csv(csv_path)
    # print(df)
    return df

def fps_to_ms(fps: int) -> int:
    '''
    Convert FPS to a millisecond interval.
    Args:
        fps: Input FPS as integer.
    Returns:
        Interval in milliseconds as integer number.
    '''
    return math.floor((1 / fps) * 1000)


def plot_metrics(df,fig_name):
    model_list = df['model'].unique()
    # print('model_list:',model_list)
    font_size = 10

    for i in range(0, len(model_list)):
        label_list = df[df['model'] == model_list[i]]['branch'].tolist()
        ms_list = df[df['model'] == model_list[i]]['ms'].values
        fps_list = df[df['model'] == model_list[i]]['fps'].values
        map_list = df[df['model'] == model_list[i]]['mAP'].values
        # print('label_list', label_list, 'ms',ms_list, 'fps', fps_list, 'map', map_list)
        y_list = map_list
        t_list = []
        # print('fps_list[0]', fps_list[0])

        if fps_list[0] == -1:
            x_list = ms_list
        else:
            for j in fps_list:
                j = fps_to_ms(j)
                t_list.append(j)
            x_list = t_list

        plt.plot(x_list, y_list, marker='.', markersize=10)
        plt.title("yolov8 metrics")
        plt.xlabel('PyTorch FP16 RTX3080(ms/img)')  # x轴标题
        plt.ylabel('COCO Mask AP val')  # y轴标题
        for ms, map, label in zip(x_list, y_list, label_list):
            plt.text(ms, map, label, ha='center', va='bottom', fontsize=font_size)
    # legend
    plt.legend(model_list, loc='lower right')

    # save
    plt.savefig(fig_name, dpi=640)
    # show
    plt.show()

if __name__ == '__main__':
    csv_path = '/home/linxu/PycharmProjects/Yolov8_Efficient/log/yolo_model_data.csv'
    fig_name = 'plot_metrics.jpg'
    df = pd_read_csv(csv_path)
    plot_metrics(df, fig_name)