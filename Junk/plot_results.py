import matplotlib.pyplot as plt
import numpy as np
import json

def plot_data(title, xs, ys, labels, labels2, x_axis_label, y_axis_label):
    for y_, label2 in zip(ys, labels2):
        plt.plot(xs, y_, label=f'{label2}')

    plt.ylabel(y_axis_label)
    plt.xlabel(x_axis_label)
    plt.legend()
    plt.title(title)
    plt.show()

def plot_data2(title, xs, ys, labels, labels2, x_axis_label, y_axis_label):
    for y_, label2 in zip(ys, labels2):
        for y, label in zip(y_, labels):
            plt.plot(xs, y_, label=f'{label2}-{label}')

    plt.ylabel(y_axis_label)
    plt.xlabel(x_axis_label)
    plt.legend()
    plt.title(title)
    plt.show()

def read_log_data(fps):
    all_runs_data = []
    for fp in fps:
        with open(f"{fp}", "r") as file:
            log_data = file.read()

        #turn it into valid json format
        log_data = log_data.replace("'", '"')
        log_data = log_data.replace("}\n{", '},\n{')
        log_data = f'[{log_data.strip()}]'

        json_data = json.loads(log_data)
        run_data = {}

        #Transpose data
        for entry in json_data:
            for key, value in entry.items():
                if key not in run_data:
                    run_data[key] = []
                run_data[key].append(value)

        all_runs_data.append(run_data)

    return all_runs_data

if __name__ == '__main__':
    title = ['U-Net', 'Ensemble']
    labels = ['IoU', 'BIoU', 'Score']
    labels2 = ['Validation', 'Train']
    runs = ['run_29_BCE_DICE_LOSS', 'run_32']
    fps = [f'team_template/src/runs/task_1/{run}/run.log' for run in runs]
    all_data = read_log_data(fps)
    key_pairs = {
        'testloss': ['trainloss', 'Loss'],
        'testscore': ['trainscore', 'Score'],
        'testiou': ['testbiou', 'testscore', 'trainiou', 'trainbiou','trainscore']
    }
    header = {'testloss': ['Loss', 'Loss'], 'testscore': ['Score', 'Score']}

    for i, data in enumerate(all_data):
        for key, value in data.items():
            if key == 'epoch':
                continue
            if key in key_pairs:
                if len(key_pairs[key]) == 2:
                    if title[i] == 'Ensemble':
                        plot_data(f'{title[i]}: {key_pairs[key][1]}', data['epoch'][:20],
                                  [data[key][:20], data[key_pairs[key][0]][:20]], header[key], ['Validation', 'Train'], 'Epochs',
                                  key_pairs[key][1])  # key_pairs[key][1], )
                    else:
                        plot_data(f'{title[i]}: {key_pairs[key][1]}', data['epoch'], [data[key], data[key_pairs[key][0]]], header[key], ['Validation', 'Train'],  'Epochs', key_pairs[key][1])#key_pairs[key][1], )
