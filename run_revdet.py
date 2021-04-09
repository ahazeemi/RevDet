
import argparse
import os
import algorithm
import evaluate_algorithm
import plot_results
import pickle


parser = argparse.ArgumentParser()

parser.add_argument(
    '--inputchains',
    default='redundancy_removed_chains/',
    type=str,
    help='Directory for redundancy removed input event chains'
)
parser.add_argument(
    '--outputchains',
    default='output_chains/',
    type=str,
    help='Directory for output event chains'
)
parser.add_argument(
    '--perdaydata',
    default='per_day_data/',
    type=str,
    help='Directory for per day data'
)
parser.add_argument(
    '--plotgraph',
    action='store_true',
    help='Run on multiple inputs to generate a graph of f_measure against window size'
)
parser.add_argument(
    '--plotactivechains',
    action='store_true',
    help='Plot active event chains per day compared with ground truth grounds.'
)
parser.add_argument(
    '--birch_thresh',
    default=2.3,
    type=float,
    help='Threshold for the birch algorithm. Default 2.3.'
)
parser.add_argument(
    '--window_size',
    default=8,
    type=int,
    help='Window size for revdet algorithm. Default 8.'
)


def delete_files(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)


def main(args):

    birch_thresh = args.birch_thresh
    window_size = args.window_size
    input_directory = args.inputchains
    output_directory = args.outputchains
    per_day_data = args.perdaydata

    print("Running algorithm")

    """if args.plotgraph:
        result = []
        precision = []
        recall = []
        f_measure = []
        highest_f1_score = 0

        window_sizes = range(2, 21, 2)
        for window_size in window_sizes:
            print("Running for window Size", window_size)
            algorithm.run(per_day_data, output_directory,
                          birch_thresh, window_size)
            temp_result = evaluate_algorithm.run(
                input_directory, output_directory)

            precision.append(temp_result[0])
            recall.append(temp_result[1])
            f_measure.append(temp_result[2])

            if temp_result[3] > highest_f1_score:
                highest_f1_score = temp_result[3]
                result = temp_result
                result.insert(0, birch_thresh)
                result.insert(0, window_size)
            delete_files(output_directory)

        with open('windowsizes', 'wb') as fp:
            pickle.dump(window_sizes, fp)

        with open('precision', 'wb') as fp:
            pickle.dump(precision, fp)

        with open('recall', 'wb') as fp:
            pickle.dump(recall, fp)

        with open('fmeasure', 'wb') as fp:
            pickle.dump(f_measure, fp)

        plot_results.plot_score_with_window_size()

        print('Highest F1 Score for these parameters: Window Size: {}, Birch Threshold: {}. Result-  Precision: {:.2f}, Recall: {:.2f}, F1-Score: {:.2f}, NMI: {:.2f}, ARI: {:.2f}'.format(
            result[0], result[1], result[2], result[3], result[4], result[5], result[6]))"""

    #else:
    algorithm.run(per_day_data, output_directory,
                      birch_thresh, window_size)
    """result = evaluate_algorithm.run(input_directory, output_directory)
        print('Window Size: {}, Birch Threshold: {}, Precision: {:.2f}, Recall: {:.2f}, F1-Score: {:.2f}, NMI: {:.2f}, ARI: {:.2f}'.format(
            window_size, birch_thresh, result[0], result[1], result[2], result[3], result[4]))

        if args.plotactivechains:
            plot_results.plot_active_events(input_directory, output_directory)"""


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
